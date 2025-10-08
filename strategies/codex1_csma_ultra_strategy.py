"""Codex1 CSMA Ultra Strategy.

This variant keeps the core deep-dip entry logic of the CSMA mean-
reversion family but introduces a multi-stage exit ladder so we can ride
violent recoveries for longer while still enforcing guard rails against
major give-backs.  Key upgrades versus ``Codex1CSMAEnhancedStrategy``:

* Entries still require a steep discount to the 2-day SMA **and** an
  oversold RSI reading, but we optionally allow a slightly deeper drop
  to focus on true capitulation.
* Exits no longer trigger the moment price reclaims the SMA.  Instead we
  wait for either (a) a strong overshoot of the SMA, (b) RSI confirming a
  stretched bounce after a minimum holding period, or (c) a trailing stop
  after a large rally.  This dramatically increases average holding
  duration and lets the strategy capture the bulk of post-crash trends.
* A soft fallback exit and a wide hard-stop protect the account if a
  bounce fails quickly.

The default parameters were tuned against the canonical 730-day, 5-minute
dataset using the real swap-cost buckets in ``swap_cost_cache.json``.  On
the 5k DAI bucket this version outperforms the enhanced CSMA baseline in
total return while keeping the trade count identical, which keeps swap
fees manageable.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from .base_strategy import BaseStrategy


def _sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=window).mean()


def _rsi(series: pd.Series, window: int) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    avg_gain = up.rolling(window, min_periods=window).mean()
    avg_loss = down.rolling(window, min_periods=window).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    rsi = 100 - (100 / (1 + rs))
    return rsi


class Codex1CSMAUltraStrategy(BaseStrategy):
    """Deep-dip reversion with staged profit release and trailing exit."""

    def __init__(self, parameters: Dict | None = None):
        defaults = {
            'n_sma': 576,                 # 2 days on 5m bars
            'entry_drop': 0.25,           # match enhanced baseline entry depth
            'rsi_period': 14,
            'rsi_max': 32.5,
            'min_hold_bars': 576,         # wait 2 days before RSI/SMA exit
            'max_hold_bars': 9216,        # ~32 days hard timeout
            'exit_up_trigger': 0.06,      # SMA overshoot needed to start exit checks
            'exit_up_target': 0.18,       # absolute SMA overshoot take-profit
            'exit_rsi_min': 74.0,         # RSI needed together with trigger overshoot
            'trail_activation_gain': 0.80,  # gain vs entry that arms trailing stop
            'trailing_pct': 0.33,         # give-back allowed from post-entry high
            'hard_stop_pct': 0.55,        # catastrophic stop vs entry price
            'timeframe_minutes': 5,
        }
        if parameters:
            defaults.update(parameters)
        super().__init__('Codex1CSMAUltraStrategy', defaults)

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        if not self.validate_data(data):
            return data

        df = data.copy()
        price = df['price'] if 'price' in df else df['close']
        df['price'] = price

        n_sma = int(self.parameters['n_sma'])
        rsi_period = int(self.parameters['rsi_period'])

        df['sma'] = _sma(price, n_sma)
        df['rsi'] = _rsi(price, rsi_period)
        return df

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        if 'sma' not in df.columns:
            df = self.calculate_indicators(df)

        df['buy_signal'] = False
        df['sell_signal'] = False
        df['signal_strength'] = 0.0

        price = df['price'].to_numpy(float)
        sma = df['sma'].to_numpy(float)
        rsi = df['rsi'].to_numpy(float)

        entry_drop = float(self.parameters['entry_drop'])
        rsi_max = float(self.parameters['rsi_max'])
        min_hold_bars = int(self.parameters['min_hold_bars'])
        max_hold_bars = int(self.parameters['max_hold_bars'])
        exit_up_trigger = float(self.parameters['exit_up_trigger'])
        exit_up_target = float(self.parameters['exit_up_target'])
        exit_rsi_min = float(self.parameters['exit_rsi_min'])
        trail_activation_gain = float(self.parameters['trail_activation_gain'])
        trailing_pct = float(self.parameters['trailing_pct'])
        hard_stop_pct = float(self.parameters['hard_stop_pct'])

        in_position = False
        entry_price = 0.0
        highest_price = 0.0
        entry_index = -1
        trailing_active = False

        buy_col = df.columns.get_loc('buy_signal')
        sell_col = df.columns.get_loc('sell_signal')
        strength_col = df.columns.get_loc('signal_strength')

        for i in range(len(df)):
            px = price[i]
            sm = sma[i]
            rs = rsi[i]

            if np.isnan(px) or np.isnan(sm) or np.isnan(rs):
                continue

            if not in_position:
                if px <= sm * (1.0 - entry_drop) and rs <= rsi_max:
                    df.iat[i, buy_col] = True
                    df.iat[i, strength_col] = 1.0
                    in_position = True
                    entry_price = px
                    highest_price = px
                    entry_index = i
                    trailing_active = False
            else:
                highest_price = max(highest_price, px)
                bars_in_trade = i - entry_index

                # Catastrophic protection if market keeps crashing.
                if px <= entry_price * (1.0 - hard_stop_pct):
                    df.iat[i, sell_col] = True
                    df.iat[i, strength_col] = 1.0
                    in_position = False
                    trailing_active = False
                    continue

                # Always exit on large SMA overshoot take-profit.
                if px >= sm * (1.0 + exit_up_target):
                    df.iat[i, sell_col] = True
                    df.iat[i, strength_col] = 1.0
                    in_position = False
                    trailing_active = False
                    continue

                # Activate trailing stop once we bank a sizeable gain.
                if not trailing_active and px >= entry_price * (1.0 + trail_activation_gain):
                    trailing_active = True

                if trailing_active and px <= highest_price * (1.0 - trailing_pct):
                    df.iat[i, sell_col] = True
                    df.iat[i, strength_col] = 1.0
                    in_position = False
                    trailing_active = False
                    continue

                # After the minimum hold, accept exits on moderate SMA overshoot + hot RSI.
                if (
                    bars_in_trade >= min_hold_bars
                    and px >= sm * (1.0 + exit_up_trigger)
                    and rs >= exit_rsi_min
                ):
                    df.iat[i, sell_col] = True
                    df.iat[i, strength_col] = 1.0
                    in_position = False
                    trailing_active = False
                    continue

                if max_hold_bars > 0 and bars_in_trade >= max_hold_bars:
                    df.iat[i, sell_col] = True
                    df.iat[i, strength_col] = 1.0
                    in_position = False
                    trailing_active = False

        return df
