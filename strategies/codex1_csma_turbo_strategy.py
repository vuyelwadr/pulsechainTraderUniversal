"""Adaptive deep-dip mean reversion with profit locking.

This strategy builds on the existing C-SMA reversion concept but keeps
positions open through the bulk of the crash recovery rally.  The key
ideas are:

* Demand a violent drawdown versus the 2‑day SMA **and** the recent
  rolling high before entering.
* Use RSI as a sanity check so we only deploy after true capitulation.
* Once a position is open, allow it to breathe.  We first look for a
  quick reversion back towards the SMA, but if price launches into a
  sustained rally we switch to a trailing stop that only exits after a
  large give-back from the post-entry high.  This captures the large
  multi‑day bounces that dominate performance in HEX.

The resulting behaviour is more selective on entries and dramatically
extends the average holding period, which reduces swap fee drag.  The
parameters below were tuned with respect to the real swap cost cache in
``swap_cost_cache.json`` using a 1,000 DAI account and a 5k trade size
bucket.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from .base_strategy import BaseStrategy


def _sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=window).mean()


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def _rsi(series: pd.Series, window: int) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    avg_gain = up.rolling(window, min_periods=window).mean()
    avg_loss = down.rolling(window, min_periods=window).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    rsi = 100 - (100 / (1 + rs))
    return rsi


class Codex1CSMATurboStrategy(BaseStrategy):
    """Deep crash reversion with adaptive profit lock and trailing exit."""

    def __init__(self, parameters: Dict | None = None):
        defaults = {
            'n_sma': 576,              # ≈ 2 days on 5m bars
            'entry_drop': 0.30,        # % below SMA required to enter
            'entry_dd_lookback': 2880, # ≈ 10 days, drawdown filter
            'entry_dd_threshold': -0.62,
            'rsi_period': 14,
            'rsi_max': 32.0,
            'profit_target': 1.60,     # 160% of entry price triggers trailer
            'trailing_pct': 0.30,      # give-back from run-up before exit
            'hard_stop': 0.40,         # protective stop from entry
            'max_hold_bars': 8064,     # ≈ 28 days
            'min_hold_bars_for_exit': 1152,  # wait ~4 days before time exit
            'timeframe_minutes': 5,
        }
        if parameters:
            defaults.update(parameters)
        super().__init__('Codex1CSMATurboStrategy', defaults)

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        if not self.validate_data(data):
            return data

        df = data.copy()
        price = df['price'] if 'price' in df else df['close']
        df['price'] = price

        n_sma = int(self.parameters['n_sma'])
        rsi_period = int(self.parameters['rsi_period'])
        entry_dd_lookback = int(self.parameters['entry_dd_lookback'])

        df['sma'] = _sma(price, n_sma)
        df['ema_fast'] = _ema(price, max(96, n_sma // 2))
        df['rsi'] = _rsi(price, rsi_period)
        df['rolling_max'] = price.rolling(entry_dd_lookback, min_periods=entry_dd_lookback).max()
        df['drawdown_from_high'] = price / df['rolling_max'] - 1.0
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
        ema_fast = df['ema_fast'].to_numpy(float)
        rsi = df['rsi'].to_numpy(float)
        drawdown = df['drawdown_from_high'].to_numpy(float)

        entry_drop = float(self.parameters['entry_drop'])
        entry_dd_threshold = float(self.parameters['entry_dd_threshold'])
        rsi_max = float(self.parameters['rsi_max'])
        profit_target = float(self.parameters['profit_target'])
        trailing_pct = float(self.parameters['trailing_pct'])
        hard_stop = float(self.parameters['hard_stop'])
        max_hold_bars = int(self.parameters['max_hold_bars'])
        min_hold_bars_for_exit = int(self.parameters['min_hold_bars_for_exit'])

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
            dd = drawdown[i]

            if np.isnan(px) or np.isnan(sm) or np.isnan(rs):
                continue

            if not in_position:
                if (
                    px <= sm * (1.0 - entry_drop)
                    and rs <= rsi_max
                    and (not np.isnan(dd) and dd <= entry_dd_threshold)
                ):
                    df.iat[i, buy_col] = True
                    df.iat[i, strength_col] = 1.0
                    in_position = True
                    entry_price = px
                    highest_price = px
                    entry_index = i
                    trailing_active = False
            else:
                highest_price = max(highest_price, px)

                # Hard stop to limit catastrophic collapse
                if px <= entry_price * (1.0 - hard_stop):
                    df.iat[i, sell_col] = True
                    df.iat[i, strength_col] = 1.0
                    in_position = False
                    trailing_active = False
                    continue

                # Profit target reached – activate trailing logic
                if px >= entry_price * (1.0 + profit_target):
                    trailing_active = True

                # EMA breakdown as secondary confirmation exit
                if trailing_active and px <= highest_price * (1.0 - trailing_pct):
                    df.iat[i, sell_col] = True
                    df.iat[i, strength_col] = 1.0
                    in_position = False
                    trailing_active = False
                    continue

                if px < ema_fast[i] and trailing_active:
                    df.iat[i, sell_col] = True
                    df.iat[i, strength_col] = 1.0
                    in_position = False
                    trailing_active = False
                    continue

                if (
                    max_hold_bars > 0
                    and i - entry_index >= max(max_hold_bars, min_hold_bars_for_exit)
                ):
                    df.iat[i, sell_col] = True
                    df.iat[i, strength_col] = 1.0
                    in_position = False
                    trailing_active = False
                    continue

                if (
                    not trailing_active
                    and min_hold_bars_for_exit > 0
                    and i - entry_index >= min_hold_bars_for_exit
                    and px >= sm
                    and px > entry_price * 1.05
                ):
                    df.iat[i, sell_col] = True
                    df.iat[i, strength_col] = 1.0
                    in_position = False
                    trailing_active = False

        return df
