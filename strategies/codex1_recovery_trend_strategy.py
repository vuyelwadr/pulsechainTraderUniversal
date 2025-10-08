"""Codex1 Recovery Trend Strategy.

Hybrid of deep-dip mean reversion and momentum trend following.  The
strategy first looks for the classic CSMA capitulation setup (price well
below a 2-day SMA with oversold RSI).  After entry it either:

* Takes quick profits when price snaps back to the SMA; or
* If medium-term trend flips positive, transitions into a trend mode and
  rides the recovery using a wide trailing stop.

This keeps the attractive crash-recovery behaviour of CSMARevert while
capturing much more upside when HEX enters multi-week rallies.
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


class Codex1RecoveryTrendStrategy(BaseStrategy):
    """Deep crash entry with trend mode continuation."""

    def __init__(self, parameters: Dict | None = None):
        defaults = {
            'n_sma': 576,
            'entry_drop': 0.25,
            'rsi_period': 14,
            'rsi_max': 30.0,
            'early_exit_up': 0.05,
            'trend_fast': 288,
            'trend_slow': 1152,
            'trend_gap': 0.012,
            'trend_trail_pct': 0.35,
            'hard_stop': 0.35,
            'max_hold_bars': 12096,
            'timeframe_minutes': 5,
        }
        if parameters:
            defaults.update(parameters)
        super().__init__('Codex1RecoveryTrendStrategy', defaults)

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        if not self.validate_data(data):
            return data

        df = data.copy()
        price = df['price'] if 'price' in df else df['close']
        df['price'] = price

        n_sma = int(self.parameters['n_sma'])
        trend_fast = int(self.parameters['trend_fast'])
        trend_slow = int(self.parameters['trend_slow'])
        rsi_period = int(self.parameters['rsi_period'])

        df['sma'] = _sma(price, n_sma)
        df['ema_fast'] = _ema(price, trend_fast)
        df['ema_slow'] = _ema(price, trend_slow)
        df['rsi'] = _rsi(price, rsi_period)
        df['ema_fast_slope'] = df['ema_fast'] - df['ema_fast'].shift(12)
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
        ema_fast = df['ema_fast'].to_numpy(float)
        ema_slow = df['ema_slow'].to_numpy(float)
        ema_fast_slope = df['ema_fast_slope'].to_numpy(float)

        entry_drop = float(self.parameters['entry_drop'])
        rsi_max = float(self.parameters['rsi_max'])
        early_exit_up = float(self.parameters['early_exit_up'])
        trend_gap = float(self.parameters['trend_gap'])
        trend_trail_pct = float(self.parameters['trend_trail_pct'])
        hard_stop = float(self.parameters['hard_stop'])
        max_hold_bars = int(self.parameters['max_hold_bars'])

        in_position = False
        mode_trend = False
        entry_price = 0.0
        highest_price = 0.0
        entry_index = -1

        buy_col = df.columns.get_loc('buy_signal')
        sell_col = df.columns.get_loc('sell_signal')
        strength_col = df.columns.get_loc('signal_strength')

        for i in range(len(df)):
            px = price[i]
            sm = sma[i]
            rs = rsi[i]
            ef = ema_fast[i]
            es = ema_slow[i]
            slope = ema_fast_slope[i]

            if np.isnan(px) or np.isnan(sm) or np.isnan(rs) or np.isnan(ef) or np.isnan(es):
                continue

            if not in_position:
                if px <= sm * (1.0 - entry_drop) and rs <= rsi_max:
                    df.iat[i, buy_col] = True
                    df.iat[i, strength_col] = 1.0
                    in_position = True
                    mode_trend = False
                    entry_price = px
                    highest_price = px
                    entry_index = i
            else:
                highest_price = max(highest_price, px)

                if px <= entry_price * (1.0 - hard_stop):
                    df.iat[i, sell_col] = True
                    df.iat[i, strength_col] = 1.0
                    in_position = False
                    mode_trend = False
                    continue

                if not mode_trend:
                    if px >= sm * (1.0 + early_exit_up):
                        if (
                            ef > es
                            and slope > 0
                            and px >= es * (1.0 + trend_gap)
                        ):
                            mode_trend = True
                        else:
                            df.iat[i, sell_col] = True
                            df.iat[i, strength_col] = 1.0
                            in_position = False
                            continue
                else:
                    if px <= highest_price * (1.0 - trend_trail_pct):
                        df.iat[i, sell_col] = True
                        df.iat[i, strength_col] = 1.0
                        in_position = False
                        mode_trend = False
                        continue

                    if ef <= es:
                        df.iat[i, sell_col] = True
                        df.iat[i, strength_col] = 1.0
                        in_position = False
                        mode_trend = False
                        continue

                    if max_hold_bars > 0 and i - entry_index >= max_hold_bars:
                        df.iat[i, sell_col] = True
                        df.iat[i, strength_col] = 1.0
                        in_position = False
                        mode_trend = False

        return df
