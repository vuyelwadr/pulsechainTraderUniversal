"""Hybrid mean-reversion + trend breakout strategy (V2)."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from .base_strategy import BaseStrategy


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=max(1, span), adjust=False).mean()


def _rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    avg_gain = up.rolling(period, min_periods=period).mean()
    avg_loss = down.rolling(period, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    return 100 - (100 / (1 + rs))


class HybridV2Strategy(BaseStrategy):
    """Combined deep-dip swing entry + breakout trend riding with adaptive exits."""

    def __init__(self, parameters: Dict | None = None):
        defaults = {
            'timeframe_minutes': 5,
            'n_sma': 576,
            'entry_drop': 0.25,
            'rsi_period': 14,
            'rsi_max': 30.0,
            'base_exit_up': 0.05,
            'z_mult': 0.03,
            'exit_cap': 0.20,
            'max_hold_bars': 720,
            'ema_fast': 96,
            'ema_slow': 288,
            'gap_min': 0.012,
            'n_break': 96,
            'trail_uptrend': 0.22,
        }
        if parameters:
            defaults.update(parameters)
        super().__init__('HybridV2Strategy', defaults)

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        if not self.validate_data(data):
            return data

        df = data.copy()
        price = df['price'] if 'price' in df else df['close']
        high = df['high'] if 'high' in df else price
        low = df['low'] if 'low' in df else price

        df['price'] = price
        timeframe = float(self.parameters['timeframe_minutes'])
        n_sma = int(self.parameters['n_sma'])
        df['sma'] = price.rolling(n_sma, min_periods=n_sma).mean()
        df['sma_std'] = price.rolling(n_sma, min_periods=n_sma).std(ddof=0)
        df['z_score'] = (price - df['sma']) / (df['sma_std'] + 1e-12)
        df['rsi'] = _rsi(price, int(self.parameters['rsi_period']))

        ema_fast = int(self.parameters['ema_fast'])
        ema_slow = int(self.parameters['ema_slow'])
        df['ema_fast'] = _ema(price, ema_fast)
        df['ema_slow'] = _ema(price, ema_slow)
        slope_lb = max(2, ema_fast // 2)
        df['ema_fast_slope'] = df['ema_fast'] - df['ema_fast'].shift(slope_lb)
        df['ema_gap'] = (df['ema_fast'] - df['ema_slow']) / (df['ema_slow'] + 1e-12)

        break_bars = int(self.parameters['n_break'])
        df['donchian_high'] = high.rolling(break_bars, min_periods=break_bars).max()
        df['donchian_low'] = low.rolling(break_bars, min_periods=break_bars).min()

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
        z_score = df['z_score'].to_numpy(float)
        ema_fast = df['ema_fast'].to_numpy(float)
        ema_slow = df['ema_slow'].to_numpy(float)
        ema_slope = df['ema_fast_slope'].to_numpy(float)
        ema_gap = df['ema_gap'].to_numpy(float)
        donchian_high = df['donchian_high'].to_numpy(float)
        donchian_low = df['donchian_low'].to_numpy(float)

        entry_drop = float(self.parameters['entry_drop'])
        rsi_max = float(self.parameters['rsi_max'])
        base_exit_up = float(self.parameters['base_exit_up'])
        z_mult = float(self.parameters['z_mult'])
        exit_cap = float(self.parameters['exit_cap'])
        max_hold = int(self.parameters['max_hold_bars'])
        gap_min = float(self.parameters['gap_min'])
        trail = float(self.parameters['trail_uptrend'])

        in_position = False
        peak_price = 0.0
        entry_index = None
        entry_mode = None

        for i in range(len(df)):
            px = price[i]
            if np.isnan(px):
                continue

            sm = sma[i]
            if np.isnan(sm):
                continue

            is_uptrend = (
                not np.isnan(ema_fast[i])
                and not np.isnan(ema_slow[i])
                and ema_fast[i] > ema_slow[i]
                and ema_slope[i] > 0
                and ema_gap[i] >= gap_min
                and not np.isnan(donchian_high[i])
            )

            mean_revert_entry = (
                not is_uptrend
                and px <= sm * (1.0 - entry_drop)
                and rsi[i] <= rsi_max
            )
            breakout_entry = is_uptrend and px >= donchian_high[i]

            if not in_position:
                if breakout_entry or mean_revert_entry:
                    df.iat[i, df.columns.get_loc('buy_signal')] = True
                    df.iat[i, df.columns.get_loc('signal_strength')] = 1.0
                    in_position = True
                    peak_price = px
                    entry_index = i
                    entry_mode = 'trend' if breakout_entry else 'mean_revert'
            else:
                peak_price = max(peak_price, px)
                should_exit = False

                if entry_mode == 'trend':
                    if trail > 0 and px <= peak_price * (1.0 - trail):
                        should_exit = True
                else:
                    z_val = max(0.0, z_score[i] if not np.isnan(z_score[i]) else 0.0)
                    dynamic_exit_up = base_exit_up + min(exit_cap - base_exit_up, z_mult * z_val)
                    exit_level = sm * (1.0 + dynamic_exit_up)
                    time_exit = entry_index is not None and (i - entry_index) >= max_hold and px >= sm
                    if px >= exit_level or time_exit:
                        should_exit = True

                if should_exit:
                    df.iat[i, df.columns.get_loc('sell_signal')] = True
                    df.iat[i, df.columns.get_loc('signal_strength')] = 1.0
                    in_position = False
                    peak_price = 0.0
                    entry_index = None
                    entry_mode = None

        return df

