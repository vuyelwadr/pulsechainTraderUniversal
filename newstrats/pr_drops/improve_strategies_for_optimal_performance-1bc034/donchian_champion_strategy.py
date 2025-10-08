"""Donchian breakout "champion" strategies derived from newstrats notes."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from .base_strategy import BaseStrategy


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=max(1, span), adjust=False).mean()


def _atr_ratio(df: pd.DataFrame, period: int) -> pd.Series:
    high = df['high'] if 'high' in df.columns else df['price']
    low = df['low'] if 'low' in df.columns else df['price']
    close = df['price'] if 'price' in df.columns else df['close']
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(period, min_periods=period).mean()
    return (atr / close).fillna(0.0)


class DonchianChampionStrategy(BaseStrategy):
    """Long-only Donchian breakout with EMA-confirmed exits and optional trail."""

    def __init__(self, parameters: Dict | None = None):
        defaults = {
            'entry_days': 11.0,
            'exit_days': 2.0,
            'ema_exit_days': 3.0,
            'trailing_drawdown_pct': 0.0,
            'timeframe_minutes': 5,
        }
        if parameters:
            defaults.update(parameters)
        super().__init__('DonchianChampionStrategy', defaults)

    def _window_bars(self, days: float) -> int:
        timeframe = float(self.parameters.get('timeframe_minutes', 5))
        bars = int(round(days * 24 * 60 / timeframe))
        return max(1, bars)

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        if not self.validate_data(data):
            return data

        df = data.copy()
        price = df['price'] if 'price' in df else df['close']
        df['price'] = price

        entry_bars = self._window_bars(float(self.parameters['entry_days']))
        exit_bars = self._window_bars(float(self.parameters['exit_days']))
        ema_span = self._window_bars(float(self.parameters['ema_exit_days']))

        df['donchian_high'] = price.rolling(entry_bars, min_periods=entry_bars).max().shift(1)
        df['donchian_low'] = price.rolling(exit_bars, min_periods=exit_bars).min().shift(1)
        df['ema_exit'] = _ema(price, ema_span)
        return df

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        if 'donchian_high' not in df.columns:
            df = self.calculate_indicators(df)

        df['buy_signal'] = False
        df['sell_signal'] = False
        df['signal_strength'] = 0.0

        price = df['price'].to_numpy(float)
        highs = df['donchian_high'].to_numpy(float)
        lows = df['donchian_low'].to_numpy(float)
        ema_exit = df['ema_exit'].to_numpy(float)
        trail_pct = float(self.parameters['trailing_drawdown_pct'])

        in_position = False
        peak_price = 0.0

        for i in range(len(df)):
            px = price[i]
            hi = highs[i]
            lo = lows[i]
            ema_val = ema_exit[i]

            if np.isnan(px) or np.isnan(hi):
                continue

            if not in_position:
                if px >= hi:
                    df.iat[i, df.columns.get_loc('buy_signal')] = True
                    df.iat[i, df.columns.get_loc('signal_strength')] = 1.0
                    in_position = True
                    peak_price = px
            else:
                peak_price = max(peak_price, px)
                exit_donchian = (not np.isnan(lo)) and px <= lo and (not np.isnan(ema_val)) and px < ema_val
                exit_trail = trail_pct > 0 and peak_price > 0 and px <= peak_price * (1.0 - trail_pct)
                if exit_donchian or exit_trail:
                    df.iat[i, df.columns.get_loc('sell_signal')] = True
                    df.iat[i, df.columns.get_loc('signal_strength')] = 1.0
                    in_position = False
                    peak_price = 0.0

        return df


class DonchianChampionAggressiveStrategy(DonchianChampionStrategy):
    """Champion v2 with default 20% trailing stop."""

    def __init__(self, parameters: Dict | None = None):
        defaults = {
            'entry_days': 11.0,
            'exit_days': 2.0,
            'ema_exit_days': 3.0,
            'trailing_drawdown_pct': 0.20,
            'timeframe_minutes': 5,
        }
        if parameters:
            defaults.update(parameters)
        super().__init__(defaults)


class DonchianChampionDynamicStrategy(DonchianChampionStrategy):
    """Champion v4 with dynamic drawdown based on ATR ratio."""

    def __init__(self, parameters: Dict | None = None):
        defaults = {
            'entry_days': 11.0,
            'exit_days': 2.0,
            'ema_exit_days': 3.0,
            'dd_base': 0.16,
            'dd_k': 0.5,
            'gain_weight': 0.10,
            'dd_min': 0.10,
            'dd_max': 0.45,
            'atr_days': 1.0,
            'entry_buffer_frac': 0.0,
            'timeframe_minutes': 5,
        }
        if parameters:
            defaults.update(parameters)
        super().__init__(defaults)

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        df = super().calculate_indicators(data)
        atr_period = self._window_bars(float(self.parameters['atr_days']))
        df['atr_ratio'] = _atr_ratio(df, atr_period)
        return df

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        if 'atr_ratio' not in df.columns:
            df = self.calculate_indicators(df)

        df['buy_signal'] = False
        df['sell_signal'] = False
        df['signal_strength'] = 0.0

        price = df['price'].to_numpy(float)
        highs = df['donchian_high'].to_numpy(float)
        lows = df['donchian_low'].to_numpy(float)
        ema_exit = df['ema_exit'].to_numpy(float)
        atr_ratio = df['atr_ratio'].to_numpy(float)

        dd_base = float(self.parameters['dd_base'])
        dd_k = float(self.parameters['dd_k'])
        gain_weight = float(self.parameters.get('gain_weight', 0.0))
        dd_min = float(self.parameters['dd_min'])
        dd_max = float(self.parameters['dd_max'])
        entry_buffer = float(self.parameters.get('entry_buffer_frac', 0.0))

        in_position = False
        peak_price = 0.0
        entry_price = 0.0

        for i in range(len(df)):
            px = price[i]
            hi = highs[i]
            lo = lows[i]
            ema_val = ema_exit[i]

            if np.isnan(px) or np.isnan(hi):
                continue

            if not in_position:
                if px >= hi * (1.0 + entry_buffer):
                    df.iat[i, df.columns.get_loc('buy_signal')] = True
                    df.iat[i, df.columns.get_loc('signal_strength')] = 1.0
                    in_position = True
                    peak_price = px
                    entry_price = px
            else:
                peak_price = max(peak_price, px)
                dd_dyn = dd_base + dd_k * (atr_ratio[i] if not np.isnan(atr_ratio[i]) else 0.0)
                if entry_price > 0 and peak_price > entry_price and gain_weight > 0:
                    gain = (peak_price / entry_price) - 1.0
                    if gain > 0:
                        dd_dyn += gain_weight * gain
                dd_dyn = min(max(dd_dyn, dd_min), dd_max)
                exit_donchian = (not np.isnan(lo)) and px <= lo and (not np.isnan(ema_val)) and px < ema_val
                exit_drawdown = peak_price > 0 and px <= peak_price * (1.0 - dd_dyn)
                if exit_donchian or exit_drawdown:
                    df.iat[i, df.columns.get_loc('sell_signal')] = True
                    df.iat[i, df.columns.get_loc('signal_strength')] = 1.0
                    in_position = False
                    peak_price = 0.0
                    entry_price = 0.0

        return df


class DonchianChampionSupremeCodex1Strategy(DonchianChampionDynamicStrategy):
    """Codex1-tuned Donchian dynamic variant with higher net return."""

    def __init__(self, parameters: Dict | None = None):
        defaults = {
            'entry_days': 11.0,
            'exit_days': 2.0,
            'ema_exit_days': 3.0,
            'dd_base': 0.14,
            'dd_k': 0.40,
            'gain_weight': 0.12,
            'dd_min': 0.08,
            'dd_max': 0.45,
            'atr_days': 1.0,
            'entry_buffer_frac': 0.0,
            'timeframe_minutes': 5,
        }
        if parameters:
            defaults.update(parameters)
        super().__init__(defaults)
