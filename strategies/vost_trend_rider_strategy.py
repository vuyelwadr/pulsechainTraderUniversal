"""VOST Trend Rider Strategy.

Long-only trend follower that relies on the Volatility Optimized SuperTrend
(VOST) to define the dominant regime, then buys pullbacks that respect the
dynamic stop line. All signals are derived from real HEX/DAI price data.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from .base_strategy import BaseStrategy
from .vost_indicator import VOSTResult, compute_vost


class VOSTTrendRiderStrategy(BaseStrategy):
    """Pullback entries during bullish VOST regimes with tight risk controls."""

    def __init__(self, parameters: Dict | None = None):
        defaults = {
            'atr_period': 14,
            'base_multiplier': 2.35,
            'vol_period': 48,
            'vol_smooth': 240,
            'vol_ratio_floor': 0.65,
            'vol_ratio_cap': 2.4,
            'multiplier_power': 1.1,
            'pullback_buffer_pct': 0.02,  # tolerate 2% excursion around VOST line
            'max_pullback_from_high': 0.22,
            'vol_entry_cap': 2.0,
            'min_trend_duration': 6,
            'min_regime_bars': 3,
            'exit_buffer_pct': 0.01,
            'trail_lookback': 288,  # 1 day of 5m bars
            'trail_drop_pct': 0.18,
            'hard_stop_pct': 0.25,
            'cooldown_bars': 24,
            'timeframe_minutes': 5,
        }
        if parameters:
            defaults.update(parameters)
        super().__init__('VOSTTrendRiderStrategy', defaults)

    @staticmethod
    def _calc_atr(df: pd.DataFrame, period: int) -> pd.Series:
        high = df['high']
        low = df['low']
        close = df['close']
        prev_close = close.shift(1)
        tr = pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
        return tr.ewm(span=max(2, period), adjust=False).mean()

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        if not self.validate_data(data):
            return data

        df = data.copy()

        vost: VOSTResult | None = compute_vost(
            df,
            atr_period=int(self.parameters['atr_period']),
            base_multiplier=float(self.parameters['base_multiplier']),
            vol_period=int(self.parameters['vol_period']),
            vol_smooth=int(self.parameters['vol_smooth']),
            vol_ratio_floor=float(self.parameters['vol_ratio_floor']),
            vol_ratio_cap=float(self.parameters['vol_ratio_cap']),
            multiplier_power=float(self.parameters['multiplier_power']),
        )
        if vost is None:
            return df

        df['vost_line'] = vost.line
        df['vost_trend'] = vost.trend
        df['vost_upper'] = vost.upper
        df['vost_lower'] = vost.lower
        df['vost_multiplier'] = vost.multiplier
        df['vost_vol_ratio'] = vost.vol_ratio

        # Track pullback depth vs recent swing high to avoid catching deep crashes.
        lookback = max(12, int(self.parameters['trail_lookback']))
        df['rolling_max_close'] = df['close'].rolling(lookback).max().ffill()
        df['pullback_from_high'] = 1.0 - (df['close'] / df['rolling_max_close']).clip(upper=1.0)

        # Trend duration: consecutive bars where VOST stays bullish/bearish.
        trend_duration = np.zeros(len(df), dtype=int)
        for i in range(1, len(df)):
            if df['vost_trend'].iat[i] > 0 and df['vost_trend'].iat[i - 1] > 0:
                trend_duration[i] = trend_duration[i - 1] + 1
            elif df['vost_trend'].iat[i] < 0 and df['vost_trend'].iat[i - 1] < 0:
                trend_duration[i] = trend_duration[i - 1] + 1
            else:
                trend_duration[i] = 1 if df['vost_trend'].iat[i] != 0 else 0
        df['trend_duration'] = trend_duration

        df['atr'] = self._calc_atr(df, int(self.parameters['atr_period']))

        return df

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        if 'vost_trend' not in df.columns:
            df = self.calculate_indicators(df)

        df['buy_signal'] = False
        df['sell_signal'] = False
        df['signal_strength'] = 0.0

        if 'vost_trend' not in df.columns:
            return df

        pullback_buffer = float(self.parameters['pullback_buffer_pct'])
        vol_entry_cap = float(self.parameters['vol_entry_cap'])
        min_trend_duration = int(self.parameters['min_trend_duration'])
        max_pullback_from_high = float(self.parameters['max_pullback_from_high'])
        exit_buffer = float(self.parameters['exit_buffer_pct'])
        trail_drop_pct = float(self.parameters['trail_drop_pct'])

        vost_line = df['vost_line'].to_numpy()
        trend = df['vost_trend'].to_numpy()
        close = df['close'].to_numpy()
        vol_ratio = df.get('vost_vol_ratio', pd.Series(1.0, index=df.index)).to_numpy()
        pullback_from_high = df['pullback_from_high'].to_numpy()
        trend_duration = df['trend_duration'].to_numpy()

        buy_flags = np.zeros(len(df), dtype=bool)
        sell_flags = np.zeros(len(df), dtype=bool)
        strengths = np.zeros(len(df))

        in_position = False
        peak_price = 0.0
        entry_price = 0.0
        hard_stop_pct = float(self.parameters['hard_stop_pct'])
        cooldown_counter = 0
        cooldown_bars = int(self.parameters['cooldown_bars'])

        for i in range(len(df)):
            current_trend = trend[i]
            prev_trend = trend[i - 1] if i > 0 else 0

            if cooldown_counter > 0:
                cooldown_counter -= 1

            if not in_position:
                reentry_cross = (
                    current_trend > 0
                    and close[i] > vost_line[i]
                    and (i > 0 and close[i - 1] <= vost_line[i - 1])
                )
                trend_flip_up = current_trend > 0 and prev_trend <= 0
                allow_entry = False
                if trend_flip_up and vol_ratio[i] <= vol_entry_cap:
                    allow_entry = pullback_from_high[i] <= max_pullback_from_high
                elif reentry_cross and vol_ratio[i] <= vol_entry_cap:
                    allow_entry = (
                        pullback_from_high[i] <= max_pullback_from_high
                        and trend_duration[i] >= min_trend_duration
                    )

                if allow_entry and cooldown_counter == 0:
                    buy_flags[i] = True
                    in_position = True
                    entry_price = close[i]
                    peak_price = close[i]
                    pullback_depth = 0.0 if close[i] == 0 else abs(close[i] - vost_line[i]) / close[i]
                    strengths[i] = np.clip(1.0 - pullback_depth / max(pullback_buffer, 1e-4), 0.0, 1.0)
                    continue
            else:
                peak_price = max(peak_price, close[i])
                exit_due_to_trend = current_trend < 0
                exit_due_to_breach = close[i] < vost_line[i] * (1.0 - exit_buffer)
                exit_due_to_trail = peak_price > 0 and close[i] < peak_price * (1.0 - trail_drop_pct)
                exit_due_to_stop = entry_price > 0 and close[i] <= entry_price * (1.0 - hard_stop_pct)

                if exit_due_to_trend or exit_due_to_breach or exit_due_to_trail or exit_due_to_stop:
                    sell_flags[i] = True
                    strengths[i] = 1.0
                    in_position = False
                    peak_price = 0.0
                    entry_price = 0.0
                    cooldown_counter = cooldown_bars

        df['buy_signal'] = buy_flags
        df['sell_signal'] = sell_flags
        df['signal_strength'] = strengths

        return df
