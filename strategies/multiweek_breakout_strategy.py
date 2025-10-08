"""Multi-week breakout trend strategy.

Focuses on rare, strong upside expansions to keep turnover low and pay the high
swap costs only when momentum is broad-based.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from .base_strategy import BaseStrategy


class MultiWeekBreakoutStrategy(BaseStrategy):
    """Enter on multi-week breakout with regime confirmation; exit on breakdown or trail."""

    def __init__(self, parameters: Dict | None = None):
        defaults = {
            'lookback_breakout': 7200,  # ~3.5 weeks of 5m bars
            'confirmation_window': 576,  # 2 days
            'exit_lookback': 576,
            'trail_drawdown_pct': 0.26,
            'min_hold_bars': 2304,        # 8 days
            'volume_window': 576,
            'volume_multiplier': 1.1,
            'regime_fast_ema': 288,
            'regime_slow_ema': 1440,
            'regime_slope_min': 0.0,
            'recovery_window': 2880,
            'recovery_drawup_threshold': 1.1,
            'short_drawdown_window': 2016,
            'short_drawdown_limit': -0.7,
            'timeframe_minutes': 5,
        }
        if parameters:
            defaults.update(parameters)
        super().__init__('MultiWeekBreakoutStrategy', defaults)

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        if not self.validate_data(data):
            return data

        df = data.copy()
        price = df['close'] if 'close' in df else df['price']
        df['price'] = price

        breakout_window = int(self.parameters['lookback_breakout'])
        confirm_window = int(self.parameters['confirmation_window'])
        exit_window = int(self.parameters['exit_lookback'])
        volume_window = int(self.parameters['volume_window'])

        df['rolling_high'] = price.rolling(breakout_window).max().shift(1)
        df['rolling_low'] = price.rolling(exit_window).min().shift(1)
        df['confirm_high'] = price.rolling(confirm_window).max().shift(1)
        df['volume_ma'] = df['volume'].rolling(volume_window).mean()

        # Regime filter
        fast_span = int(self.parameters['regime_fast_ema'])
        slow_span = int(self.parameters['regime_slow_ema'])
        df['ema_fast'] = price.ewm(span=fast_span, adjust=False).mean()
        df['ema_slow'] = price.ewm(span=slow_span, adjust=False).mean()
        df['ema_fast_slope'] = df['ema_fast'].diff()
        slope_min = float(self.parameters['regime_slope_min'])
        df['regime_ok'] = (df['ema_fast'] > df['ema_slow']) & (df['ema_fast_slope'] >= slope_min)

        # Recovery gating
        recovery_window = int(self.parameters['recovery_window'])
        df['recovery_low'] = price.rolling(recovery_window).min()
        df['drawup'] = (price / df['recovery_low']).replace([np.inf, -np.inf], np.nan)
        threshold = float(self.parameters['recovery_drawup_threshold'])
        df['recovery_ok'] = df['drawup'] >= threshold

        short_window = int(self.parameters['short_drawdown_window'])
        df['short_max'] = price.rolling(short_window).max()
        df['short_drawdown'] = (price / df['short_max']) - 1.0

        return df

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        if 'rolling_high' not in df.columns:
            df = self.calculate_indicators(df)

        df['buy_signal'] = False
        df['sell_signal'] = False
        df['signal_strength'] = 0.0

        trail_pct = float(self.parameters['trail_drawdown_pct'])
        min_hold = int(self.parameters['min_hold_bars'])
        vol_multiplier = float(self.parameters['volume_multiplier'])
        short_dd_limit = float(self.parameters['short_drawdown_limit'])

        price = df['price'].to_numpy()
        rolling_high = df['rolling_high'].to_numpy()
        confirm_high = df['confirm_high'].to_numpy()
        rolling_low = df['rolling_low'].to_numpy()
        volume = df['volume'].to_numpy()
        volume_ma = df['volume_ma'].to_numpy()
        regime_ok = df['regime_ok'].to_numpy()
        recovery_ok = df['recovery_ok'].to_numpy()
        short_drawdown = df['short_drawdown'].to_numpy()

        buy_flags = np.zeros(len(df), dtype=bool)
        sell_flags = np.zeros(len(df), dtype=bool)
        strength = np.zeros(len(df))

        in_position = False
        entry_index = -1
        peak_price = 0.0

        for i in range(len(df)):
            if np.isnan(rolling_high[i]) or np.isnan(confirm_high[i]):
                continue

            vol_ok = True
            if not np.isnan(volume_ma[i]) and volume_ma[i] > 0:
                vol_ok = volume[i] >= volume_ma[i] * vol_multiplier

            if not in_position:
                breakout = price[i] > rolling_high[i] and price[i] > confirm_high[i]
                if (
                    breakout
                    and vol_ok
                    and regime_ok[i]
                    and recovery_ok[i]
                    and short_drawdown[i] >= short_dd_limit
                ):
                    buy_flags[i] = True
                    strength[i] = 0.8
                    in_position = True
                    entry_index = i
                    peak_price = price[i]
            else:
                peak_price = max(peak_price, price[i])
                hold_duration = i - entry_index
                breakdown = not np.isnan(rolling_low[i]) and price[i] < rolling_low[i]
                trail_hit = peak_price > 0 and price[i] <= peak_price * (1.0 - trail_pct)

                if (hold_duration >= min_hold and breakdown) or trail_hit:
                    sell_flags[i] = True
                    strength[i] = 1.0
                    in_position = False
                    entry_index = -1
                    peak_price = 0.0

        df['buy_signal'] = buy_flags
        df['sell_signal'] = sell_flags
        df['signal_strength'] = strength

        return df
