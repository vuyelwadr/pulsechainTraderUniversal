"""Post-crash momentum strategy.

Stays in cash until price has rallied sharply off a multi-month low, then rides
the recovery with a wide trailing stop. Keeps turnover tiny to survive large
swap costs.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from .base_strategy import BaseStrategy


class PostCrashMomentumStrategy(BaseStrategy):
    """Enter once recovery momentum is confirmed after a deep crash."""

    def __init__(self, parameters: Dict | None = None):
        defaults = {
            'crash_window': 51840,  # 180 days of 5m bars
            'entry_drawup': 4.0,    # price up 400% off the low
            'momentum_span': 2880,  # 10-day EMA slope for confirmation
            'trail_drawdown_pct': 0.45,
            'stop_from_low_pct': 0.35,
            'timeframe_minutes': 5,
        }
        if parameters:
            defaults.update(parameters)
        super().__init__('PostCrashMomentumStrategy', defaults)

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        if not self.validate_data(data):
            return data

        df = data.copy()
        price = df['close'] if 'close' in df else df['price']
        df['price'] = price

        crash_window = int(self.parameters['crash_window'])
        rolling_min = price.rolling(crash_window).min()
        df['rolling_min'] = rolling_min
        df['drawup'] = price / rolling_min - 1.0

        span = int(self.parameters['momentum_span'])
        ema = price.ewm(span=span, adjust=False).mean()
        df['ema'] = ema
        df['ema_slope'] = ema.diff()

        return df

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        if 'drawup' not in df.columns:
            df = self.calculate_indicators(df)

        df['buy_signal'] = False
        df['sell_signal'] = False
        df['signal_strength'] = 0.0

        entry_drawup = float(self.parameters['entry_drawup'])
        trail_pct = float(self.parameters['trail_drawdown_pct'])
        stop_from_low = float(self.parameters['stop_from_low_pct'])

        price = df['price'].to_numpy()
        drawup = df['drawup'].to_numpy()
        ema_slope = df['ema_slope'].to_numpy()
        rolling_min = df['rolling_min'].to_numpy()

        buy_flags = np.zeros(len(df), dtype=bool)
        sell_flags = np.zeros(len(df), dtype=bool)
        strength = np.zeros(len(df))

        in_position = False
        peak_price = 0.0
        entry_price = 0.0
        low_anchor = 0.0

        for i in range(len(df)):
            if np.isnan(drawup[i]) or np.isnan(ema_slope[i]) or np.isnan(rolling_min[i]):
                continue

            if not in_position:
                recovering = drawup[i] >= entry_drawup and ema_slope[i] > 0
                if recovering:
                    buy_flags[i] = True
                    strength[i] = min(1.0, drawup[i] / (entry_drawup * 1.5))
                    in_position = True
                    peak_price = price[i]
                    entry_price = price[i]
                    low_anchor = rolling_min[i]
            else:
                peak_price = max(peak_price, price[i])
                stop_level = max(low_anchor * (1.0 + stop_from_low), peak_price * (1.0 - trail_pct))
                if price[i] <= stop_level:
                    sell_flags[i] = True
                    strength[i] = 1.0
                    in_position = False
                    peak_price = 0.0
                    entry_price = 0.0
                    low_anchor = 0.0

        df['buy_signal'] = buy_flags
        df['sell_signal'] = sell_flags
        df['signal_strength'] = strength

        return df

