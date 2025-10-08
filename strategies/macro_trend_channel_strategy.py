"""Macro trend channel strategy with very low turnover."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from .base_strategy import BaseStrategy


class MacroTrendChannelStrategy(BaseStrategy):
    """Trades only when 30-day trend decisively clears 120-day baseline."""

    def __init__(self, parameters: Dict | None = None):
        defaults = {
            'ma_fast': 8640,    # 30 days of 5m bars
            'ma_slow': 34560,   # 120 days
            'entry_buffer': 0.02,
            'exit_buffer': 0.005,
            'trail_drawdown_pct': 0.32,
            'min_hold_bars': 2016,  # one week
            'timeframe_minutes': 5,
        }
        if parameters:
            defaults.update(parameters)
        super().__init__('MacroTrendChannelStrategy', defaults)

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        if not self.validate_data(data):
            return data

        df = data.copy()
        price = df['close'] if 'close' in df else df['price']
        df['price'] = price

        ma_fast = int(self.parameters['ma_fast'])
        ma_slow = int(self.parameters['ma_slow'])

        df['ma_fast'] = price.rolling(ma_fast).mean()
        df['ma_slow'] = price.rolling(ma_slow).mean()
        df['ma_gap'] = (df['ma_fast'] - df['ma_slow']) / df['ma_slow']
        return df

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        if 'ma_fast' not in df.columns:
            df = self.calculate_indicators(df)

        df['buy_signal'] = False
        df['sell_signal'] = False
        df['signal_strength'] = 0.0

        entry_buffer = float(self.parameters['entry_buffer'])
        exit_buffer = float(self.parameters['exit_buffer'])
        trail_pct = float(self.parameters['trail_drawdown_pct'])
        min_hold = int(self.parameters['min_hold_bars'])

        ma_fast = df['ma_fast'].to_numpy()
        ma_slow = df['ma_slow'].to_numpy()
        ma_gap = df['ma_gap'].to_numpy()
        price = df['price'].to_numpy()

        buy_flags = np.zeros(len(df), dtype=bool)
        sell_flags = np.zeros(len(df), dtype=bool)
        strength = np.zeros(len(df))

        in_position = False
        entry_index = -1
        peak_price = 0.0

        for i in range(len(df)):
            if np.isnan(ma_fast[i]) or np.isnan(ma_slow[i]):
                continue

            if not in_position:
                strong_trend = ma_gap[i] >= entry_buffer and ma_fast[i] > ma_slow[i]
                if strong_trend:
                    buy_flags[i] = True
                    denom = max(abs(entry_buffer), 1e-9) * 1.5
                    strength[i] = min(1.0, ma_gap[i] / denom)
                    in_position = True
                    entry_index = i
                    peak_price = price[i]
            else:
                peak_price = max(peak_price, price[i])
                hold_duration = i - entry_index
                breakdown = ma_gap[i] <= -exit_buffer
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
