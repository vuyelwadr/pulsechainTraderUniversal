"""Long-term regime hold strategy.

Designed to mirror buy-and-hold during sustained bull markets while providing a
hard stop if the long regime breaks down badly. Keeps turnover extremely low to
survive large swap costs per roundtrip.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from .base_strategy import BaseStrategy


class LongTermRegimeStrategy(BaseStrategy):
    """Low-frequency long-only strategy with regime and trailing protections."""

    def __init__(self, parameters: Dict | None = None):
        defaults = {
            'ema_short': 720,   # ~2.5 days of 5m bars
            'ema_long': 2880,   # ~10 days
            'entry_buffer': 0.003,
            'exit_buffer': 0.01,
            'trail_drawdown_pct': 0.45,
            'min_hold_bars': 1440,  # at least 5 days
            'timeframe_minutes': 5,
        }
        if parameters:
            defaults.update(parameters)
        super().__init__('LongTermRegimeStrategy', defaults)

    @staticmethod
    def _ema(series: pd.Series, span: int) -> pd.Series:
        return series.ewm(span=max(2, span), adjust=False).mean()

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        if not self.validate_data(data):
            return data

        df = data.copy()
        price = df['close'] if 'close' in df else df['price']
        df['price'] = price
        ema_short = int(self.parameters['ema_short'])
        ema_long = int(self.parameters['ema_long'])
        df['ema_short'] = self._ema(price, ema_short)
        df['ema_long'] = self._ema(price, ema_long)
        df['ema_gap'] = (df['ema_short'] - df['ema_long']) / df['ema_long']
        return df

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        if 'ema_short' not in df.columns:
            df = self.calculate_indicators(df)

        df['buy_signal'] = False
        df['sell_signal'] = False
        df['signal_strength'] = 0.0

        entry_buffer = float(self.parameters['entry_buffer'])
        exit_buffer = float(self.parameters['exit_buffer'])
        trail_pct = float(self.parameters['trail_drawdown_pct'])
        min_hold = int(self.parameters['min_hold_bars'])

        ema_short = df['ema_short'].to_numpy()
        ema_long = df['ema_long'].to_numpy()
        ema_gap = df['ema_gap'].to_numpy()
        price = df['price'].to_numpy()

        buy_flags = np.zeros(len(df), dtype=bool)
        sell_flags = np.zeros(len(df), dtype=bool)
        strength = np.zeros(len(df))

        in_position = False
        entry_index = -1
        peak_price = 0.0

        for i in range(len(df)):
            if np.isnan(ema_short[i]) or np.isnan(ema_long[i]):
                continue

            if not in_position:
                bullish = ema_short[i] >= ema_long[i] * (1 + entry_buffer) and ema_gap[i] > 0
                if bullish:
                    buy_flags[i] = True
                    strength[i] = min(1.0, ema_gap[i] / (entry_buffer * 2))
                    in_position = True
                    entry_index = i
                    peak_price = price[i]
            else:
                peak_price = max(peak_price, price[i])
                hold_duration = i - entry_index
                regime_break = ema_short[i] <= ema_long[i] * (1 - exit_buffer)
                trail_hit = peak_price > 0 and price[i] <= peak_price * (1.0 - trail_pct)

                if (hold_duration >= min_hold and regime_break) or trail_hit:
                    sell_flags[i] = True
                    strength[i] = 1.0
                    in_position = False
                    entry_index = -1
                    peak_price = 0.0

        df['buy_signal'] = buy_flags
        df['sell_signal'] = sell_flags
        df['signal_strength'] = strength

        return df

