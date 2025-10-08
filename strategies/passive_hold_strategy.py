"""Passive hold strategy: enter once and hold for entire dataset."""

from __future__ import annotations

import pandas as pd

from .base_strategy import BaseStrategy


class PassiveHoldStrategy(BaseStrategy):
    """Buy on first bar and never exit."""

    def __init__(self):
        super().__init__('PassiveHoldStrategy', {'timeframe_minutes': 5})

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        return data

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        df['buy_signal'] = False
        df['sell_signal'] = False
        df['signal_strength'] = 0.0
        if not df.empty:
            df.iat[0, df.columns.get_loc('buy_signal')] = True
            df.iat[0, df.columns.get_loc('signal_strength')] = 1.0
        return df

