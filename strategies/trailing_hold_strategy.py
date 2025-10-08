"""Passive hold with trailing stop to manage catastrophic selloffs."""

from __future__ import annotations

import pandas as pd

from .base_strategy import BaseStrategy


class TrailingHoldStrategy(BaseStrategy):
    """Buy once, ride trend, exit if price collapses beyond trailing percentage."""

    def __init__(self, trail_drawdown_pct: float = 0.8):
        super().__init__('TrailingHoldStrategy', {
            'trail_drawdown_pct': trail_drawdown_pct,
            'timeframe_minutes': 5,
        })
        self._peak_price = None
        self._in_position = False

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        return data

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        df['buy_signal'] = False
        df['sell_signal'] = False
        df['signal_strength'] = 0.0

        if df.empty:
            return df

        price = df['close'] if 'close' in df else df['price']
        trail_pct = float(self.parameters['trail_drawdown_pct'])

        if not self._in_position:
            df.iat[0, df.columns.get_loc('buy_signal')] = True
            df.iat[0, df.columns.get_loc('signal_strength')] = 1.0
            self._in_position = True
            self._peak_price = float(price.iloc[0])

        for i in range(len(df)):
            current_price = float(price.iloc[i])
            if self._in_position:
                self._peak_price = max(self._peak_price, current_price)
                if current_price <= self._peak_price * (1.0 - trail_pct):
                    df.iat[i, df.columns.get_loc('sell_signal')] = True
                    df.iat[i, df.columns.get_loc('signal_strength')] = 1.0
                    self._in_position = False
                    self._peak_price = None

        return df
