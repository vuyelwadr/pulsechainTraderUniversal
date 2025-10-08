"""Low-turnover EMA regime hold strategy.

Goes long only during strong bullish regimes defined by wide EMA separation and
positive slope, keeping turnover small to survive high swap costs.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from .base_strategy import BaseStrategy


class EMATrendHoldStrategy(BaseStrategy):
    """Enter long on strong EMA regime, exit on regime breakdown or deep trailing drop."""

    def __init__(self, parameters: Dict | None = None):
        defaults = {
            'ema_fast': 288,   # 1 day of 5m bars
            'ema_slow': 1440,  # 5 days
            'confirm_threshold': 0.005,
            'exit_threshold': 0.002,
            'trail_drawdown_pct': 0.35,
            'min_hold_bars': 2880,
            'timeframe_minutes': 5,
        }
        if parameters:
            defaults.update(parameters)
        super().__init__('EMATrendHoldStrategy', defaults)

    @staticmethod
    def _ema(series: pd.Series, span: int) -> pd.Series:
        return series.ewm(span=max(2, span), adjust=False).mean()

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        if not self.validate_data(data):
            return data

        df = data.copy()
        price = df['close'] if 'close' in df else df['price']
        df['price'] = price

        fast = int(self.parameters['ema_fast'])
        slow = int(self.parameters['ema_slow'])
        df['ema_fast'] = self._ema(price, fast)
        df['ema_slow'] = self._ema(price, slow)
        df['ema_spread'] = (df['ema_fast'] - df['ema_slow']) / df['ema_slow']
        df['ema_slope'] = df['ema_fast'].diff()
        df['rolling_max'] = price.cummax()

        return df

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        if 'ema_fast' not in df.columns:
            df = self.calculate_indicators(df)

        df['buy_signal'] = False
        df['sell_signal'] = False
        df['signal_strength'] = 0.0

        spread_confirm = float(self.parameters['confirm_threshold'])
        exit_threshold = float(self.parameters['exit_threshold'])
        trail_pct = float(self.parameters['trail_drawdown_pct'])
        min_hold = int(self.parameters['min_hold_bars'])

        ema_fast = df['ema_fast'].to_numpy()
        ema_slow = df['ema_slow'].to_numpy()
        ema_spread = df['ema_spread'].to_numpy()
        ema_slope = df['ema_slope'].to_numpy()
        price = df['price'].to_numpy()

        buy_flags = np.zeros(len(df), dtype=bool)
        sell_flags = np.zeros(len(df), dtype=bool)
        strength = np.zeros(len(df))

        in_position = False
        peak_price = 0.0
        entry_index = -1

        for i in range(len(df)):
            if np.isnan(ema_fast[i]) or np.isnan(ema_slow[i]):
                continue

            if not in_position:
                bullish = ema_spread[i] >= spread_confirm and ema_slope[i] > 0
                if bullish:
                    buy_flags[i] = True
                    strength[i] = min(1.0, ema_spread[i] / (spread_confirm * 2))
                    in_position = True
                    peak_price = price[i]
                    entry_index = i
            else:
                peak_price = max(peak_price, price[i])
                hold_duration = i - entry_index
                bearish = ema_spread[i] <= -exit_threshold or ema_fast[i] < ema_slow[i]
                trail_hit = peak_price > 0 and price[i] <= peak_price * (1.0 - trail_pct)
                exit_ready = hold_duration >= min_hold or trail_hit

                if exit_ready and (bearish or trail_hit):
                    sell_flags[i] = True
                    strength[i] = 1.0
                    in_position = False
                    peak_price = 0.0
                    entry_index = -1

        df['buy_signal'] = buy_flags
        df['sell_signal'] = sell_flags
        df['signal_strength'] = strength

        return df
