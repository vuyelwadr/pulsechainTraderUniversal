"""Recovery momentum strategy with single post-crash entry."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from .base_strategy import BaseStrategy


class RecoveryMomentumStrategy(BaseStrategy):
    """Enter once price rebounds strongly from a multi-month low, then trail."""

    def __init__(self, parameters: Dict | None = None):
        defaults = {
            'low_window': 23040,      # ~80 days of 5m bars
            'entry_multiplier': 8.0,  # price must be 8x the rolling low
            'ema_span': 2880,         # 10-day EMA for trend confirmation
            'trail_drawdown_pct': 0.5,
            'timeframe_minutes': 5,
        }
        if parameters:
            defaults.update(parameters)
        super().__init__('RecoveryMomentumStrategy', defaults)

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        if not self.validate_data(data):
            return data

        df = data.copy()
        price = df['close'] if 'close' in df else df['price']
        df['price'] = price

        low_window = int(self.parameters['low_window'])
        df['rolling_low'] = price.rolling(low_window).min()
        df['drawup'] = price / df['rolling_low']

        ema_span = int(self.parameters['ema_span'])
        df['ema'] = price.ewm(span=ema_span, adjust=False).mean()

        return df

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        if 'drawup' not in df.columns:
            df = self.calculate_indicators(df)

        df['buy_signal'] = False
        df['sell_signal'] = False
        df['signal_strength'] = 0.0

        entry_multiplier = float(self.parameters['entry_multiplier'])
        trail_pct = float(self.parameters['trail_drawdown_pct'])

        price = df['price'].to_numpy()
        drawup = df['drawup'].to_numpy()
        ema = df['ema'].to_numpy()

        in_position = False
        peak_price = 0.0

        for i in range(len(df)):
            if np.isnan(drawup[i]) or np.isnan(ema[i]):
                continue

            if not in_position:
                if drawup[i] >= entry_multiplier and price[i] > ema[i] and (i == 0 or drawup[i - 1] < entry_multiplier):
                    df.iat[i, df.columns.get_loc('buy_signal')] = True
                    df.iat[i, df.columns.get_loc('signal_strength')] = 1.0
                    in_position = True
                    peak_price = price[i]
            else:
                peak_price = max(peak_price, price[i])
                stop_level = peak_price * (1.0 - trail_pct)
                if price[i] <= stop_level:
                    df.iat[i, df.columns.get_loc('sell_signal')] = True
                    df.iat[i, df.columns.get_loc('signal_strength')] = 1.0
                    in_position = False
                    peak_price = 0.0

        return df

