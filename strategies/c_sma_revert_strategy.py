"""Cost-aware SMA Reversion strategy adapted for BaseStrategy interface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

from .base_strategy import BaseStrategy


def _sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=window).mean()


def _rsi(series: pd.Series, window: int) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    avg_gain = up.rolling(window, min_periods=window).mean()
    avg_loss = down.rolling(window, min_periods=window).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    rsi = 100 - (100 / (1 + rs))
    return rsi


class CSMARevertStrategy(BaseStrategy):
    """SMA reversion with RSI filter and configurable entry/exit bands."""

    def __init__(self, parameters: Dict | None = None):
        defaults = {
            'n_sma': 576,          # ~2 days on 5m bars
            'entry_drop': 0.25,    # 25% below SMA to enter
            'exit_up': 0.048,      # tuned exit overshoot ~=4.8%
            'rsi_period': 14,
            'rsi_max': 32.0,
            'timeframe_minutes': 5,
        }
        if parameters:
            defaults.update(parameters)
        super().__init__('CSMARevertStrategy', defaults)

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        if not self.validate_data(data):
            return data

        df = data.copy()
        price = df['price'] if 'price' in df else df['close']
        df['price'] = price
        n_sma = int(self.parameters['n_sma'])
        rsi_period = int(self.parameters['rsi_period'])

        df['sma'] = _sma(price, n_sma)
        df['rsi'] = _rsi(price, rsi_period)
        return df

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        if 'sma' not in df.columns:
            df = self.calculate_indicators(df)

        df['buy_signal'] = False
        df['sell_signal'] = False
        df['signal_strength'] = 0.0

        price = df['price'].to_numpy(float)
        sma = df['sma'].to_numpy(float)
        rsi = df['rsi'].to_numpy(float)

        entry_drop = float(self.parameters['entry_drop'])
        exit_up = float(self.parameters['exit_up'])
        rsi_max = float(self.parameters['rsi_max'])

        in_position = False
        for i in range(len(df)):
            px = price[i]
            sm = sma[i]
            rs = rsi[i]

            if np.isnan(sm) or np.isnan(rs):
                continue

            if not in_position:
                if px <= sm * (1.0 - entry_drop) and rs <= rsi_max:
                    df.iat[i, df.columns.get_loc('buy_signal')] = True
                    df.iat[i, df.columns.get_loc('signal_strength')] = 1.0
                    in_position = True
            else:
                if px >= sm * (1.0 + exit_up):
                    df.iat[i, df.columns.get_loc('sell_signal')] = True
                    df.iat[i, df.columns.get_loc('signal_strength')] = 1.0
                    in_position = False

        return df
