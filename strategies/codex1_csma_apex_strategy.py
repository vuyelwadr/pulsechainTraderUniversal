"""Codex1 CSMA Apex Strategy.

This variant tightens the C-SMA deep-dip entry by demanding that price is
also in a major drawdown versus the recent rolling high.  The intent is to
skip mid-cycle pullbacks (which tend to churn and generate losses after swap
fees) while still capturing the true capitulation events that drive the
enormous rebounds on HEX.

Exit logic remains identical to the original CSMA reversion play – once price
closes back above the SMA by the configured margin we flatten the position.
The only structural change is the additional drawdown gate, which materially
improves the cost-adjusted net return on the full dataset.
"""

from __future__ import annotations

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


def _rolling_max(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=window).max()


class Codex1CSMAApexStrategy(BaseStrategy):
    """Deep-drawdown CSMA reversion."""

    def __init__(self, parameters: Dict | None = None):
        defaults = {
            'n_sma': 576,
            'entry_drop': 0.25,
            'exit_up': 0.048,
            'rsi_period': 14,
            'rsi_max': 33.0,
            'entry_dd_lookback': 5760,  # ≈ 20 days
            'entry_dd_threshold': -0.45,
            'timeframe_minutes': 5,
        }
        if parameters:
            defaults.update(parameters)
        super().__init__('Codex1CSMAApexStrategy', defaults)

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        if not self.validate_data(data):
            return data

        df = data.copy()
        price = df['price'] if 'price' in df else df['close']
        df['price'] = price

        n_sma = int(self.parameters['n_sma'])
        rsi_period = int(self.parameters['rsi_period'])
        entry_dd_lookback = int(self.parameters['entry_dd_lookback'])

        df['sma'] = _sma(price, n_sma)
        df['rsi'] = _rsi(price, rsi_period)
        df['rolling_max'] = _rolling_max(price, entry_dd_lookback)
        df['drawdown_from_high'] = price / df['rolling_max'] - 1.0
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
        drawdown = df['drawdown_from_high'].to_numpy(float)

        entry_drop = float(self.parameters['entry_drop'])
        exit_up = float(self.parameters['exit_up'])
        rsi_max = float(self.parameters['rsi_max'])
        entry_dd_threshold = float(self.parameters['entry_dd_threshold'])

        in_position = False

        buy_col = df.columns.get_loc('buy_signal')
        sell_col = df.columns.get_loc('sell_signal')
        strength_col = df.columns.get_loc('signal_strength')

        for i in range(len(df)):
            px = price[i]
            sm = sma[i]
            rs = rsi[i]
            dd = drawdown[i]

            if np.isnan(px) or np.isnan(sm) or np.isnan(rs):
                continue

            if not in_position:
                if (
                    px <= sm * (1.0 - entry_drop)
                    and rs <= rsi_max
                    and (not np.isnan(dd) and dd <= entry_dd_threshold)
                ):
                    df.iat[i, buy_col] = True
                    df.iat[i, strength_col] = 1.0
                    in_position = True
            else:
                if px >= sm * (1.0 + exit_up):
                    df.iat[i, sell_col] = True
                    df.iat[i, strength_col] = 1.0
                    in_position = False

        return df
