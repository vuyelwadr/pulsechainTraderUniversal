#!/usr/bin/env python3
"""
Strategy 020: True Strength Index (TSI)

LazyBear Name: True Strength Index (TSI)
TradingView URL: https://www.tradingview.com/v/UQjj3yax/
Type: momentum/oscillator (Ehlers)

Description:
Implements the True Strength Index using double-smoothed momentum and absolute
momentum. Common defaults: long=25, short=13, signal=7. Generates buy when TSI
crosses above its signal and sell when TSI crosses below, with optional level
filtering.
"""

import pandas as pd
import numpy as np
from typing import Dict
import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)
from strategies.base_strategy import BaseStrategy

try:
    sys.path.insert(0, os.path.join(project_root, 'src'))
    from utils.vectorized_helpers import crossover, crossunder, apply_position_constraints, calculate_signal_strength
except Exception:  # pragma: no cover
    def crossover(a, b):
        return (a > b) & (a.shift(1) <= b.shift(1))
    def crossunder(a, b):
        return (a < b) & (a.shift(1) >= b.shift(1))
    def apply_position_constraints(buy, sell, allow_short=False):
        return buy, sell
    def calculate_signal_strength(factors, weights=None):
        if not factors:
            return pd.Series(0.0)
        df = pd.concat(factors, axis=1)
        return df.mean(axis=1).clip(0, 1)


class Strategy020TrueStrengthIndex(BaseStrategy):
    """True Strength Index with signal line and basic level filters."""

    def __init__(self, parameters: Dict = None):
        default_params = {
            'long_period': 25,
            'short_period': 13,
            'signal_period': 7,
            'overbought': 25.0,
            'oversold': -25.0,
            'signal_threshold': 0.6,
            'use_trend_filter': True,
            'trend_ma_period': 50,
        }
        if parameters:
            default_params.update(parameters)

        super().__init__(
            name="Strategy_020_TrueStrengthIndex",
            parameters=default_params,
        )

    @staticmethod
    def _ema(series: pd.Series, period: int) -> pd.Series:
        return series.ewm(span=period, adjust=False, min_periods=1).mean()

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        if 'close' not in data.columns and 'price' in data.columns:
            data['close'] = data['price']

        p = self.parameters
        m = data['close'].diff()
        abs_m = m.abs()

        ema1_m = self._ema(m, p['long_period'])
        ema2_m = self._ema(ema1_m, p['short_period'])

        ema1_abs = self._ema(abs_m, p['long_period'])
        ema2_abs = self._ema(ema1_abs, p['short_period'])

        tsi = 100 * (ema2_m / (ema2_abs.replace(0, np.nan)))
        tsi = tsi.fillna(0)
        tsi_signal = self._ema(tsi, p['signal_period'])

        data['tsi'] = tsi
        data['tsi_signal'] = tsi_signal

        if p['use_trend_filter']:
            data['trend_sma'] = data['close'].rolling(p['trend_ma_period'], min_periods=1).mean()

        self.indicators = data[['tsi', 'tsi_signal']].copy()
        return data

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        p = self.parameters
        data['buy_signal'] = False
        data['sell_signal'] = False
        data['signal_strength'] = 0.0

        cross_up = crossover(data['tsi'], data['tsi_signal'])
        cross_dn = crossunder(data['tsi'], data['tsi_signal'])

        buy_conditions = [cross_up]
        sell_conditions = [cross_dn]

        if p['use_trend_filter'] and 'trend_sma' in data.columns:
            uptrend = data['close'] > data['trend_sma']
            downtrend = data['close'] < data['trend_sma']
            buy_conditions.append(uptrend)
            sell_conditions.append(downtrend)

        # Optional level filters
        buy_conditions.append(data['tsi'] > p['oversold'])
        sell_conditions.append(data['tsi'] < p['overbought'])

        data['buy_signal'] = pd.concat(buy_conditions, axis=1).all(axis=1)
        data['sell_signal'] = pd.concat(sell_conditions, axis=1).all(axis=1)

        # Strength factors: divergence from signal and distance from zero
        div = (data['tsi'] - data['tsi_signal']).abs()
        div_norm = (div / (div.rolling(50, min_periods=1).max() + 1e-9)).clip(0, 1)
        level = (data['tsi'].abs() / (data['tsi'].abs().rolling(50, min_periods=1).max() + 1e-9)).clip(0, 1)

        data['signal_strength'] = calculate_signal_strength([div_norm, level], weights=[0.6, 0.4])

        weak = data['signal_strength'] < p['signal_threshold']
        data.loc[weak, ['buy_signal', 'sell_signal']] = False
        data['buy_signal'], data['sell_signal'] = apply_position_constraints(
            data['buy_signal'], data['sell_signal'], allow_short=False
        )

        self.signals = data[['buy_signal', 'sell_signal', 'signal_strength']].copy()
        return data

