#!/usr/bin/env python3
"""
Strategy 013: RSI Enclosed by Bollinger Bands

LazyBear Name: RSI enclosed by Bollinger Bands
TradingView URL: https://www.tradingview.com/v/4hhFyZwm/
Type: oscillator/momentum

Description:
Calculates RSI on close prices, then applies Bollinger Bands on the RSI series.
Signals are generated when RSI crosses above the lower band (potential buy) or
below the upper band (potential sell), with optional trend and momentum filters.

Notes:
- Implemented via pandas; no TA-Lib dependency.
- Uses vectorized_helpers for robust crossover/crossunder logic.
- Designed for hyperparameter tuning via the parameters dict.
"""

import pandas as pd
import numpy as np
from typing import Dict
import sys
import os

# Ensure repository root on path for BaseStrategy import
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)
from strategies.base_strategy import BaseStrategy

# Try vectorized helpers; fall back to simple versions if unavailable
try:
    sys.path.insert(0, os.path.join(project_root, 'src'))
    from utils.vectorized_helpers import (
        crossover, crossunder, apply_position_constraints,
        calculate_signal_strength, pine_rma
    )
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


class Strategy013RsiBollingerBands(BaseStrategy):
    """
    RSI with Bollinger Bands on RSI values. Generates:
    - Buy when RSI crosses up above the lower band (optionally above RSI SMA)
    - Sell when RSI crosses down below the upper band (optionally below RSI SMA)
    Signal strength increases with distance from band midline and momentum alignment.
    """

    def __init__(self, parameters: Dict = None):
        default_params = {
            'rsi_period': 14,
            'bb_period': 20,
            'bb_std_dev': 2.0,
            'signal_threshold': 0.6,
            'use_trend_filter': True,
            'trend_ma_period': 50,
            'momentum_lookback': 10,
        }
        if parameters:
            default_params.update(parameters)

        super().__init__(
            name="Strategy_013_RSI_BollingerBands",
            parameters=default_params,
        )

    @staticmethod
    def _ema(series: pd.Series, period: int) -> pd.Series:
        return series.ewm(span=period, adjust=False, min_periods=1).mean()

    def _rsi(self, close: pd.Series, period: int) -> pd.Series:
        # Pine ta.rsi uses Wilder's RMA (pine_rma)
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = pine_rma(gain, period)
        avg_loss = pine_rma(loss, period)
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        if 'close' not in data.columns and 'price' in data.columns:
            data['close'] = data['price']

        p = self.parameters
        rsi = self._rsi(data['close'], p['rsi_period'])
        rsi_ma = rsi.rolling(p['bb_period'], min_periods=1).mean()
        rsi_std = rsi.rolling(p['bb_period'], min_periods=1).std(ddof=0)
        bb_upper = rsi_ma + p['bb_std_dev'] * rsi_std
        bb_lower = rsi_ma - p['bb_std_dev'] * rsi_std

        data['rsi'] = rsi
        data['rsi_bb_upper'] = bb_upper
        data['rsi_bb_middle'] = rsi_ma
        data['rsi_bb_lower'] = bb_lower

        # Optional filters
        if p['use_trend_filter']:
            data['trend_sma'] = data['close'].rolling(p['trend_ma_period'], min_periods=1).mean()
        data['momentum'] = data['close'] - data['close'].shift(p['momentum_lookback'])

        self.indicators = data[['rsi', 'rsi_bb_upper', 'rsi_bb_middle', 'rsi_bb_lower']].copy()
        return data

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        p = self.parameters
        data['buy_signal'] = False
        data['sell_signal'] = False
        data['signal_strength'] = 0.0

        # Primary conditions
        buy_cross = crossover(data['rsi'], data['rsi_bb_lower'])
        sell_cross = crossunder(data['rsi'], data['rsi_bb_upper'])

        buy_conditions = [buy_cross]
        sell_conditions = [sell_cross]

        # Trend filter
        if p['use_trend_filter'] and 'trend_sma' in data.columns:
            uptrend = data['close'] > data['trend_sma']
            downtrend = data['close'] < data['trend_sma']
            buy_conditions.append(uptrend)
            sell_conditions.append(downtrend)

        # Momentum confirmation
        mom_up = data['momentum'] > 0
        mom_down = data['momentum'] < 0
        buy_conditions.append(mom_up)
        sell_conditions.append(mom_down)

        data['buy_signal'] = pd.concat(buy_conditions, axis=1).all(axis=1)
        data['sell_signal'] = pd.concat(sell_conditions, axis=1).all(axis=1)

        # Signal strength factors
        dist_to_mid = 1 - (np.abs(data['rsi'] - data['rsi_bb_middle']) / (data['rsi_bb_upper'] - data['rsi_bb_lower']).replace(0, np.nan))
        dist_to_mid = dist_to_mid.fillna(0).clip(0, 1)
        momentum_strength = (np.abs(data['momentum']) / (data['close'].rolling(20, min_periods=1).std(ddof=0) + 1e-9)).clip(0, 1)

        data['signal_strength'] = calculate_signal_strength([
            dist_to_mid, momentum_strength
        ], weights=[0.6, 0.4])

        # Apply threshold and constraints
        weak = data['signal_strength'] < p['signal_threshold']
        data.loc[weak, ['buy_signal', 'sell_signal']] = False
        data['buy_signal'], data['sell_signal'] = apply_position_constraints(
            data['buy_signal'], data['sell_signal'], allow_short=False
        )

        self.signals = data[['buy_signal', 'sell_signal', 'signal_strength']].copy()
        return data
