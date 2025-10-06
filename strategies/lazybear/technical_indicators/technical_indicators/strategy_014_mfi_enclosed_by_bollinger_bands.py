#!/usr/bin/env python3
"""
Strategy 014: MFI Enclosed by Bollinger Bands

LazyBear Name: MFI enclosed by Bollinger Bands
TradingView URL: https://www.tradingview.com/v/4hhFyZwm/
Type: volume/momentum

Description:
Computes Money Flow Index (MFI) and applies Bollinger Bands on the MFI series.
Generates buy when MFI crosses above the lower band (potential rebound) and sell
when MFI crosses below the upper band (potential rejection), with optional trend
and momentum filters.
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


class Strategy014MfiBollingerBands(BaseStrategy):
    """MFI with Bollinger Bands on MFI values for reversal-style entries/exits."""

    def __init__(self, parameters: Dict = None):
        default_params = {
            'mfi_period': 14,
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
            name="Strategy_014_MFI_BollingerBands",
            parameters=default_params,
        )

    @staticmethod
    def _ema(series: pd.Series, period: int) -> pd.Series:
        return series.ewm(span=period, adjust=False, min_periods=1).mean()

    def _mfi(self, df: pd.DataFrame, period: int) -> pd.Series:
        # Typical price and raw money flow
        tp = (df['high'] + df['low'] + df['close']) / 3.0
        rmf = tp * df['volume']
        # Positive/negative money flow
        delta_tp = tp.diff()
        pos_mf = rmf.where(delta_tp > 0, 0.0)
        neg_mf = rmf.where(delta_tp < 0, 0.0)
        # Sum over period (SMA window)
        pos_sum = pos_mf.rolling(period, min_periods=1).sum()
        neg_sum = neg_mf.rolling(period, min_periods=1).sum()
        # Money Flow Index
        mfr = pos_sum / (neg_sum.replace(0, np.nan))
        mfi = 100 - (100 / (1 + mfr))
        return mfi.fillna(50)

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        if 'close' not in data.columns and 'price' in data.columns:
            data['close'] = data['price']
        for col in ('high', 'low', 'open'):
            if col not in data.columns:
                data[col] = data['close']

        p = self.parameters
        mfi = self._mfi(data, p['mfi_period'])
        mfi_ma = mfi.rolling(p['bb_period'], min_periods=1).mean()
        mfi_std = mfi.rolling(p['bb_period'], min_periods=1).std(ddof=0)
        bb_upper = mfi_ma + p['bb_std_dev'] * mfi_std
        bb_lower = mfi_ma - p['bb_std_dev'] * mfi_std

        data['mfi'] = mfi
        data['mfi_bb_upper'] = bb_upper
        data['mfi_bb_middle'] = mfi_ma
        data['mfi_bb_lower'] = bb_lower

        if p['use_trend_filter']:
            data['trend_sma'] = data['close'].rolling(p['trend_ma_period'], min_periods=1).mean()
        data['momentum'] = data['close'] - data['close'].shift(p['momentum_lookback'])

        self.indicators = data[['mfi', 'mfi_bb_upper', 'mfi_bb_middle', 'mfi_bb_lower']].copy()
        return data

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        p = self.parameters
        data['buy_signal'] = False
        data['sell_signal'] = False
        data['signal_strength'] = 0.0

        buy_cross = crossover(data['mfi'], data['mfi_bb_lower'])
        sell_cross = crossunder(data['mfi'], data['mfi_bb_upper'])

        buy_conditions = [buy_cross]
        sell_conditions = [sell_cross]

        if p['use_trend_filter'] and 'trend_sma' in data.columns:
            uptrend = data['close'] > data['trend_sma']
            downtrend = data['close'] < data['trend_sma']
            buy_conditions.append(uptrend)
            sell_conditions.append(downtrend)

        mom_up = data['momentum'] > 0
        mom_down = data['momentum'] < 0
        buy_conditions.append(mom_up)
        sell_conditions.append(mom_down)

        data['buy_signal'] = pd.concat(buy_conditions, axis=1).all(axis=1)
        data['sell_signal'] = pd.concat(sell_conditions, axis=1).all(axis=1)

        # Strength: distance from mid band and momentum confirmation
        width = (data['mfi_bb_upper'] - data['mfi_bb_lower']).replace(0, np.nan)
        dist_to_mid = 1 - (np.abs(data['mfi'] - data['mfi_bb_middle']) / width)
        dist_to_mid = dist_to_mid.fillna(0).clip(0, 1)
        momentum_strength = (np.abs(data['momentum']) / (data['close'].rolling(20, min_periods=1).std(ddof=0) + 1e-9)).clip(0, 1)

        data['signal_strength'] = calculate_signal_strength([
            dist_to_mid, momentum_strength
        ], weights=[0.6, 0.4])

        weak = data['signal_strength'] < p['signal_threshold']
        data.loc[weak, ['buy_signal', 'sell_signal']] = False
        data['buy_signal'], data['sell_signal'] = apply_position_constraints(
            data['buy_signal'], data['sell_signal'], allow_short=False
        )

        self.signals = data[['buy_signal', 'sell_signal', 'signal_strength']].copy()
        return data

