#!/usr/bin/env python3
"""
Strategy 048: 11x MA/EMA with Cross Highlights

TradingView Reference: "11x MA [LazyBear]" (Pine v2)
- Mirrors the Pine defaults exactly (use_ema default false → SMA by default)
- Computes 11 moving averages with configurable lengths
- Pine draws cross shapes when adjacent MAs cross; here we translate that intent
  into a deterministic trading rule based on ordered MA alignment.

Deterministic trading rule (minimal, consistent with intent):
- Bullish alignment when ma1 > ma2 > ... > ma11; emit buy on first bar alignment becomes true
- Bearish alignment when ma1 < ma2 < ... < ma11; emit sell on first bar alignment becomes true
- Signal strength is the fraction of adjacent pairs already aligned in the signal direction

Notes:
- No synthetic data; works on real OHLCV. If only `price` is present, it is used for `close`.
- Vectorized, no look-ahead (uses shift(1) for alignment start detection).
"""

import pandas as pd
import numpy as np
from typing import Dict
import sys, os

# Project root for BaseStrategy import
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)
from strategies.base_strategy import BaseStrategy

# Try vectorized helpers; fall back to minimal locals if unavailable
try:
    sys.path.insert(0, os.path.join(project_root, 'src'))
    from utils.vectorized_helpers import apply_position_constraints, calculate_signal_strength
except Exception:
    def apply_position_constraints(b, s, allow_short: bool = False):
        return b, s
    def calculate_signal_strength(fs, weights=None):
        df = pd.concat(fs, axis=1)
        if weights is None:
            weights = [1.0 / df.shape[1]] * df.shape[1]
        out = pd.Series(0.0, index=df.index)
        for col, w in zip(df.columns, weights):
            series = df[col]
            if series.dtype == bool:
                series = series.astype(float)
            out = out.add(series.fillna(0) * w, fill_value=0)
        return out.clip(0, 1)


class Strategy048EMA11xCrossHighlights(BaseStrategy):
    def __init__(self, parameters: Dict = None):
        params = {
            'use_ema': False,         # Pine default: SMA unless toggled
            'draw_cross': False,      # Visualization-only in Pine
            'ma1_p': 25,
            'ma2_p': 30,
            'ma3_p': 35,
            'ma4_p': 40,
            'ma5_p': 45,
            'ma6_p': 50,
            'ma7_p': 55,
            'ma8_p': 60,
            'ma9_p': 65,
            'ma10_p': 70,
            'ma11_p': 75,
            'signal_threshold': 0.6,
        }
        if parameters:
            params.update(parameters)
        super().__init__('Strategy_048_11x_MA_CrossHighlights', params)

    @staticmethod
    def _sma(s: pd.Series, n: int) -> pd.Series:
        return s.rolling(max(1, int(n)), min_periods=1).mean()

    @staticmethod
    def _ema(s: pd.Series, n: int) -> pd.Series:
        return s.ewm(span=max(1, int(n)), adjust=False, min_periods=1).mean()

    def _ma(self, s: pd.Series, n: int) -> pd.Series:
        return (self._ema if bool(self.parameters.get('use_ema', False)) else self._sma)(s, n)

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        # Ensure we have close
        if 'close' not in data.columns and 'price' in data.columns:
            data['close'] = data['price']

        # Compute 11 moving averages with Pine defaults
        lens = [
            self.parameters['ma1_p'], self.parameters['ma2_p'], self.parameters['ma3_p'],
            self.parameters['ma4_p'], self.parameters['ma5_p'], self.parameters['ma6_p'],
            self.parameters['ma7_p'], self.parameters['ma8_p'], self.parameters['ma9_p'],
            self.parameters['ma10_p'], self.parameters['ma11_p'],
        ]
        mas = []
        for i, n in enumerate(lens, start=1):
            col = f'ma{i}'
            data[col] = self._ma(data['close'], int(n))
            mas.append(col)

        # Store for potential debugging
        self.indicators = data[mas].copy()
        return data

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        # Adjacent-pair ordering for alignment
        ma_cols = [f'ma{i}' for i in range(1, 12)]
        pairs_up = []
        pairs_dn = []
        for i in range(1, 11):
            a = data[f'ma{i}']
            b = data[f'ma{i+1}']
            pairs_up.append(a > b)
            pairs_dn.append(a < b)

        # Alignment booleans (all adjacent pairs in order)
        align_up = pd.concat(pairs_up, axis=1).all(axis=1)
        align_dn = pd.concat(pairs_dn, axis=1).all(axis=1)

        # Trigger only on first bar of (new) alignment
        buy_raw = align_up & (~align_up.shift(1).fillna(False))
        sell_raw = align_dn & (~align_dn.shift(1).fillna(False))

        # Signal strength: fraction of pairs already aligned in the signal direction
        frac_up = pd.concat(pairs_up, axis=1).mean(axis=1)
        frac_dn = pd.concat(pairs_dn, axis=1).mean(axis=1)
        strength = pd.Series(0.0, index=data.index)
        strength[buy_raw] = frac_up[buy_raw]
        strength[sell_raw] = frac_dn[sell_raw]

        # Apply minimum threshold only at signal bars; otherwise zero to avoid phantom trades
        threshold = float(self.parameters.get('signal_threshold', 0.6))
        sig_strength = strength.copy()
        sig_strength[(buy_raw | sell_raw) & (sig_strength < threshold)] = threshold
        sig_strength[~(buy_raw | sell_raw)] = 0.0

        # Position constraints (long-only)
        buy_sig, sell_sig = apply_position_constraints(buy_raw, sell_raw, allow_short=False)

        data['buy_signal'] = buy_sig
        data['sell_signal'] = sell_sig
        data['signal_strength'] = sig_strength.clip(0, 1).fillna(0)
        return data

