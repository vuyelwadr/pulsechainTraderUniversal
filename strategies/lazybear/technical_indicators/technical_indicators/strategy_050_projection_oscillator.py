#!/usr/bin/env python3
"""
Strategy 050: Projection Oscillator

Pine source is incomplete in the repo; consistent with Projection Bands,
we implement the canonical oscillator:
  proj_osc = (close - lpb) / (upb - lpb)
where upb/lpb are Projection Bands (Strategy 049). This yields 0..1 scale.

Deterministic trading rule:
- Buy when proj_osc crosses above high_threshold (default 0.8)
- Sell when proj_osc crosses below low_threshold (default 0.2)
"""

import pandas as pd
import numpy as np
from typing import Dict
import sys, os

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)
from strategies.base_strategy import BaseStrategy

try:
    sys.path.insert(0, os.path.join(project_root, 'src'))
    from utils.vectorized_helpers import crossover, crossunder, calculate_signal_strength
except Exception:
    def crossover(a,b): return (a>b) & (a.shift(1)<=b)
    def crossunder(a,b): return (a<b) & (a.shift(1)>=b)
    def calculate_signal_strength(fs,weights=None):
        df=pd.concat(fs,axis=1); return df.mean(axis=1).clip(0,1)


class Strategy050ProjectionOscillator(BaseStrategy):
    def __init__(self, parameters: Dict = None):
        params = {
            'length': 14,
            'high_threshold': 0.8,
            'low_threshold': 0.2,
            'signal_threshold': 0.6,
        }
        if parameters:
            params.update(parameters)
        super().__init__('Strategy_050_ProjectionOscillator', params)

    @staticmethod
    def _rolling_slope(y: pd.Series, n: int, idx: pd.Series) -> pd.Series:
        sum_x = idx.rolling(n, min_periods=1).sum()
        sum_x2 = (idx*idx).rolling(n, min_periods=1).sum()
        sum_y = y.rolling(n, min_periods=1).sum()
        sum_xy = (idx*y).rolling(n, min_periods=1).sum()
        denom = (n*sum_x2 - (sum_x**2)).replace(0,np.nan)
        slope = (n*sum_xy - sum_x*sum_y) / denom
        return slope.fillna(0)

    def _projection_bands(self, df: pd.DataFrame, n: int) -> pd.DataFrame:
        idx = pd.Series(np.arange(len(df)) + 1, index=df.index, dtype=float)
        rlh = self._rolling_slope(df['high'], n, idx)
        rll = self._rolling_slope(df['low'], n, idx)
        ups = [df['high']]
        dws = [df['low']]
        for k in range(1, n):
            ups.append(df['high'].shift(k) + k*rlh)
            dws.append(df['low'].shift(k) + k*rll)
        upb = pd.concat(ups, axis=1).max(axis=1)
        lpb = pd.concat(dws, axis=1).min(axis=1)
        return upb, lpb

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        for c in ('open','high','low','close'):
            if c not in data.columns and 'price' in data.columns:
                data[c] = data['price']
        n = int(self.parameters['length'])
        upb, lpb = self._projection_bands(data, n)
        width = (upb - lpb).replace(0, np.nan)
        osc = ((data['close'] - lpb) / width).clip(0,1)
        data['proj_up'] = upb
        data['proj_dn'] = lpb
        data['proj_osc'] = osc.fillna(0)
        return data

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        hi = float(self.parameters['high_threshold'])
        lo = float(self.parameters['low_threshold'])
        buy = crossover(data['proj_osc'], hi)
        sell = crossunder(data['proj_osc'], lo)
        data['buy_signal'] = buy
        data['sell_signal'] = sell
        st = pd.Series(0.0, index=data.index)
        st[buy] = (data['proj_osc'] - hi).clip(0,1)[buy]
        st[sell] = (lo - data['proj_osc']).clip(0,1)[sell]
        strength = calculate_signal_strength([st.fillna(0)],[1.0])
        thr = float(self.parameters['signal_threshold'])
        strength[(buy|sell) & (strength < thr)] = thr
        strength[~(buy|sell)] = 0.0
        data['signal_strength'] = strength.clip(0,1)
        return data

