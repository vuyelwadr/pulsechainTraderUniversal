#!/usr/bin/env python3
"""
Strategy 049: Projection Bands

Pine reference: pine_scripts/049_projection_bands.pine
Math mirrors the Pine code:
- length is fixed at 14 in Pine due to lack of loops
- rlh/rll are rolling linear regression slopes of high/low w.r.t. bar index
- upb = max_k(high[k] + k*rlh) for k in 0..13
- lpb = min_k(low[k] + k*rll) for k in 0..13

Deterministic trading rule consistent with band intent:
- Buy when close crosses above upb after being inside the band on the prior bar
- Sell when close crosses below lpb after being inside the band on the prior bar
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
    def crossover(a,b): return (a>b) & (a.shift(1)<= (b.shift(1) if isinstance(b, pd.Series) else b))
    def crossunder(a,b): return (a<b) & (a.shift(1)>= (b.shift(1) if isinstance(b, pd.Series) else b))
    def calculate_signal_strength(fs, weights=None):
        df = pd.concat(fs, axis=1); return df.mean(axis=1).clip(0,1)


class Strategy049ProjectionBands(BaseStrategy):
    def __init__(self, parameters: Dict = None):
        params = {
            'length': 14,            # fixed in Pine script
            'signal_threshold': 0.6,
        }
        if parameters:
            params.update(parameters)
        super().__init__('Strategy_049_ProjectionBands', params)

    @staticmethod
    def _rolling_slope(y: pd.Series, n: int, idx: pd.Series) -> pd.Series:
        # slope = (n*sum(x*y) - sum(x)*sum(y)) / (n*sum(x^2) - (sum(x))^2)
        sum_x = idx.rolling(n, min_periods=1).sum()
        sum_x2 = (idx*idx).rolling(n, min_periods=1).sum()
        sum_y = y.rolling(n, min_periods=1).sum()
        sum_xy = (idx*y).rolling(n, min_periods=1).sum()
        denom = (n*sum_x2 - (sum_x**2))
        denom = denom.replace(0, np.nan)
        slope = (n*sum_xy - sum_x*sum_y) / denom
        return slope.fillna(0)

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        for c in ('open','high','low','close'):
            if c not in data.columns and 'price' in data.columns:
                data[c] = data['price']

        n = int(self.parameters['length'])
        idx = pd.Series(np.arange(len(data)) + 1, index=data.index, dtype=float)

        rlh = self._rolling_slope(data['high'], n, idx)
        rll = self._rolling_slope(data['low'], n, idx)

        # Build candidate projections for k in 0..n-1
        up_candidates = [data['high']]
        low_candidates = [data['low']]
        for k in range(1, n):
            up_candidates.append(data['high'].shift(k) + k*rlh)
            low_candidates.append(data['low'].shift(k) + k*rll)

        upb = pd.concat(up_candidates, axis=1).max(axis=1)
        lpb = pd.concat(low_candidates, axis=1).min(axis=1)

        data['pb_up'] = upb
        data['pb_dn'] = lpb
        return data

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        inside_prev = (data['close'].shift(1) <= data['pb_up'].shift(1)) & (data['close'].shift(1) >= data['pb_dn'].shift(1))
        buy = crossover(data['close'], data['pb_up']) & inside_prev
        sell = crossunder(data['close'], data['pb_dn']) & inside_prev
        data['buy_signal'] = buy
        data['sell_signal'] = sell

        width = (data['pb_up'] - data['pb_dn']).replace(0, np.nan)
        dist = pd.Series(0.0, index=data.index)
        dist[buy] = ((data['close'] - data['pb_up']) / width).clip(0,1)[buy]
        dist[sell] = ((data['pb_dn'] - data['close']) / width).clip(0,1)[sell]
        strength = calculate_signal_strength([dist.fillna(0)], [1.0])
        # Enforce minimum threshold only on signal bars
        thr = float(self.parameters['signal_threshold'])
        strength[(buy | sell) & (strength < thr)] = thr
        strength[~(buy | sell)] = 0.0
        data['signal_strength'] = strength.clip(0,1)
        return data

