#!/usr/bin/env python3
"""
Strategy 063: Accumulative Swing Index (ASI)

Implements Welles Wilder's Swing Index (SI) and Accumulative Swing Index (ASI).
Signals when smoothed ASI crosses zero.
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
    from utils.vectorized_helpers import crossover, crossunder
except Exception:
    def crossover(a,b): return (a>b) & (a.shift(1)<=b)
    def crossunder(a,b): return (a<b) & (a.shift(1)>=b)


class Strategy063AccumulativeSwingIndex(BaseStrategy):
    def __init__(self, parameters: Dict = None):
        params = {
            'limit_move': 0.25,   # instrument-specific
            'smooth_period': 5,
            'signal_threshold': 0.6,
        }
        if parameters:
            params.update(parameters)
        super().__init__('Strategy_063_ASI', params)

    @staticmethod
    def _ema(s: pd.Series, n: int) -> pd.Series:
        return s.ewm(span=max(1,int(n)), adjust=False, min_periods=1).mean()

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        for c in ('open','high','low','close'):
            if c not in data.columns and 'price' in data.columns:
                data[c]=data['price']
        tr = pd.concat([
            data['high'] - data['low'],
            (data['high'] - data['close'].shift(1)).abs(),
            (data['low'] - data['close'].shift(1)).abs()
        ], axis=1).max(axis=1)
        k_factor = np.maximum(tr, float(self.parameters['limit_move']))

        cy = data['close'] - data['close'].shift(1)
        oy = data['open'] - data['close'].shift(1)
        hy = data['high'] - data['close'].shift(1)
        ly = data['low'] - data['close'].shift(1)
        r = np.where(hy.abs() > ly.abs(), hy, ly)
        si = 50.0 * (cy + 0.5*oy + 0.25*r) / (pd.Series(k_factor, index=data.index).replace(0,np.nan))
        asi = si.cumsum().fillna(0)
        asi_s = self._ema(asi, int(self.parameters['smooth_period']))
        data['si'] = si
        data['asi'] = asi
        data['asi_s'] = asi_s
        return data

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        buy = crossover(data['asi_s'], 0.0)
        sell = crossunder(data['asi_s'], 0.0)
        data['buy_signal'] = buy
        data['sell_signal'] = sell
        st = data['asi_s'].abs()
        # Normalize by rolling std dev
        denom = data['asi_s'].rolling(50, min_periods=10).std().replace(0,np.nan)
        st = (st / denom).fillna(0).clip(0,1)
        st = st.where(buy|sell, 0.0)
        thr = float(self.parameters['signal_threshold'])
        st[(buy|sell) & (st < thr)] = thr
        data['signal_strength'] = st.clip(0,1)
        return data

