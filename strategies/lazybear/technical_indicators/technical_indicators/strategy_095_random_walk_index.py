#!/usr/bin/env python3
"""
Strategy 095: Random Walk Index (RWI)

Computes RWI High and Low per standard formula. Signals long when RWI High
crosses above RWI Low; signals short when inverse crosses.
"""

import pandas as pd
import numpy as np
from typing import Dict
import os, sys, math

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)
from strategies.base_strategy import BaseStrategy

try:
    sys.path.insert(0, os.path.join(project_root, 'src'))
    from utils.vectorized_helpers import crossover, crossunder
except Exception:
    def crossover(a,b): return (a>b) & (a.shift(1)<=b)
    def crossunder(a,b): return (a<b) & (a.shift(1)>=b)


class Strategy095RandomWalkIndex(BaseStrategy):
    def __init__(self, parameters: Dict=None):
        params = {
            'max_n': 14,
            'signal_threshold': 0.6,
        }
        if parameters:
            params.update(parameters)
        super().__init__('Strategy_095_RandomWalkIndex', params)

    @staticmethod
    def _atr(h: pd.Series, l: pd.Series, c: pd.Series, n: int) -> pd.Series:
        pc=c.shift(1)
        tr=pd.concat([(h-l),(h-pc).abs(),(l-pc).abs()],axis=1).max(axis=1)
        return tr.rolling(max(1,int(n)),min_periods=1).mean()

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        for c in ('high','low','close'):
            if c not in data:
                data[c]=data.get('close', pd.Series(index=data.index)).fillna(method='ffill')
        max_n=int(self.parameters['max_n'])
        atr = self._atr(data['high'], data['low'], data['close'], max_n)
        rwi_high = pd.Series(0.0, index=data.index)
        rwi_low = pd.Series(0.0, index=data.index)
        for n in range(2, max_n+1):
            hh = data['high'].rolling(n, min_periods=1).max()
            ll = data['low'].rolling(n, min_periods=1).min()
            rh = (data['high'] - ll.shift(n-1)).abs() / (atr * math.sqrt(n)).replace(0,np.nan)
            rl = (hh.shift(n-1) - data['low']).abs() / (atr * math.sqrt(n)).replace(0,np.nan)
            rwi_high = pd.concat([rwi_high, rh], axis=1).max(axis=1)
            rwi_low = pd.concat([rwi_low, rl], axis=1).max(axis=1)
        data['rwi_high'] = rwi_high.replace([np.inf,-np.inf], np.nan).fillna(0)
        data['rwi_low'] = rwi_low.replace([np.inf,-np.inf], np.nan).fillna(0)
        return data

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        buy = crossover(data['rwi_high'], data['rwi_low'])
        sell = crossunder(data['rwi_high'], data['rwi_low'])
        data['buy_signal'] = buy
        data['sell_signal'] = sell
        st = (data['rwi_high'] - data['rwi_low']).abs()
        st = (st / (st.rolling(50, min_periods=10).std(ddof=0).replace(0,np.nan))).clip(0,1).fillna(0)
        st = st.where(buy|sell, 0.0)
        thr=float(self.parameters['signal_threshold'])
        st[(buy|sell) & (st < thr)] = thr
        data['signal_strength'] = st
        return data

