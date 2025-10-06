#!/usr/bin/env python3
"""
Strategy 120: Ehlers Smoothed Stochastic

Computes Stochastic on EMA-smoothed price and further smooths %K/%D.
Signals on %K crossing %D.
"""

import pandas as pd
import numpy as np
from typing import Dict
import os, sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)
from strategies.base_strategy import BaseStrategy

try:
    sys.path.insert(0, os.path.join(project_root, 'src'))
    from utils.vectorized_helpers import crossover, crossunder
except Exception:
    def crossover(a,b): return (a>b) & (a.shift(1)<=b)
    def crossunder(a,b): return (a<b) & (a.shift(1)>=b)


class Strategy120EhlersSmoothedStochastic(BaseStrategy):
    def __init__(self, parameters: Dict=None):
        params = {
            'length': 14,
            'ema_len': 5,
            'smooth_k': 3,
            'smooth_d': 3,
            'signal_threshold': 0.6,
        }
        if parameters:
            params.update(parameters)
        super().__init__('Strategy_120_Ehlers_Smoothed_Stochastic', params)

    @staticmethod
    def _ema(s: pd.Series, n: int) -> pd.Series:
        return s.ewm(span=max(1,int(n)), adjust=False, min_periods=1).mean()

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        for c in ('high','low','close'):
            if c not in data:
                data[c]=data.get('close', pd.Series(index=data.index)).fillna(method='ffill')
        p=self.parameters
        sm_close = self._ema(data['close'], int(p['ema_len']))
        hh = sm_close.rolling(int(p['length']), min_periods=1).max()
        ll = sm_close.rolling(int(p['length']), min_periods=1).min()
        k = 100.0 * (sm_close - ll) / (hh - ll).replace(0,np.nan)
        k = self._ema(k, int(p['smooth_k']))
        d = self._ema(k, int(p['smooth_d']))
        data['stk_k'] = k.fillna(50)
        data['stk_d'] = d.fillna(50)
        return data

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        buy = crossover(data['stk_k'], data['stk_d'])
        sell = crossunder(data['stk_k'], data['stk_d'])
        data['buy_signal'] = buy
        data['sell_signal'] = sell
        st = (data['stk_k'] - data['stk_d']).abs() / 100.0
        st = st.clip(0,1).fillna(0).where(buy|sell, 0.0)
        thr=float(self.parameters['signal_threshold'])
        st[(buy|sell) & (st < thr)] = thr
        data['signal_strength'] = st
        return data

