#!/usr/bin/env python3
"""
Strategy 104: Kaufman Stress Indicator (single-instrument proxy)

Original uses a second instrument; this proxy computes two stochastic values on
the same instrument with different lookbacks and takes their difference as stress.
Signals when stress crosses a midline.
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


class Strategy104KaufmanStressIndicator(BaseStrategy):
    def __init__(self, parameters: Dict=None):
        params = {
            'len_fast': 20,
            'len_slow': 60,
            'signal_threshold': 0.6,
        }
        if parameters:
            params.update(parameters)
        super().__init__('Strategy_104_Kaufman_Stress_Indicator', params)

    def _stoch(self, c: pd.Series, h: pd.Series, l: pd.Series, n: int) -> pd.Series:
        hh=h.rolling(max(1,int(n)),min_periods=1).max(); ll=l.rolling(max(1,int(n)),min_periods=1).min()
        return ((c - ll) / (hh - ll).replace(0,np.nan) * 100.0).fillna(50)

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        for c in ('high','low','close'):
            if c not in data:
                data[c]=data.get('close', pd.Series(index=data.index)).fillna(method='ffill')
        s1=self._stoch(data['close'], data['high'], data['low'], int(self.parameters['len_fast']))
        s2=self._stoch(data['close'], data['high'], data['low'], int(self.parameters['len_slow']))
        data['ksi_stress'] = s1 - s2
        return data

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        buy=crossover(data['ksi_stress'], 0.0)
        sell=crossunder(data['ksi_stress'], 0.0)
        data['buy_signal']=buy
        data['sell_signal']=sell
        st=(data['ksi_stress'].abs()/ (data['ksi_stress'].rolling(50,min_periods=10).std(ddof=0).replace(0,np.nan))).clip(0,1).fillna(0)
        st=st.where(buy|sell,0.0)
        thr=float(self.parameters['signal_threshold'])
        st[(buy|sell)&(st<thr)]=thr
        data['signal_strength']=st
        return data

