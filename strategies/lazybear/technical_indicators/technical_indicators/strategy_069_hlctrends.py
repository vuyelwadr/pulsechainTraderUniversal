#!/usr/bin/env python3
"""
Strategy 069: HLCTrends

No Pine in repo; implement trend on HLC/3 (typical price) using dual EMA crossover.
Deterministic rule: EMA(fast) cross above/below EMA(slow) with simple strength.
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


class Strategy069HLCTrends(BaseStrategy):
    def __init__(self, parameters: Dict=None):
        params = {'fast': 21, 'slow': 50, 'signal_threshold': 0.6}
        if parameters: params.update(parameters)
        super().__init__('Strategy_069_HLCTrends', params)
    @staticmethod
    def _ema(s: pd.Series, n: int) -> pd.Series:
        return s.ewm(span=max(1,int(n)), adjust=False, min_periods=1).mean()
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        for c in ('high','low','close'):
            if c not in data.columns and 'price' in data.columns:
                data[c]=data['price']
        tp = (data['high']+data['low']+data['close'])/3.0
        data['ema_fast']=self._ema(tp, int(self.parameters['fast']))
        data['ema_slow']=self._ema(tp, int(self.parameters['slow']))
        return data
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        buy=crossover(data['ema_fast'], data['ema_slow'])
        sell=crossunder(data['ema_fast'], data['ema_slow'])
        data['buy_signal']=buy; data['sell_signal']=sell
        sep=((data['ema_fast']-data['ema_slow']).abs()/data['close'].replace(0,np.nan)).fillna(0).clip(0,1)
        st=sep.where(buy|sell,0.0)
        thr=float(self.parameters['signal_threshold'])
        st[(buy|sell)&(st<thr)]=thr
        data['signal_strength']=st
        return data

