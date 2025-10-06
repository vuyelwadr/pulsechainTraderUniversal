#!/usr/bin/env python3
"""
Strategy 070: TRIX Ribbon (simplified)

No Pine file; implement TRIX (triple-smoothed EMA rate-of-change) with signal EMA.
Deterministic rule: TRIX crosses its signal.
"""

import pandas as pd
import numpy as np
from typing import Dict
import sys, os

project_root=os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)
from strategies.base_strategy import BaseStrategy

try:
    sys.path.insert(0, os.path.join(project_root,'src'))
    from utils.vectorized_helpers import crossover, crossunder
except Exception:
    def crossover(a,b): return (a>b)&(a.shift(1)<=b)
    def crossunder(a,b): return (a<b)&(a.shift(1)>=b)


class Strategy070TrixRibbon(BaseStrategy):
    def __init__(self, parameters: Dict=None):
        params={'length':15,'signal':9,'signal_threshold':0.6}
        if parameters: params.update(parameters)
        super().__init__('Strategy_070_TRIX_Ribbon', params)
    @staticmethod
    def _ema(s: pd.Series, n:int)->pd.Series:
        return s.ewm(span=max(1,int(n)), adjust=False, min_periods=1).mean()
    def calculate_indicators(self, data:pd.DataFrame)->pd.DataFrame:
        if 'close' not in data.columns and 'price' in data.columns: data['close']=data['price']
        n=int(self.parameters['length'])
        e1=self._ema(data['close'], n)
        e2=self._ema(e1, n)
        e3=self._ema(e2, n)
        trix = e3.pct_change().fillna(0)
        data['trix']=trix
        data['trix_sig']=self._ema(trix, int(self.parameters['signal']))
        return data
    def generate_signals(self, data:pd.DataFrame)->pd.DataFrame:
        buy=crossover(data['trix'], data['trix_sig'])
        sell=crossunder(data['trix'], data['trix_sig'])
        data['buy_signal']=buy; data['sell_signal']=sell
        st=(data['trix']-data['trix_sig']).abs().clip(0,1)
        st=st.where(buy|sell,0.0)
        thr=float(self.parameters['signal_threshold']); st[(buy|sell)&(st<thr)]=thr
        data['signal_strength']=st
        return data

