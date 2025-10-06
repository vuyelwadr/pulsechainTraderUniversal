#!/usr/bin/env python3
"""
Strategy 087: Volume-Weighted MACD Histogram (adapted)

Compute VWMA-based EMAs for MACD (fast/slow) and histogram vs signal.
Signals on histogram zero-line crosses.
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


class Strategy087VolumeWeightedMACDHistogram(BaseStrategy):
    def __init__(self, parameters: Dict=None):
        params={'fast':12,'slow':26,'signal':9,'vwma_len':20,'signal_threshold':0.6}
        if parameters: params.update(parameters)
        super().__init__('Strategy_087_VolumeWeighted_MACD_Hist', params)
    def _vwma(self, price: pd.Series, volume: pd.Series, n:int)->pd.Series:
        n=max(1,int(n)); pv=(price*volume).rolling(n, min_periods=1).sum(); v=volume.rolling(n, min_periods=1).sum().replace(0,np.nan)
        return (pv/v).fillna(method='ffill').fillna(0)
    def _ema(self, s:pd.Series, n:int)->pd.Series:
        return s.ewm(span=max(1,int(n)), adjust=False, min_periods=1).mean()
    def calculate_indicators(self, data:pd.DataFrame)->pd.DataFrame:
        for c in ('close','volume'):
            if c not in data.columns:
                if c=='close' and 'price' in data.columns: data['close']=data['price']
                else: data['volume']=0
        vw=self._vwma(data['close'], data['volume'], int(self.parameters['vwma_len']))
        macd=self._ema(vw, int(self.parameters['fast'])) - self._ema(vw, int(self.parameters['slow']))
        sig=self._ema(macd, int(self.parameters['signal']))
        data['vw_macd_hist']=macd - sig
        return data
    def generate_signals(self, data:pd.DataFrame)->pd.DataFrame:
        buy=crossover(data['vw_macd_hist'], 0.0); sell=crossunder(data['vw_macd_hist'], 0.0)
        data['buy_signal']=buy; data['sell_signal']=sell
        st=data['vw_macd_hist'].abs()
        denom=data['vw_macd_hist'].rolling(50, min_periods=10).std(ddof=0).replace(0,np.nan)
        st=(st/denom).fillna(0).clip(0,1).where(buy|sell,0.0)
        thr=float(self.parameters['signal_threshold']); st[(buy|sell)&(st<thr)]=thr
        data['signal_strength']=st
        return data

