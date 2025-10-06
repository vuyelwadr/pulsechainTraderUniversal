#!/usr/bin/env python3
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

class Strategy113DEnvelopeBandwidth(BaseStrategy):
    def __init__(self, parameters: Dict=None):
        params={'lb':20,'de':2.0,'signal_threshold':0.6}
        if parameters: params.update(parameters)
        super().__init__('Strategy_113_DEnvelope_Bandwidth', params)
    @staticmethod
    def _ema(s:pd.Series,n:int)->pd.Series:
        return s.ewm(span=max(1,int(n)), adjust=False, min_periods=1).mean()
    def calculate_indicators(self, data:pd.DataFrame)->pd.DataFrame:
        for c in ('high','low','close'):
            if c not in data.columns and 'price' in data.columns: data[c]=data['price']
        src=(data['high']+data['low']+data['close'])/3.0
        n=int(self.parameters['lb']); de=float(self.parameters['de'])
        dt=self._ema(src, n)
        dt2=self._ema((src-dt).abs(), n)
        up=dt+de*dt2; dn=dt-de*dt2
        data['denv_bw']=((up-dn)/dt.replace(0,np.nan)).replace([np.inf,-np.inf],np.nan).fillna(0)
        return data
    def generate_signals(self, data:pd.DataFrame)->pd.DataFrame:
        buy=crossover(data['denv_bw'], data['denv_bw'].rolling(20, min_periods=1).mean())
        sell=crossunder(data['denv_bw'], data['denv_bw'].rolling(20, min_periods=1).mean())
        data['buy_signal']=buy; data['sell_signal']=sell
        st=(data['denv_bw']-data['denv_bw'].rolling(20, min_periods=1).mean()).abs()
        st=(st / (st.rolling(50, min_periods=10).std(ddof=0).replace(0,np.nan))).fillna(0).clip(0,1)
        st=st.where(buy|sell,0.0)
        thr=float(self.parameters['signal_threshold']); st[(buy|sell)&(st<thr)]=thr
        data['signal_strength']=st
        return data

