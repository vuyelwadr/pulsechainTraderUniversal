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

class Strategy137LeaderOfMACD(BaseStrategy):
    def __init__(self, parameters: Dict=None):
        params={'fast':12,'slow':26,'signal':9,'lead_len':5,'signal_threshold':0.6}
        if parameters: params.update(parameters)
        super().__init__('Strategy_137_Leader_of_MACD', params)
    @staticmethod
    def _ema(s:pd.Series,n:int)->pd.Series:
        return s.ewm(span=max(1,int(n)), adjust=False, min_periods=1).mean()
    def calculate_indicators(self, data:pd.DataFrame)->pd.DataFrame:
        if 'close' not in data.columns and 'price' in data.columns: data['close']=data['price']
        macd=self._ema(data['close'], int(self.parameters['fast'])) - self._ema(data['close'], int(self.parameters['slow']))
        macd_sig=self._ema(macd, int(self.parameters['signal']))
        # Leader: short EMA of MACD histogram
        hist=macd - macd_sig
        leader=hist.ewm(span=max(1,int(self.parameters['lead_len'])), adjust=False, min_periods=1).mean()
        data['macd_leader']=leader
        data['macd_hist']=hist
        return data
    def generate_signals(self, data:pd.DataFrame)->pd.DataFrame:
        buy=crossover(data['macd_leader'], 0.0)
        sell=crossunder(data['macd_leader'], 0.0)
        data['buy_signal']=buy; data['sell_signal']=sell
        st=(data['macd_leader'].abs() / (data['macd_hist'].rolling(50, min_periods=10).std(ddof=0).replace(0,np.nan))).fillna(0).clip(0,1)
        st=st.where(buy|sell,0.0)
        thr=float(self.parameters['signal_threshold']); st[(buy|sell)&(st<thr)]=thr
        data['signal_strength']=st
        return data

