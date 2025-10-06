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

class Strategy138ZeroLagEMAKAMAMACDLeader(BaseStrategy):
    def __init__(self, parameters: Dict=None):
        params={'length':20,'fast':12,'slow':26,'signal':9,'signal_threshold':0.6}
        if parameters: params.update(parameters)
        super().__init__('Strategy_138_ZeroLag_EMA_KAMA_MACD_Leader', params)
    @staticmethod
    def _ema(s:pd.Series,n:int)->pd.Series:
        return s.ewm(span=max(1,int(n)), adjust=False, min_periods=1).mean()
    def _zlema(self, s:pd.Series, n:int)->pd.Series:
        ema1=self._ema(s,n); ema2=self._ema(ema1,n); return 2*ema1 - ema2
    def _kama(self, s:pd.Series, n:int)->pd.Series:
        # Simple KAMA approximation
        change=s.diff(n).abs()
        volatility=s.diff().abs().rolling(n, min_periods=1).sum().replace(0,np.nan)
        er=(change/volatility).fillna(0)
        fast=2/(2+1); slow=2/(30+1)
        sc=(er*(fast-slow)+slow)**2
        kama=np.zeros(len(s)); arr=s.to_numpy(); sc_arr=sc.to_numpy()
        for i in range(len(s)):
            if i==0: kama[i]=arr[i]
            else: kama[i]=kama[i-1]+sc_arr[i]*(arr[i]-kama[i-1])
        return pd.Series(kama, index=s.index)
    def calculate_indicators(self, data:pd.DataFrame)->pd.DataFrame:
        if 'close' not in data.columns and 'price' in data.columns: data['close']=data['price']
        zle=self._zlema(data['close'], int(self.parameters['fast']))
        ka=self._kama(data['close'], int(self.parameters['slow']))
        macd=zle - ka
        sig=self._ema(macd, int(self.parameters['signal']))
        leader=(macd - sig).ewm(span=5, adjust=False, min_periods=1).mean()
        data['zk_macd']=macd; data['zk_signal']=sig; data['zk_leader']=leader
        return data
    def generate_signals(self, data:pd.DataFrame)->pd.DataFrame:
        buy=crossover(data['zk_leader'], 0.0)
        sell=crossunder(data['zk_leader'], 0.0)
        data['buy_signal']=buy; data['sell_signal']=sell
        st=(data['zk_leader'].abs() / (data['zk_leader'].rolling(50, min_periods=10).std(ddof=0).replace(0,np.nan))).fillna(0).clip(0,1)
        st=st.where(buy|sell,0.0)
        thr=float(self.parameters['signal_threshold']); st[(buy|sell)&(st<thr)]=thr
        data['signal_strength']=st
        return data

