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

class Strategy130MACZVWAP(BaseStrategy):
    def __init__(self, parameters: Dict=None):
        params={'fast':12,'slow':25,'signal':9,'z_len':20,'stdev_len':25,'A':1.0,'B':1.0,'signal_threshold':0.6}
        if parameters: params.update(parameters)
        super().__init__('Strategy_130_MACZ_VWAP', params)
    @staticmethod
    def _ema(s:pd.Series,n:int)->pd.Series:
        return s.ewm(span=max(1,int(n)), adjust=False, min_periods=1).mean()
    def calculate_indicators(self, data:pd.DataFrame)->pd.DataFrame:
        for c in ('close','volume'):
            if c not in data.columns:
                if c=='close' and 'price' in data.columns: data['close']=data['price']
                else: data['volume']=0
        # MACD
        macd=self._ema(data['close'], int(self.parameters['fast'])) - self._ema(data['close'], int(self.parameters['slow']))
        macd_sig=self._ema(macd, int(self.parameters['signal']))
        # z-VWAP
        n=int(self.parameters['z_len'])
        mean=(data['volume']*data['close']).rolling(n, min_periods=1).sum()/data['volume'].rolling(n, min_periods=1).sum().replace(0,np.nan)
        vwapsd=np.sqrt(data['close'].sub(mean).pow(2).rolling(n, min_periods=1).mean())
        zvwap=(data['close']-mean)/vwapsd.replace(0,np.nan)
        a=float(self.parameters['A']); b=float(self.parameters['B'])
        macz=a*macd + b*zvwap
        data['macz_vwap']=macz.replace([np.inf,-np.inf],np.nan).fillna(0)
        data['macz_signal']=self._ema(data['macz_vwap'], int(self.parameters['signal']))
        return data
    def generate_signals(self, data:pd.DataFrame)->pd.DataFrame:
        buy=crossover(data['macz_vwap'], data['macz_signal'])
        sell=crossunder(data['macz_vwap'], data['macz_signal'])
        data['buy_signal']=buy; data['sell_signal']=sell
        st=(data['macz_vwap']-data['macz_signal']).abs()
        st=(st/(st.rolling(50, min_periods=10).std(ddof=0).replace(0,np.nan))).fillna(0).clip(0,1)
        st=st.where(buy|sell,0.0)
        thr=float(self.parameters['signal_threshold']); st[(buy|sell)&(st<thr)]=thr
        data['signal_strength']=st
        return data

