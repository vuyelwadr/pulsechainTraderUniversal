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

class Strategy134VervoortSmoothedPercentB(BaseStrategy):
    def __init__(self, parameters: Dict=None):
        params={'length':18,'tema_len':8,'stdev_high':1.6,'stdev_low':1.6,'stdev_len':200,'signal_threshold':0.6}
        if parameters: params.update(parameters)
        super().__init__('Strategy_134_Vervoort_Smoothed_PercentB', params)
    @staticmethod
    def _ema(s:pd.Series,n:int)->pd.Series:
        return s.ewm(span=max(1,int(n)), adjust=False, min_periods=1).mean()
    def _tema(self, s:pd.Series, n:int)->pd.Series:
        e1=self._ema(s,n); e2=self._ema(e1,n); e3=self._ema(e2,n)
        return 3*(e1-e2)+e3
    def calculate_indicators(self, data:pd.DataFrame)->pd.DataFrame:
        for c in ('open','high','low','close'):
            if c not in data.columns and 'price' in data.columns: data[c]=data['price']
        n=int(self.parameters['length']); tlen=int(self.parameters['tema_len'])
        ha_open=(data[['open','high','low','close']].mean(axis=1).shift(1)+data.get('haOpen', pd.Series(0,index=data.index)).shift(1)).div(2).fillna(method='ffill').fillna(data['open'])
        ha_c=(data[['open','high','low','close']].mean(axis=1)+ha_open+data['high'].where(data['high']>ha_open,ha_open)+data['low'].where(data['low']<ha_open,ha_open))/4
        tma1=self._tema(ha_c, tlen)
        tma2=self._tema(tma1, tlen)
        zl=tma1+(tma1-tma2)
        tema_zl=self._tema(zl, tlen)
        std=self._tema(zl, tlen).rolling(n, min_periods=1).std(ddof=0)
        percb=((tema_zl + 2*std - tema_zl.rolling(n, min_periods=1).mean())/(4*std.replace(0,np.nan))*100).replace([np.inf,-np.inf],np.nan).fillna(0)
        data['vervoort_percb']=percb
        # Dynamic bands
        sh=float(self.parameters['stdev_high']); sl=float(self.parameters['stdev_low']); slen=int(self.parameters['stdev_len'])
        ub=50+sh*percb.rolling(slen, min_periods=5).std(ddof=0)
        lb=50-sl*percb.rolling(slen, min_periods=5).std(ddof=0)
        data['vervoort_ub']=ub; data['vervoort_lb']=lb
        return data
    def generate_signals(self, data:pd.DataFrame)->pd.DataFrame:
        buy=crossover(data['vervoort_percb'], data['vervoort_lb'])
        sell=crossunder(data['vervoort_percb'], data['vervoort_ub'])
        data['buy_signal']=buy; data['sell_signal']=sell
        st=(data['vervoort_percb']-50).abs()/50.0
        st=st.clip(0,1).where(buy|sell,0.0)
        thr=float(self.parameters['signal_threshold']); st[(buy|sell)&(st<thr)]=thr
        data['signal_strength']=st
        return data

