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

class Strategy139VolatilityBasedTrailingStops(BaseStrategy):
    def __init__(self, parameters: Dict=None):
        params={'atr_len':14,'mult':3.0,'signal_threshold':0.6}
        if parameters: params.update(parameters)
        super().__init__('Strategy_139_Volatility_Trailing_Stops', params)
    def calculate_indicators(self, data:pd.DataFrame)->pd.DataFrame:
        for c in ('high','low','close'):
            if c not in data.columns and 'price' in data.columns: data[c]=data['price']
        n=int(self.parameters['atr_len']); mult=float(self.parameters['mult'])
        tr=pd.concat([
            data['high']-data['low'],
            (data['high']-data['close'].shift(1)).abs(),
            (data['low']-data['close'].shift(1)).abs()
        ],axis=1).max(axis=1)
        atr=tr.rolling(n, min_periods=1).mean()
        # Long trailing stop
        tsl=np.zeros(len(data)); close=data['close'].to_numpy(); atr_arr=atr.to_numpy()
        for i in range(len(data)):
            if i==0:
                tsl[i]=close[i]-mult*atr_arr[i]
            else:
                tsl[i]=max(tsl[i-1], close[i]-mult*atr_arr[i])
        data['trail_stop_long']=pd.Series(tsl,index=data.index)
        return data
    def generate_signals(self, data:pd.DataFrame)->pd.DataFrame:
        buy=crossover(data['close'], data['trail_stop_long'])
        sell=crossunder(data['close'], data['trail_stop_long'])
        data['buy_signal']=buy; data['sell_signal']=sell
        st=((data['close']-data['trail_stop_long']).abs()/data['close'].replace(0,np.nan)).fillna(0).clip(0,1)
        st=st.where(buy|sell,0.0)
        thr=float(self.parameters['signal_threshold']); st[(buy|sell)&(st<thr)]=thr
        data['signal_strength']=st
        return data

