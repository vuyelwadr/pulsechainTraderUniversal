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

class Strategy132ElasticVWMA(BaseStrategy):
    def __init__(self, parameters: Dict=None):
        params={'length':20,'use_cumulative_volume':False,'signal_threshold':0.6}
        if parameters: params.update(parameters)
        super().__init__('Strategy_132_Elastic_VWMA', params)
    def calculate_indicators(self, data:pd.DataFrame)->pd.DataFrame:
        for c in ('close','high','low','volume'):
            if c not in data.columns:
                if c=='volume': data['volume']=0
                elif 'price' in data.columns: data[c]=data['price']
        n=int(self.parameters['length'])
        nbfs = data['volume'].cumsum() if self.parameters.get('use_cumulative_volume', False) else data['volume'].rolling(n, min_periods=1).sum()
        evwma=np.zeros(len(data))
        nbfs_arr=nbfs.to_numpy(); vol_arr=data['volume'].to_numpy(); price_arr=data['close'].to_numpy()
        for i in range(len(data)):
            if nbfs_arr[i] == 0:
                evwma[i]=price_arr[i] if i==0 else evwma[i-1]
            else:
                prev = evwma[i-1] if i>0 else price_arr[i]
                evwma[i] = prev * (nbfs_arr[i] - vol_arr[i]) / nbfs_arr[i] + (vol_arr[i]*price_arr[i] / nbfs_arr[i])
        data['evwma']=pd.Series(evwma, index=data.index)
        return data
    def generate_signals(self, data:pd.DataFrame)->pd.DataFrame:
        buy=crossover(data['close'], data['evwma'])
        sell=crossunder(data['close'], data['evwma'])
        data['buy_signal']=buy; data['sell_signal']=sell
        st=((data['close']-data['evwma']).abs()/data['close'].replace(0,np.nan)).fillna(0).clip(0,1)
        st=st.where(buy|sell,0.0)
        thr=float(self.parameters['signal_threshold']); st[(buy|sell)&(st<thr)]=thr
        data['signal_strength']=st
        return data

