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

class Strategy127ZDistanceFromVWAP(BaseStrategy):
    def __init__(self, parameters: Dict=None):
        params={'z_len':20,'signal_threshold':0.6}
        if parameters: params.update(parameters)
        super().__init__('Strategy_127_Z_Distance_From_VWAP', params)
    def calculate_indicators(self, data:pd.DataFrame)->pd.DataFrame:
        for c in ('close','volume'):
            if c not in data.columns:
                if c=='close' and 'price' in data.columns: data['close']=data['price']
                else: data['volume']=0
        cum_pv=(data['close']*data['volume']).cumsum(); cum_v=data['volume'].cumsum().replace(0,np.nan)
        vwap=(cum_pv/cum_v).fillna(method='ffill').fillna(data['close'])
        diff=data['close']-vwap
        z=diff/ diff.rolling(int(self.parameters['z_len']), min_periods=2).std(ddof=0).replace(0,np.nan)
        data['z_vwap']=z.replace([np.inf,-np.inf],np.nan).fillna(0)
        return data
    def generate_signals(self, data:pd.DataFrame)->pd.DataFrame:
        buy=crossover(data['z_vwap'], 0.0)
        sell=crossunder(data['z_vwap'], 0.0)
        data['buy_signal']=buy; data['sell_signal']=sell
        st=data['z_vwap'].abs().clip(0,3)/3.0
        st=st.where(buy|sell,0.0)
        thr=float(self.parameters['signal_threshold']); st[(buy|sell)&(st<thr)]=thr
        data['signal_strength']=st.clip(0,1)
        return data

