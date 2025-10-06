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

class Strategy135McGinleyDynamicCD(BaseStrategy):
    def __init__(self, parameters: Dict=None):
        params={'length':14,'k':125.0,'signal_threshold':0.6}
        if parameters: params.update(parameters)
        super().__init__('Strategy_135_McGinleyDynamic_CD', params)
    def calculate_indicators(self, data:pd.DataFrame)->pd.DataFrame:
        if 'close' not in data.columns and 'price' in data.columns: data['close']=data['price']
        n=int(self.parameters['length']); k=float(self.parameters['k'])
        md=np.zeros(len(data)); price=data['close'].to_numpy()
        for i in range(len(data)):
            if i==0: md[i]=price[i]
            else:
                ratio = (price[i]/md[i-1]) if md[i-1]!=0 else 1.0
                md[i] = md[i-1] + (price[i]-md[i-1]) / (k*ratio**4)
        data['mcg']=pd.Series(md,index=data.index)
        data['mcg_diff']=data['close']-data['mcg']
        return data
    def generate_signals(self, data:pd.DataFrame)->pd.DataFrame:
        buy=crossover(data['mcg_diff'], 0.0)
        sell=crossunder(data['mcg_diff'], 0.0)
        data['buy_signal']=buy; data['sell_signal']=sell
        st=(data['mcg_diff'].abs()/data['close'].replace(0,np.nan)).fillna(0).clip(0,1)
        st=st.where(buy|sell,0.0)
        thr=float(self.parameters['signal_threshold']); st[(buy|sell)&(st<thr)]=thr
        data['signal_strength']=st
        return data

