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
    def crossover(a,b): return (a>b) & (a.shift(1)<=b)
    def crossunder(a,b): return (a<b) & (a.shift(1)>=b)

class Strategy110BBImpulseIndicator(BaseStrategy):
    def __init__(self, parameters: Dict=None):
        params={'length':20,'mult':2.0,'signal_threshold':0.6}
        if parameters: params.update(parameters)
        super().__init__('Strategy_110_BBImpulse', params)
    def calculate_indicators(self, data:pd.DataFrame)->pd.DataFrame:
        if 'close' not in data.columns and 'price' in data.columns: data['close']=data['price']
        n=int(self.parameters['length'])
        basis=data['close'].rolling(n, min_periods=1).mean()
        dev=self.parameters['mult']*data['close'].rolling(n, min_periods=1).std(ddof=0)
        upper=basis+dev; lower=basis-dev
        impulse=(data['close']-basis)/dev.replace(0,np.nan)
        data['bbimpulse']=impulse.replace([np.inf,-np.inf],np.nan).fillna(0)
        return data
    def generate_signals(self, data:pd.DataFrame)->pd.DataFrame:
        buy=crossover(data['bbimpulse'], 0.0)
        sell=crossunder(data['bbimpulse'], 0.0)
        data['buy_signal']=buy; data['sell_signal']=sell
        st=data['bbimpulse'].abs().clip(0,1).where(buy|sell,0.0)
        thr=float(self.parameters['signal_threshold']); st[(buy|sell)&(st<thr)]=thr
        data['signal_strength']=st
        return data

