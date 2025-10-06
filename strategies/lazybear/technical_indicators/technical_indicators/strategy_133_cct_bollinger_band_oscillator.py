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

class Strategy133CCTBBOscillator(BaseStrategy):
    def __init__(self, parameters: Dict=None):
        params={'length':21,'lengthMA':13,'signal_threshold':0.6}
        if parameters: params.update(parameters)
        super().__init__('Strategy_133_CCT_BB_Oscillator', params)
    def calculate_indicators(self, data:pd.DataFrame)->pd.DataFrame:
        if 'close' not in data.columns and 'price' in data.columns: data['close']=data['price']
        n=int(self.parameters['length'])
        stdev=data['close'].rolling(n, min_periods=1).std(ddof=0)
        cctbbo=100*(data['close'] + 2*stdev - data['close'].rolling(n, min_periods=1).mean())/(4*stdev.replace(0,np.nan))
        data['cct_bbo']=cctbbo.replace([np.inf,-np.inf],np.nan).fillna(0)
        data['cct_bbo_ma']=data['cct_bbo'].ewm(span=max(1,int(self.parameters['lengthMA'])), adjust=False, min_periods=1).mean()
        return data
    def generate_signals(self, data:pd.DataFrame)->pd.DataFrame:
        buy=crossover(data['cct_bbo'], data['cct_bbo_ma'])
        sell=crossunder(data['cct_bbo'], data['cct_bbo_ma'])
        data['buy_signal']=buy; data['sell_signal']=sell
        st=(data['cct_bbo']-data['cct_bbo_ma']).abs()
        st=(st/(st.rolling(50, min_periods=10).std(ddof=0).replace(0,np.nan))).fillna(0).clip(0,1)
        st=st.where(buy|sell,0.0)
        thr=float(self.parameters['signal_threshold']); st[(buy|sell)&(st<thr)]=thr
        data['signal_strength']=st
        return data

