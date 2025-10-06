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
    from utils.vectorized_helpers import pine_rma, crossover, crossunder
except Exception:
    def pine_rma(s,n): return s.ewm(alpha=1/max(1,int(n)), adjust=False, min_periods=1).mean()
    def crossover(a,b): return (a>b)&(a.shift(1)<=b)
    def crossunder(a,b): return (a<b)&(a.shift(1)>=b)

class Strategy136DTOscillator(BaseStrategy):
    def __init__(self, parameters: Dict=None):
        params={'rsi_period':14,'stoch_len':10,'smooth_k':3,'smooth_d':3,'signal_threshold':0.6}
        if parameters: params.update(parameters)
        super().__init__('Strategy_136_DT_Oscillator', params)
    def calculate_indicators(self, data:pd.DataFrame)->pd.DataFrame:
        if 'close' not in data.columns and 'price' in data.columns: data['close']=data['price']
        n=int(self.parameters['rsi_period'])
        delta=data['close'].diff(); gain=delta.clip(lower=0); loss=-delta.clip(upper=0)
        avg_g=pine_rma(gain,n); avg_l=pine_rma(loss,n)
        rs=avg_g/avg_l.replace(0,np.nan); rsi=(100-100/(1+rs)).fillna(50)
        length=int(self.parameters['stoch_len'])
        ll=rsi.rolling(length, min_periods=1).min(); hh=rsi.rolling(length, min_periods=1).max()
        k=100*(rsi-ll)/(hh-ll).replace(0,np.nan)
        k_s=k.rolling(int(self.parameters['smooth_k']), min_periods=1).mean()
        d=k_s.rolling(int(self.parameters['smooth_d']), min_periods=1).mean()
        data['dt_k']=k_s.fillna(0); data['dt_d']=d.fillna(0)
        return data
    def generate_signals(self, data:pd.DataFrame)->pd.DataFrame:
        buy=crossover(data['dt_k'], data['dt_d'])
        sell=crossunder(data['dt_k'], data['dt_d'])
        data['buy_signal']=buy; data['sell_signal']=sell
        st=(data['dt_k']-data['dt_d']).abs()/100.0
        st=st.clip(0,1).where(buy|sell,0.0)
        thr=float(self.parameters['signal_threshold']); st[(buy|sell)&(st<thr)]=thr
        data['signal_strength']=st
        return data

