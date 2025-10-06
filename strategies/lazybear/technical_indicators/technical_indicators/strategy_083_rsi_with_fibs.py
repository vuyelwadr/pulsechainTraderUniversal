#!/usr/bin/env python3
"""
Strategy 083: RSI with Fibs

Compute RSI and use Fibonacci levels (38.2, 61.8) as dynamic zones.
Signals: cross up through 38.2 (recovering) and cross down through 61.8 (weakening).
"""

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


class Strategy083RSIWithFibs(BaseStrategy):
    def __init__(self, parameters: Dict=None):
        params={'period':14,'fib_low':38.2,'fib_high':61.8,'signal_threshold':0.6}
        if parameters: params.update(parameters)
        super().__init__('Strategy_083_RSI_With_Fibs', params)
    def calculate_indicators(self, data:pd.DataFrame)->pd.DataFrame:
        if 'close' not in data.columns and 'price' in data.columns: data['close']=data['price']
        n=int(self.parameters['period'])
        delta=data['close'].diff(); gain=delta.clip(lower=0); loss=-delta.clip(upper=0)
        avg_g=pine_rma(gain,n); avg_l=pine_rma(loss,n)
        rs=avg_g/avg_l.replace(0,np.nan); data['rsi']=(100-100/(1+rs)).fillna(50)
        return data
    def generate_signals(self, data:pd.DataFrame)->pd.DataFrame:
        fl=float(self.parameters['fib_low']); fh=float(self.parameters['fib_high'])
        buy=crossover(data['rsi'], fl); sell=crossunder(data['rsi'], fh)
        data['buy_signal']=buy; data['sell_signal']=sell
        st=((data['rsi']-50).abs()/50.0).clip(0,1).where(buy|sell,0.0)
        thr=float(self.parameters['signal_threshold']); st[(buy|sell)&(st<thr)]=thr
        data['signal_strength']=st
        return data

