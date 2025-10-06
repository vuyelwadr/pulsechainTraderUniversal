#!/usr/bin/env python3
"""
Strategy 079: RSI using SMA smoothing (not RMA)

Compute RSI with SMA of gains/losses.
Signals: RSI crosses OB/OS levels.
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
    from utils.vectorized_helpers import crossover, crossunder
except Exception:
    def crossover(a,b): return (a>b)&(a.shift(1)<=b)
    def crossunder(a,b): return (a<b)&(a.shift(1)>=b)


class Strategy079RSIUsingSMA(BaseStrategy):
    def __init__(self, parameters: Dict=None):
        params={'period':14,'overbought':70,'oversold':30,'signal_threshold':0.6}
        if parameters: params.update(parameters)
        super().__init__('Strategy_079_RSI_Using_SMA', params)
    def calculate_indicators(self, data:pd.DataFrame)->pd.DataFrame:
        if 'close' not in data.columns and 'price' in data.columns: data['close']=data['price']
        n=int(self.parameters['period'])
        delta=data['close'].diff(); gain=delta.clip(lower=0); loss=-delta.clip(upper=0)
        avg_g=gain.rolling(n, min_periods=1).mean(); avg_l=loss.rolling(n, min_periods=1).mean()
        rs=avg_g/avg_l.replace(0,np.nan); data['rsi_sma']=(100-100/(1+rs)).fillna(50)
        return data
    def generate_signals(self, data:pd.DataFrame)->pd.DataFrame:
        ob=float(self.parameters['overbought']); os_=float(self.parameters['oversold'])
        buy=crossover(data['rsi_sma'], os_); sell=crossunder(data['rsi_sma'], ob)
        data['buy_signal']=buy; data['sell_signal']=sell
        st=((data['rsi_sma']-50).abs()/50.0).clip(0,1).where(buy|sell,0.0)
        thr=float(self.parameters['signal_threshold']); st[(buy|sell)&(st<thr)]=thr
        data['signal_strength']=st
        return data

