#!/usr/bin/env python3
"""
Strategy 082: RSI of MACD

Compute MACD line then RSI of that line. Signals on OB/OS crosses.
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


class Strategy082RSIofMACD(BaseStrategy):
    def __init__(self, parameters: Dict=None):
        params={'fast':12,'slow':26,'signal':9,'rsi_period':14,'overbought':70,'oversold':30,'signal_threshold':0.6}
        if parameters: params.update(parameters)
        super().__init__('Strategy_082_RSI_of_MACD', params)
    @staticmethod
    def _ema(s:pd.Series,n:int)->pd.Series:
        return s.ewm(span=max(1,int(n)), adjust=False, min_periods=1).mean()
    def calculate_indicators(self, data:pd.DataFrame)->pd.DataFrame:
        if 'close' not in data.columns and 'price' in data.columns: data['close']=data['price']
        fast=self._ema(data['close'], int(self.parameters['fast']))
        slow=self._ema(data['close'], int(self.parameters['slow']))
        macd=fast-slow
        n=int(self.parameters['rsi_period'])
        delta=macd.diff(); gain=delta.clip(lower=0); loss=-delta.clip(upper=0)
        avg_g=pine_rma(gain,n); avg_l=pine_rma(loss,n)
        rs=avg_g/avg_l.replace(0,np.nan); data['rsi_macd']=(100-100/(1+rs)).fillna(50)
        return data
    def generate_signals(self, data:pd.DataFrame)->pd.DataFrame:
        ob=float(self.parameters['overbought']); os_=float(self.parameters['oversold'])
        buy=crossover(data['rsi_macd'], os_); sell=crossunder(data['rsi_macd'], ob)
        data['buy_signal']=buy; data['sell_signal']=sell
        st=((data['rsi_macd']-50).abs()/50.0).clip(0,1).where(buy|sell,0.0)
        thr=float(self.parameters['signal_threshold']); st[(buy|sell)&(st<thr)]=thr
        data['signal_strength']=st
        return data

