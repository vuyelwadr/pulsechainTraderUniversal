#!/usr/bin/env python3
"""
Strategy 078: RSI with Volume (adapted)

Compute RSI and use a volume ratio filter to confirm signals.
Deterministic rule: RSI crosses OB/OS with vol_ratio > 1 for buys and >1 for sells
(volume confirmation for both directions).
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


class Strategy078RSIWithVolume(BaseStrategy):
    def __init__(self, parameters: Dict=None):
        params={'rsi_period':14,'overbought':70,'oversold':30,'vol_len':20,'signal_threshold':0.6}
        if parameters: params.update(parameters)
        super().__init__('Strategy_078_RSI_With_Volume', params)
    def calculate_indicators(self, data:pd.DataFrame)->pd.DataFrame:
        if 'close' not in data.columns and 'price' in data.columns: data['close']=data['price']
        if 'volume' not in data.columns: data['volume']=0
        n=int(self.parameters['rsi_period'])
        delta=data['close'].diff(); gain=delta.clip(lower=0); loss=-delta.clip(upper=0)
        avg_g=pine_rma(gain,n); avg_l=pine_rma(loss,n)
        rs=avg_g/avg_l.replace(0,np.nan); data['rsi']=(100-100/(1+rs)).fillna(50)
        vlen=int(self.parameters['vol_len'])
        vma=data['volume'].rolling(vlen, min_periods=1).mean().replace(0,np.nan)
        data['vol_ratio']=(data['volume']/vma).replace([np.inf,-np.inf],np.nan).fillna(0)
        return data
    def generate_signals(self, data:pd.DataFrame)->pd.DataFrame:
        ob=float(self.parameters['overbought']); os_=float(self.parameters['oversold'])
        vr_ok=data['vol_ratio']>1.0
        buy=crossover(data['rsi'], os_) & vr_ok
        sell=crossunder(data['rsi'], ob) & vr_ok
        data['buy_signal']=buy; data['sell_signal']=sell
        st=((data['rsi']-50).abs()/50.0).clip(0,1).where(buy|sell,0.0)
        thr=float(self.parameters['signal_threshold']); st[(buy|sell)&(st<thr)]=thr
        data['signal_strength']=st
        return data

