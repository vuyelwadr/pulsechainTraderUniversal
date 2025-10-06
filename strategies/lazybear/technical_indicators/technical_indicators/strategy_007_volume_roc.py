#!/usr/bin/env python3
"""
Strategy 007: Volume ROC

TradingView URL: https://www.tradingview.com/v/dB56X9AU/
Type: volume/momentum
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
    from utils.vectorized_helpers import crossover, crossunder, calculate_signal_strength, apply_position_constraints
except Exception:
    def crossover(a,b): return (a>b)&(a.shift(1)<=b.shift(1))
    def crossunder(a,b): return (a<b)&(a.shift(1)>=b.shift(1))
    def calculate_signal_strength(fs,weights=None):
        df=pd.concat(fs,axis=1); return df.mean(axis=1).clip(0,1)
    def apply_position_constraints(b,s,allow_short=False): return b,s

class Strategy007VolumeROC(BaseStrategy):
    def __init__(self, parameters: Dict=None):
        params={'period':14,'pos_th':5.0,'neg_th':-5.0,'signal_threshold':0.6}
        if parameters: params.update(parameters)
        super().__init__('Strategy_007_VolumeROC', params)
    def calculate_indicators(self, data:pd.DataFrame)->pd.DataFrame:
        if 'volume' not in data.columns:
            data['volume']=0
        v=data['volume']
        n=self.parameters['period']
        vroc=((v - v.shift(n)) / v.shift(n).replace(0,np.nan) * 100.0).fillna(0)
        data['vroc']=vroc
        return data
    def generate_signals(self, data:pd.DataFrame)->pd.DataFrame:
        p=self.parameters
        data['buy_signal']=False; data['sell_signal']=False; data['signal_strength']=0.0
        up=crossover(data['vroc'], p['pos_th'])
        dn=crossunder(data['vroc'], p['neg_th'])
        data['buy_signal']=up; data['sell_signal']=dn
        mag=(data['vroc'].abs()/100.0).clip(0,1)
        data['signal_strength']=calculate_signal_strength([mag],[1.0])
        weak=data['signal_strength']<p['signal_threshold']
        data.loc[weak,['buy_signal','sell_signal']]=False
        data['buy_signal'],data['sell_signal']=apply_position_constraints(data['buy_signal'],data['sell_signal'])
        return data

