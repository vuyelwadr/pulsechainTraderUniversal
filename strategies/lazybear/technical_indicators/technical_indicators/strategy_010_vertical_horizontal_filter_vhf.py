#!/usr/bin/env python3
"""
Strategy 010: Vertical Horizontal Filter (VHF)

TradingView URL: https://www.tradingview.com/v/cGtwC2C9/
Type: trend/strength
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

class Strategy010VHF(BaseStrategy):
    def __init__(self, parameters: Dict=None):
        params={'period':28,'threshold':0.2,'signal_threshold':0.6}
        if parameters: params.update(parameters)
        super().__init__('Strategy_010_VHF', params)
    def calculate_indicators(self, data:pd.DataFrame)->pd.DataFrame:
        if 'close' not in data.columns and 'price' in data.columns: data['close']=data['price']
        n=self.parameters['period']
        hh=data['close'].rolling(n,min_periods=1).max()
        ll=data['close'].rolling(n,min_periods=1).min()
        denom=(data['close'].diff().abs()).rolling(n,min_periods=1).sum().replace(0,np.nan)
        data['vhf']=((hh-ll)/denom).fillna(0)
        return data
    def generate_signals(self, data:pd.DataFrame)->pd.DataFrame:
        p=self.parameters
        data['buy_signal']=crossover(data['vhf'], p['threshold'])
        data['sell_signal']=crossunder(data['vhf'], p['threshold'])
        data['signal_strength']=calculate_signal_strength([data['vhf'].clip(0,1)],[1.0])
        weak=data['signal_strength']<p['signal_threshold']
        data.loc[weak,['buy_signal','sell_signal']]=False
        return data

