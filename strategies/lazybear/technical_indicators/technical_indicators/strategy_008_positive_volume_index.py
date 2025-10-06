#!/usr/bin/env python3
"""
Strategy 008: Positive Volume Index (PVI)

TradingView URL: https://www.tradingview.com/v/GMW5uOvc/
Type: volume/trend
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

class Strategy008PositiveVolumeIndex(BaseStrategy):
    def __init__(self, parameters: Dict=None):
        params={'ma_period':255,'signal_threshold':0.6}
        if parameters: params.update(parameters)
        super().__init__('Strategy_008_PVI', params)
    def calculate_indicators(self, data:pd.DataFrame)->pd.DataFrame:
        for c in ('close','volume'):
            if c not in data.columns and 'price' in data.columns:
                data[c]=data['price'] if c=='close' else 0
        pvi=pd.Series(index=data.index, dtype=float)
        pvi.iloc[0]=1000.0
        for i in range(1,len(data)):
            if data['volume'].iloc[i] > data['volume'].iloc[i-1]:
                change=(data['close'].iloc[i]-data['close'].iloc[i-1])/data['close'].iloc[i-1]
                pvi.iloc[i]=pvi.iloc[i-1]*(1+change)
            else:
                pvi.iloc[i]=pvi.iloc[i-1]
        data['pvi']=pvi
        data['pvi_ma']=data['pvi'].rolling(self.parameters['ma_period'],min_periods=1).mean()
        return data
    def generate_signals(self, data:pd.DataFrame)->pd.DataFrame:
        p=self.parameters
        data['buy_signal']=crossover(data['pvi'], data['pvi_ma'])
        data['sell_signal']=crossunder(data['pvi'], data['pvi_ma'])
        data['signal_strength']=calculate_signal_strength([((data['pvi']-data['pvi_ma']).abs()/data['pvi_ma'].replace(0,np.nan)).fillna(0).clip(0,1)],[1.0])
        weak=data['signal_strength']<p['signal_threshold']
        data.loc[weak,['buy_signal','sell_signal']]=False
        return data

