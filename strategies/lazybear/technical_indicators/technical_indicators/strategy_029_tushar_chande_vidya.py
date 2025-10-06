#!/usr/bin/env python3
"""
Strategy 029: Tushar Chande VIDYA (Variable Index Dynamic Average)

TradingView URL: https://www.tradingview.com/v/wZGOIz9r/
Type: trend/volatility-adaptive

Description:
VIDYA adapts its smoothing based on CMO. Implements a common formulation:
 alpha = (|CMO(n)|/100) * 2/(length+1); vidya = alpha*close + (1-alpha)*vidya_prev
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


class Strategy029VIDYA(BaseStrategy):
    def __init__(self, parameters: Dict=None):
        params={'length':20,'cmo_period':9,'signal_ma':50,'signal_threshold':0.6}
        if parameters: params.update(parameters)
        super().__init__('Strategy_029_VIDYA', params)

    def calculate_indicators(self, data:pd.DataFrame)->pd.DataFrame:
        if 'close' not in data.columns and 'price' in data.columns:
            data['close']=data['price']
        # CMO
        n=self.parameters['cmo_period']
        delta=data['close'].diff()
        up=delta.clip(lower=0).rolling(n,min_periods=1).sum()
        dn=(-delta.clip(upper=0)).rolling(n,min_periods=1).sum()
        cmo=100*(up-dn)/(up+dn).replace(0,np.nan)
        cmo=cmo.fillna(0)
        # alpha
        alpha=(cmo.abs()/100.0)*(2/(self.parameters['length']+1))
        # VIDYA recursive (vectorized via loop acceptable for single column)
        vidya=pd.Series(index=data.index, dtype=float)
        prev=data['close'].iloc[0]
        for i,(cl,a) in enumerate(zip(data['close'], alpha)):
            prev=a*cl+(1-a)*prev
            vidya.iloc[i]=prev
        data['vidya']=vidya
        data['signal_ma']=data['close'].rolling(self.parameters['signal_ma'],min_periods=1).mean()
        return data

    def generate_signals(self, data:pd.DataFrame)->pd.DataFrame:
        data['buy_signal']=False; data['sell_signal']=False; data['signal_strength']=0.0
        up=crossover(data['close'], data['vidya'])
        dn=crossunder(data['close'], data['vidya'])
        # Optional trend filter via signal_ma
        up &= data['close']>data['signal_ma']
        dn &= data['close']<data['signal_ma']
        data['buy_signal']=up; data['sell_signal']=dn
        slope=(data['vidya']-data['vidya'].shift(5)).abs()
        strength=(slope/(data['close'].rolling(50,min_periods=1).std(ddof=0)+1e-9)).clip(0,1)
        data['signal_strength']=calculate_signal_strength([strength],[1.0])
        weak=data['signal_strength']<self.parameters['signal_threshold']
        data.loc[weak,['buy_signal','sell_signal']]=False
        return data

