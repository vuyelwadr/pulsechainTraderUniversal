#!/usr/bin/env python3
"""
Strategy 016: Bears/Bulls Power (Elder)

TradingView URL: https://www.tradingview.com/v/0t5z9xe2/
Type: momentum/volume

Description:
Elder Bulls/Bears Power: Bulls = High - EMA(n), Bears = Low - EMA(n).
Signals:
 - Buy when Bulls cross above 0 and Bears increasing (less negative)
 - Sell when Bulls cross below 0 and Bears decreasing
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


class Strategy016BearsBullsPower(BaseStrategy):
    def __init__(self, parameters: Dict=None):
        params={'ema_period':13,'signal_threshold':0.6}
        if parameters: params.update(parameters)
        super().__init__('Strategy_016_BearsBullsPower',params)

    @staticmethod
    def _ema(s:pd.Series,n:int)->pd.Series:
        return s.ewm(span=n,adjust=False,min_periods=1).mean()

    def calculate_indicators(self, data:pd.DataFrame)->pd.DataFrame:
        for c in ('high','low','close'):
            if c not in data.columns and 'price' in data.columns:
                data[c]=data['price']
        ema=self._ema(data['close'], self.parameters['ema_period'])
        data['bulls']=data['high']-ema
        data['bears']=data['low']-ema
        return data

    def generate_signals(self, data:pd.DataFrame)->pd.DataFrame:
        data['buy_signal']=False; data['sell_signal']=False; data['signal_strength']=0.0
        bulls=data['bulls']; bears=data['bears']
        buy=crossover(bulls,0) & (bears.diff()>0)
        sell=crossunder(bulls,0) & (bears.diff()<0)
        data['buy_signal']=buy; data['sell_signal']=sell
        # Strength: magnitude relative to ATR proxy (std of returns)
        mag=((bulls.abs()+(-bears).clip(lower=0))/2.0)
        strength=(mag/(data['close'].rolling(20,min_periods=1).std(ddof=0)+1e-9)).clip(0,1)
        data['signal_strength']=calculate_signal_strength([strength],[1.0])
        weak=data['signal_strength']<self.parameters['signal_threshold']
        data.loc[weak,['buy_signal','sell_signal']]=False
        data['buy_signal'],data['sell_signal']=apply_position_constraints(data['buy_signal'],data['sell_signal'])
        return data

