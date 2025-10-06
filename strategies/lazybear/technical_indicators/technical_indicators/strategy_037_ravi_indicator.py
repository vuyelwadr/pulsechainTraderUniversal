#!/usr/bin/env python3
"""
Strategy 037: RAVI Indicator (Range Action Verification Index)

TradingView URL: https://www.tradingview.com/v/GhxOzF0z/
Type: trend

Description:
RAVI = 100 * (EMA(short) - EMA(long)) / EMA(long). Typical: short=7, long=65.
Signals when RAVI crosses threshold bands.
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


class Strategy037RAVI(BaseStrategy):
    def __init__(self, parameters: Dict=None):
        params={'short':7,'long':65,'upper':0.3,'lower':-0.3,'signal_threshold':0.6}
        if parameters: params.update(parameters)
        super().__init__('Strategy_037_RAVI', params)

    @staticmethod
    def _ema(s:pd.Series,n:int)->pd.Series:
        return s.ewm(span=n,adjust=False,min_periods=1).mean()

    def calculate_indicators(self, data:pd.DataFrame)->pd.DataFrame:
        if 'close' not in data.columns and 'price' in data.columns:
            data['close']=data['price']
        ema_s=self._ema(data['close'], self.parameters['short'])
        ema_l=self._ema(data['close'], self.parameters['long'])
        data['ravi']=100.0*(ema_s-ema_l)/ema_l.replace(0,np.nan)
        return data

    def generate_signals(self, data:pd.DataFrame)->pd.DataFrame:
        p=self.parameters
        data['buy_signal']=False; data['sell_signal']=False; data['signal_strength']=0.0
        up=crossover(data['ravi'], p['upper'])
        dn=crossunder(data['ravi'], p['lower'])
        data['buy_signal']=up; data['sell_signal']=dn
        mag=(data['ravi'].abs()/data['ravi'].abs().rolling(50,min_periods=1).max().replace(0,np.nan)).fillna(0).clip(0,1)
        data['signal_strength']=calculate_signal_strength([mag],[1.0])
        weak=data['signal_strength']<p['signal_threshold']
        data.loc[weak,['buy_signal','sell_signal']]=False
        data['buy_signal'],data['sell_signal']=apply_position_constraints(data['buy_signal'],data['sell_signal'])
        return data

