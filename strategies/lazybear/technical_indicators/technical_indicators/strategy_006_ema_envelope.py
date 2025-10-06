#!/usr/bin/env python3
"""
Strategy 006: EMA Envelope

TradingView URL: https://www.tradingview.com/v/jRrLUjPJ/
Type: trend/bands
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

class Strategy006EMAEnvelope(BaseStrategy):
    def __init__(self, parameters: Dict=None):
        params={'period':20,'percent':2.0,'signal_threshold':0.6}
        if parameters: params.update(parameters)
        super().__init__('Strategy_006_EMAEnvelope', params)
    @staticmethod
    def _ema(s:pd.Series,n:int)->pd.Series:
        return s.ewm(span=n,adjust=False,min_periods=1).mean()
    def calculate_indicators(self, data:pd.DataFrame)->pd.DataFrame:
        if 'close' not in data.columns and 'price' in data.columns:
            data['close']=data['price']
        # Pine parity: e=ema(close,length), eu=ema(high,length), el=ema(low,length)
        length=int(self.parameters['period'])
        e=self._ema(data['close'], length)
        eu=self._ema(data['high'], length)
        el=self._ema(data['low'], length)
        data['env_mid']=e
        data['env_up']=eu
        data['env_dn']=el
        return data
    def generate_signals(self, data:pd.DataFrame)->pd.DataFrame:
        p=self.parameters
        data['buy_signal']=False; data['sell_signal']=False; data['signal_strength']=0.0
        # Do not derive trading logic beyond pine intent; emit no trades by default
        data['buy_signal']=False
        data['sell_signal']=False
        data['signal_strength']=0.0
        return data
