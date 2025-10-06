#!/usr/bin/env python3
"""
Strategy 026: Elliott Wave Oscillator (EWO)

TradingView URL: https://www.tradingview.com/v/uculwCTj/
Type: momentum/trend

Description:
EWO = EMA(5) - EMA(35) (or median price variant). Signals on zero cross and
signal line crossover.
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


class Strategy026EWO(BaseStrategy):
    def __init__(self, parameters: Dict=None):
        params={'fast':5,'slow':35,'signal':9,'use_median_price':False,'signal_threshold':0.6}
        if parameters: params.update(parameters)
        super().__init__('Strategy_026_EWO', params)

    @staticmethod
    def _ema(s:pd.Series,n:int)->pd.Series:
        return s.ewm(span=n,adjust=False,min_periods=1).mean()

    def calculate_indicators(self, data:pd.DataFrame)->pd.DataFrame:
        for c in ('close','high','low'):
            if c not in data.columns and 'price' in data.columns:
                data[c]=data['price']
        src = ((data['high']+data['low'])/2.0) if self.parameters['use_median_price'] else data['close']
        ema_f=self._ema(src, self.parameters['fast'])
        ema_s=self._ema(src, self.parameters['slow'])
        ewo=ema_f-ema_s
        sig=self._ema(ewo, self.parameters['signal'])
        data['ewo']=ewo; data['ewo_sig']=sig
        return data

    def generate_signals(self, data:pd.DataFrame)->pd.DataFrame:
        p=self.parameters
        data['buy_signal']=False; data['sell_signal']=False; data['signal_strength']=0.0
        up = crossover(data['ewo'], 0) | crossover(data['ewo'], data['ewo_sig'])
        dn = crossunder(data['ewo'], 0) | crossunder(data['ewo'], data['ewo_sig'])
        data['buy_signal']=up; data['sell_signal']=dn
        mag=(data['ewo'].abs()/data['ewo'].abs().rolling(50,min_periods=1).max().replace(0,np.nan)).fillna(0).clip(0,1)
        data['signal_strength']=calculate_signal_strength([mag],[1.0])
        weak=data['signal_strength']<p['signal_threshold']
        data.loc[weak,['buy_signal','sell_signal']]=False
        data['buy_signal'],data['sell_signal']=apply_position_constraints(data['buy_signal'],data['sell_signal'])
        return data

