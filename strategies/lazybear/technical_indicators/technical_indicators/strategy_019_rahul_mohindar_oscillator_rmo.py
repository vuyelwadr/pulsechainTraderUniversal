#!/usr/bin/env python3
"""
Strategy 019: Rahul Mohindar Oscillator (RMO) - Approximation

TradingView URL: https://www.tradingview.com/v/efHJsedw/
Type: momentum/trend

Description:
Approximate RMO using smoothed momentum: EMA of momentum and its signal line.
Signals on oscillator zero-cross and signal crossover with trend filter.

Note: RMO is proprietary; this is a pragmatic, vectorized approximation pending
exact Pine parity.
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


class Strategy019RMO(BaseStrategy):
    def __init__(self, parameters: Dict=None):
        params={'mom_period':20,'ema1':10,'ema2':30,'signal':9,'trend_ma':50,'signal_threshold':0.6}
        if parameters: params.update(parameters)
        super().__init__('Strategy_019_RMO', params)

    @staticmethod
    def _ema(s:pd.Series,n:int)->pd.Series:
        return s.ewm(span=n,adjust=False,min_periods=1).mean()

    def calculate_indicators(self, data:pd.DataFrame)->pd.DataFrame:
        if 'close' not in data.columns and 'price' in data.columns:
            data['close']=data['price']
        mom=data['close']-data['close'].shift(self.parameters['mom_period'])
        em1=self._ema(mom, self.parameters['ema1'])
        em2=self._ema(em1, self.parameters['ema2'])
        osc=em2
        sig=self._ema(osc, self.parameters['signal'])
        data['rmo_osc']=osc
        data['rmo_sig']=sig
        data['trend_sma']=data['close'].rolling(self.parameters['trend_ma'],min_periods=1).mean()
        return data

    def generate_signals(self, data:pd.DataFrame)->pd.DataFrame:
        p=self.parameters
        data['buy_signal']=False; data['sell_signal']=False; data['signal_strength']=0.0
        up=(crossover(data['rmo_osc'], 0) | crossover(data['rmo_osc'], data['rmo_sig'])) & (data['close']>data['trend_sma'])
        dn=(crossunder(data['rmo_osc'], 0) | crossunder(data['rmo_osc'], data['rmo_sig'])) & (data['close']<data['trend_sma'])
        data['buy_signal']=up; data['sell_signal']=dn
        mag=(data['rmo_osc'].abs()/data['rmo_osc'].abs().rolling(50,min_periods=1).max().replace(0,np.nan)).fillna(0).clip(0,1)
        data['signal_strength']=calculate_signal_strength([mag],[1.0])
        weak=data['signal_strength']<p['signal_threshold']
        data.loc[weak,['buy_signal','sell_signal']]=False
        data['buy_signal'],data['sell_signal']=apply_position_constraints(data['buy_signal'],data['sell_signal'])
        return data

