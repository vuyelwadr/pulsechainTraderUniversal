#!/usr/bin/env python3
"""
Strategy 022: 4MACD

TradingView URL: https://www.tradingview.com/v/nbx4UFZ6/
Type: momentum/trend

Description:
Compute four MACD configurations; signal when majority agree with crossover.
This approximates LazyBear's 4MACD ensemble.
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


class Strategy022FourMACD(BaseStrategy):
    def __init__(self, parameters: Dict=None):
        params={
            'configs': [(12,26,9),(5,35,5),(8,21,9),(20,50,9)],
            'signal_threshold':0.6
        }
        if parameters: params.update(parameters)
        super().__init__('Strategy_022_4MACD', params)

    @staticmethod
    def _ema(s:pd.Series,n:int)->pd.Series:
        return s.ewm(span=n,adjust=False,min_periods=1).mean()

    def _macd(self, close:pd.Series, fast:int, slow:int, signal:int):
        ema_fast=self._ema(close, fast)
        ema_slow=self._ema(close, slow)
        macd=ema_fast-ema_slow
        macd_signal=self._ema(macd, signal)
        return macd, macd_signal

    def calculate_indicators(self, data:pd.DataFrame)->pd.DataFrame:
        if 'close' not in data.columns and 'price' in data.columns:
            data['close']=data['price']
        for i,(f,s,g) in enumerate(self.parameters['configs']):
            m,ms=self._macd(data['close'], f,s,g)
            data[f'macd_{i}']=m; data[f'msig_{i}']=ms
        return data

    def generate_signals(self, data:pd.DataFrame)->pd.DataFrame:
        cfgs=self.parameters['configs']
        data['buy_signal']=False; data['sell_signal']=False; data['signal_strength']=0.0
        buys=[]; sells=[]
        for i in range(len(cfgs)):
            buys.append(crossover(data[f'macd_{i}'], data[f'msig_{i}']))
            sells.append(crossunder(data[f'macd_{i}'], data[f'msig_{i}']))
        buy_agree=pd.concat(buys,axis=1).sum(axis=1)
        sell_agree=pd.concat(sells,axis=1).sum(axis=1)
        data['buy_signal']=buy_agree>=3
        data['sell_signal']=sell_agree>=3
        consensus_str=((buy_agree.where(data['buy_signal'],0)+sell_agree.where(data['sell_signal'],0))/len(cfgs)).clip(0,1)
        data['signal_strength']=calculate_signal_strength([consensus_str],[1.0])
        data['buy_signal'],data['sell_signal']=apply_position_constraints(data['buy_signal'],data['sell_signal'])
        return data

