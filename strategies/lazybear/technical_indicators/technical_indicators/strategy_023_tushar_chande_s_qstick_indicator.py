#!/usr/bin/env python3
"""
Strategy 023: Tushar Chande's QStick Indicator

TradingView URL: https://www.tradingview.com/v/ssL68jQu/
Type: momentum

Description:
QStick = MA(close - open) over a lookback window. Signals on zero cross with
trend filter.
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


class Strategy023QStick(BaseStrategy):
    def __init__(self, parameters: Dict=None):
        params={'period':14,'ma_type':'ema','signal_threshold':0.6,'trend_ma':50}
        if parameters: params.update(parameters)
        super().__init__('Strategy_023_QStick', params)

    def _ma(self, s:pd.Series, n:int, kind:str)->pd.Series:
        return s.ewm(span=n,adjust=False,min_periods=1).mean() if kind=='ema' else s.rolling(n,min_periods=1).mean()

    def calculate_indicators(self, data:pd.DataFrame)->pd.DataFrame:
        for c in ('open','close'):
            if c not in data.columns and 'price' in data.columns:
                data[c]=data['price']
        q=(data['close']-data['open'])
        data['qstick']=self._ma(q, self.parameters['period'], self.parameters['ma_type'])
        data['trend_sma']=data['close'].rolling(self.parameters['trend_ma'], min_periods=1).mean()
        return data

    def generate_signals(self, data:pd.DataFrame)->pd.DataFrame:
        p=self.parameters
        data['buy_signal']=False; data['sell_signal']=False; data['signal_strength']=0.0
        up=crossover(data['qstick'],0) & (data['close']>data['trend_sma'])
        dn=crossunder(data['qstick'],0) & (data['close']<data['trend_sma'])
        data['buy_signal']=up; data['sell_signal']=dn
        mag=(data['qstick'].abs()/(data['qstick'].abs().rolling(50,min_periods=1).max()+1e-9)).clip(0,1)
        data['signal_strength']=calculate_signal_strength([mag],[1.0])
        weak=data['signal_strength']<p['signal_threshold']
        data.loc[weak,['buy_signal','sell_signal']]=False
        return data

