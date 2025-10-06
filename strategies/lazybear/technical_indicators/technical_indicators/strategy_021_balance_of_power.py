#!/usr/bin/env python3
"""
Strategy 021: Balance Of Power (BOP)

TradingView URL: https://www.tradingview.com/v/tzZx7dTy/
Type: momentum

Description:
BOP = (Close - Open) / (High - Low). Smoothed with SMA. Signals on zero cross
with optional trend confirmation.
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


class Strategy021BalanceOfPower(BaseStrategy):
    def __init__(self, parameters: Dict=None):
        params={'sma_period':14,'signal_threshold':0.6,'use_trend_filter':True,'trend_ma':50}
        if parameters: params.update(parameters)
        super().__init__('Strategy_021_BalanceOfPower', params)

    def calculate_indicators(self, data:pd.DataFrame)->pd.DataFrame:
        for c in ('open','high','low','close'):
            if c not in data.columns and 'price' in data.columns:
                data[c]=data['price']
        rng=(data['high']-data['low']).replace(0,np.nan)
        data['bop_raw']=(data['close']-data['open'])/rng
        data['bop']=data['bop_raw'].rolling(self.parameters['sma_period'],min_periods=1).mean().fillna(0)
        if self.parameters['use_trend_filter']:
            data['trend_sma']=data['close'].rolling(self.parameters['trend_ma'],min_periods=1).mean()
        return data

    def generate_signals(self, data:pd.DataFrame)->pd.DataFrame:
        p=self.parameters
        data['buy_signal']=False; data['sell_signal']=False; data['signal_strength']=0.0
        up = crossover(data['bop'], 0)
        dn = crossunder(data['bop'], 0)
        if p['use_trend_filter']:
            up &= data['close']>data['trend_sma']
            dn &= data['close']<data['trend_sma']
        data['buy_signal']=up; data['sell_signal']=dn
        strength=(data['bop'].abs()/data['bop'].abs().rolling(50,min_periods=1).max().replace(0,np.nan)).fillna(0).clip(0,1)
        data['signal_strength']=calculate_signal_strength([strength],[1.0])
        weak=data['signal_strength']<p['signal_threshold']
        data.loc[weak,['buy_signal','sell_signal']]=False
        data['buy_signal'],data['sell_signal']=apply_position_constraints(data['buy_signal'],data['sell_signal'])
        return data

