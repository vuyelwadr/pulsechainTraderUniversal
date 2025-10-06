#!/usr/bin/env python3
"""
Strategy 035: STARC Bands

TradingView URL: https://www.tradingview.com/v/yTMOV9NM/
Type: volatility/bands

Description:
STARC Bands around a moving average using ATR multiplier. Signals on touches
and reverts with momentum confirmation.
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


class Strategy035STARC(BaseStrategy):
    def __init__(self, parameters: Dict=None):
        params={'ma_period':20,'atr_period':14,'atr_mult':2.0,'signal_threshold':0.6}
        if parameters: params.update(parameters)
        super().__init__('Strategy_035_STARC', params)

    def calculate_indicators(self, data:pd.DataFrame)->pd.DataFrame:
        for c in ('high','low','close','open'):
            if c not in data.columns and 'price' in data.columns:
                data[c]=data['price']
        ma=data['close'].rolling(self.parameters['ma_period'],min_periods=1).mean()
        tr=pd.concat([
            data['high']-data['low'],
            (data['high']-data['close'].shift(1)).abs(),
            (data['low']-data['close'].shift(1)).abs()
        ],axis=1).max(axis=1)
        atr=tr.rolling(self.parameters['atr_period'],min_periods=1).mean()
        data['starc_mid']=ma
        data['starc_up']=ma + self.parameters['atr_mult']*atr
        data['starc_dn']=ma - self.parameters['atr_mult']*atr
        data['momentum']=data['close']-data['close'].shift(5)
        return data

    def generate_signals(self, data:pd.DataFrame)->pd.DataFrame:
        p=self.parameters
        data['buy_signal']=False; data['sell_signal']=False; data['signal_strength']=0.0
        buy=(crossunder(data['close'], data['starc_dn']) | crossover(data['close'], data['starc_dn'])) & (data['momentum']>0)
        sell=(crossover(data['close'], data['starc_up']) | crossunder(data['close'], data['starc_up'])) & (data['momentum']<0)
        data['buy_signal']=buy; data['sell_signal']=sell
        # Strength: distance outside bands
        dist_buy=((data['starc_mid']-data['close'])/(data['starc_mid']-data['starc_dn']).replace(0,np.nan)).clip(0,1).fillna(0)
        dist_sell=((data['close']-data['starc_mid'])/(data['starc_up']-data['starc_mid']).replace(0,np.nan)).clip(0,1).fillna(0)
        strength=np.where(buy, dist_buy, 0)+np.where(sell, dist_sell, 0)
        data['signal_strength']=pd.Series(strength,index=data.index).clip(0,1)
        weak=data['signal_strength']<p['signal_threshold']
        data.loc[weak,['buy_signal','sell_signal']]=False
        data['buy_signal'],data['sell_signal']=apply_position_constraints(data['buy_signal'],data['sell_signal'])
        return data

