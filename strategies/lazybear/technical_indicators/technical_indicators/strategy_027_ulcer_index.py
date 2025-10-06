#!/usr/bin/env python3
"""
Strategy 027: Ulcer Index (UI)

TradingView URL: https://www.tradingview.com/v/QuqgdJgF/
Type: risk/volatility

Description:
Ulcer Index measures downside risk via drawdown depth and duration. We compute
UI over a rolling window and generate signals when UI declines (risk easing) in
an uptrend, or spikes (risk rising) in a downtrend.
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


class Strategy027UlcerIndex(BaseStrategy):
    def __init__(self, parameters: Dict=None):
        params={'period':14,'trend_ma':50,'threshold':1.0,'signal_threshold':0.6}
        if parameters: params.update(parameters)
        super().__init__('Strategy_027_UlcerIndex', params)

    def calculate_indicators(self, data:pd.DataFrame)->pd.DataFrame:
        if 'close' not in data.columns and 'price' in data.columns:
            data['close']=data['price']
        n=self.parameters['period']
        rolling_max=data['close'].rolling(n, min_periods=1).max()
        drawdown=100.0*(data['close']-rolling_max)/rolling_max.replace(0,np.nan)
        ui=np.sqrt((drawdown**2).rolling(n, min_periods=1).mean())
        data['ui']=ui.fillna(0)
        data['trend_sma']=data['close'].rolling(self.parameters['trend_ma'],min_periods=1).mean()
        return data

    def generate_signals(self, data:pd.DataFrame)->pd.DataFrame:
        p=self.parameters
        data['buy_signal']=False; data['sell_signal']=False; data['signal_strength']=0.0
        # Buy when UI crosses down below threshold and uptrend
        buy=crossunder(data['ui'], p['threshold']) & (data['close']>data['trend_sma'])
        # Sell when UI crosses up above threshold and downtrend
        sell=crossover(data['ui'], p['threshold']) & (data['close']<data['trend_sma'])
        data['buy_signal']=buy; data['sell_signal']=sell
        # Strength: relative UI change
        change=(data['ui'].shift(1)-data['ui']) # falling increases buy strength
        strength=((change.abs())/(data['ui'].rolling(50,min_periods=1).max()+1e-9)).clip(0,1)
        data['signal_strength']=calculate_signal_strength([strength],[1.0])
        weak=data['signal_strength']<p['signal_threshold']
        data.loc[weak,['buy_signal','sell_signal']]=False
        data['buy_signal'],data['sell_signal']=apply_position_constraints(data['buy_signal'],data['sell_signal'])
        return data

