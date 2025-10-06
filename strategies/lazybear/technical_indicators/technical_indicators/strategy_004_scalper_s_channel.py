#!/usr/bin/env python3
"""
Strategy 004: Scalper's Channel

TradingView URL: https://www.tradingview.com/v/hmfZ7SUf/
Type: channel/volatility (fast EMA channel for scalping)

Derived trading rule:
- Build a tight channel around a fast EMA using ATR multiplier.
- Buy when price crosses above upper band from inside the channel (momentum burst).
- Sell when price crosses below lower band from inside the channel.
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
    from utils.vectorized_helpers import crossover, crossunder, calculate_signal_strength
except Exception:
    def crossover(a,b): return (a>b) & (a.shift(1)<=b.shift(1))
    def crossunder(a,b): return (a<b) & (a.shift(1)>=b.shift(1))
    def calculate_signal_strength(fs,weights=None):
        import pandas as pd
        df=pd.concat(fs,axis=1); return df.mean(axis=1).clip(0,1)


class Strategy004ScalpersChannel(BaseStrategy):
    def __init__(self, parameters: Dict=None):
        params={
            'ema_fast': 9,
            'atr_period': 14,
            'mult': 0.7,
            'signal_threshold': 0.6,
        }
        if parameters: params.update(parameters)
        super().__init__('Strategy_004_ScalpersChannel', params)

    @staticmethod
    def _ema(s: pd.Series, n: int) -> pd.Series:
        return s.ewm(span=max(1,int(n)), adjust=False, min_periods=1).mean()

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        for c in ('open','high','low','close'):
            if c not in data.columns and 'price' in data.columns:
                data[c]=data['price']
        basis=self._ema(data['close'], int(self.parameters['ema_fast']))
        tr=pd.concat([
            data['high']-data['low'],
            (data['high']-data['close'].shift(1)).abs(),
            (data['low']-data['close'].shift(1)).abs()
        ],axis=1).max(axis=1)
        atr=tr.rolling(int(self.parameters['atr_period']), min_periods=1).mean()
        m=float(self.parameters['mult'])
        data['sc_mid']=basis
        data['sc_up']=basis + m*atr
        data['sc_dn']=basis - m*atr
        return data

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        inside_prev=(data['close'].shift(1)<=data['sc_up'].shift(1)) & (data['close'].shift(1)>=data['sc_dn'].shift(1))
        buy=crossover(data['close'], data['sc_up']) & inside_prev
        sell=crossunder(data['close'], data['sc_dn']) & inside_prev
        data['buy_signal']=buy
        data['sell_signal']=sell
        width=(data['sc_up']-data['sc_mid']).replace(0,np.nan)
        dist=pd.Series(0.0,index=data.index)
        dist[buy]=((data['close']-data['sc_up'])/width).clip(0,1)[buy]
        dist[sell]=((data['sc_dn']-data['close'])/width).clip(0,1)[sell]
        strength=calculate_signal_strength([dist.fillna(0)],[1.0])
        data['signal_strength']=strength.mask(~(buy|sell),0.0).mask((buy|sell)&(strength<self.parameters['signal_threshold']), self.parameters['signal_threshold'])
        return data

