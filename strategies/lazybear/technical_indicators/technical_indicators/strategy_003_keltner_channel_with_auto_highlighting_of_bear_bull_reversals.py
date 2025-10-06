#!/usr/bin/env python3
"""
Strategy 003: Keltner Channel (auto highlighting of Bull/Bear reversals)

TradingView URL: https://www.tradingview.com/v/WH1PVDTO/
Type: channel/volatility

Derived trading rule consistent with highlighting intent:
- Compute Keltner Channel (EMA basis, ATR multiplier).
- Bull reversal: close crosses up through upper band after being within/below channel.
- Bear reversal: close crosses down through lower band after being within/above channel.
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


class Strategy003KeltnerChannel(BaseStrategy):
    def __init__(self, parameters: Dict=None):
        params={
            'length': 20,
            'atr_period': 20,
            'mult': 1.5,
            'signal_threshold': 0.6,
        }
        if parameters: params.update(parameters)
        super().__init__('Strategy_003_KeltnerChannel', params)

    @staticmethod
    def _ema(s: pd.Series, n: int) -> pd.Series:
        return s.ewm(span=max(1,int(n)), adjust=False, min_periods=1).mean()

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        for c in ('open','high','low','close'):
            if c not in data.columns and 'price' in data.columns:
                data[c]=data['price']
        basis = self._ema(data['close'], int(self.parameters['length']))
        tr = pd.concat([
            data['high']-data['low'],
            (data['high']-data['close'].shift(1)).abs(),
            (data['low']-data['close'].shift(1)).abs()
        ],axis=1).max(axis=1)
        atr = tr.rolling(int(self.parameters['atr_period']), min_periods=1).mean()
        mult = float(self.parameters['mult'])
        data['kc_mid'] = basis
        data['kc_up'] = basis + mult*atr
        data['kc_dn'] = basis - mult*atr
        return data

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        inside_prev = (data['close'].shift(1) <= data['kc_up'].shift(1)) & (data['close'].shift(1) >= data['kc_dn'].shift(1))
        buy = crossover(data['close'], data['kc_up']) & inside_prev
        sell = crossunder(data['close'], data['kc_dn']) & inside_prev
        data['buy_signal']=buy
        data['sell_signal']=sell
        # Strength by distance outside band normalized by ATR magnitude
        width = (data['kc_up']-data['kc_mid']).replace(0,np.nan)
        dist = pd.Series(0.0, index=data.index)
        dist[buy] = ((data['close']-data['kc_up'])/width).clip(0,1)[buy]
        dist[sell]= ((data['kc_dn']-data['close'])/width).clip(0,1)[sell]
        data['signal_strength']=calculate_signal_strength([dist.fillna(0)],[1.0])
        # Floor to ensure execution
        data.loc[buy|sell,'signal_strength']=data.loc[buy|sell,'signal_strength'].clip(lower=float(self.parameters['signal_threshold']))
        return data

