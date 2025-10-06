#!/usr/bin/env python3
"""
Strategy 192: Premier RSI Oscillator

Pine reference: pine_scripts/192_premier_rsi.pine

Signals: buy when %K crosses above %D; sell when crosses below.
"""

import pandas as pd
import numpy as np
from typing import Dict
import os, sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)
from strategies.base_strategy import BaseStrategy

try:
    sys.path.insert(0, os.path.join(project_root, 'src'))
    from utils.vectorized_helpers import crossover, crossunder
except Exception:
    def crossover(a,b): return (a>b) & (a.shift(1)<=b)
    def crossunder(a,b): return (a<b) & (a.shift(1)>=b)


class Strategy192PremierRSIOscillator(BaseStrategy):
    def __init__(self, parameters: Dict=None):
        params = {
            'length': 14,
            'smoothK': 3,
            'smoothD': 3,
            'signal_threshold': 0.6,
        }
        if parameters:
            params.update(parameters)
        super().__init__('Strategy_192_PremierRSIOscillator', params)

    @staticmethod
    def _sma(s: pd.Series, n: int) -> pd.Series:
        return s.rolling(max(1,int(n)), min_periods=1).mean()

    @staticmethod
    def _rsi(s: pd.Series, n: int) -> pd.Series:
        n=max(1,int(n))
        d=s.diff()
        up=d.clip(lower=0); dn=-d.clip(upper=0)
        roll_up=up.ewm(alpha=1.0/n, adjust=False, min_periods=1).mean()
        roll_dn=dn.ewm(alpha=1.0/n, adjust=False, min_periods=1).mean()
        rs=roll_up/roll_dn.replace(0,np.nan)
        return (100 - 100/(1+rs)).fillna(50)

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        if 'close' not in data and 'price' in data:
            data['close']=data['price']
        p=self.parameters
        rsi = self._rsi(data['close'], int(p['length']))
        k = self._sma(rsi.rolling(int(p['length']), min_periods=1).apply(lambda x: (x.iloc[-1] - x.min())/(x.max()-x.min())*100 if (x.max()-x.min())!=0 else 50, raw=False), int(p['smoothK']))
        d = self._sma(k, int(p['smoothD']))
        data['prsi_k'] = k
        data['prsi_d'] = d
        return data

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        buy = crossover(data['prsi_k'], data['prsi_d'])
        sell = crossunder(data['prsi_k'], data['prsi_d'])
        data['buy_signal'] = buy
        data['sell_signal'] = sell
        st = (data['prsi_k'] - data['prsi_d']).abs()
        st = (st / (data['prsi_k'].rolling(50, min_periods=10).std(ddof=0).replace(0,np.nan))).clip(0,1).fillna(0)
        st = st.where(buy|sell, 0.0)
        thr=float(self.parameters['signal_threshold'])
        st[(buy|sell) & (st < thr)] = thr
        data['signal_strength'] = st
        return data

