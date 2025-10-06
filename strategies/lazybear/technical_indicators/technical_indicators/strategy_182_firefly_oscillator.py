#!/usr/bin/env python3
"""
Strategy 182: Firefly Oscillator

Pine reference: pine_scripts/182_firefly_osc.pine

Signals: buy when Firefly value crosses above 50; sell when crosses below 50.
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


class Strategy182FireflyOscillator(BaseStrategy):
    def __init__(self, parameters: Dict=None):
        params = {
            'm': 10,
            'n1': 3,
            'as_double': False,
            'use_zlema': False,
            'signal_threshold': 0.6,
        }
        if parameters:
            params.update(parameters)
        super().__init__('Strategy_182_FireflyOscillator', params)

    @staticmethod
    def _ema(s: pd.Series, n: int) -> pd.Series:
        return s.ewm(span=max(1,int(n)), adjust=False, min_periods=1).mean()

    def _zlema(self, s: pd.Series, n: int) -> pd.Series:
        e1 = self._ema(s, n)
        e2 = self._ema(e1, n)
        d = e1 - e2
        return e1 + d

    def _ma(self, s: pd.Series, n: int) -> pd.Series:
        return self._ema(s, n) if not bool(self.parameters.get('use_zlema', False)) else self._zlema(s, n)

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        for c in ('high','low','close'):
            if c not in data:
                data[c]=data.get('close', pd.Series(index=data.index)).fillna(method='ffill')
        m = int(self.parameters['m']); n1 = int(self.parameters['n1'])
        v2 = (data['high'] + data['low'] + 2*data['close'])/4.0
        v3 = self._ma(v2, m)
        v4 = v2.rolling(m, min_periods=1).std(ddof=0)
        v5 = (v2 - v3) * 100.0 / v4.replace(0,np.nan)
        v6 = self._ma(v5, n1)
        v7 = self._ma(v6, n1) if bool(self.parameters.get('as_double', False)) else v6
        ww = (self._ma(v7, m) + 100.0)/2.0 - 4.0
        data['firefly'] = ww
        return data

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        buy = crossover(data['firefly'], 50.0)
        sell = crossunder(data['firefly'], 50.0)
        data['buy_signal'] = buy
        data['sell_signal'] = sell
        st = (data['firefly'] - 50.0).abs() / 50.0
        st = st.clip(0,1).fillna(0).where(buy|sell, 0.0)
        thr=float(self.parameters['signal_threshold'])
        st[(buy|sell) & (st < thr)] = thr
        data['signal_strength'] = st
        return data

