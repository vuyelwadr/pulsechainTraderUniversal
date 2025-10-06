#!/usr/bin/env python3
"""
Strategy 201: Apirine Slow RSI

Pine reference: pine_scripts/201_apirine_slow_rsi.pine

Signals: buy on SlowRSI crossing above 50; sell on crossing below 50.
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


class Strategy201ApirineSlowRSI(BaseStrategy):
    def __init__(self, parameters: Dict=None):
        params = {
            'periods': 6,
            'smooth': 14,
            'signal_threshold': 0.6,
        }
        if parameters:
            params.update(parameters)
        super().__init__('Strategy_201_ApirineSlowRSI', params)

    @staticmethod
    def _ema(s: pd.Series, n: int) -> pd.Series:
        return s.ewm(span=max(1,int(n)), adjust=False, min_periods=1).mean()

    @staticmethod
    def _wima(s: pd.Series, n: int) -> pd.Series:
        n=max(1,int(n)); alpha=1.0/n
        return s.ewm(alpha=alpha, adjust=False, min_periods=1).mean()

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        if 'close' not in data and 'price' in data:
            data['close']=data['price']
        periods=int(self.parameters['periods']); smooth=int(self.parameters['smooth'])
        r1 = self._ema(data['close'], periods)
        r2 = (data['close'] - r1).clip(lower=0)
        r3 = (r1 - data['close']).clip(lower=0)
        r4 = self._wima(r2, smooth)
        r5 = self._wima(r3, smooth)
        rr = 100.0 - (100.0 / (1.0 + (r4 / r5.replace(0,np.nan))))
        data['slow_rsi'] = rr.fillna(50)
        return data

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        buy = crossover(data['slow_rsi'], 50.0)
        sell = crossunder(data['slow_rsi'], 50.0)
        data['buy_signal'] = buy
        data['sell_signal'] = sell
        st = (data['slow_rsi'] - 50.0).abs() / 50.0
        st = st.clip(0,1).fillna(0).where(buy|sell, 0.0)
        thr=float(self.parameters['signal_threshold'])
        st[(buy|sell) & (st < thr)] = thr
        data['signal_strength'] = st
        return data

