#!/usr/bin/env python3
"""
Strategy 193: Ehlers Center of Gravity Oscillator (COG)

Implements CoG as weighted average of recent prices normalized by sum of prices.
Signals on oscillator crossing its trigger (1-bar lag) or zero.
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


class Strategy193EhlersCOG(BaseStrategy):
    def __init__(self, parameters: Dict=None):
        params = {
            'length': 10,
            'signal_threshold': 0.6,
        }
        if parameters:
            params.update(parameters)
        super().__init__('Strategy_193_Ehlers_Center_of_Gravity_Oscillator', params)

    def _cog(self, src: pd.Series, length: int) -> pd.Series:
        n=max(1,int(length))
        def _f(window):
            w = window.astype(float)
            s = w.sum()
            if s == 0:
                return 0.0
            # Use weights matching current window length (most recent weight = 1)
            idx = np.arange(1, len(w)+1, dtype=float)
            return np.dot(w[::-1], idx) / s
        return src.rolling(n, min_periods=1).apply(_f, raw=True)

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        if 'close' not in data and 'price' in data:
            data['close']=data['price']
        src = (data['high'] + data['low'])/2.0 if 'high' in data and 'low' in data else data['close']
        length = int(self.parameters['length'])
        cog = self._cog(src, length)
        # Detrend around midline by subtracting rolling median of index
        cog_norm = cog - cog.rolling(length, min_periods=1).mean()
        data['cog'] = cog_norm.fillna(0)
        data['cog_trig'] = data['cog'].shift(1).fillna(0)
        return data

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        buy = crossover(data['cog'], data['cog_trig'])
        sell = crossunder(data['cog'], data['cog_trig'])
        data['buy_signal'] = buy
        data['sell_signal'] = sell
        st = (data['cog'].abs() / (data['cog'].rolling(50, min_periods=10).std(ddof=0).replace(0,np.nan))).clip(0,1).fillna(0)
        st = st.where(buy|sell, 0.0)
        thr=float(self.parameters['signal_threshold'])
        st[(buy|sell) & (st < thr)] = thr
        data['signal_strength'] = st
        return data
