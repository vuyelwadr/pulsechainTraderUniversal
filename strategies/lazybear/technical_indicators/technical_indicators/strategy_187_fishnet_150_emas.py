#!/usr/bin/env python3
"""
Strategy 187: Fishnet (1..150 EMAs)

Pine reference: pine_scripts/187_fishnet_150emas.pine

Signals: buy when EMA(1) crosses above EMA(150); sell on cross below.
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


class Strategy187Fishnet150EMAs(BaseStrategy):
    def __init__(self, parameters: Dict=None):
        params = {
            'fast': 1,
            'slow': 150,
            'signal_threshold': 0.6,
        }
        if parameters:
            params.update(parameters)
        super().__init__('Strategy_187_Fishnet150EMAs', params)

    @staticmethod
    def _ema(s: pd.Series, n: int) -> pd.Series:
        return s.ewm(span=max(1,int(n)), adjust=False, min_periods=1).mean()

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        if 'close' not in data and 'price' in data:
            data['close']=data['price']
        data['ema_fast'] = self._ema(data['close'], int(self.parameters['fast']))
        data['ema_slow'] = self._ema(data['close'], int(self.parameters['slow']))
        return data

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        buy = crossover(data['ema_fast'], data['ema_slow'])
        sell = crossunder(data['ema_fast'], data['ema_slow'])
        data['buy_signal'] = buy
        data['sell_signal'] = sell
        st = (data['ema_fast'] - data['ema_slow']).abs() / (data['close'].rolling(50, min_periods=10).std(ddof=0).replace(0,np.nan))
        st = st.clip(0,1).fillna(0).where(buy|sell, 0.0)
        thr=float(self.parameters['signal_threshold'])
        st[(buy|sell) & (st < thr)] = thr
        data['signal_strength'] = st
        return data

