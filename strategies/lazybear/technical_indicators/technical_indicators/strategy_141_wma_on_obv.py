#!/usr/bin/env python3
"""
Strategy 141: WMA on OBV

Pine reference: pine_scripts/141_wma_on_obv.pine

Signals: buy when OBV crosses above WMA(len); sell when crosses below.
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


class Strategy141WMAOnOBV(BaseStrategy):
    def __init__(self, parameters: Dict=None):
        params = {
            'length': 13,
            'signal_threshold': 0.6,
        }
        if parameters:
            params.update(parameters)
        super().__init__('Strategy_141_WMA_On_OBV', params)

    @staticmethod
    def _wma(s: pd.Series, n: int) -> pd.Series:
        n = max(1,int(n))
        def _f(window):
            wlen = len(window)
            weights = np.arange(1, wlen+1, dtype=float)
            w = window.astype(float)
            return (w * weights).sum() / weights.sum()
        return s.rolling(n, min_periods=1).apply(_f, raw=True)

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        for c in ('close','volume'):
            if c not in data:
                data[c] = 0 if c=='volume' else data.get('price', pd.Series(index=data.index)).fillna(method='ffill')
        # OBV
        chg = data['close'].diff()
        obv = (np.where(chg > 0, data['volume'], np.where(chg < 0, -data['volume'], 0))).cumsum()
        data['obv'] = pd.Series(obv, index=data.index)
        data['obv_wma'] = self._wma(data['obv'], int(self.parameters['length']))
        return data

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        buy = crossover(data['obv'], data['obv_wma'])
        sell = crossunder(data['obv'], data['obv_wma'])
        data['buy_signal'] = buy
        data['sell_signal'] = sell
        st = (data['obv'] - data['obv_wma']).abs()
        st = (st / (data['obv'].rolling(50, min_periods=10).std(ddof=0).replace(0,np.nan))).clip(0,1).fillna(0)
        st = st.where(buy|sell, 0.0)
        thr=float(self.parameters['signal_threshold'])
        st[(buy|sell) & (st < thr)] = thr
        data['signal_strength'] = st
        return data
