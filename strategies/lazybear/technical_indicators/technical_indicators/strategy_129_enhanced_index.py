#!/usr/bin/env python3
"""
Strategy 129: Enhanced Index

Pine reference: pine_scripts/129_enhanced_index.pine

Signals: buy on closewr crossing above 0; sell on crossing below 0.
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


class Strategy129EnhancedIndex(BaseStrategy):
    def __init__(self, parameters: Dict=None):
        params = {
            'length': 14,
            'lengthMA': 8,
            'signal_threshold': 0.6,
        }
        if parameters:
            params.update(parameters)
        super().__init__('Strategy_129_EnhancedIndex', params)

    @staticmethod
    def _ema(s: pd.Series, n: int) -> pd.Series:
        return s.ewm(span=max(1,int(n)), adjust=False, min_periods=1).mean()

    @staticmethod
    def _sma(s: pd.Series, n: int) -> pd.Series:
        return s.rolling(max(1,int(n)), min_periods=1).mean()

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        if 'close' not in data and 'price' in data:
            data['close'] = data['price']
        length = int(self.parameters['length'])
        dnm = (data['close'].rolling(length, min_periods=1).max() - data['close'].rolling(length, min_periods=1).min()).replace(0, np.nan)
        closewr = 2.0 * (data['close'] - self._sma(data['close'], round(length/2))) / dnm
        data['closewr'] = closewr.fillna(0)
        data['closewr_ma'] = self._ema(data['closewr'], int(self.parameters['lengthMA']))
        return data

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        buy = crossover(data['closewr'], 0.0)
        sell = crossunder(data['closewr'], 0.0)
        data['buy_signal'] = buy
        data['sell_signal'] = sell
        st = (data['closewr'].abs() / (data['closewr'].rolling(50, min_periods=10).std(ddof=0).replace(0,np.nan))).clip(0,1).fillna(0)
        st = st.where(buy|sell, 0.0)
        thr=float(self.parameters['signal_threshold'])
        st[(buy|sell) & (st < thr)] = thr
        data['signal_strength'] = st
        return data

