#!/usr/bin/env python3
"""
Strategy 044: Wilder's MA (RMA)

Implements Wilder's Moving Average of price.

Signals: buy when price crosses above RMA; sell when crosses below.
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


class Strategy044WildersMA(BaseStrategy):
    def __init__(self, parameters: Dict=None):
        params = {
            'length': 14,
            'source': 'close',
            'signal_threshold': 0.6,
        }
        if parameters:
            params.update(parameters)
        super().__init__('Strategy_044_WildersMA', params)

    @staticmethod
    def _rma(s: pd.Series, n: int) -> pd.Series:
        n = max(1,int(n))
        alpha = 1.0 / n
        return s.ewm(alpha=alpha, adjust=False, min_periods=1).mean()

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        src_name = self.parameters.get('source','close')
        if src_name not in data.columns:
            if src_name=='close' and 'price' in data:
                data['close']=data['price']
            else:
                data[src_name]=data.get('close', pd.Series(index=data.index)).fillna(method='ffill')
        data['rma'] = self._rma(data[src_name], int(self.parameters['length']))
        return data

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        src_name = self.parameters.get('source','close')
        buy = crossover(data[src_name], data['rma'])
        sell = crossunder(data[src_name], data['rma'])
        data['buy_signal'] = buy
        data['sell_signal'] = sell
        dist = (data[src_name]-data['rma']).abs()/(data[src_name].rolling(50, min_periods=10).std(ddof=0).replace(0,np.nan))
        st = dist.clip(0,1).fillna(0)
        st = st.where(buy|sell, 0.0)
        thr = float(self.parameters['signal_threshold'])
        st[(buy|sell) & (st < thr)] = thr
        data['signal_strength'] = st
        return data

