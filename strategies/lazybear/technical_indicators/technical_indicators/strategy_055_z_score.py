#!/usr/bin/env python3
"""
Strategy 055: Z-Score

Pine source not available; implement standard z-score of close:
  z = (close - SMA(close,n)) / stdev(close,n)

Deterministic rule:
- Buy when z crosses above +1 (momentum breakout)
- Sell when z crosses below -1
"""

import pandas as pd
import numpy as np
from typing import Dict
import sys, os

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)
from strategies.base_strategy import BaseStrategy

try:
    sys.path.insert(0, os.path.join(project_root, 'src'))
    from utils.vectorized_helpers import crossover, crossunder
except Exception:
    def crossover(a,b): return (a>b) & (a.shift(1)<=b)
    def crossunder(a,b): return (a<b) & (a.shift(1)>=b)


class Strategy055ZScore(BaseStrategy):
    def __init__(self, parameters: Dict = None):
        params = {
            'length': 20,
            'upper': 1.0,
            'lower': -1.0,
            'signal_threshold': 0.6,
        }
        if parameters:
            params.update(parameters)
        super().__init__('Strategy_055_ZScore', params)

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        if 'close' not in data.columns and 'price' in data.columns:
            data['close']=data['price']
        n = int(self.parameters['length'])
        mean = data['close'].rolling(n, min_periods=1).mean()
        std = data['close'].rolling(n, min_periods=1).std(ddof=0).replace(0,np.nan)
        z = ((data['close'] - mean) / std).replace([np.inf,-np.inf], np.nan).fillna(0)
        data['zscore'] = z
        return data

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        up = float(self.parameters['upper'])
        lo = float(self.parameters['lower'])
        buy = crossover(data['zscore'], up)
        sell = crossunder(data['zscore'], lo)
        data['buy_signal'] = buy
        data['sell_signal'] = sell
        st = data['zscore'].abs() / 3.0  # normalize roughly
        st = st.clip(0,1)
        st = st.where(buy|sell, 0.0)
        thr = float(self.parameters['signal_threshold'])
        st[(buy|sell) & (st < thr)] = thr
        data['signal_strength'] = st.clip(0,1)
        return data

