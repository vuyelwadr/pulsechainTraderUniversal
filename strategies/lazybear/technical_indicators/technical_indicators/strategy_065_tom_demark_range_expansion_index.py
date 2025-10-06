#!/usr/bin/env python3
"""
Strategy 065: Tom Demark Range Expansion Index (REI)

No Pine file in repo; implement standard REI-style measure as in our Agent06
approximation: compute directional range expansion over a lookback, then derive
an index between 0..100 with EMA smoothing.

Deterministic rule:
- Buy when smoothed REI crosses above threshold (default 55)
- Sell when smoothed REI crosses below (100 - threshold) (default 45)
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


class Strategy065TomDemarkREI(BaseStrategy):
    def __init__(self, parameters: Dict = None):
        params = {
            'period': 8,
            'smooth_period': 5,
            'threshold': 55.0,
            'signal_threshold': 0.6,
        }
        if parameters:
            params.update(parameters)
        super().__init__('Strategy_065_TomDemark_REI', params)

    @staticmethod
    def _ema(s: pd.Series, n: int) -> pd.Series:
        return s.ewm(span=max(1,int(n)), adjust=False, min_periods=1).mean()

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        for c in ('high','low','close'):
            if c not in data.columns and 'price' in data.columns:
                data[c]=data['price']
        n = int(self.parameters['period'])
        # Range expansion components (approximation)
        rei_up = (data['high'] - data['high'].shift(n)).clip(lower=0)
        rei_dn = (data['low'].shift(n) - data['low']).clip(lower=0)
        up_sum = rei_up.rolling(n, min_periods=1).sum()
        dn_sum = rei_dn.rolling(n, min_periods=1).sum()
        rei = (100.0 * up_sum / (up_sum + dn_sum + 1e-6)).fillna(50.0)
        rei_s = self._ema(rei, int(self.parameters['smooth_period']))
        data['rei'] = rei
        data['rei_s'] = rei_s
        return data

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        th = float(self.parameters['threshold'])
        sell_th = 100.0 - th
        buy = crossover(data['rei_s'], th)
        sell = crossunder(data['rei_s'], sell_th)
        data['buy_signal'] = buy
        data['sell_signal'] = sell
        # Strength based on distance from 50 scaled
        st = (data['rei_s'] - 50.0).abs() / 50.0
        st = st.clip(0,1).where(buy|sell, 0.0)
        thr = float(self.parameters['signal_threshold'])
        st[(buy|sell) & (st < thr)] = thr
        data['signal_strength'] = st.clip(0,1)
        return data

