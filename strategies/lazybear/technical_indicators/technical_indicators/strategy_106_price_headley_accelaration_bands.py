#!/usr/bin/env python3
"""
Strategy 106: Price Headley Acceleration Bands

Pine reference: pine_scripts/106_headley_bands.pine

Signals: buy when close crosses above upper band; sell when crosses below lower band.
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


class Strategy106PriceHeadleyAccelerationBands(BaseStrategy):
    def __init__(self, parameters: Dict=None):
        params = {
            'length': 20,
            'signal_threshold': 0.6,
        }
        if parameters:
            params.update(parameters)
        super().__init__('Strategy_106_PriceHeadleyAccelerationBands', params)

    @staticmethod
    def _sma(s: pd.Series, n: int) -> pd.Series:
        return s.rolling(max(1,int(n)), min_periods=1).mean()

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        for c in ('high','low','close'):
            if c not in data:
                data[c] = data.get('close', pd.Series(index=data.index)).fillna(method='ffill')
        length = int(self.parameters['length'])
        center = (data['high'] + data['low']) / 2.0
        rng = ((data['high'] - data['low']) / center.replace(0, np.nan)) * 1000.0 * 0.001
        ub = data['high'] * (1 + 2 * rng)
        lb = data['low'] * (1 - 2 * rng)
        data['phab_upper'] = self._sma(ub, length)
        data['phab_lower'] = self._sma(lb, length)
        data['phab_mid'] = (data['phab_upper'] + data['phab_lower']) / 2.0
        return data

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        buy = crossover(data['close'], data['phab_upper'])
        sell = crossunder(data['close'], data['phab_lower'])
        data['buy_signal'] = buy
        data['sell_signal'] = sell
        bw = (data['phab_upper'] - data['phab_lower']).replace(0, np.nan)
        st = (data['close'] - data['phab_mid']).abs() / bw
        st = st.clip(0,1).fillna(0).where(buy|sell, 0.0)
        thr=float(self.parameters['signal_threshold'])
        st[(buy|sell) & (st < thr)] = thr
        data['signal_strength'] = st
        return data

