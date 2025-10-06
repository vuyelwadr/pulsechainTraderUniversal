#!/usr/bin/env python3
"""
Strategy 186: Chande Composite Momentum Index (CCMI)

Pine reference: pine_scripts/186_composite_momentum.pine

Signals: buy when DMI crosses above 0; sell when crosses below 0.
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


class Strategy186ChandeCompositeMomentumIndex(BaseStrategy):
    def __init__(self, parameters: Dict=None):
        params = {
            'lenSmooth': 3,
            'trigg': 5,
            'signal_threshold': 0.6,
        }
        if parameters:
            params.update(parameters)
        super().__init__('Strategy_186_ChandeCompositeMomentumIndex', params)

    @staticmethod
    def _ema(s: pd.Series, n: int) -> pd.Series:
        return s.ewm(span=max(1,int(n)), adjust=False, min_periods=1).mean()

    def _dema(self, s: pd.Series, n: int) -> pd.Series:
        e1 = self._ema(s, n)
        e2 = self._ema(e1, n)
        return 2*e1 - e2

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        if 'close' not in data and 'price' in data:
            data['close'] = data['price']
        src = data['close']
        def cmo(src, n):
            up = pd.Series(0.0, index=src.index)
            dn = pd.Series(0.0, index=src.index)
            diff = src.diff()
            up = diff.clip(lower=0).rolling(n, min_periods=1).sum()
            dn = (-diff.clip(upper=0)).rolling(n, min_periods=1).sum()
            return self._dema(100 * ((up - dn) / (up + dn).replace(0,np.nan)), 3)
        cmo5 = cmo(src, 5)
        cmo10 = cmo(src, 10)
        cmo20 = cmo(src, 20)
        dmi = (src.rolling(5, min_periods=1).std(ddof=0)*cmo5 + src.rolling(10, min_periods=1).std(ddof=0)*cmo10 + src.rolling(20, min_periods=1).std(ddof=0)*cmo20)
        denom = (src.rolling(5, min_periods=1).std(ddof=0) + src.rolling(10, min_periods=1).std(ddof=0) + src.rolling(20, min_periods=1).std(ddof=0)).replace(0,np.nan)
        dmi = dmi / denom
        data['ccmi'] = dmi.fillna(0)
        return data

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        buy = crossover(data['ccmi'], 0.0)
        sell = crossunder(data['ccmi'], 0.0)
        data['buy_signal'] = buy
        data['sell_signal'] = sell
        st = (data['ccmi'].abs() / (data['ccmi'].rolling(50, min_periods=10).std(ddof=0).replace(0,np.nan))).clip(0,1).fillna(0)
        st = st.where(buy|sell, 0.0)
        thr=float(self.parameters['signal_threshold'])
        st[(buy|sell) & (st < thr)] = thr
        data['signal_strength'] = st
        return data

