#!/usr/bin/env python3
"""
Strategy 203: DecisionPoint Price Momentum Oscillator (DPMO)

Pine reference: pine_scripts/203_decisionpoint_pmo.pine

Signals: buy when PMO crosses above Signal; sell when crosses below.
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


class Strategy203DecisionpointPMO(BaseStrategy):
    def __init__(self, parameters: Dict=None):
        params = {
            'length1': 35,
            'length2': 20,
            'siglength': 10,
            'signal_threshold': 0.6,
        }
        if parameters:
            params.update(parameters)
        super().__init__('Strategy_203_DecisionpointPMO', params)

    def _csf(self, src: pd.Series, length: int) -> pd.Series:
        sm = 2.0 / float(length)
        csf = pd.Series(index=src.index, dtype=float)
        for i in range(len(src)):
            prev = 0.0 if i == 0 else csf.iloc[i-1]
            csf.iloc[i] = (src.iloc[i] - (src.shift(1).iloc[i] if i>0 else src.iloc[i])) * sm + prev
        return csf

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        if 'close' not in data and 'price' in data:
            data['close']=data['price']
        i = (data['close'] / data['close'].shift(1).fillna(data['close'])) * 100.0
        pmol2 = self._csf(i - 100.0, int(self.parameters['length1']))
        pmol = self._csf(10.0 * pmol2, int(self.parameters['length2']))
        pmols = pmol.ewm(span=int(self.parameters['siglength']), adjust=False, min_periods=1).mean()
        data['pmo'] = pmol
        data['pmo_sig'] = pmols
        data['pmo_diff'] = pmol - pmols
        return data

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        buy = crossover(data['pmo'], data['pmo_sig'])
        sell = crossunder(data['pmo'], data['pmo_sig'])
        data['buy_signal'] = buy
        data['sell_signal'] = sell
        st = (data['pmo_diff'].abs() / (data['pmo'].rolling(50, min_periods=10).std(ddof=0).replace(0,np.nan))).clip(0,1).fillna(0)
        st = st.where(buy|sell, 0.0)
        thr=float(self.parameters['signal_threshold'])
        st[(buy|sell) & (st < thr)] = thr
        data['signal_strength'] = st
        return data

