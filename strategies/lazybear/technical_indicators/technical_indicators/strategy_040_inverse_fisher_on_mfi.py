#!/usr/bin/env python3
"""
Strategy 040: Inverse Fisher on MFI

Computes Money Flow Index and applies inverse Fisher transform to magnify turning
points. Signals on crossing of midline (0) and optional threshold bands.
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


class Strategy040InverseFisherMFI(BaseStrategy):
    def __init__(self, parameters: Dict=None):
        params = {
            'length': 14,
            'signal_threshold': 0.6,
        }
        if parameters:
            params.update(parameters)
        super().__init__('Strategy_040_InverseFisherMFI', params)

    @staticmethod
    def _mfi(h: pd.Series, l: pd.Series, c: pd.Series, v: pd.Series, n: int) -> pd.Series:
        tp = (h + l + c) / 3.0
        rmf = tp * v
        pos = pd.Series(0.0, index=tp.index)
        neg = pd.Series(0.0, index=tp.index)
        delta = tp.diff()
        pos[delta > 0] = rmf[delta > 0]
        neg[delta < 0] = rmf[delta < 0].abs()
        mr = pos.rolling(max(1,int(n)), min_periods=1).sum() / (neg.rolling(max(1,int(n)), min_periods=1).sum().replace(0,np.nan))
        mfi = 100.0 - (100.0 / (1.0 + mr))
        return mfi.fillna(50)

    @staticmethod
    def _invfisher(x: pd.Series) -> pd.Series:
        y = 0.1 * (x - 50.0)
        return (np.exp(2*y) - 1) / (np.exp(2*y) + 1)

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        for c in ('high','low','close','volume'):
            if c not in data:
                data[c] = 0 if c=='volume' else data.get('close', pd.Series(index=data.index)).fillna(method='ffill')
        n=int(self.parameters['length'])
        mfi = self._mfi(data['high'], data['low'], data['close'], data['volume'], n)
        data['if_mfi'] = self._invfisher(mfi)
        return data

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        buy = crossover(data['if_mfi'], 0.0)
        sell = crossunder(data['if_mfi'], 0.0)
        data['buy_signal'] = buy
        data['sell_signal'] = sell
        st = data['if_mfi'].abs().clip(0,1).fillna(0)
        st = st.where(buy|sell, 0.0)
        thr=float(self.parameters['signal_threshold'])
        st[(buy|sell) & (st < thr)] = thr
        data['signal_strength'] = st
        return data

