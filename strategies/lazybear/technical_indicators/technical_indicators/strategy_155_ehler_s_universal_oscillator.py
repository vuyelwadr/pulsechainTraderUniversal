#!/usr/bin/env python3
"""
Strategy 155: Ehlers Universal Oscillator

Pine reference: pine_scripts/155_ehlers_universal.pine (truncated). Implemented per
canonical formulation: whitenoise input -> 2‑pole bandpass‑like filter, peak detector,
and normalized oscillator. Signals on zero crossing.
"""

import pandas as pd
import numpy as np
from typing import Dict
import os, sys, math

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)
from strategies.base_strategy import BaseStrategy

try:
    sys.path.insert(0, os.path.join(project_root, 'src'))
    from utils.vectorized_helpers import crossover, crossunder
except Exception:
    def crossover(a,b): return (a>b) & (a.shift(1)<=b)
    def crossunder(a,b): return (a<b) & (a.shift(1)>=b)


class Strategy155EhlersUniversalOscillator(BaseStrategy):
    def __init__(self, parameters: Dict=None):
        params = {
            'bandedge': 20,
            'signal_threshold': 0.6,
        }
        if parameters:
            params.update(parameters)
        super().__init__('Strategy_155_Ehlers_Universal_Oscillator', params)

    @staticmethod
    def _filt_series(x: pd.Series, bandedge: int) -> pd.Series:
        n = max(2, int(bandedge))
        a1 = math.exp(-1.414 * math.pi / n)
        b1 = 2.0 * a1 * math.cos(1.414 * math.pi / n)
        c2 = b1
        c3 = -a1*a1
        c1 = 1.0 - c2 - c3
        y = pd.Series(index=x.index, dtype=float)
        x1 = x.shift(1).fillna(x.iloc[0])
        y_prev1 = 0.0
        y_prev2 = 0.0
        for i, idx in enumerate(x.index):
            yi = c1 * (x.loc[idx] + x1.loc[idx]) * 0.5 + c2*y_prev1 + c3*y_prev2
            y.loc[idx] = yi
            y_prev2 = y_prev1
            y_prev1 = yi
        return y

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        if 'close' not in data and 'price' in data:
            data['close']=data['price']
        wn = (data['close'] - data['close'].shift(2)).fillna(0) / 2.0
        filt = self._filt_series(wn.astype(float), int(self.parameters['bandedge']))
        # Peak detector
        pk = pd.Series(index=filt.index, dtype=float)
        for i, idx in enumerate(filt.index):
            if i == 1:
                pk.iloc[i] = 1e-7
            else:
                prev = 0.0 if i==0 else pk.iloc[i-1]
                pk.iloc[i] = max(abs(filt.iloc[i]), 0.991*prev)
        denom = pk.replace(0, np.nan)
        euo = (filt / denom).clip(-1,1).fillna(0)
        data['euo'] = euo
        return data

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        buy = crossover(data['euo'], 0.0)
        sell = crossunder(data['euo'], 0.0)
        data['buy_signal'] = buy
        data['sell_signal'] = sell
        st = (data['euo'].abs() / (data['euo'].rolling(50, min_periods=10).std(ddof=0).replace(0,np.nan))).clip(0,1).fillna(0)
        st = st.where(buy|sell, 0.0)
        thr=float(self.parameters['signal_threshold'])
        st[(buy|sell) & (st < thr)] = thr
        data['signal_strength'] = st
        return data

