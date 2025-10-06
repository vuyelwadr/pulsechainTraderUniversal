#!/usr/bin/env python3
"""
Strategy 084: Volume Zone Indicator (VZO)

Pine reference: pine_scripts/084_volume_zone_oscillator_lazybear.pine
vzo = 100 * EMA(sign(close-close[1]) * volume, length) / EMA(volume, length)
Signals: cross key zones; we use zero-line crosses for deterministic entries.
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


class Strategy084VolumeZoneIndicator(BaseStrategy):
    def __init__(self, parameters: Dict = None):
        params = {
            'length': 20,
            'signal_threshold': 0.6,
        }
        if parameters:
            params.update(parameters)
        super().__init__('Strategy_084_VZO', params)

    @staticmethod
    def _ema(s: pd.Series, n: int) -> pd.Series:
        return s.ewm(span=max(1,int(n)), adjust=False, min_periods=1).mean()

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        for c in ('close','volume'):
            if c not in data.columns:
                if c=='close' and 'price' in data.columns:
                    data['close']=data['price']
                else:
                    data['volume']=0
        n = int(self.parameters['length'])
        dvol = np.sign(data['close'].diff()).fillna(0) * data['volume']
        dvma = self._ema(pd.Series(dvol, index=data.index), n)
        vma = self._ema(data['volume'], n).replace(0,np.nan)
        data['vzo'] = (100.0 * dvma / vma).replace([np.inf,-np.inf], np.nan).fillna(0)
        return data

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        buy = crossover(data['vzo'], 0.0)
        sell = crossunder(data['vzo'], 0.0)
        data['buy_signal'] = buy
        data['sell_signal'] = sell
        st = (data['vzo'].abs() / 100.0).clip(0,1)
        st = st.where(buy|sell, 0.0)
        thr = float(self.parameters['signal_threshold'])
        st[(buy|sell) & (st < thr)] = thr
        data['signal_strength'] = st
        return data

