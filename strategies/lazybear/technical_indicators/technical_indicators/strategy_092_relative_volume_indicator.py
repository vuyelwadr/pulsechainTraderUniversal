#!/usr/bin/env python3
"""
Strategy 092: Relative Volume Indicator

Computes relative volume as ratio of current volume to its rolling average.
Signals when RVOL crosses above and below dynamic bands.
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


class Strategy092RelativeVolumeIndicator(BaseStrategy):
    def __init__(self, parameters: Dict=None):
        params = {
            'avg_len': 20,
            'band_mult': 1.5,
            'signal_threshold': 0.6,
        }
        if parameters:
            params.update(parameters)
        super().__init__('Strategy_092_RelativeVolumeIndicator', params)

    @staticmethod
    def _sma(s: pd.Series, n: int) -> pd.Series:
        return s.rolling(max(1,int(n)), min_periods=1).mean()

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        if 'volume' not in data:
            data['volume'] = 0
        n=int(self.parameters['avg_len'])
        vma = self._sma(data['volume'], n)
        rvol = (data['volume'] / vma.replace(0,np.nan)).replace([np.inf,-np.inf], np.nan).fillna(0)
        data['rvol'] = rvol
        data['rvol_mid'] = 1.0
        data['rvol_up'] = 1.0 + float(self.parameters['band_mult']) * rvol.rolling(n, min_periods=1).std(ddof=0)
        data['rvol_dn'] = (1.0 - float(self.parameters['band_mult']) * rvol.rolling(n, min_periods=1).std(ddof=0)).clip(lower=0)
        return data

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        buy = crossover(data['rvol'], data['rvol_up'])
        sell = crossunder(data['rvol'], data['rvol_dn'])
        data['buy_signal'] = buy
        data['sell_signal'] = sell
        st = (data['rvol'] - 1.0).abs()
        st = (st / (data['rvol'].rolling(50, min_periods=10).std(ddof=0).replace(0,np.nan))).clip(0,1).fillna(0)
        st = st.where(buy|sell, 0.0)
        thr=float(self.parameters['signal_threshold'])
        st[(buy|sell) & (st < thr)] = thr
        data['signal_strength'] = st
        return data

