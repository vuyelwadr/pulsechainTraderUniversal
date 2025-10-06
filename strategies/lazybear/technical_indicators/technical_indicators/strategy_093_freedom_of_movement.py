#!/usr/bin/env python3
"""
Strategy 093: Freedom Of Movement (approximation)

Similar to 'Ease of Movement'—price movement adjusted by range and volume.
Signals on FOM crossing 0.
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


class Strategy093FreedomOfMovement(BaseStrategy):
    def __init__(self, parameters: Dict=None):
        params = {
            'length': 14,
            'signal_threshold': 0.6,
        }
        if parameters:
            params.update(parameters)
        super().__init__('Strategy_093_Freedom_of_Movement', params)

    @staticmethod
    def _sma(s: pd.Series, n: int) -> pd.Series:
        return s.rolling(max(1,int(n)), min_periods=1).mean()

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        for c in ('high','low','close','volume'):
            if c not in data:
                data[c]=0 if c=='volume' else data.get('close', pd.Series(index=data.index)).fillna(method='ffill')
        mid = (data['high'] + data['low'])/2.0
        dm = mid.diff().fillna(0)
        rng = (data['high'] - data['low']).replace(0,np.nan)
        fom = (dm / rng) / (data['volume'].replace(0,np.nan))
        data['fom'] = self._sma(fom.replace([np.inf,-np.inf], np.nan).fillna(0), int(self.parameters['length']))
        return data

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        buy = crossover(data['fom'], 0.0)
        sell = crossunder(data['fom'], 0.0)
        data['buy_signal'] = buy
        data['sell_signal'] = sell
        st = data['fom'].abs()
        st = (st / (st.rolling(50, min_periods=10).std(ddof=0).replace(0,np.nan))).clip(0,1).fillna(0)
        st = st.where(buy|sell, 0.0)
        thr=float(self.parameters['signal_threshold'])
        st[(buy|sell) & (st<thr)] = thr
        data['signal_strength'] = st
        return data

