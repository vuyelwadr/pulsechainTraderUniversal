#!/usr/bin/env python3
"""
Strategy 173: Volume Accumulation Percentage Indicator (VAPI)

Pine reference: pine_scripts/173_volume_accumulation.pine

Signals: buy when VA crosses above 0; sell when crosses below 0.
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


class Strategy173VolumeAccumulationPercentageIndicator(BaseStrategy):
    def __init__(self, parameters: Dict=None):
        params = {
            'length': 10,
            'signal_threshold': 0.6,
        }
        if parameters:
            params.update(parameters)
        super().__init__('Strategy_173_VAPI', params)

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        for c in ('high','low','close','volume'):
            if c not in data:
                data[c] = 0 if c=='volume' else data.get('close', pd.Series(index=data.index)).fillna(method='ffill')
        n = int(self.parameters['length'])
        x = (2*data['close'] - data['high'] - data['low']) / (data['high'] - data['low']).replace(0, np.nan)
        tva = (data['volume'] * x).rolling(n, min_periods=1).sum()
        tv = data['volume'].rolling(n, min_periods=1).sum().replace(0, np.nan)
        data['va'] = 100.0 * tva / tv
        data['va'] = data['va'].replace([np.inf,-np.inf], np.nan).fillna(0)
        return data

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        buy = crossover(data['va'], 0.0)
        sell = crossunder(data['va'], 0.0)
        data['buy_signal'] = buy
        data['sell_signal'] = sell
        st = (data['va'].abs() / (data['va'].rolling(50, min_periods=10).std(ddof=0).replace(0,np.nan))).clip(0,1).fillna(0)
        st = st.where(buy|sell, 0.0)
        thr=float(self.parameters['signal_threshold'])
        st[(buy|sell) & (st < thr)] = thr
        data['signal_strength'] = st
        return data

