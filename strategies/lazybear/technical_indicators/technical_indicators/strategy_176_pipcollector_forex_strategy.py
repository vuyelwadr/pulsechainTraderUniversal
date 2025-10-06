#!/usr/bin/env python3
"""
Strategy 176: Pip Collector (simplified, single timeframe)

Pine reference: pine_scripts/176_pipcollector_forex.pine
Simplified to current timeframe: center EMA and symmetric bands. Signals: 
buy when price crosses above center EMA; sell when crosses below.
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


class Strategy176PipCollector(BaseStrategy):
    def __init__(self, parameters: Dict=None):
        params = {
            'lengthCenter': 50,
            'lengthLower': 20,  # treated as fixed offset units
            'signal_threshold': 0.6,
        }
        if parameters:
            params.update(parameters)
        super().__init__('Strategy_176_PipCollector', params)

    @staticmethod
    def _ema(s: pd.Series, n: int) -> pd.Series:
        return s.ewm(span=max(1,int(n)), adjust=False, min_periods=1).mean()

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        if 'close' not in data and 'price' in data:
            data['close']=data['price']
        center = self._ema(data['close'], int(self.parameters['lengthCenter']))
        offset = float(self.parameters['lengthLower']) * (data['close'].diff().abs().rolling(100, min_periods=1).median().fillna(0))
        data['pc_center'] = center
        data['pc_lower'] = center - offset
        data['pc_upper'] = center + offset
        return data

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        buy = crossover(data['close'], data['pc_center'])
        sell = crossunder(data['close'], data['pc_center'])
        data['buy_signal'] = buy
        data['sell_signal'] = sell
        st = (data['close'] - data['pc_center']).abs() / (data['close'].rolling(50, min_periods=10).std(ddof=0).replace(0,np.nan))
        st = st.clip(0,1).fillna(0).where(buy|sell, 0.0)
        thr=float(self.parameters['signal_threshold'])
        st[(buy|sell) & (st < thr)] = thr
        data['signal_strength'] = st
        return data

