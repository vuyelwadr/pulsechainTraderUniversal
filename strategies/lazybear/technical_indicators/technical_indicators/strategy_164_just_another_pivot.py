#!/usr/bin/env python3
"""
Strategy 164: Just Another Pivot (simplified)

Pine reference: pine_scripts/164_just_another_pivot.pine
Computes hlc3 pivot (no MTF). Signals: buy when price crosses above pivot; sell when crosses below.
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


class Strategy164JustAnotherPivot(BaseStrategy):
    def __init__(self, parameters: Dict=None):
        params = {
            'signal_threshold': 0.6,
        }
        if parameters:
            params.update(parameters)
        super().__init__('Strategy_164_JustAnotherPivot', params)

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        for c in ('high','low','close'):
            if c not in data:
                data[c] = data.get('close', pd.Series(index=data.index)).fillna(method='ffill')
        data['pivot'] = (data['high'] + data['low'] + data['close'])/3.0
        return data

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        buy = crossover(data['close'], data['pivot'])
        sell = crossunder(data['close'], data['pivot'])
        data['buy_signal'] = buy
        data['sell_signal'] = sell
        st = (data['close'] - data['pivot']).abs() / (data['close'].rolling(50, min_periods=10).std(ddof=0).replace(0,np.nan))
        st = st.clip(0,1).fillna(0).where(buy|sell, 0.0)
        thr=float(self.parameters['signal_threshold'])
        st[(buy|sell) & (st < thr)] = thr
        data['signal_strength'] = st
        return data

