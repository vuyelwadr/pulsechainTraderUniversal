#!/usr/bin/env python3
"""
Strategy 189: Absolute Strength Index

Pine reference: pine_scripts/189_absolute_strength_index.pine

Signals: buy when ABSSI crosses above 50; sell when crosses below 50.
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


class Strategy189AbsoluteStrengthIndex(BaseStrategy):
    def __init__(self, parameters: Dict=None):
        params = {
            'lma': 21,
            'ld': 34,
            'osl': 10,
            'signal_threshold': 0.6,
        }
        if parameters:
            params.update(parameters)
        super().__init__('Strategy_189_AbsoluteStrengthIndex', params)

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        if 'close' not in data and 'price' in data:
            data['close']=data['price']
        c = data['close']
        osl = float(self.parameters['osl'])
        A = pd.Series(0.0, index=c.index)
        M = pd.Series(0.0, index=c.index)
        D = pd.Series(0.0, index=c.index)
        for i in range(1, len(c)):
            if c.iloc[i] > c.iloc[i-1]:
                A.iloc[i] = A.iloc[i-1] + (c.iloc[i]/c.iloc[i-1] - 1.0)
            elif c.iloc[i] < c.iloc[i-1]:
                D.iloc[i] = D.iloc[i-1] + (c.iloc[i-1]/c.iloc[i] - 1.0)
            else:
                M.iloc[i] = M.iloc[i-1] + 1.0/osl
        denom = (D + M/2.0).replace(0, np.nan)
        abssi = 100.0 - 100.0/(1.0 + (A + M/2.0)/denom)
        data['abssi'] = abssi.replace([np.inf,-np.inf], np.nan).fillna(50)
        return data

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        buy = crossover(data['abssi'], 50.0)
        sell = crossunder(data['abssi'], 50.0)
        data['buy_signal'] = buy
        data['sell_signal'] = sell
        st = (data['abssi'] - 50.0).abs() / 50.0
        st = st.clip(0,1).fillna(0).where(buy|sell, 0.0)
        thr=float(self.parameters['signal_threshold'])
        st[(buy|sell) & (st < thr)] = thr
        data['signal_strength'] = st
        return data

