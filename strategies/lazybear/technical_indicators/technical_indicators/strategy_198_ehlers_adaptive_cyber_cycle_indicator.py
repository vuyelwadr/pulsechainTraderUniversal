#!/usr/bin/env python3
"""
Strategy 198: Ehlers Adaptive Cyber Cycle Indicator

Pine reference: pine_scripts/198_ehlers_adaptive_cyber.pine

Signals: buy when CyberCycle crosses above Smoothed; sell when crosses below.
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


class Strategy198EhlersAdaptiveCyberCycle(BaseStrategy):
    def __init__(self, parameters: Dict=None):
        params = {
            'length': 20,
            'smoothingLength': 9,
            'signal_threshold': 0.6,
        }
        if parameters:
            params.update(parameters)
        super().__init__('Strategy_198_EhlersAdaptiveCyberCycle', params)

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        if 'close' not in data and 'price' in data:
            data['close']=data['price']
        length = int(self.parameters['length'])
        alpha = 2.0 / (length + 1.0)
        cc = pd.Series(index=data.index, dtype=float)
        for i in range(len(data)):
            if i == 0:
                cc.iloc[i] = data['close'].iloc[i]
            else:
                cc.iloc[i] = alpha * (data['close'].iloc[i] - cc.iloc[i-1]) + cc.iloc[i-1]
        smoothed = cc.rolling(int(self.parameters['smoothingLength']), min_periods=1).mean()
        data['cyber'] = cc
        data['smoothed'] = smoothed
        return data

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        buy = crossover(data['cyber'], data['smoothed'])
        sell = crossunder(data['cyber'], data['smoothed'])
        data['buy_signal'] = buy
        data['sell_signal'] = sell
        st = (data['cyber'] - data['smoothed']).abs()
        st = (st / (data['cyber'].rolling(50, min_periods=10).std(ddof=0).replace(0,np.nan))).clip(0,1).fillna(0)
        st = st.where(buy|sell, 0.0)
        thr=float(self.parameters['signal_threshold'])
        st[(buy|sell) & (st < thr)] = thr
        data['signal_strength'] = st
        return data

