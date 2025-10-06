#!/usr/bin/env python3
"""
Strategy 174: Belkhayate Timing

Pine reference: pine_scripts/174_belkhayate_timing.pine

Signals: buy when HT crosses above -Range1; sell when HT crosses below +Range1.
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


class Strategy174BelkhayateTiming(BaseStrategy):
    def __init__(self, parameters: Dict=None):
        params = {
            'Range1': 4.0,
            'Range2': 8.0,
            'showHLC': True,
            'signal_threshold': 0.6,
        }
        if parameters:
            params.update(parameters)
        super().__init__('Strategy_174_BelkhayateTiming', params)

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        for c in ('open','high','low','close'):
            if c not in data:
                data[c] = data.get('close', pd.Series(index=data.index)).fillna(method='ffill')
        # Middle and scale from last 5 bars
        mid = ((data['high'] + data['low'] + data['high'].shift(1) + data['low'].shift(1) + data['high'].shift(2) + data['low'].shift(2) + data['high'].shift(3) + data['low'].shift(3) + data['high'].shift(4) + data['low'].shift(4)) / 2.0) / 5.0
        scale = (((data['high'] - data['low']) + (data['high'].shift(1) - data['low'].shift(1)) + (data['high'].shift(2) - data['low'].shift(2)) + (data['high'].shift(3) - data['low'].shift(3)) + (data['high'].shift(4) - data['low'].shift(4))) / 5.0) * 0.2
        h = (data['high'] - mid) / scale.replace(0,np.nan)
        l = (data['low'] - mid) / scale.replace(0,np.nan)
        c = (data['close'] - mid) / scale.replace(0,np.nan)
        showHLC = bool(self.parameters.get('showHLC', True))
        ht = (h + l + c)/3.0 if showHLC else c
        data['bt_ht'] = ht.replace([np.inf,-np.inf], np.nan).fillna(0)
        data['bt_buy_line'] = -float(self.parameters['Range1'])
        data['bt_sell_line'] = float(self.parameters['Range1'])
        return data

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        buy_line = data['bt_buy_line']
        sell_line = data['bt_sell_line']
        buy = crossover(data['bt_ht'], buy_line)
        sell = crossunder(data['bt_ht'], sell_line)
        data['buy_signal'] = buy
        data['sell_signal'] = sell
        st = (data['bt_ht'].abs() / (data['bt_ht'].rolling(50, min_periods=10).std(ddof=0).replace(0,np.nan))).clip(0,1).fillna(0)
        st = st.where(buy|sell, 0.0)
        thr=float(self.parameters['signal_threshold'])
        st[(buy|sell) & (st < thr)] = thr
        data['signal_strength'] = st
        return data

