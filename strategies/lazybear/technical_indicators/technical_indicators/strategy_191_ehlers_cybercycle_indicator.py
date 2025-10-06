#!/usr/bin/env python3
"""
Strategy 191: Ehlers Cyber Cycle Indicator

Pine reference: pine_scripts/191_ehlers_cybercycle.pine

Signals: buy when cycle crosses above trigger; sell when crosses below.
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


class Strategy191EhlersCyberCycle(BaseStrategy):
    def __init__(self, parameters: Dict=None):
        params = {
            'alpha': 0.07,
            'signal_threshold': 0.6,
        }
        if parameters:
            params.update(parameters)
        super().__init__('Strategy_191_EhlersCyberCycle', params)

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        if 'high' not in data or 'low' not in data or 'close' not in data:
            if 'close' in data:
                data['high']=data['close']; data['low']=data['close']
            elif 'price' in data:
                data['close']=data['price']; data['high']=data['price']; data['low']=data['price']
        src = (data['high'] + data['low'])/2.0
        a = float(self.parameters['alpha'])
        smooth = (src + 2*src.shift(1) + 2*src.shift(2) + src.shift(3)) / 6.0
        cycle = pd.Series(0.0, index=src.index)
        for i in range(len(src)):
            if i < 7:
                cycle.iloc[i] = (src.iloc[i] - 2*src.shift(1).iloc[i] + src.shift(2).iloc[i]) / 4.0 if i>=2 else 0.0
            else:
                cycle.iloc[i] = (1-0.5*a)*(1-0.5*a)*(smooth.iloc[i]-2*smooth.shift(1).iloc[i]+smooth.shift(2).iloc[i]) + 2*(1-a)*cycle.iloc[i-1] - (1-a)*(1-a)*cycle.iloc[i-2]
        t = cycle.shift(1)
        data['cycle'] = cycle
        data['trigger'] = t
        return data

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        buy = crossover(data['cycle'], data['trigger'])
        sell = crossunder(data['cycle'], data['trigger'])
        data['buy_signal'] = buy
        data['sell_signal'] = sell
        st = (data['cycle'] - data['trigger']).abs()
        st = (st / (data['cycle'].rolling(50, min_periods=10).std(ddof=0).replace(0,np.nan))).clip(0,1).fillna(0)
        st = st.where(buy|sell, 0.0)
        thr=float(self.parameters['signal_threshold'])
        st[(buy|sell) & (st < thr)] = thr
        data['signal_strength'] = st
        return data

