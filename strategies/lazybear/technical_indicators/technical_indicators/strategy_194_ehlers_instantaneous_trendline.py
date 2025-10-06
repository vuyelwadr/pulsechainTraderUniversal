#!/usr/bin/env python3
"""
Strategy 194: Ehlers Instantaneous Trendline

Pine reference: pine_scripts/194_ehlers_instant_trend.pine

Signals: buy when IT crosses above Trigger (lag); sell when crosses below.
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


class Strategy194EhlersInstantTrend(BaseStrategy):
    def __init__(self, parameters: Dict=None):
        params = {
            'alpha': 0.07,
            'signal_threshold': 0.6,
        }
        if parameters:
            params.update(parameters)
        super().__init__('Strategy_194_EhlersInstantTrend', params)

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        for c in ('high','low','close'):
            if c not in data:
                data[c]=data.get('close', pd.Series(index=data.index)).fillna(method='ffill')
        src = (data['high'] + data['low'])/2.0
        a = float(self.parameters['alpha'])
        it = pd.Series(index=src.index, dtype=float)
        for i in range(len(src)):
            if i < 2:
                it.iloc[i] = ((src + 2*src.shift(1) + src.shift(2))/4.0).iloc[i] if i>=2 else src.iloc[i]
            else:
                it.iloc[i] = (a - (a*a)/4.0)*src.iloc[i] + 0.5*a*a*src.shift(1).iloc[i] - (a - 0.75*a*a)*src.shift(2).iloc[i] + 2*(1-a)*it.iloc[i-1] - (1-a)*(1-a)*it.iloc[i-2]
        lag = 2.0*it - it.shift(2)
        data['it'] = it
        data['lag'] = lag
        return data

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        buy = crossover(data['it'], data['lag'])
        sell = crossunder(data['it'], data['lag'])
        data['buy_signal'] = buy
        data['sell_signal'] = sell
        st = (data['it'] - data['lag']).abs()
        st = (st / (data['it'].rolling(50, min_periods=10).std(ddof=0).replace(0,np.nan))).clip(0,1).fillna(0)
        st = st.where(buy|sell, 0.0)
        thr=float(self.parameters['signal_threshold'])
        st[(buy|sell) & (st < thr)] = thr
        data['signal_strength'] = st
        return data

