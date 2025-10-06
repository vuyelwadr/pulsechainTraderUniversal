#!/usr/bin/env python3
"""
Strategy 188: Absolute Strength Index Oscillator

Pine reference: pine_scripts/188_absolute_strength_osc.pine

Signals: buy when ABSSIO crosses above signal; sell when crosses below.
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


class Strategy188AbsoluteStrengthOscillator(BaseStrategy):
    def __init__(self, parameters: Dict=None):
        params = {
            'lma': 21,
            'ld': 34,
            'osl': 10,
            'signal_threshold': 0.6,
        }
        if parameters:
            params.update(parameters)
        super().__init__('Strategy_188_AbsoluteStrengthOsc', params)

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
                M.iloc[i] = M.iloc[i-1]
                D.iloc[i] = D.iloc[i-1]
            elif c.iloc[i] < c.iloc[i-1]:
                D.iloc[i] = D.iloc[i-1] + (c.iloc[i-1]/c.iloc[i] - 1.0)
                M.iloc[i] = M.iloc[i-1]
                A.iloc[i] = A.iloc[i-1]
            else:
                M.iloc[i] = M.iloc[i-1] + 1.0/osl
                A.iloc[i] = A.iloc[i-1]
                D.iloc[i] = D.iloc[i-1]
        denom = (D + M/2.0).replace(0, np.nan)
        abssi = 1.0 - 1.0/(1.0 + (A + M/2.0)/denom)
        abssio = abssi - abssi.ewm(span=int(self.parameters['lma']), adjust=False, min_periods=1).mean()
        alp = 2.0 / (float(self.parameters['ld']) + 1.0)
        mt = pd.Series(0.0, index=c.index)
        ut = pd.Series(0.0, index=c.index)
        for i in range(1, len(c)):
            mt.iloc[i] = alp*abssio.iloc[i] + (1-alp)*mt.iloc[i-1]
            ut.iloc[i] = alp*mt.iloc[i] + (1-alp)*ut.iloc[i-1]
        s = ((2 - alp) * mt - ut) / (1 - alp)
        data['abssio'] = abssio
        data['abssio_sig'] = s
        return data

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        buy = crossover(data['abssio'], data['abssio_sig'])
        sell = crossunder(data['abssio'], data['abssio_sig'])
        data['buy_signal'] = buy
        data['sell_signal'] = sell
        st = (data['abssio'] - data['abssio_sig']).abs()
        st = (st / (data['abssio'].rolling(50, min_periods=10).std(ddof=0).replace(0,np.nan))).clip(0,1).fillna(0)
        st = st.where(buy|sell, 0.0)
        thr=float(self.parameters['signal_threshold'])
        st[(buy|sell) & (st < thr)] = thr
        data['signal_strength'] = st
        return data

