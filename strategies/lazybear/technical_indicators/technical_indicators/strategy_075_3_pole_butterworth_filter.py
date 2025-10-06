#!/usr/bin/env python3
"""
Strategy 075: 3‑Pole Butterworth Low‑Pass Filter

Implements a 3-pole Butterworth by cascading a canonical 2-pole Butterworth
with a 1-pole EMA stage at the same cutoff. Signals on price cross vs filter.
"""

import pandas as pd
import numpy as np
from typing import Dict
import os, sys, math

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)
from strategies.base_strategy import BaseStrategy

try:
    sys.path.insert(0, os.path.join(project_root, 'src'))
    from utils.vectorized_helpers import crossover, crossunder
except Exception:
    def crossover(a,b): return (a>b) & (a.shift(1)<=b)
    def crossunder(a,b): return (a<b) & (a.shift(1)>=b)


class Strategy075Butterworth3Pole(BaseStrategy):
    def __init__(self, parameters: Dict = None):
        params = {
            'period': 20,
            'source': 'close',
            'signal_threshold': 0.6,
        }
        if parameters:
            params.update(parameters)
        super().__init__('Strategy_075_Butterworth_3Pole', params)

    @staticmethod
    def _butterworth_2pole(x: pd.Series, period: int) -> pd.Series:
        n = max(1, int(period))
        c = 1.0 / math.tan(math.pi / n)
        a0 = 1.0 / (1.0 + math.sqrt(2.0)*c + c*c)
        a1 = 2.0 * a0
        a2 = a0
        b1 = 2.0 * a0 * (1.0 - c*c)
        b2 = a0 * (1.0 - math.sqrt(2.0)*c + c*c)
        y = pd.Series(index=x.index, dtype=float)
        x0=x; x1=x.shift(1).fillna(x.iloc[0]); x2=x.shift(2).fillna(x.iloc[0])
        y_prev1 = 0.0
        y_prev2 = 0.0
        for i, idx in enumerate(x.index):
            xi = x0.loc[idx]
            xim1 = x1.loc[idx]
            xim2 = x2.loc[idx]
            yi = a0*xi + a1*xim1 + a2*xim2 + b1*y_prev1 - b2*y_prev2
            y.loc[idx] = yi
            y_prev2 = y_prev1
            y_prev1 = yi
        return y

    @staticmethod
    def _ema(s: pd.Series, n: int) -> pd.Series:
        return s.ewm(span=max(1,int(n)), adjust=False, min_periods=1).mean()

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        src_name = self.parameters.get('source', 'close')
        if src_name not in data:
            if src_name == 'close' and 'price' in data:
                data['close'] = data['price']
            else:
                data[src_name] = data.get('close', pd.Series(index=data.index)).fillna(method='ffill')
        stage1 = self._butterworth_2pole(data[src_name].astype(float), int(self.parameters['period']))
        data['bw3'] = self._ema(stage1, int(self.parameters['period']))
        return data

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        src_name = self.parameters.get('source', 'close')
        buy = crossover(data[src_name], data['bw3'])
        sell = crossunder(data[src_name], data['bw3'])
        data['buy_signal'] = buy
        data['sell_signal'] = sell
        st = (data[src_name] - data['bw3']).abs() / (data[src_name].rolling(50, min_periods=10).std(ddof=0).replace(0, np.nan))
        st = st.clip(0, 1).fillna(0).where(buy | sell, 0.0)
        thr = float(self.parameters['signal_threshold'])
        st[(buy | sell) & (st < thr)] = thr
        data['signal_strength'] = st
        return data

