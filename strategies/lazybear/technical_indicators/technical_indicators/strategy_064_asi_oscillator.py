#!/usr/bin/env python3
"""
Strategy 064: ASI Oscillator

Oscillator derived from Swing Index (SI):
  asi_osc = (SI - SMA(SI,n)) / STD(SI,n)
Smoothed and used with threshold crossings.
"""

import pandas as pd
import numpy as np
from typing import Dict
import sys, os

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)
from strategies.base_strategy import BaseStrategy

try:
    sys.path.insert(0, os.path.join(project_root, 'src'))
    from utils.vectorized_helpers import crossover, crossunder
except Exception:
    def crossover(a,b): return (a>b) & (a.shift(1)<=b)
    def crossunder(a,b): return (a<b) & (a.shift(1)>=b)


class Strategy064ASIOscillator(BaseStrategy):
    def __init__(self, parameters: Dict = None):
        params = {
            'limit_move': 0.25,
            'oscillator_period': 20,
            'smooth_period': 5,
            'signal_threshold': 0.6,
        }
        if parameters:
            params.update(parameters)
        super().__init__('Strategy_064_ASI_Oscillator', params)

    @staticmethod
    def _ema(s: pd.Series, n: int) -> pd.Series:
        return s.ewm(span=max(1,int(n)), adjust=False, min_periods=1).mean()

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        for c in ('open','high','low','close'):
            if c not in data.columns and 'price' in data.columns:
                data[c]=data['price']
        tr = pd.concat([
            data['high'] - data['low'],
            (data['high'] - data['close'].shift(1)).abs(),
            (data['low'] - data['close'].shift(1)).abs()
        ], axis=1).max(axis=1)
        k_factor = np.maximum(tr, float(self.parameters['limit_move']))
        cy = data['close'] - data['close'].shift(1)
        oy = data['open'] - data['close'].shift(1)
        hy = data['high'] - data['close'].shift(1)
        ly = data['low'] - data['close'].shift(1)
        r = np.where(hy.abs() > ly.abs(), hy, ly)
        si = 50.0 * (cy + 0.5*oy + 0.25*r) / (pd.Series(k_factor, index=data.index).replace(0,np.nan))
        n = int(self.parameters['oscillator_period'])
        si_sma = si.rolling(n, min_periods=1).mean()
        si_std = si.rolling(n, min_periods=1).std(ddof=0).replace(0,np.nan)
        asi_osc = ((si - si_sma) / si_std).replace([np.inf,-np.inf], np.nan).fillna(0)
        asi_osc_s = self._ema(asi_osc, int(self.parameters['smooth_period']))
        data['si'] = si
        data['asi_osc'] = asi_osc
        data['asi_osc_s'] = asi_osc_s
        return data

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        # Cross +/-1 bands
        buy = crossover(data['asi_osc_s'], -1.0)
        sell = crossunder(data['asi_osc_s'], 1.0)
        data['buy_signal'] = buy
        data['sell_signal'] = sell
        st = data['asi_osc_s'].abs().clip(0,2) / 2.0
        st = st.where(buy|sell, 0.0)
        thr = float(self.parameters['signal_threshold'])
        st[(buy|sell) & (st < thr)] = thr
        data['signal_strength'] = st.clip(0,1)
        return data

