#!/usr/bin/env python3
"""
Strategy 199: Ehlers Adaptive CG Oscillator (approximation)

Implements an adaptive CG oscillator by estimating a dominant cycle period from
price changes and adapting the smoothing factor accordingly. Signals on crossing
zero.
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


class Strategy199EhlersAdaptiveCGOscillator(BaseStrategy):
    def __init__(self, parameters: Dict=None):
        params = {
            'alpha': 0.07,
            'signal_threshold': 0.6,
        }
        if parameters:
            params.update(parameters)
        super().__init__('Strategy_199_Ehlers_Adaptive_CG_Oscillator', params)

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        if 'close' not in data and 'price' in data:
            data['close']=data['price']
        src = (data['high'] + data['low'])/2.0 if 'high' in data and 'low' in data else data['close']
        a = float(self.parameters['alpha'])
        # Pre-smoothing
        s = (src + 2*src.shift(1) + 2*src.shift(2) + src.shift(3)) / 6.0
        c = pd.Series(index=src.index, dtype=float)
        for i in range(len(src)):
            if i < 2:
                c.iloc[i] = (src.iloc[i] - 2*src.shift(1).fillna(src.iloc[i]).iloc[i] + src.shift(2).fillna(src.iloc[i]).iloc[i]) / 4.0
            else:
                c.iloc[i] = (1 - 0.5*a)*(1 - 0.5*a)*(s.iloc[i]-2*s.shift(1).iloc[i]+s.shift(2).iloc[i]) + 2*(1-a)*c.iloc[i-1] - (1-a)*(1-a)*c.iloc[i-2]
        # Quadrature component approximation
        q1 = (.0962*c + 0.5769*c.shift(2).bfill() - 0.5769*c.shift(4).bfill() - .0962*c.shift(6).bfill()) * (0.5 + .08)
        I1 = c.shift(3).bfill()
        # Phase advance estimator
        dp_ = (I1 / q1.replace(0,np.nan) - I1.shift(1).bfill()/ q1.shift(1).replace(0,np.nan)) / (1 + (I1*I1.shift(1).bfill())/(q1*q1.shift(1).replace(0,np.nan)))
        dp = dp_.clip(0.1, 1.1).fillna(0.1)
        # Median smoothing for dominant cycle
        md = pd.concat([dp, dp.shift(1), dp.shift(2), dp.shift(3), dp.shift(4)], axis=1).median(axis=1)
        dc = (6.28318 / md + 0.5).clip(5, 50)  # bounded dominant cycle
        # Adaptive filter alpha from dc
        alpha_ad = 2.0 / (dc + 1.0)
        acg = pd.Series(index=src.index, dtype=float)
        for i in range(len(src)):
            prev = src.iloc[i] if i==0 else acg.iloc[i-1]
            acg.iloc[i] = alpha_ad.iloc[i] * (src.iloc[i] - prev) + prev
        data['acg'] = acg.fillna(method='ffill').fillna(src)
        return data

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        diff = data['acg'].diff().fillna(0)
        buy = crossover(diff, 0.0)
        sell = crossunder(diff, 0.0)
        data['buy_signal'] = buy
        data['sell_signal'] = sell
        st = (diff.abs() / (diff.rolling(50, min_periods=10).std(ddof=0).replace(0,np.nan))).clip(0,1).fillna(0)
        st = st.where(buy|sell, 0.0)
        thr=float(self.parameters['signal_threshold'])
        st[(buy|sell) & (st < thr)] = thr
        data['signal_strength'] = st
        return data
