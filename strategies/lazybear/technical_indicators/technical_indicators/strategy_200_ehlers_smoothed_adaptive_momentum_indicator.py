#!/usr/bin/env python3
"""
Strategy 200: Ehlers Smoothed Adaptive Momentum (approximation)

Implements a smoothed adaptive momentum using pre-smoothing, quadrature estimate,
and a phase-adaptive step, following the structure of the Pine.
Signals on momentum crossing zero.
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


class Strategy200EhlersSmoothedAdaptiveMomentum(BaseStrategy):
    def __init__(self, parameters: Dict=None):
        params = {
            'alpha': 0.07,
            'cutoff': 8.0,
            'signal_threshold': 0.6,
        }
        if parameters:
            params.update(parameters)
        super().__init__('Strategy_200_Ehlers_Smoothed_Adaptive_Momentum', params)

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        src = (data['high'] + data['low'])/2.0 if 'high' in data and 'low' in data else data['close']
        a = float(self.parameters['alpha'])
        # Pre-smoothing s
        s = (src + 2*src.shift(1) + 2*src.shift(2) + src.shift(3)) / 6.0
        # Core component c (instantaneous trend proxy)
        c = pd.Series(index=src.index, dtype=float)
        for i in range(len(src)):
            if i < 2:
                c.iloc[i] = (src.iloc[i] - 2*src.shift(1).fillna(src.iloc[i]).iloc[i] + src.shift(2).fillna(src.iloc[i]).iloc[i]) / 4.0
            else:
                c.iloc[i] = (1 - 0.5*a)*(1 - 0.5*a)*(s.iloc[i]-2*s.shift(1).iloc[i]+s.shift(2).iloc[i]) + 2*(1-a)*c.iloc[i-1] - (1-a)*(1-a)*c.iloc[i-2]
        # Quadrature estimate
        q1 = (.0962*c + 0.5769*c.shift(2).bfill() - 0.5769*c.shift(4).bfill() - .0962*c.shift(6).bfill()) * (0.5+.08)
        I1 = c.shift(3).bfill()
        momentum = I1 - I1.shift(1).fillna(I1)
        # Smoothed adaptive momentum via cutoff
        co = float(self.parameters['cutoff'])
        k = 2.0 / (co + 1.0)
        sam = pd.Series(index=src.index, dtype=float)
        for i in range(len(src)):
            prev = momentum.iloc[i] if i==0 else sam.iloc[i-1]
            sam.iloc[i] = prev + k*(momentum.iloc[i] - prev)
        data['sam'] = sam.fillna(0)
        return data

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        buy = crossover(data['sam'], 0.0)
        sell = crossunder(data['sam'], 0.0)
        data['buy_signal'] = buy
        data['sell_signal'] = sell
        st = (data['sam'].abs() / (data['sam'].rolling(50, min_periods=10).std(ddof=0).replace(0,np.nan))).clip(0,1).fillna(0)
        st = st.where(buy|sell, 0.0)
        thr=float(self.parameters['signal_threshold'])
        st[(buy|sell) & (st < thr)] = thr
        data['signal_strength'] = st
        return data
