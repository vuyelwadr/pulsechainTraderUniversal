#!/usr/bin/env python3
"""
Strategy 177: JMA RSX Clone (approximation)

Implements an RSX-like oscillator by computing Ehlers-style RSI (RSX) and
optionally smoothing with a zero-lag EMA. Signals on RSX crossing midline (50).
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


class Strategy177JmaRsxClone(BaseStrategy):
    def __init__(self, parameters: Dict=None):
        params = {
            'length': 14,
            'use_zlema': True,
            'signal_threshold': 0.6,
        }
        if parameters:
            params.update(parameters)
        super().__init__('Strategy_177_JMA_RSX_Clone', params)

    @staticmethod
    def _rsx(s: pd.Series, n: int) -> pd.Series:
        n=max(1,int(n))
        d = s.diff()
        up = d.clip(lower=0)
        dn = -d.clip(upper=0)
        alpha = 1.0 / n
        cu = up.ewm(alpha=alpha, adjust=False, min_periods=1).mean()
        cd = dn.ewm(alpha=alpha, adjust=False, min_periods=1).mean()
        rs = cu / cd.replace(0, np.nan)
        return (100 - 100/(1+rs)).fillna(50)

    @staticmethod
    def _ema(s: pd.Series, n: int) -> pd.Series:
        return s.ewm(span=max(1,int(n)), adjust=False, min_periods=1).mean()

    def _zlema(self, s: pd.Series, n: int) -> pd.Series:
        e1=self._ema(s,n); e2=self._ema(e1,n); return e1 + (e1 - e2)

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        if 'close' not in data and 'price' in data:
            data['close']=data['price']
        r = self._rsx(data['close'].astype(float), int(self.parameters['length']))
        data['rsx'] = self._zlema(r, int(self.parameters['length'])) if bool(self.parameters.get('use_zlema', True)) else r
        return data

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        buy = crossover(data['rsx'], 50.0)
        sell = crossunder(data['rsx'], 50.0)
        data['buy_signal'] = buy
        data['sell_signal'] = sell
        st = (data['rsx'] - 50.0).abs() / 50.0
        st = st.clip(0,1).fillna(0).where(buy|sell, 0.0)
        thr=float(self.parameters['signal_threshold'])
        st[(buy|sell) & (st < thr)] = thr
        data['signal_strength'] = st
        return data

