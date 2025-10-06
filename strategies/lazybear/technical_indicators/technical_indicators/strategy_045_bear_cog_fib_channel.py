#!/usr/bin/env python3
"""
Strategy 045: COG Fibs (Bear COG Fib Channel)

Pine reference: pine_scripts/045_bear_cog_fib_channel.pine

Signals: buy/sell on midline (mp) crosses as a minimal deterministic rule.
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


class Strategy045BearCOGFibChannel(BaseStrategy):
    def __init__(self, parameters: Dict=None):
        params = {
            'length': 20,
            'signal_threshold': 0.6,
        }
        if parameters:
            params.update(parameters)
        super().__init__('Strategy_045_BearCOGFibChannel', params)

    @staticmethod
    def _sma(s: pd.Series, n: int) -> pd.Series:
        return s.rolling(max(1,int(n)), min_periods=1).mean()

    @staticmethod
    def _wima(s: pd.Series, n: int) -> pd.Series:
        n = max(1,int(n))
        alpha = 1.0 / n
        return s.ewm(alpha=alpha, adjust=False, min_periods=1).mean()

    @staticmethod
    def _linreg(s: pd.Series, n: int) -> pd.Series:
        n = max(1,int(n))
        if n == 1:
            return s
        def _f(window):
            wlen = len(window)
            x = np.arange(wlen, dtype=float)
            y = window.astype(float)
            x_mean = x.mean(); y_mean = y.mean()
            cov = ((x - x_mean) * (y - y_mean)).sum()
            var = ((x - x_mean) ** 2).sum()
            slope = 0.0 if var == 0 else cov/var
            intercept = y_mean - slope*x_mean
            return intercept + slope*(wlen-1)
        return s.rolling(n, min_periods=1).apply(_f, raw=True)

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        for c in ('high','low','close'):
            if c not in data:
                data[c] = data.get('close', pd.Series(index=data.index)).fillna(method='ffill')
        length = int(self.parameters['length'])
        th = self._linreg(data['high'], length)
        tl = self._linreg(data['low'], length)
        tr_c = (th - tl).abs()
        tra = self._wima(tr_c, length)
        mp = self._sma(data['close'], length)
        data['cog_mp'] = mp
        for val, name in [(4.2360,'4p2360'),(3.6180,'3p6180'),(2.6180,'2p6180'),(1.6180,'1p6180'),(0.618,'0p618')]:
            data[f'cog_ub_{name}'] = mp + (val * tra)
            data[f'cog_lb_{name}'] = mp - (val * tra)
        return data

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        buy = crossover(data['close'], data['cog_mp'])
        sell = crossunder(data['close'], data['cog_mp'])
        data['buy_signal'] = buy
        data['sell_signal'] = sell
        # Strength via distance from mp normalized by 1.618 band
        bw = (data['cog_ub_1p6180'] - data['cog_lb_1p6180']).replace(0, np.nan)
        st = (data['close'] - data['cog_mp']).abs() / bw
        st = st.clip(0,1).fillna(0)
        st = st.where(buy|sell, 0.0)
        thr=float(self.parameters['signal_threshold'])
        st[(buy|sell) & (st < thr)] = thr
        data['signal_strength'] = st
        return data
