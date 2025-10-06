#!/usr/bin/env python3
"""
Strategy 042: Bear COG Channel (COG Double Channel)

Pine reference: pine_scripts/042_bear_cog.pine
- basis = linreg(close, length)
- stdev band: basis ± mult * stdev(close, length)
- ATR band (Starc): basis ± 2*sma(tr, length)

Signals (deterministic, indicator-driven):
- Buy when close crosses above either upper band (breakout)
- Sell when close crosses below either lower band (breakdown)
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


class Strategy042BearCOGChannel(BaseStrategy):
    def __init__(self, parameters: Dict = None):
        params = {
            'length': 34,
            'mult': 2.5,
            'signal_threshold': 0.6,
        }
        if parameters:
            params.update(parameters)
        super().__init__('Strategy_042_BearCOGChannel', params)

    @staticmethod
    def _stdev(s: pd.Series, n: int) -> pd.Series:
        return s.rolling(max(1,int(n)), min_periods=1).std(ddof=0)

    @staticmethod
    def _sma(s: pd.Series, n: int) -> pd.Series:
        return s.rolling(max(1,int(n)), min_periods=1).mean()

    @staticmethod
    def _tr(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        prev_close = close.shift(1)
        tr = pd.concat([
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ], axis=1).max(axis=1)
        return tr.fillna(0)

    @staticmethod
    def _linreg(s: pd.Series, n: int) -> pd.Series:
        n = max(1, int(n))
        if n == 1:
            return s
        def _f(window):
            wlen = len(window)
            x = np.arange(wlen, dtype=float)
            y = window.astype(float)
            x_mean = x.mean(); y_mean = y.mean()
            cov = ((x - x_mean) * (y - y_mean)).sum()
            var = ((x - x_mean) ** 2).sum()
            slope = 0.0 if var == 0 else cov / var
            intercept = y_mean - slope * x_mean
            return intercept + slope * (wlen - 1)
        return s.rolling(n, min_periods=1).apply(_f, raw=True)

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        if 'close' not in data and 'price' in data:
            data['close'] = data['price']
        for c in ('open','high','low','close'):
            if c not in data:
                data[c] = data.get('close', pd.Series(index=data.index)).fillna(method='ffill')

        length = int(self.parameters['length'])
        mult = float(self.parameters['mult'])

        basis = self._linreg(data['close'], length)
        dev = mult * self._stdev(data['close'], length)
        data['cog_ul'] = basis + dev
        data['cog_ll'] = basis - dev

        tr = self._tr(data['high'], data['low'], data['close'])
        acustom = 2.0 * self._sma(tr, length)
        data['cog_uls'] = basis + acustom
        data['cog_lls'] = basis - acustom
        data['cog_basis'] = basis
        return data

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        buy = crossover(data['close'], data['cog_ul']) | crossover(data['close'], data['cog_uls'])
        sell = crossunder(data['close'], data['cog_ll']) | crossunder(data['close'], data['cog_lls'])
        data['buy_signal'] = buy
        data['sell_signal'] = sell
        # Strength via distance from basis normalized by band width
        bw = (data['cog_ul'] - data['cog_ll']).replace(0, np.nan)
        dist = (data['close'] - data['cog_basis']).abs() / bw
        st = dist.clip(0,1).fillna(0)
        st = st.where(buy|sell, 0.0)
        thr = float(self.parameters['signal_threshold'])
        st[(buy|sell) & (st < thr)] = thr
        data['signal_strength'] = st
        return data
