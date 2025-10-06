#!/usr/bin/env python3
"""
Strategy 152: Vervoort Heiken‑Ashi Candlestick Oscillator (HACO)

Pine reference: pine_scripts/152_vervoort_ha_osc.pine (truncated). Implemented using
the canonical Vervoort zero‑lag TEMA approach on Heiken‑Ashi candles.

Signals: buy when HACO crosses above 0; sell when crosses below 0.
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


class Strategy152VervoortHeikenAshiCandlestickOscillator(BaseStrategy):
    def __init__(self, parameters: Dict=None):
        params = {
            'avgup': 34,
            'avgdn': 34,
            'signal_threshold': 0.6,
        }
        if parameters:
            params.update(parameters)
        super().__init__('Strategy_152_Vervoort_HeikenAshi_Candlestick_Oscillator', params)

    @staticmethod
    def _ema(s: pd.Series, n: int) -> pd.Series:
        return s.ewm(span=max(1,int(n)), adjust=False, min_periods=1).mean()

    def _tema(self, s: pd.Series, n: int) -> pd.Series:
        e1 = self._ema(s, n)
        e2 = self._ema(e1, n)
        e3 = self._ema(e2, n)
        return 3*(e1 - e2) + e3

    def _zltema(self, s: pd.Series, n: int) -> pd.Series:
        t1 = self._tema(s, n)
        t2 = self._tema(t1, n)
        return t1 + (t1 - t2)

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        for c in ('open','high','low','close'):
            if c not in data:
                data[c] = data.get('close', pd.Series(index=data.index)).fillna(method='ffill')
        ohlc4 = data[['open','high','low','close']].mean(axis=1)
        ha_open = (ohlc4.shift(1) + ohlc4.shift(1)) / 2.0
        ha_c = (ohlc4 + ha_open + data[['high', 'low']].max(axis=1) + data[['high','low']].min(axis=1)) / 4.0
        p=self.parameters
        up = self._zltema(ha_c, int(p['avgup']))
        dn = self._zltema(ohlc4, int(p['avgdn']))
        data['haco'] = (up - dn)
        return data

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        buy = crossover(data['haco'], 0.0)
        sell = crossunder(data['haco'], 0.0)
        data['buy_signal'] = buy
        data['sell_signal'] = sell
        st = (data['haco'].abs() / (data['haco'].rolling(50, min_periods=10).std(ddof=0).replace(0,np.nan))).clip(0,1).fillna(0)
        st = st.where(buy|sell, 0.0)
        thr=float(self.parameters['signal_threshold'])
        st[(buy|sell) & (st < thr)] = thr
        data['signal_strength'] = st
        return data

