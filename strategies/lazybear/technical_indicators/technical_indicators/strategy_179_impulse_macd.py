#!/usr/bin/env python3
"""
Strategy 179: Impulse MACD

Pine reference: pine_scripts/179_impulse_macd.pine

Signals: buy when MD crosses above SB (signal); sell when MD crosses below SB.
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


class Strategy179ImpulseMACD(BaseStrategy):
    def __init__(self, parameters: Dict=None):
        params = {
            'lengthMA': 34,
            'lengthSignal': 9,
            'signal_threshold': 0.6,
        }
        if parameters:
            params.update(parameters)
        super().__init__('Strategy_179_ImpulseMACD', params)

    @staticmethod
    def _sma(s: pd.Series, n: int) -> pd.Series:
        return s.rolling(max(1,int(n)), min_periods=1).mean()

    @staticmethod
    def _zlema(s: pd.Series, n: int) -> pd.Series:
        e1 = s.ewm(span=max(1,int(n)), adjust=False, min_periods=1).mean()
        e2 = e1.ewm(span=max(1,int(n)), adjust=False, min_periods=1).mean()
        d = e1 - e2
        return e1 + d

    @staticmethod
    def _smma(s: pd.Series, n: int) -> pd.Series:
        n = max(1,int(n))
        out = pd.Series(index=s.index, dtype=float)
        out.iloc[0] = s.iloc[0]
        alpha = 1.0 / n
        for i in range(1, len(s)):
            prev = out.iloc[i-1]
            out.iloc[i] = prev * (n - 1)/n + s.iloc[i]/n
        return out

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        for c in ('high','low','close'):
            if c not in data:
                data[c] = data.get('close', pd.Series(index=data.index)).fillna(method='ffill')
        n = int(self.parameters['lengthMA'])
        src = data[['high','low','close']].mean(axis=1)
        hi = self._smma(data['high'], n)
        lo = self._smma(data['low'], n)
        mi = self._zlema(src, n)
        md = np.where(mi>hi, mi-hi, np.where(mi<lo, mi-lo, 0.0))
        md = pd.Series(md, index=data.index)
        sb = self._sma(md, int(self.parameters['lengthSignal']))
        sh = md - sb
        data['imacd_md'] = md
        data['imacd_sb'] = sb
        data['imacd_sh'] = sh
        return data

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        buy = crossover(data['imacd_md'], data['imacd_sb'])
        sell = crossunder(data['imacd_md'], data['imacd_sb'])
        data['buy_signal'] = buy
        data['sell_signal'] = sell
        st = (data['imacd_sh'].abs() / (data['imacd_sh'].rolling(50, min_periods=10).std(ddof=0).replace(0,np.nan))).clip(0,1).fillna(0)
        st = st.where(buy|sell, 0.0)
        thr=float(self.parameters['signal_threshold'])
        st[(buy|sell) & (st < thr)] = thr
        data['signal_strength'] = st
        return data

