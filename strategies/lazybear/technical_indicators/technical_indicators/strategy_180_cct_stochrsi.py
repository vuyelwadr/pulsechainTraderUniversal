#!/usr/bin/env python3
"""
Strategy 180: CCT StochRSI

Pine reference: pine_scripts/180_cct_stochrsi.pine

Signals: buy when %K crosses above 50; sell when crosses below 50 (custom type).
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


class Strategy180CCTStochRSI(BaseStrategy):
    def __init__(self, parameters: Dict=None):
        params = {
            'lr': 8,   # RSI Length
            'le': 3,   # EMA Length
            'ls': 9,   # Signal Length
            'signal_threshold': 0.6,
        }
        if parameters:
            params.update(parameters)
        super().__init__('Strategy_180_CCT_StochRSI', params)

    @staticmethod
    def _ema(s: pd.Series, n: int) -> pd.Series:
        return s.ewm(span=max(1,int(n)), adjust=False, min_periods=1).mean()

    @staticmethod
    def _rsi(s: pd.Series, n: int) -> pd.Series:
        n = max(1,int(n))
        d = s.diff()
        up = d.clip(lower=0)
        dn = -d.clip(upper=0)
        roll_up = up.ewm(alpha=1.0/n, adjust=False, min_periods=1).mean()
        roll_dn = dn.ewm(alpha=1.0/n, adjust=False, min_periods=1).mean()
        rs = roll_up / roll_dn.replace(0, np.nan)
        return (100 - 100/(1+rs)).fillna(50)

    @staticmethod
    def _stoch(src: pd.Series, n: int) -> pd.Series:
        hh = src.rolling(max(1,int(n)), min_periods=1).max()
        ll = src.rolling(max(1,int(n)), min_periods=1).min()
        return 100.0 * (src - ll) / (hh - ll).replace(0, np.nan)

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        if 'close' not in data and 'price' in data:
            data['close'] = data['price']
        p = self.parameters
        r = self._rsi(data['close'], int(p['lr']))
        k = self._ema(self._stoch(r, int(p['lr'])), int(p['le']))
        d = self._ema(k, int(p['ls']))
        data['cct_k'] = k
        data['cct_d'] = d
        return data

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        buy = crossover(data['cct_k'], 50.0)
        sell = crossunder(data['cct_k'], 50.0)
        data['buy_signal'] = buy
        data['sell_signal'] = sell
        st = (data['cct_k'] - 50.0).abs() / 50.0
        st = st.clip(0,1).fillna(0).where(buy|sell, 0.0)
        thr=float(self.parameters['signal_threshold'])
        st[(buy|sell) & (st < thr)] = thr
        data['signal_strength'] = st
        return data

