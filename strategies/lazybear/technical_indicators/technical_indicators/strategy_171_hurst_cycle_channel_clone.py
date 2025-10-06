#!/usr/bin/env python3
"""
Strategy 171: Hurst Cycle Channel Clone

Pine reference: pine_scripts/171_hurst_cycle_clone.pine

Signals: buy when close crosses above short-cycle top; sell when close crosses below short-cycle bottom.
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


class Strategy171HurstCycleChannelClone(BaseStrategy):
    def __init__(self, parameters: Dict=None):
        params = {
            'scl_t': 10,
            'mcl_t': 30,
            'scm': 1.0,
            'mcm': 3.0,
            'signal_threshold': 0.6,
        }
        if parameters:
            params.update(parameters)
        super().__init__('Strategy_171_HurstCycleChannelClone', params)

    @staticmethod
    def _rma(s: pd.Series, n: int) -> pd.Series:
        n = max(1,int(n))
        alpha = 1.0 / n
        return s.ewm(alpha=alpha, adjust=False, min_periods=1).mean()

    @staticmethod
    def _atr(h: pd.Series, l: pd.Series, c: pd.Series, n: int) -> pd.Series:
        pc = c.shift(1)
        tr = pd.concat([(h-l), (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
        return tr.rolling(max(1,int(n)), min_periods=1).mean()

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        for c in ('high','low','close'):
            if c not in data:
                data[c]=data.get('close', pd.Series(index=data.index)).fillna(method='ffill')
        p = self.parameters
        scl = int(p['scl_t'])//2
        mcl = int(p['mcl_t'])//2
        ma_scl = self._rma(data['close'], scl)
        ma_mcl = self._rma(data['close'], mcl)
        scm_off = float(p['scm']) * self._atr(data['high'], data['low'], data['close'], scl)
        mcm_off = float(p['mcm']) * self._atr(data['high'], data['low'], data['close'], mcl)
        sct = ma_scl.shift(max(1,scl//2)).fillna(data['close']) + scm_off
        scb = ma_scl.shift(max(1,scl//2)).fillna(data['close']) - scm_off
        mct = ma_mcl.shift(max(1,mcl//2)).fillna(data['close']) + mcm_off
        mcb = ma_mcl.shift(max(1,mcl//2)).fillna(data['close']) - mcm_off
        data['sct'] = sct
        data['scb'] = scb
        data['mct'] = mct
        data['mcb'] = mcb
        return data

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        buy = crossover(data['close'], data['sct'])
        sell = crossunder(data['close'], data['scb'])
        data['buy_signal'] = buy
        data['sell_signal'] = sell
        bw = (data['sct'] - data['scb']).replace(0, np.nan)
        st = (data['close'] - (data['sct']+data['scb'])/2.0).abs() / bw
        st = st.clip(0,1).fillna(0).where(buy|sell, 0.0)
        thr=float(self.parameters['signal_threshold'])
        st[(buy|sell) & (st < thr)] = thr
        data['signal_strength'] = st
        return data

