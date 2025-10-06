#!/usr/bin/env python3
"""
Strategy 125: Color Coded SMIIO (TSI-based)

Pine reference: pine_scripts/125_colored_smiio.pine

Signals: buy when OSC (erg - sig) crosses above 0; sell when it crosses below 0.
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


class Strategy125ColoredSMIIO(BaseStrategy):
    def __init__(self, parameters: Dict=None):
        params = {
            'shortlen': 10,
            'longlen': 4,
            'siglen': 10,
            'signal_threshold': 0.6,
        }
        if parameters:
            params.update(parameters)
        super().__init__('Strategy_125_Colored_SMIIO', params)

    @staticmethod
    def _ema(s: pd.Series, n: int) -> pd.Series:
        return s.ewm(span=max(1,int(n)), adjust=False, min_periods=1).mean()

    @staticmethod
    def _tsi(src: pd.Series, short: int, long: int) -> pd.Series:
        mtm = src.diff()
        ema1 = mtm.ewm(span=max(1,int(long)), adjust=False, min_periods=1).mean()
        ema2 = ema1.ewm(span=max(1,int(short)), adjust=False, min_periods=1).mean()
        abs_mtm = mtm.abs()
        ema1d = abs_mtm.ewm(span=max(1,int(long)), adjust=False, min_periods=1).mean()
        ema2d = ema1d.ewm(span=max(1,int(short)), adjust=False, min_periods=1).mean()
        return (ema2 / ema2d.replace(0, np.nan)).fillna(0)

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        if 'close' not in data and 'price' in data:
            data['close'] = data['price']
        p = self.parameters
        erg = self._tsi(data['close'], int(p['shortlen']), int(p['longlen']))
        sig = self._ema(erg, int(p['siglen']))
        data['erg'] = erg
        data['erg_sig'] = sig
        data['erg_osc'] = erg - sig
        return data

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        buy = crossover(data['erg_osc'], 0.0)
        sell = crossunder(data['erg_osc'], 0.0)
        data['buy_signal'] = buy
        data['sell_signal'] = sell
        st = (data['erg_osc'].abs() / (data['erg_osc'].rolling(50, min_periods=10).std(ddof=0).replace(0,np.nan))).clip(0,1).fillna(0)
        st = st.where(buy|sell, 0.0)
        thr=float(self.parameters['signal_threshold'])
        st[(buy|sell) & (st < thr)] = thr
        data['signal_strength'] = st
        return data

