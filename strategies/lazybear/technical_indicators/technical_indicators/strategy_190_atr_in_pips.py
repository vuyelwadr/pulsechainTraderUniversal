#!/usr/bin/env python3
"""
Strategy 190: ATR in Pips (simplified)

Pine reference: pine_scripts/190_atr_in_pips.pine

Signals: buy when ATR-series crosses above its upper BB; sell when crosses below lower BB.
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


class Strategy190ATRPips(BaseStrategy):
    def __init__(self, parameters: Dict=None):
        params = {
            'length': 20,
            'mult': 0.7,
            'bb_length': 34,
            'bb_mult': 2.0,
            'signal_threshold': 0.6,
        }
        if parameters:
            params.update(parameters)
        super().__init__('Strategy_190_ATR_Pips', params)

    @staticmethod
    def _sma(s: pd.Series, n: int) -> pd.Series:
        return s.rolling(max(1,int(n)), min_periods=1).mean()

    @staticmethod
    def _atr(h: pd.Series, l: pd.Series, c: pd.Series, n: int) -> pd.Series:
        pc = c.shift(1)
        tr = pd.concat([(h-l), (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
        return tr.rolling(max(1,int(n)), min_periods=1).mean()

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        for c in ('high','low','close'):
            if c not in data:
                data[c]=data.get('close', pd.Series(index=data.index)).fillna(method='ffill')
        atr = self._atr(data['high'], data['low'], data['close'], int(self.parameters['length']))
        s = float(self.parameters['mult']) * 100.0 * atr
        data['atrpips'] = s
        basis = self._sma(s, int(self.parameters['bb_length']))
        dev = float(self.parameters['bb_mult']) * s.rolling(int(self.parameters['bb_length']), min_periods=1).std(ddof=0)
        data['atrpips_mid'] = basis
        data['atrpips_up'] = basis + dev
        data['atrpips_dn'] = basis - dev
        return data

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        buy = crossover(data['atrpips'], data['atrpips_up'])
        sell = crossunder(data['atrpips'], data['atrpips_dn'])
        data['buy_signal'] = buy
        data['sell_signal'] = sell
        st = (data['atrpips'] - data['atrpips_mid']).abs() / (data['atrpips'].rolling(50, min_periods=10).std(ddof=0).replace(0,np.nan))
        st = st.clip(0,1).fillna(0).where(buy|sell, 0.0)
        thr=float(self.parameters['signal_threshold'])
        st[(buy|sell) & (st < thr)] = thr
        data['signal_strength'] = st
        return data

