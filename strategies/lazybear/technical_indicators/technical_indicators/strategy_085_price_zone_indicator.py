#!/usr/bin/env python3
"""
Strategy 085: Price Zone Indicator (PZI)

Approximation: Define price zones around an EMA centerline using ATR-based
envelopes. Signals when price enters upper/lower zones from neutral.
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


class Strategy085PriceZoneIndicator(BaseStrategy):
    def __init__(self, parameters: Dict=None):
        params = {
            'ema_len': 20,
            'atr_len': 14,
            'mult': 1.5,
            'signal_threshold': 0.6,
        }
        if parameters:
            params.update(parameters)
        super().__init__('Strategy_085_PriceZoneIndicator', params)

    @staticmethod
    def _ema(s: pd.Series, n: int) -> pd.Series:
        return s.ewm(span=max(1,int(n)), adjust=False, min_periods=1).mean()

    @staticmethod
    def _atr(h: pd.Series, l: pd.Series, c: pd.Series, n: int) -> pd.Series:
        pc = c.shift(1)
        tr = pd.concat([(h-l), (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
        return tr.rolling(max(1,int(n)), min_periods=1).mean()

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        for c in ('high','low','close'):
            if c not in data:
                data[c]=data.get('close', pd.Series(index=data.index)).fillna(method='ffill')
        ema = self._ema(data['close'], int(self.parameters['ema_len']))
        atr = self._atr(data['high'], data['low'], data['close'], int(self.parameters['atr_len']))
        mult = float(self.parameters['mult'])
        data['pzi_mid'] = ema
        data['pzi_up'] = ema + mult * atr
        data['pzi_dn'] = ema - mult * atr
        return data

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        # Zones: upper if close > up; lower if close < dn; neutral otherwise
        upper = data['close'] > data['pzi_up']
        lower = data['close'] < data['pzi_dn']
        buy = upper & (~upper.shift(1).fillna(False))
        sell = lower & (~lower.shift(1).fillna(False))
        data['buy_signal'] = buy
        data['sell_signal'] = sell
        bw = (data['pzi_up'] - data['pzi_dn']).replace(0,np.nan)
        st = (data['close'] - data['pzi_mid']).abs() / bw
        st = st.clip(0,1).fillna(0).where(buy|sell, 0.0)
        thr=float(self.parameters['signal_threshold'])
        st[(buy|sell) & (st<thr)] = thr
        data['signal_strength'] = st
        return data

