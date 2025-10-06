#!/usr/bin/env python3
"""
Strategy 144: Vervoort Volatility Bands

Pine reference: pine_scripts/144_vervoort_volatility.pine

Signals: buy when price crosses above UpperBand; sell when crosses below LowerBand.
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


class Strategy144VervoortVolatilityBands(BaseStrategy):
    def __init__(self, parameters: Dict=None):
        params = {
            'al': 8,
            'vl': 13,
            'df': 3.55,
            'lba': 0.9,
            'signal_threshold': 0.6,
        }
        if parameters:
            params.update(parameters)
        super().__init__('Strategy_144_VervoortVolatilityBands', params)

    @staticmethod
    def _sma(s: pd.Series, n: int) -> pd.Series:
        return s.rolling(max(1,int(n)), min_periods=1).mean()

    @staticmethod
    def _ema(s: pd.Series, n: int) -> pd.Series:
        return s.ewm(span=max(1,int(n)), adjust=False, min_periods=1).mean()

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        for c in ('high','low','close'):
            if c not in data:
                data[c] = data.get('close', pd.Series(index=data.index)).fillna(method='ffill')
        src = (data['high'] + data['low'] + data['close'])/3.0
        al = int(self.parameters['al'])
        vl = int(self.parameters['vl'])
        df = float(self.parameters['df'])
        lba = float(self.parameters['lba'])
        typical = np.where(src >= src.shift(1), src - data['low'].shift(1), src.shift(1) - data['low'])
        typical = pd.Series(typical, index=data.index)
        deviation = df * self._sma(typical, vl)
        devHigh = self._ema(deviation, al)
        devLow = lba * devHigh
        medianAvg = self._ema(src, al)
        mid = self._sma(medianAvg, al)
        up = self._ema(medianAvg, al) + devHigh
        dn = self._ema(medianAvg, al) - devLow
        data['vvb_mid'] = mid
        data['vvb_up'] = up
        data['vvb_dn'] = dn
        return data

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        buy = crossover(data['close'], data['vvb_up'])
        sell = crossunder(data['close'], data['vvb_dn'])
        data['buy_signal'] = buy
        data['sell_signal'] = sell
        bw = (data['vvb_up'] - data['vvb_dn']).replace(0, np.nan)
        st = (data['close'] - data['vvb_mid']).abs() / bw
        st = st.clip(0,1).fillna(0).where(buy|sell, 0.0)
        thr=float(self.parameters['signal_threshold'])
        st[(buy|sell) & (st < thr)] = thr
        data['signal_strength'] = st
        return data

