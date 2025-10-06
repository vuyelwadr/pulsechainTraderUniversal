#!/usr/bin/env python3
"""
Strategy 100: Rainbow Charts Oscillator

Pine reference: pine_scripts/100_rainbow_osc.pine

Signals: buy on ROsc crossing above 0, sell on crossing below 0.
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


class Strategy100RainbowChartsOscillator(BaseStrategy):
    def __init__(self, parameters: Dict=None):
        params = {
            'signal_threshold': 0.6,
        }
        if parameters:
            params.update(parameters)
        super().__init__('Strategy_100_RainbowChartsOscillator', params)

    @staticmethod
    def _sma(s: pd.Series, n: int) -> pd.Series:
        return s.rolling(max(1,int(n)), min_periods=1).mean()

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        if 'close' not in data and 'price' in data:
            data['close'] = data['price']
        c = data['close']
        sma2 = self._sma(c, 2)
        dsma2 = self._sma(sma2, 2)
        tsma2 = self._sma(dsma2, 2)
        qsma2 = self._sma(tsma2, 2)
        psma2 = self._sma(qsma2, 2)
        ssma2 = self._sma(psma2, 2)
        s2sma2 = self._sma(ssma2, 2)
        osma2 = self._sma(s2sma2, 2)
        o2sma2 = self._sma(osma2, 2)
        desma2 = self._sma(o2sma2, 2)

        rmax = pd.concat([sma2, dsma2, tsma2, qsma2, psma2, ssma2, s2sma2, osma2, o2sma2, desma2], axis=1).max(axis=1)
        rmin = pd.concat([sma2, dsma2, tsma2, qsma2, psma2, ssma2, s2sma2, osma2, o2sma2, desma2], axis=1).min(axis=1)
        denom = (data['close'].rolling(10, min_periods=1).max() - data['close'].rolling(10, min_periods=1).min()).replace(0, np.nan)
        avg10 = (sma2 + dsma2 + tsma2 + qsma2 + psma2 + ssma2 + s2sma2 + osma2 + o2sma2 + desma2) / 10.0
        data['rosc'] = 100.0 * (data['close'] - avg10) / denom
        data['rbl'] = -100.0 * (rmax - rmin) / denom
        data['rbu'] = -data['rbl']
        data[['rosc','rbl','rbu']] = data[['rosc','rbl','rbu']].replace([np.inf,-np.inf], np.nan).fillna(0)
        return data

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        buy = crossover(data['rosc'], 0.0)
        sell = crossunder(data['rosc'], 0.0)
        data['buy_signal'] = buy
        data['sell_signal'] = sell
        st = (data['rosc'].abs() / (data['rosc'].rolling(50, min_periods=10).std(ddof=0).replace(0,np.nan))).clip(0,1).fillna(0)
        st = st.where(buy|sell, 0.0)
        thr=float(self.parameters['signal_threshold'])
        st[(buy|sell) & (st < thr)] = thr
        data['signal_strength'] = st
        return data

