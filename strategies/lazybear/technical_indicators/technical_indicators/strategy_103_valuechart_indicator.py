#!/usr/bin/env python3
"""
Strategy 103: ValueChart Indicator (simplified)

Implements a ValueChart-style oscillator by normalizing price deviation from a
rolling mean using ATR as a scale. Signals occur when the oscillator exits and
re-enters fair value (zero) zone.
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


class Strategy103ValueChartIndicator(BaseStrategy):
    def __init__(self, parameters: Dict = None):
        params = {
            'length': 5,
            'signal_threshold': 0.6,
        }
        if parameters:
            params.update(parameters)
        super().__init__('Strategy_103_ValueChartIndicator', params)

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
                data[c] = data.get('close', pd.Series(index=data.index)).fillna(method='ffill')
        n = int(self.parameters['length'])
        vc_mid = self._sma((data['high']+data['low'])/2.0, n)
        scale = self._atr(data['high'], data['low'], data['close'], n)
        vc = (data['close'] - vc_mid) / scale.replace(0,np.nan) * 10.0
        data['vc'] = vc.replace([np.inf,-np.inf], np.nan).fillna(0)
        data['vc_upper'] = 8.0
        data['vc_lower'] = -8.0
        return data

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        # Deterministic: buy when vc crosses up through 0 from below fair; sell when crosses down
        buy = crossover(data['vc'], 0.0)
        sell = crossunder(data['vc'], 0.0)
        data['buy_signal'] = buy
        data['sell_signal'] = sell
        st = (data['vc'].abs() / (data['vc'].rolling(50, min_periods=10).std(ddof=0).replace(0,np.nan))).clip(0,1).fillna(0)
        st = st.where(buy|sell, 0.0)
        thr = float(self.parameters['signal_threshold'])
        st[(buy|sell) & (st < thr)] = thr
        data['signal_strength'] = st
        return data

