#!/usr/bin/env python3
"""
Strategy 094: Chartmill Value Indicator (CVI)

Pine reference: pine_scripts/094_chartmill_value.pine

Signals: buy on CVI crossing above 0 (and above ob1 threshold),
sell on CVI crossing below 0 (and below os1 threshold).
"""

import pandas as pd
import numpy as np
from typing import Dict
import os, sys, math

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)
from strategies.base_strategy import BaseStrategy

try:
    sys.path.insert(0, os.path.join(project_root, 'src'))
    from utils.vectorized_helpers import crossover, crossunder
except Exception:
    def crossover(a,b): return (a>b) & (a.shift(1)<=b)
    def crossunder(a,b): return (a<b) & (a.shift(1)>=b)


class Strategy094ChartmillValueIndicator(BaseStrategy):
    def __init__(self, parameters: Dict=None):
        params = {
            'length': 3,
            'use_modified': False,  # use atr*sqrt(length)
            'os1': -0.51,
            'ob1': 0.43,
            'signal_threshold': 0.6,
        }
        if parameters:
            params.update(parameters)
        super().__init__('Strategy_094_ChartmillValueIndicator', params)

    @staticmethod
    def _sma(s: pd.Series, n: int) -> pd.Series:
        return s.rolling(max(1, int(n)), min_periods=1).mean()

    @staticmethod
    def _tr(h: pd.Series, l: pd.Series, c: pd.Series) -> pd.Series:
        pc = c.shift(1)
        tr = pd.concat([(h-l), (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
        return tr.fillna(0)

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        for c in ('high','low','close'):
            if c not in data:
                data[c] = data.get('close', pd.Series(index=data.index)).fillna(method='ffill')
        length = int(self.parameters['length'])
        vc = self._sma((data['high'] + data['low'])/2.0, length)
        atr = self._sma(self._tr(data['high'], data['low'], data['close']), length)
        denom = atr * (math.sqrt(length) if bool(self.parameters.get('use_modified', False)) else 1.0)
        denom = denom.replace(0, np.nan)
        data['cvi'] = ((data['close'] - vc) / denom).replace([np.inf,-np.inf], np.nan).fillna(0)
        return data

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        ob1 = float(self.parameters['ob1'])
        os1 = float(self.parameters['os1'])
        buy = crossover(data['cvi'], 0.0) & (data['cvi'] > ob1)
        sell = crossunder(data['cvi'], 0.0) & (data['cvi'] < os1)
        data['buy_signal'] = buy
        data['sell_signal'] = sell
        st = data['cvi'].abs() / (data['cvi'].rolling(50, min_periods=10).std(ddof=0).replace(0,np.nan))
        st = st.clip(0,1).fillna(0)
        st = st.where(buy|sell, 0.0)
        thr=float(self.parameters['signal_threshold'])
        st[(buy|sell) & (st < thr)] = thr
        data['signal_strength'] = st
        return data

