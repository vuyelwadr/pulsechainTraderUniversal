#!/usr/bin/env python3
"""
Strategy 047: Volatility Quality Index

Pine reference: pine_scripts/047_volatility_quality_index.pine
Implements cumulative VQI and two SMAs (fast/slow).

Signals: buy on VQI_sum crossing above fast SMA; sell on crossing below fast SMA.
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


class Strategy047VolatilityQualityIndex(BaseStrategy):
    def __init__(self, parameters: Dict=None):
        params = {
            'length_fast': 9,
            'length_slow': 200,
            'signal_threshold': 0.6,
        }
        if parameters:
            params.update(parameters)
        super().__init__('Strategy_047_VolatilityQualityIndex', params)

    @staticmethod
    def _sma(s: pd.Series, n: int) -> pd.Series:
        return s.rolling(max(1,int(n)), min_periods=1).mean()

    @staticmethod
    def _tr(h: pd.Series, l: pd.Series, c: pd.Series) -> pd.Series:
        pc = c.shift(1)
        tr = pd.concat([(h-l), (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
        return tr.fillna(0)

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        for c in ('open','high','low','close'):
            if c not in data:
                data[c] = data.get('close', pd.Series(index=data.index)).fillna(method='ffill')
        tr = self._tr(data['high'], data['low'], data['close'])
        hl_range = (data['high'] - data['low']).replace(0, np.nan)
        comp = (((data['close'] - data['close'].shift(1)) / tr.replace(0,np.nan)) +
                ((data['close'] - data['open']) / hl_range)) * 0.5
        comp = comp.replace([np.inf, -np.inf], np.nan)
        vqi_t = comp.fillna(method='ffill').fillna(0)
        vqi = vqi_t.abs() * ((data['close'] - data['close'].shift(1) + (data['close'] - data['open'])) * 0.5)
        data['vqi_sum'] = vqi.cumsum().fillna(0)
        data['vqi_fast'] = self._sma(data['vqi_sum'], int(self.parameters['length_fast']))
        data['vqi_slow'] = self._sma(data['vqi_sum'], int(self.parameters['length_slow']))
        return data

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        buy = crossover(data['vqi_sum'], data['vqi_fast'])
        sell = crossunder(data['vqi_sum'], data['vqi_fast'])
        data['buy_signal'] = buy
        data['sell_signal'] = sell
        dev = (data['vqi_sum'] - data['vqi_fast']).abs()
        norm = dev / (data['vqi_sum'].rolling(50, min_periods=10).std(ddof=0).replace(0,np.nan))
        st = norm.clip(0,1).fillna(0)
        st = st.where(buy|sell, 0.0)
        thr=float(self.parameters['signal_threshold'])
        st[(buy|sell) & (st < thr)] = thr
        data['signal_strength'] = st
        return data

