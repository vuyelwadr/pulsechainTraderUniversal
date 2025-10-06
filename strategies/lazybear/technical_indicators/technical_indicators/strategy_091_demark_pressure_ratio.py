#!/usr/bin/env python3
"""
Strategy 091: DeMark Pressure Ratio (approximation)

Defines buy and sell pressure from price changes normalized by ATR, computes a
ratio and signals on crossing midline (0).
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


class Strategy091DemarkPressureRatio(BaseStrategy):
    def __init__(self, parameters: Dict=None):
        params = {
            'length': 14,
            'signal_threshold': 0.6,
        }
        if parameters:
            params.update(parameters)
        super().__init__('Strategy_091_Demark_Pressure_Ratio', params)

    @staticmethod
    def _atr(h: pd.Series, l: pd.Series, c: pd.Series, n: int) -> pd.Series:
        pc=c.shift(1)
        tr=pd.concat([(h-l),(h-pc).abs(),(l-pc).abs()],axis=1).max(axis=1)
        return tr.rolling(max(1,int(n)),min_periods=1).mean()

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        for c in ('high','low','close'):
            if c not in data:
                data[c]=data.get('close', pd.Series(index=data.index)).fillna(method='ffill')
        n=int(self.parameters['length'])
        atr=self._atr(data['high'], data['low'], data['close'], n).replace(0,np.nan)
        delta = data['close'].diff().fillna(0)
        pos = delta.clip(lower=0) / atr
        neg = (-delta.clip(upper=0)) / atr
        pr = (pos.rolling(n, min_periods=1).sum() - neg.rolling(n, min_periods=1).sum()) / (
             (pos.rolling(n, min_periods=1).sum() + neg.rolling(n, min_periods=1).sum()).replace(0,np.nan))
        data['dpr'] = pr.fillna(0)
        return data

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        buy = crossover(data['dpr'], 0.0)
        sell = crossunder(data['dpr'], 0.0)
        data['buy_signal'] = buy
        data['sell_signal'] = sell
        st = data['dpr'].abs().clip(0,1).fillna(0)
        st = st.where(buy|sell, 0.0)
        thr=float(self.parameters['signal_threshold'])
        st[(buy|sell) & (st<thr)] = thr
        data['signal_strength'] = st
        return data

