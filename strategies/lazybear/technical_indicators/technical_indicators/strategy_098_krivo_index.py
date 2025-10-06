#!/usr/bin/env python3
"""
Strategy 098: Krivo Index (Single‑Instrument Proxy)

Original KI sums scores across multiple EUR pairs relative to MA.
Proxy: sum of signs of price vs SMA across a set of lookback lengths for the
single instrument. Optionally normalize to [-1,1].

Signals: KI crosses zero; optional signal smoothing.
"""

from typing import Dict, List
import pandas as pd
import numpy as np
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


class Strategy098KrivoIndexProxy(BaseStrategy):
    def __init__(self, parameters: Dict = None):
        params = {
            'ma_lengths': [10, 20, 50, 100, 200],
            'normalize': True,
            'signal_len': 5,
            'min_strength': 0.2,
        }
        if parameters:
            params.update(parameters)
        super().__init__('Strategy_098_Krivo_Index_Proxy', params)

    @staticmethod
    def _sma(s: pd.Series, n: int) -> pd.Series:
        return s.rolling(max(1,int(n)), min_periods=1).mean()

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        df=data.copy()
        if 'close' not in df and 'price' in df:
            df['close']=df['price']
        lengths: List[int] = self.parameters.get('ma_lengths', [10,20,50,100,200])
        scores = []
        for n in lengths:
            ma = self._sma(df['close'].astype(float), int(n))
            score = np.where(df['close'] >= ma, 1.0, -1.0)
            scores.append(pd.Series(score, index=df.index))
        ki = pd.concat(scores, axis=1).sum(axis=1)
        if self.parameters.get('normalize', True) and len(lengths)>0:
            ki = ki / float(len(lengths))  # range approximately [-1,1]
        df['ki'] = pd.Series(ki, index=df.index)
        df['ki_sig'] = df['ki'].ewm(span=max(1,int(self.parameters['signal_len'])), adjust=False).mean()
        return df

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df=data.copy()
        buy = crossover(df['ki'], 0.0)
        sell = crossunder(df['ki'], 0.0)
        df['buy_signal']=buy
        df['sell_signal']=sell
        strength = (df['ki'].abs()).clip(0,1).fillna(0.0)
        min_st=float(self.parameters['min_strength'])
        strength[(buy|sell) & (strength<min_st)] = min_st
        df['signal_strength']=strength.where(buy|sell, 0.0)
        return df

