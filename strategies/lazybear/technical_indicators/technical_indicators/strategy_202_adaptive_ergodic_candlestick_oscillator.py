#!/usr/bin/env python3
"""
Strategy 202: Adaptive Ergodic Candlestick Oscillator (AECO)

Pine reference: pine_scripts/202_adaptive_ergodic.pine

Signals: buy when AECO crosses above its EMA signal; sell on cross below.
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


class Strategy202AdaptiveErgodicCandlestickOscillator(BaseStrategy):
    def __init__(self, parameters: Dict=None):
        params = {
            'length': 14,
            'sl': 9,
            'ep': 5,
            'signal_threshold': 0.6,
        }
        if parameters:
            params.update(parameters)
        super().__init__('Strategy_202_AdaptiveErgodicCandlestickOscillator', params)

    @staticmethod
    def _ema(s: pd.Series, n: int) -> pd.Series:
        return s.ewm(span=max(1,int(n)), adjust=False, min_periods=1).mean()

    @staticmethod
    def _stoch(close: pd.Series, high: pd.Series, low: pd.Series, n: int) -> pd.Series:
        hh=high.rolling(max(1,int(n)), min_periods=1).max()
        ll=low.rolling(max(1,int(n)), min_periods=1).min()
        return 100.0 * (close - ll) / (hh - ll).replace(0,np.nan)

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        for c in ('open','high','low','close'):
            if c not in data:
                data[c] = data.get('close', pd.Series(index=data.index)).fillna(method='ffill')
        length=int(self.parameters['length']); sl=int(self.parameters['sl']); ep=int(self.parameters['ep'])
        vrb = (self._stoch(data['close'], data['high'], data['low'], length) - 50.0).abs() / 50.0
        mep = 2.0 / (ep + 1.0)
        # came1/came2 accumulate adaptively
        came1 = pd.Series(index=data.index, dtype=float)
        came2 = pd.Series(index=data.index, dtype=float)
        came11 = pd.Series(index=data.index, dtype=float)
        came22 = pd.Series(index=data.index, dtype=float)
        for i in range(len(data)):
            if i <= (length + ep)*2:
                came1.iloc[i] = (data['close'] - data['open']).iloc[i]
                came2.iloc[i] = (data['high'] - data['low']).iloc[i]
                came11.iloc[i] = came1.iloc[i]
                came22.iloc[i] = came2.iloc[i]
            else:
                vr = vrb.iloc[i]
                came1.iloc[i] = came1.iloc[i-1] + mep*vr * ((data['close'].iloc[i] - data['open'].iloc[i]) - came1.iloc[i-1])
                came2.iloc[i] = came2.iloc[i-1] + mep*vr * ((data['high'].iloc[i] - data['low'].iloc[i]) - came2.iloc[i-1])
                came11.iloc[i] = came11.iloc[i-1] + mep*vr * (came1.iloc[i] - came11.iloc[i-1])
                came22.iloc[i] = came22.iloc[i-1] + mep*vr * (came2.iloc[i] - came22.iloc[i-1])
        eco = (came11 / came22.replace(0,np.nan)) * 100.0
        data['aeco'] = eco.fillna(0)
        data['aeco_sig'] = self._ema(data['aeco'], sl)
        return data

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        buy = crossover(data['aeco'], data['aeco_sig'])
        sell = crossunder(data['aeco'], data['aeco_sig'])
        data['buy_signal'] = buy
        data['sell_signal'] = sell
        st = (data['aeco'] - data['aeco_sig']).abs() / (data['aeco'].rolling(50, min_periods=10).std(ddof=0).replace(0,np.nan))
        st = st.clip(0,1).fillna(0).where(buy|sell, 0.0)
        thr=float(self.parameters['signal_threshold'])
        st[(buy|sell) & (st < thr)] = thr
        data['signal_strength'] = st
        return data

