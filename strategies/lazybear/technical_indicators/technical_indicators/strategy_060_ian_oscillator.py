#!/usr/bin/env python3
"""
Strategy 060: Ian Oscillator

Derived from directional movement components:
  DM+ = max(high - prev_high, 0) when greater than DM-
  DM- = max(prev_low - low, 0) when greater than DM+
  TR = max(high-low, |high-prev_close|, |low-prev_close|)
  IanOsc = EMA(DM+, n) - EMA(DM-, n) normalized by EMA(TR, n)
  Then smoothed by short EMA

Signals:
- Buy when smoothed oscillator crosses above +threshold
- Sell when it crosses below -threshold
"""

import pandas as pd
import numpy as np
from typing import Dict
import sys, os

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)
from strategies.base_strategy import BaseStrategy

try:
    sys.path.insert(0, os.path.join(project_root, 'src'))
    from utils.vectorized_helpers import crossover, crossunder
except Exception:
    def crossover(a,b): return (a>b) & (a.shift(1)<=b)
    def crossunder(a,b): return (a<b) & (a.shift(1)>=b)


class Strategy060IanOscillator(BaseStrategy):
    def __init__(self, parameters: Dict = None):
        params = {
            'period': 14,
            'smooth_period': 3,
            'threshold': 0.5,
            'signal_threshold': 0.6,
        }
        if parameters:
            params.update(parameters)
        super().__init__('Strategy_060_IanOscillator', params)

    @staticmethod
    def _ema(s: pd.Series, n: int) -> pd.Series:
        return s.ewm(span=max(1,int(n)), adjust=False, min_periods=1).mean()

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        for c in ('open','high','low','close'):
            if c not in data.columns and 'price' in data.columns:
                data[c]=data['price']
        p = int(self.parameters['period'])
        up_move = data['high'] - data['high'].shift(1)
        dn_move = data['low'].shift(1) - data['low']
        dm_pos = np.where((up_move > dn_move) & (up_move > 0), up_move, 0.0)
        dm_neg = np.where((dn_move > up_move) & (dn_move > 0), dn_move, 0.0)
        tr = pd.concat([
            data['high'] - data['low'],
            (data['high'] - data['close'].shift(1)).abs(),
            (data['low'] - data['close'].shift(1)).abs()
        ], axis=1).max(axis=1)

        dm_pos_s = self._ema(pd.Series(dm_pos, index=data.index), p)
        dm_neg_s = self._ema(pd.Series(dm_neg, index=data.index), p)
        tr_s = self._ema(tr, p).replace(0, np.nan)
        osc = ((dm_pos_s - dm_neg_s) / tr_s).replace([np.inf,-np.inf], np.nan).fillna(0)
        data['ian_osc'] = self._ema(osc, int(self.parameters['smooth_period']))
        return data

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        th = float(self.parameters['threshold'])
        buy = crossover(data['ian_osc'], th)
        sell = crossunder(data['ian_osc'], -th)
        data['buy_signal'] = buy
        data['sell_signal'] = sell
        st = data['ian_osc'].abs().clip(0,1)
        st = st.where(buy|sell, 0.0)
        min_thr = float(self.parameters['signal_threshold'])
        st[(buy|sell) & (st < min_thr)] = min_thr
        data['signal_strength'] = st.clip(0,1)
        return data

