#!/usr/bin/env python3
"""
Strategy 051: Projection Bands Bandwidth

Pine source is incomplete; we implement a bandwidth consistent with Projection Bands:
  bandwidth = (upb - lpb) / reference
Where reference = close (percent width relative to price). Signals are derived
from band breakouts following a contraction (low bandwidth).
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
    from utils.vectorized_helpers import crossover, crossunder, calculate_signal_strength
except Exception:
    def crossover(a,b): return (a>b) & (a.shift(1)<=b)
    def crossunder(a,b): return (a<b) & (a.shift(1)>=b)
    def calculate_signal_strength(fs,weights=None):
        df=pd.concat(fs,axis=1); return df.mean(axis=1).clip(0,1)


class Strategy051ProjectionBandwidth(BaseStrategy):
    def __init__(self, parameters: Dict = None):
        params = {
            'length': 14,
            'squeeze_mult': 0.7,    # bandwidth < ma*mult indicates contraction
            'ma_period': 20,
            'signal_threshold': 0.6,
        }
        if parameters:
            params.update(parameters)
        super().__init__('Strategy_051_ProjectionBandwidth', params)

    @staticmethod
    def _rolling_slope(y: pd.Series, n: int, idx: pd.Series) -> pd.Series:
        sum_x = idx.rolling(n, min_periods=1).sum()
        sum_x2 = (idx*idx).rolling(n, min_periods=1).sum()
        sum_y = y.rolling(n, min_periods=1).sum()
        sum_xy = (idx*y).rolling(n, min_periods=1).sum()
        denom = (n*sum_x2 - (sum_x**2)).replace(0,np.nan)
        return ((n*sum_xy - sum_x*sum_y) / denom).fillna(0)

    def _projection_bands(self, df: pd.DataFrame, n: int):
        idx = pd.Series(np.arange(len(df)) + 1, index=df.index, dtype=float)
        rlh = self._rolling_slope(df['high'], n, idx)
        rll = self._rolling_slope(df['low'], n, idx)
        ups = [df['high']]
        dws = [df['low']]
        for k in range(1,n):
            ups.append(df['high'].shift(k) + k*rlh)
            dws.append(df['low'].shift(k) + k*rll)
        return pd.concat(ups,axis=1).max(axis=1), pd.concat(dws,axis=1).min(axis=1)

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        for c in ('open','high','low','close'):
            if c not in data.columns and 'price' in data.columns:
                data[c]=data['price']
        n = int(self.parameters['length'])
        upb, lpb = self._projection_bands(data, n)
        width = (upb - lpb)
        ref = data['close'].replace(0,np.nan)
        bw = (width / ref).fillna(0)
        data['pb_up'] = upb
        data['pb_dn'] = lpb
        data['pb_bw'] = bw
        data['pb_bw_ma'] = bw.rolling(int(self.parameters['ma_period']), min_periods=1).mean()
        return data

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        # Contraction when bandwidth below moving average times multiplier
        squeeze = data['pb_bw'] < (data['pb_bw_ma'] * float(self.parameters['squeeze_mult']))
        # Breakout triggers
        buy = crossover(data['close'], data['pb_up']) & squeeze.shift(1).fillna(False)
        sell = crossunder(data['close'], data['pb_dn']) & squeeze.shift(1).fillna(False)
        data['buy_signal'] = buy
        data['sell_signal'] = sell
        # Strength rises with bandwidth expansion on breakout
        exp_rate = (data['pb_bw'] - data['pb_bw_ma']).clip(0)
        norm = (exp_rate / (data['pb_bw_ma'].replace(0,np.nan))).fillna(0).clip(0,1)
        st = pd.Series(0.0, index=data.index)
        st[buy | sell] = norm[buy | sell]
        strength = calculate_signal_strength([st],[1.0])
        thr = float(self.parameters['signal_threshold'])
        strength[(buy|sell) & (strength < thr)] = thr
        strength[~(buy|sell)] = 0.0
        data['signal_strength'] = strength.clip(0,1)
        return data

