#!/usr/bin/env python3
"""
Strategy 146: Waddah Attar Explosion

Pine reference: pine_scripts/146_waddah_attar_explosion.pine

Signals: buy when trendUp rises above DeadZone and ExplosionLine; sell when
trendDown rises above DeadZone and ExplosionLine.
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
    from utils.vectorized_helpers import crossover
except Exception:
    def crossover(a,b): return (a>b) & (a.shift(1)<=b)


class Strategy146WaddahAttarExplosion(BaseStrategy):
    def __init__(self, parameters: Dict=None):
        params = {
            'sensitivity': 150.0,
            'fastLength': 20,
            'slowLength': 40,
            'channelLength': 20,
            'mult': 2.0,
            'deadZone': 20.0,
            'signal_threshold': 0.6,
        }
        if parameters:
            params.update(parameters)
        super().__init__('Strategy_146_WaddahAttarExplosion', params)

    @staticmethod
    def _ema(s: pd.Series, n: int) -> pd.Series:
        return s.ewm(span=max(1,int(n)), adjust=False, min_periods=1).mean()

    @staticmethod
    def _sma(s: pd.Series, n: int) -> pd.Series:
        return s.rolling(max(1,int(n)), min_periods=1).mean()

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        if 'close' not in data and 'price' in data:
            data['close'] = data['price']
        p = self.parameters
        fast = self._ema(data['close'], int(p['fastLength']))
        slow = self._ema(data['close'], int(p['slowLength']))
        macd = fast - slow
        macd_prev = macd.shift(1)
        t1 = (macd - macd_prev) * float(p['sensitivity'])
        # Explosion line is BB width on close
        basis = self._sma(data['close'], int(p['channelLength']))
        dev = float(p['mult']) * data['close'].rolling(int(p['channelLength']), min_periods=1).std(ddof=0)
        e1 = (basis + dev) - (basis - dev)
        data['trendUp'] = t1.clip(lower=0)
        data['trendDown'] = (-t1).clip(lower=0)
        data['explosion'] = e1
        return data

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        dz = float(self.parameters['deadZone'])
        buy = (data['trendUp'] > dz) & (data['explosion'] > dz) & crossover(data['trendUp'], dz)
        sell = (data['trendDown'] > dz) & (data['explosion'] > dz) & crossover(data['trendDown'], dz)
        data['buy_signal'] = buy
        data['sell_signal'] = sell
        st = (data[['trendUp','trendDown']].max(axis=1) / (data['explosion'].replace(0, np.nan))).clip(0,1).fillna(0)
        st = st.where(buy|sell, 0.0)
        thr=float(self.parameters['signal_threshold'])
        st[(buy|sell) & (st < thr)] = thr
        data['signal_strength'] = st
        return data

