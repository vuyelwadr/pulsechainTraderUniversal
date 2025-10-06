#!/usr/bin/env python3
"""
Strategy 124: Stochastic RSI with Volatility Bands

Pine reference: pine_scripts/124_stochrsi_volatility_bands.pine

Signals: buy when %K crosses above lower band; sell when %K crosses below upper band.
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


class Strategy124StochRSIVolatilityBands(BaseStrategy):
    def __init__(self, parameters: Dict=None):
        params = {
            'lengthRSI': 14,
            'lengthStoch': 14,
            'smoothK': 3,
            'smoothD': 3,
            'bb_length': 20,
            'bb_mult': 2.0,
            'signal_threshold': 0.6,
        }
        if parameters:
            params.update(parameters)
        super().__init__('Strategy_124_StochRSI_VolatilityBands', params)

    @staticmethod
    def _sma(s: pd.Series, n: int) -> pd.Series:
        return s.rolling(max(1,int(n)), min_periods=1).mean()

    @staticmethod
    def _rsi(s: pd.Series, n: int) -> pd.Series:
        n = max(1,int(n))
        delta = s.diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        roll_up = up.ewm(alpha=1.0/n, adjust=False, min_periods=1).mean()
        roll_down = down.ewm(alpha=1.0/n, adjust=False, min_periods=1).mean()
        rs = roll_up / roll_down.replace(0, np.nan)
        return (100 - (100 / (1 + rs))).fillna(50)

    @staticmethod
    def _stoch(src: pd.Series, n: int) -> pd.Series:
        hh = src.rolling(max(1,int(n)), min_periods=1).max()
        ll = src.rolling(max(1,int(n)), min_periods=1).min()
        return 100.0 * (src - ll) / (hh - ll).replace(0, np.nan)

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        if 'close' not in data and 'price' in data:
            data['close'] = data['price']
        p = self.parameters
        rsi1 = self._rsi(data['close'], p['lengthRSI'])
        k = self._sma(self._stoch(rsi1, p['lengthStoch']), p['smoothK'])
        d = self._sma(k, p['smoothD'])
        data['stoch_k'] = k
        data['stoch_d'] = d
        # BB on %K
        length = int(p['bb_length']); mult = float(p['bb_mult'])
        basis = self._sma(k, length)
        dev = mult * k.rolling(length, min_periods=1).std(ddof=0)
        data['bb_upper_k'] = basis + dev
        data['bb_lower_k'] = basis - dev
        return data

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        buy = crossover(data['stoch_k'], data['bb_lower_k'])
        sell = crossunder(data['stoch_k'], data['bb_upper_k'])
        data['buy_signal'] = buy
        data['sell_signal'] = sell
        # Strength by band excursion
        bw = (data['bb_upper_k'] - data['bb_lower_k']).replace(0, np.nan)
        st = (data['stoch_k'] - (data['bb_upper_k'] + data['bb_lower_k'])/2.0).abs() / bw
        st = st.clip(0,1).fillna(0).where(buy|sell, 0.0)
        thr=float(self.parameters['signal_threshold'])
        st[(buy|sell) & (st < thr)] = thr
        data['signal_strength'] = st
        return data

