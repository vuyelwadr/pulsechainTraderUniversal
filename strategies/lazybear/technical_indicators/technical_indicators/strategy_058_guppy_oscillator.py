#!/usr/bin/env python3
"""
Strategy 058: Guppy Oscillator

Pine source not present; derive an oscillator consistent with GMMA:
  osc = mean(EMAs short) - mean(EMAs long)
  norm_osc = osc / close

Deterministic rule:
- Buy when norm_osc crosses above 0
- Sell when norm_osc crosses below 0
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


class Strategy058GuppyOscillator(BaseStrategy):
    def __init__(self, parameters: Dict = None):
        params = {
            'short_periods': [3,5,8,10,12,15],
            'long_periods': [30,35,40,45,50,60],
            'signal_threshold': 0.6,
        }
        if parameters:
            params.update(parameters)
        super().__init__('Strategy_058_GuppyOscillator', params)

    @staticmethod
    def _ema(s: pd.Series, n: int) -> pd.Series:
        return s.ewm(span=max(1,int(n)), adjust=False, min_periods=1).mean()

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        if 'close' not in data.columns and 'price' in data.columns:
            data['close']=data['price']
        sps = self.parameters['short_periods']
        lps = self.parameters['long_periods']
        s_emas = [self._ema(data['close'], int(n)) for n in sps]
        l_emas = [self._ema(data['close'], int(n)) for n in lps]
        short_mean = pd.concat(s_emas, axis=1).mean(axis=1)
        long_mean  = pd.concat(l_emas, axis=1).mean(axis=1)
        osc = (short_mean - long_mean) / data['close'].replace(0,np.nan)
        data['guppy_osc'] = osc.fillna(0)
        return data

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        buy = crossover(data['guppy_osc'], 0.0)
        sell = crossunder(data['guppy_osc'], 0.0)
        data['buy_signal'] = buy
        data['sell_signal'] = sell
        st = data['guppy_osc'].abs().clip(0,1)
        st = st.where(buy|sell, 0.0)
        thr = float(self.parameters['signal_threshold'])
        st[(buy|sell) & (st < thr)] = thr
        data['signal_strength'] = st.clip(0,1)
        return data

