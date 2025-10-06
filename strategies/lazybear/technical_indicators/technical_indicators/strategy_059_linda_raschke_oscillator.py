#!/usr/bin/env python3
"""
Strategy 059: Linda Raschke Oscillator (3/10 style)

Pine source not present; implement a well-known Raschke-style oscillator:
  osc = EMA(close,3) - EMA(close,10)
  signal = EMA(osc,16)

Deterministic rule:
- Buy when osc crosses above signal
- Sell when osc crosses below signal
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


class Strategy059LindaRaschkeOscillator(BaseStrategy):
    def __init__(self, parameters: Dict = None):
        params = {
            'fast': 3,
            'slow': 10,
            'signal': 16,
            'signal_threshold': 0.6,
        }
        if parameters:
            params.update(parameters)
        super().__init__('Strategy_059_LindaRaschke_Osc', params)

    @staticmethod
    def _ema(s: pd.Series, n: int) -> pd.Series:
        return s.ewm(span=max(1,int(n)), adjust=False, min_periods=1).mean()

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        if 'close' not in data.columns and 'price' in data.columns:
            data['close']=data['price']
        f = int(self.parameters['fast'])
        sl = int(self.parameters['slow'])
        sg = int(self.parameters['signal'])
        ema_f = self._ema(data['close'], f)
        ema_s = self._ema(data['close'], sl)
        osc = ema_f - ema_s
        sig = self._ema(osc, sg)
        data['lr_osc'] = osc
        data['lr_sig'] = sig
        return data

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        buy = crossover(data['lr_osc'], data['lr_sig'])
        sell = crossunder(data['lr_osc'], data['lr_sig'])
        data['buy_signal'] = buy
        data['sell_signal'] = sell
        st = (data['lr_osc'] - data['lr_sig']).abs()
        # normalize by price scale to keep 0..1
        if 'close' in data.columns:
            st = (st / data['close'].replace(0,np.nan)).fillna(0)
        st = st.clip(0,1)
        st = st.where(buy|sell, 0.0)
        thr = float(self.parameters['signal_threshold'])
        st[(buy|sell) & (st < thr)] = thr
        data['signal_strength'] = st.clip(0,1)
        return data

