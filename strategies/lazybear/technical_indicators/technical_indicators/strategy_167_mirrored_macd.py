#!/usr/bin/env python3
"""
Strategy 167: Mirrored MACD

Pine reference: pine_scripts/167_mirrored_macd.pine

Signals: buy when BullLine (EMA(close,len)-EMA(open,len)) crosses above signal;
sell when BullLine crosses below signal.
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


class Strategy167MirroredMACD(BaseStrategy):
    def __init__(self, parameters: Dict=None):
        params = {
            'length': 20,
            'siglength': 9,
            'signal_threshold': 0.6,
        }
        if parameters:
            params.update(parameters)
        super().__init__('Strategy_167_MirroredMACD', params)

    @staticmethod
    def _ema(s: pd.Series, n: int) -> pd.Series:
        return s.ewm(span=max(1,int(n)), adjust=False, min_periods=1).mean()

    @staticmethod
    def _sma(s: pd.Series, n: int) -> pd.Series:
        return s.rolling(max(1,int(n)), min_periods=1).mean()

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        for c in ('open','close'):
            if c not in data:
                data[c] = data.get('close', pd.Series(index=data.index)).fillna(method='ffill')
        l = int(self.parameters['length'])
        mac = self._ema(data['close'], l)
        mao = self._ema(data['open'], l)
        mc = mac - mao
        mo = mao - mac
        signal = self._sma(mc, int(self.parameters['siglength']))
        data['mmac_bull'] = mc
        data['mmac_bear'] = mo
        data['mmac_signal'] = signal
        return data

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        buy = crossover(data['mmac_bull'], data['mmac_signal'])
        sell = crossunder(data['mmac_bull'], data['mmac_signal'])
        data['buy_signal'] = buy
        data['sell_signal'] = sell
        st = (data['mmac_bull'].abs() / (data['mmac_bull'].rolling(50, min_periods=10).std(ddof=0).replace(0,np.nan))).clip(0,1).fillna(0)
        st = st.where(buy|sell, 0.0)
        thr=float(self.parameters['signal_threshold'])
        st[(buy|sell) & (st < thr)] = thr
        data['signal_strength'] = st
        return data

