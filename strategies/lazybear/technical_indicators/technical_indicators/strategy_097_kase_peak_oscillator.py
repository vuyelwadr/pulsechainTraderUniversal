#!/usr/bin/env python3
"""
Strategy 097: Kase Peak Oscillator (approximation)

Oscillator measuring momentum peaks by normalizing price change with ATR and
applying smoothing. Signals on zero crossings.
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


class Strategy097KasePeakOscillator(BaseStrategy):
    def __init__(self, parameters: Dict=None):
        params = {
            'atr_len': 14,
            'smooth_len': 5,
            'signal_threshold': 0.6,
        }
        if parameters:
            params.update(parameters)
        super().__init__('Strategy_097_Kase_Peak_Oscillator', params)

    @staticmethod
    def _atr(h: pd.Series, l: pd.Series, c: pd.Series, n: int) -> pd.Series:
        pc=c.shift(1)
        tr=pd.concat([(h-l),(h-pc).abs(),(l-pc).abs()],axis=1).max(axis=1)
        return tr.rolling(max(1,int(n)),min_periods=1).mean()

    @staticmethod
    def _ema(s: pd.Series, n: int) -> pd.Series:
        return s.ewm(span=max(1,int(n)), adjust=False, min_periods=1).mean()

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        for c in ('high','low','close'):
            if c not in data:
                data[c]=data.get('close', pd.Series(index=data.index)).fillna(method='ffill')
        atr=self._atr(data['high'], data['low'], data['close'], int(self.parameters['atr_len'])).replace(0,np.nan)
        mom=(data['close'] - data['close'].shift(1)) / atr
        kpo=self._ema(mom, int(self.parameters['smooth_len']))
        data['kpo']=kpo.fillna(0)
        return data

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        buy=crossover(data['kpo'], 0.0)
        sell=crossunder(data['kpo'], 0.0)
        data['buy_signal']=buy
        data['sell_signal']=sell
        st=(data['kpo'].abs()/ (data['kpo'].rolling(50,min_periods=10).std(ddof=0).replace(0,np.nan))).clip(0,1).fillna(0)
        st=st.where(buy|sell,0.0)
        thr=float(self.parameters['signal_threshold'])
        st[(buy|sell)&(st<thr)]=thr
        data['signal_strength']=st
        return data

