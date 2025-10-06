#!/usr/bin/env python3
"""
Strategy 096: KaseCD (approximation)

Volatility-adjusted MACD-like oscillator using ATR-normalized EMA difference.
Signals on oscillator crossing signal line.
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


class Strategy096KaseCD(BaseStrategy):
    def __init__(self, parameters: Dict=None):
        params = {
            'fast': 12,
            'slow': 26,
            'signal_len': 9,
            'atr_len': 14,
            'signal_threshold': 0.6,
        }
        if parameters:
            params.update(parameters)
        super().__init__('Strategy_096_KaseCD', params)

    @staticmethod
    def _ema(s: pd.Series, n: int) -> pd.Series:
        return s.ewm(span=max(1,int(n)), adjust=False, min_periods=1).mean()

    @staticmethod
    def _atr(h: pd.Series, l: pd.Series, c: pd.Series, n: int) -> pd.Series:
        pc=c.shift(1)
        tr=pd.concat([(h-l),(h-pc).abs(),(l-pc).abs()],axis=1).max(axis=1)
        return tr.rolling(max(1,int(n)),min_periods=1).mean()

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        for c in ('high','low','close'):
            if c not in data:
                data[c]=data.get('close', pd.Series(index=data.index)).fillna(method='ffill')
        fast=self._ema(data['close'], int(self.parameters['fast']))
        slow=self._ema(data['close'], int(self.parameters['slow']))
        atr=self._atr(data['high'], data['low'], data['close'], int(self.parameters['atr_len'])).replace(0,np.nan)
        kcd=(fast - slow) / atr
        data['kcd']=kcd.replace([np.inf,-np.inf], np.nan).fillna(0)
        data['kcd_sig']=self._ema(data['kcd'], int(self.parameters['signal_len']))
        return data

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        buy=crossover(data['kcd'], data['kcd_sig'])
        sell=crossunder(data['kcd'], data['kcd_sig'])
        data['buy_signal']=buy
        data['sell_signal']=sell
        st=(data['kcd']-data['kcd_sig']).abs()
        st=(st/(data['kcd'].rolling(50,min_periods=10).std(ddof=0).replace(0,np.nan))).clip(0,1).fillna(0)
        st=st.where(buy|sell,0.0)
        thr=float(self.parameters['signal_threshold'])
        st[(buy|sell)&(st<thr)]=thr
        data['signal_strength']=st
        return data

