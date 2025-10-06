#!/usr/bin/env python3
"""
Strategy 118: Vervoort Smoothed Oscillator (Proxy)

Proxy rationale: Use double-smoothed momentum to emulate Vervoort-style smoothing.

Formula:
  mom = close - close[n]
  vso  = EMA( EMA(mom, s1), s2 )
  sig  = EMA(vso, sig_len)

Signals: vso crosses above/below zero (or signal).
"""

from typing import Dict
import pandas as pd
import numpy as np
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


class Strategy118VervoortSmoothedOscillator(BaseStrategy):
    def __init__(self, parameters: Dict=None):
        params={'length':10,'smooth1':9,'smooth2':9,'sig_len':5,'signal_threshold':0.5}
        if parameters: params.update(parameters)
        super().__init__('Strategy_118_Vervoort_Smoothed_Osc', params)

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        df=data.copy();
        if 'close' not in df and 'price' in df: df['close']=df['price']
        n=int(self.parameters['length']); s1=int(self.parameters['smooth1']); s2=int(self.parameters['smooth2'])
        mom = df['close'].astype(float) - df['close'].astype(float).shift(n)
        v1 = mom.ewm(span=max(1,s1), adjust=False).mean()
        df['vso'] = v1.ewm(span=max(1,s2), adjust=False).mean()
        df['vso_sig'] = df['vso'].ewm(span=max(1,int(self.parameters['sig_len'])), adjust=False).mean()
        return df

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df=data.copy()
        buy=crossover(df['vso'], 0.0)
        sell=crossunder(df['vso'], 0.0)
        df['buy_signal']=buy; df['sell_signal']=sell
        strength=(df['vso'].abs() / (df['vso'].abs().rolling(50,min_periods=1).mean().replace(0,np.nan))).clip(0,1).fillna(0.0)
        thr=float(self.parameters['signal_threshold'])
        strength[(buy|sell) & (strength<thr)] = thr
        df['signal_strength']=strength.where(buy|sell, 0.0)
        return df

