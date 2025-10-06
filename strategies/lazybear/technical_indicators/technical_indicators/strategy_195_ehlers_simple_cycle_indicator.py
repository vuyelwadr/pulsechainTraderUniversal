#!/usr/bin/env python3
"""
Strategy 195: Ehlers Simple Cycle Indicator (Proxy)

Proxy rationale: Use a detrended price oscillator (DPO) style measure to
isolate cycle component in price. Signals on zero-line cross.

Formula:
  sma = SMA(close, period)
  shift = int(period/2)+1
  dpo = close.shift(shift) - sma
  sc = EMA(dpo, smooth)
"""

from typing import Dict
import pandas as pd
import numpy as np
import os, sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)
from strategies.base_strategy import BaseStrategy

try:
    sys.path.insert(0, os.path.join(project_root,'src'))
    from utils.vectorized_helpers import crossover, crossunder
except Exception:
    def crossover(a,b): return (a>b) & (a.shift(1)<=b)
    def crossunder(a,b): return (a<b) & (a.shift(1)>=b)


class Strategy195EhlersSimpleCycleProxy(BaseStrategy):
    def __init__(self, parameters: Dict=None):
        params={'period':20,'smooth':5,'signal_threshold':0.5}
        if parameters: params.update(parameters)
        super().__init__('Strategy_195_Ehlers_Simple_Cycle_Proxy', params)

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        df=data.copy()
        if 'close' not in df and 'price' in df: df['close']=df['price']
        n=int(self.parameters['period'])
        sma = df['close'].astype(float).rolling(max(1,n), min_periods=1).mean()
        shift = max(1, int(n/2)+1)
        dpo = df['close'].astype(float).shift(shift) - sma
        df['sc'] = dpo.ewm(span=max(1,int(self.parameters['smooth'])), adjust=False).mean()
        return df

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df=data.copy()
        buy=crossover(df['sc'], 0.0)
        sell=crossunder(df['sc'], 0.0)
        df['buy_signal']=buy; df['sell_signal']=sell
        strength=(df['sc'].abs()/(df['sc'].abs().rolling(50, min_periods=10).mean().replace(0,np.nan))).clip(0,1).fillna(0.0)
        thr=float(self.parameters['signal_threshold'])
        strength[(buy|sell) & (strength<thr)]=thr
        df['signal_strength']=strength.where(buy|sell, 0.0)
        return df

