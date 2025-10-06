#!/usr/bin/env python3
"""
Strategy 116: RSI %B

Proxy definition using RSI and static OS/OB levels:
  rsi_b = (RSI - osLevel) / (obLevel - osLevel)
Signals: rsi_b crosses 0.5 with optional thresholding.
"""

from typing import Dict
import pandas as pd
import numpy as np
import os, sys

project_root=os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)
from strategies.base_strategy import BaseStrategy

try:
    sys.path.insert(0, os.path.join(project_root,'src'))
    from utils.vectorized_helpers import crossover, crossunder
except Exception:
    def crossover(a,b): return (a>b)&(a.shift(1)<=b)
    def crossunder(a,b): return (a<b)&(a.shift(1)>=b)


class Strategy116RSIPercentB(BaseStrategy):
    def __init__(self, parameters: Dict=None):
        params={'length':14,'obLevel':70.0,'osLevel':30.0,'signal_threshold':0.4}
        if parameters: params.update(parameters)
        super().__init__('Strategy_116_RSI_PercentB', params)

    @staticmethod
    def _rsi(close: pd.Series, length: int) -> pd.Series:
        delta = close.diff()
        up = delta.clip(lower=0)
        dn = (-delta).clip(lower=0)
        rma_up = up.ewm(alpha=1/length, adjust=False, min_periods=length).mean()
        rma_dn = dn.ewm(alpha=1/length, adjust=False, min_periods=length).mean()
        rs = rma_up / rma_dn.replace(0,np.nan)
        rsi = 100 - (100/(1+rs))
        return rsi.fillna(50.0)

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        df=data.copy()
        if 'close' not in df and 'price' in df: df['close']=df['price']
        L=int(self.parameters['length'])
        rsi=self._rsi(df['close'].astype(float), L)
        ob=float(self.parameters['obLevel']); os_=float(self.parameters['osLevel'])
        width=max(1e-6, ob-os_)
        df['rsi_b']=((rsi - os_) / width).clip(0,1)
        return df

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df=data.copy()
        mid=0.5
        buy=crossover(df['rsi_b'], mid)
        sell=crossunder(df['rsi_b'], mid)
        df['buy_signal']=buy; df['sell_signal']=sell
        # Strength based on distance from mid, normalized by 0.5
        strength=(df['rsi_b']-mid).abs()/0.5
        strength=strength.clip(0,1).fillna(0.0)
        thr=float(self.parameters['signal_threshold'])
        strength[(buy|sell) & (strength<thr)] = thr
        df['signal_strength']=strength.where(buy|sell, 0.0)
        return df

