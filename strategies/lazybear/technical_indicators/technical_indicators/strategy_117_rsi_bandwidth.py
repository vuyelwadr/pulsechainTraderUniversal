#!/usr/bin/env python3
"""
Strategy 117: RSI Bandwidth

Proxy: Rolling bandwidth of RSI over a lookback window.
  rsi_bw = (max(RSI, N) - min(RSI, N)) / 100
Signals: Crosses of rsi_bw over its EMA signal.
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
    from utils.vectorized_helpers import crossover, crossunder, highest, lowest
except Exception:
    def crossover(a,b): return (a>b)&(a.shift(1)<=b)
    def crossunder(a,b): return (a<b)&(a.shift(1)>=b)
    def highest(s,n): return s.rolling(n, min_periods=1).max()
    def lowest(s,n): return s.rolling(n, min_periods=1).min()


class Strategy117RSIBandwidth(BaseStrategy):
    def __init__(self, parameters: Dict=None):
        params={'length':14,'band_lookback':20,'signal_len':10,'signal_threshold':0.4}
        if parameters: params.update(parameters)
        super().__init__('Strategy_117_RSI_Bandwidth', params)

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
        rsi=self._rsi(df['close'].astype(float), int(self.parameters['length']))
        lb=int(self.parameters['band_lookback'])
        bw=(highest(rsi, lb) - lowest(rsi, lb)) / 100.0
        df['rsi_bw']=bw.clip(0,1).fillna(0.0)
        df['rsi_bw_sig']=df['rsi_bw'].ewm(span=max(1,int(self.parameters['signal_len'])), adjust=False).mean()
        return df

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df=data.copy()
        buy=crossover(df['rsi_bw'], df['rsi_bw_sig'])
        sell=crossunder(df['rsi_bw'], df['rsi_bw_sig'])
        df['buy_signal']=buy; df['sell_signal']=sell
        strength=(df['rsi_bw']-df['rsi_bw_sig']).abs().clip(0,1).fillna(0.0)
        thr=float(self.parameters['signal_threshold'])
        strength[(buy|sell) & (strength<thr)]=thr
        df['signal_strength']=strength.where(buy|sell, 0.0)
        return df

