#!/usr/bin/env python3
"""
Strategy 140: BTC All Exchanges Volume Averaged (Single‑Instrument Proxy)

Proxy: Relative volume activity for the instrument.
  rv = volume / EMA(volume, vol_len)
Optionally smooth and generate signals on threshold crosses.
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


class Strategy140BTCAllExchangesVolumeAveragedProxy(BaseStrategy):
    def __init__(self, parameters: Dict=None):
        params={'vol_len':30,'smooth':10,'buy_thr':1.5,'sell_thr':0.8,'signal_threshold':0.4}
        if parameters: params.update(parameters)
        super().__init__('Strategy_140_BTC_All_Exchanges_Volume_Averaged_Proxy', params)

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        df=data.copy()
        if 'volume' not in df: df['volume']=0.0
        vol_ema = df['volume'].astype(float).ewm(span=max(1,int(self.parameters['vol_len'])), adjust=False).mean().replace(0,np.nan)
        rv = (df['volume'].astype(float) / vol_ema).fillna(0.0)
        df['rv'] = rv.ewm(span=max(1,int(self.parameters['smooth'])), adjust=False).mean()
        return df

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df=data.copy()
        buy = df['rv'] > float(self.parameters['buy_thr'])
        sell = df['rv'] < float(self.parameters['sell_thr'])
        df['buy_signal']=buy
        df['sell_signal']=sell
        strength=(df['rv'] - float(self.parameters['sell_thr']))/(max(1e-6,(float(self.parameters['buy_thr'])-float(self.parameters['sell_thr']))))
        strength=strength.clip(0,1).fillna(0.0)
        thr=float(self.parameters['signal_threshold'])
        strength[(buy|sell) & (strength<thr)]=thr
        df['signal_strength']=strength.where(buy|sell, 0.0)
        return df

