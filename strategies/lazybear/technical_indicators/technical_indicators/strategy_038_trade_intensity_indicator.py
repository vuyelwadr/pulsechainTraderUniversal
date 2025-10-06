#!/usr/bin/env python3
"""
Strategy 038: Trade Intensity Indicator (Proxy)

Proxy rationale: original Pine missing. Use signed return weighted by
relative volume to estimate intensity of trading pressure.

Formula:
  ret = pct_change(close)
  rv  = volume / EMA(volume, vol_len)
  ti  = EMA(ret * rv, len)
  sig = EMA(ti, signal_len)

Signals: ti crosses above/below zero (or its signal).
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


class Strategy038TradeIntensityIndicator(BaseStrategy):
    def __init__(self, parameters: Dict = None):
        params = {
            'len': 20,
            'signal_len': 9,
            'vol_len': 20,
            'signal_threshold': 0.4,
        }
        if parameters:
            params.update(parameters)
        super().__init__('Strategy_038_Trade_Intensity_Indicator', params)

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        df=data.copy()
        if 'close' not in df and 'price' in df:
            df['close']=df['price']
        if 'volume' not in df:
            df['volume']=0.0
        vol_len=int(self.parameters['vol_len'])
        length=int(self.parameters['len'])
        signal_len=int(self.parameters['signal_len'])
        ret = df['close'].astype(float).pct_change().fillna(0.0)
        vol_ema = df['volume'].astype(float).ewm(span=max(1, vol_len), adjust=False).mean().replace(0, np.nan)
        rv = (df['volume'].astype(float) / vol_ema).fillna(0.0)
        ti_raw = (ret * rv)
        df['ti'] = ti_raw.ewm(span=max(1, length), adjust=False).mean()
        df['ti_sig'] = df['ti'].ewm(span=max(1, signal_len), adjust=False).mean()
        return df

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df=data.copy()
        buy = crossover(df['ti'], 0.0)
        sell = crossunder(df['ti'], 0.0)
        df['buy_signal']=buy
        df['sell_signal']=sell
        strength=(df['ti'].abs() / (df['ti'].abs().rolling(50, min_periods=1).mean().replace(0,np.nan))).clip(0,1).fillna(0.0)
        thr=float(self.parameters['signal_threshold'])
        strength[(buy|sell) & (strength<thr)] = thr
        df['signal_strength']=strength.where(buy|sell, 0.0)
        return df

