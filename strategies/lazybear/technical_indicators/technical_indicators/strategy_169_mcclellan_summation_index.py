#!/usr/bin/env python3
"""
Strategy 169: McClellan Summation Index (Single‑Instrument Proxy)

Classical: cumulative sum of McClellan Oscillator.
Proxy: build from Strategy168 proxy MO and cumulate.

Signals: Summation index crosses its EMA; optional slope filter.
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


class Strategy169McClellanSummationIndexProxy(BaseStrategy):
    def __init__(self, parameters: Dict = None):
        params = {
            'scale': 100.0,
            'fast': 19,
            'slow': 39,
            'signal_len': 10,     # EMA on summation index
            'min_strength': 0.2,
        }
        if parameters:
            params.update(parameters)
        super().__init__('Strategy_169_McClellan_Summation_Index_Proxy', params)

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        if 'close' not in df and 'price' in df:
            df['close'] = df['price']
        ret = df['close'].astype(float).pct_change().fillna(0.0) * float(self.parameters['scale'])
        fast = int(self.parameters['fast'])
        slow = int(self.parameters['slow'])
        mo = ret.ewm(span=max(1, fast), adjust=False).mean() - ret.ewm(span=max(1, slow), adjust=False).mean()
        df['mo'] = mo
        df['msi'] = mo.cumsum()
        sig_len = int(self.parameters['signal_len'])
        df['msi_ema'] = df['msi'].ewm(span=max(1, sig_len), adjust=False).mean()
        return df

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        buy = crossover(df['msi'], df['msi_ema'])
        sell = crossunder(df['msi'], df['msi_ema'])
        df['buy_signal'] = buy
        df['sell_signal'] = sell
        # Strength from distance to EMA normalized by rolling step size
        dist = (df['msi'] - df['msi_ema']).abs()
        denom = df['mo'].abs().rolling(50, min_periods=1).mean().replace(0, np.nan)
        strength = (dist / denom).clip(0, 1).fillna(0.0)
        min_st = float(self.parameters['min_strength'])
        strength[(buy | sell) & (strength < min_st)] = min_st
        df['signal_strength'] = strength.where(buy | sell, 0.0)
        return df

