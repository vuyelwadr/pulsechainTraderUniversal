#!/usr/bin/env python3
"""
Strategy 185: High-Low Index (Single‑Instrument Proxy)

Classical: % of stocks making new 52-week highs vs lows (smoothed 10-day MA).
Proxy: For a single instrument, use position within rolling high-low range:
  HLI ≈ 100 * EMA( (close - LL) / (HH - LL), smooth )
This behaves like a Stochastic %K smoothed, serving as a conservative proxy.

Signals: HLI crosses midline (50) with band filters.
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
    from utils.vectorized_helpers import crossover, crossunder, highest, lowest
except Exception:
    def crossover(a,b): return (a>b) & (a.shift(1)<=b)
    def crossunder(a,b): return (a<b) & (a.shift(1)>=b)
    def highest(s,n): return s.rolling(n, min_periods=1).max()
    def lowest(s,n): return s.rolling(n, min_periods=1).min()


class Strategy185HighLowIndexProxy(BaseStrategy):
    def __init__(self, parameters: Dict = None):
        params = {
            'length': 252,     # proxy for 52-week if daily; works generically on intraday
            'smooth': 10,      # smoothing EMA
            'buy_level': 50.0,
            'sell_level': 50.0,
            'min_strength': 0.2,
        }
        if parameters:
            params.update(parameters)
        super().__init__('Strategy_185_High_Low_Index_Proxy', params)

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        if 'close' not in df and 'price' in df:
            df['close'] = df['price']
        n = int(self.parameters['length'])
        hh = highest(df['close'].astype(float), n)
        ll = lowest(df['close'].astype(float), n)
        rng = (hh - ll).replace(0, np.nan)
        pct = ((df['close'] - ll) / rng).clip(0, 1).fillna(0.5)
        df['hli_raw'] = pct * 100.0
        df['hli'] = df['hli_raw'].ewm(span=max(1, int(self.parameters['smooth'])), adjust=False).mean()
        return df

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        buy_lvl = float(self.parameters['buy_level'])
        sell_lvl = float(self.parameters['sell_level'])
        buy = crossover(df['hli'], buy_lvl)
        sell = crossunder(df['hli'], sell_lvl)
        df['buy_signal'] = buy
        df['sell_signal'] = sell
        # Strength: distance from 50 normalized by 50
        dist = (df['hli'] - 50.0).abs() / 50.0
        strength = dist.clip(0, 1).fillna(0.0)
        min_st = float(self.parameters['min_strength'])
        strength[(buy | sell) & (strength < min_st)] = min_st
        df['signal_strength'] = strength.where(buy | sell, 0.0)
        return df

