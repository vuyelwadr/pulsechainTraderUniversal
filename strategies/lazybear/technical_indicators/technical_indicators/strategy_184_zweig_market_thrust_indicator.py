#!/usr/bin/env python3
"""
Strategy 184: Zweig Market Thrust Indicator (Single‑Instrument Proxy)

Classical breadth thrust uses advancing/(adv+dec) over 10 days with specific thresholds.
Proxy: Use fraction of up-closes in a rolling window as 'advancing ratio'.

Signals: EMA of ratio crosses above ~0.615 (buy) and below ~0.50 (sell).
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


class Strategy184ZweigMarketThrustProxy(BaseStrategy):
    def __init__(self, parameters: Dict = None):
        params = {
            'lookback': 10,          # rolling window for up-day ratio
            'ema_len': 10,           # EMA of the ratio
            'buy_threshold': 0.615,  # classic breadth thrust threshold
            'sell_threshold': 0.50,  # fallback threshold to exit
            'min_strength': 0.2,
        }
        if parameters:
            params.update(parameters)
        super().__init__('Strategy_184_Zweig_Market_Thrust_Proxy', params)

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        if 'close' not in df and 'price' in df:
            df['close'] = df['price']
        up = (df['close'].astype(float).diff() > 0).astype(float)
        lb = int(self.parameters['lookback'])
        # Fraction of up bars in the window
        ratio = up.rolling(lb, min_periods=1).mean()
        df['thrust_ratio'] = ratio
        df['thrust_ema'] = ratio.ewm(span=max(1, int(self.parameters['ema_len'])), adjust=False).mean()
        return df

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        bt = float(self.parameters['buy_threshold'])
        st = float(self.parameters['sell_threshold'])
        buy = crossover(df['thrust_ema'], bt)
        sell = crossunder(df['thrust_ema'], st)
        df['buy_signal'] = buy
        df['sell_signal'] = sell
        # Strength normalized by band width between st and bt
        width = max(1e-6, bt - st)
        base = np.where(buy, bt, np.where(sell, st, 0.0))
        dist = (df['thrust_ema'] - base).abs()
        strength = (dist / width).clip(0, 1).fillna(0.0)
        min_st = float(self.parameters['min_strength'])
        strength[(buy | sell) & (strength < min_st)] = min_st
        df['signal_strength'] = strength.where(buy | sell, 0.0)
        return df

