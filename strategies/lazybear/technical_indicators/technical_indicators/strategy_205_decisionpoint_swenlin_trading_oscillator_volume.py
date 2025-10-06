#!/usr/bin/env python3
"""
Strategy 205: DecisionPoint Swenlin Trading Oscillator [Volume] (Proxy)

Proxy rationale: No external volume breadth feed; use signed volume flow:
  vol_flow = sign(close change) * volume
Compute STO as EMA(fast) - EMA(slow) on vol_flow.

Signals: zero-line crosses; strength from |STO| normalized.
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


class Strategy205DecisionPointSTOVolumeProxy(BaseStrategy):
    def __init__(self, parameters: Dict = None):
        params = {
            'fast': 10,
            'slow': 21,
            'min_strength': 0.2,
        }
        if parameters:
            params.update(parameters)
        super().__init__('Strategy_205_DecisionPoint_STO_Volume_Proxy', params)

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        if 'close' not in df and 'price' in df:
            df['close'] = df['price']
        if 'volume' not in df:
            df['volume'] = 0.0
        sign = np.sign(df['close'].astype(float).diff().fillna(0.0))
        vol_flow = sign * df['volume'].astype(float)
        fast = int(self.parameters['fast'])
        slow = int(self.parameters['slow'])
        df['sto_vol'] = vol_flow.ewm(span=max(1, fast), adjust=False).mean() - vol_flow.ewm(span=max(1, slow), adjust=False).mean()
        return df

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        buy = crossover(df['sto_vol'], 0.0)
        sell = crossunder(df['sto_vol'], 0.0)
        df['buy_signal'] = buy
        df['sell_signal'] = sell
        denom = df['sto_vol'].abs().rolling(50, min_periods=1).mean().replace(0, np.nan)
        strength = (df['sto_vol'].abs() / denom).clip(0, 1).fillna(0.0)
        min_st = float(self.parameters['min_strength'])
        strength[(buy | sell) & (strength < min_st)] = min_st
        df['signal_strength'] = strength.where(buy | sell, 0.0)
        return df

