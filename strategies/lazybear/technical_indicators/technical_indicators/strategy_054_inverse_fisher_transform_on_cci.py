#!/usr/bin/env python3
"""
Strategy 054: Inverse Fisher Transform on CCI

Pine source not available; implement canonical CCI and inverse Fisher transform:
  CCI = (TP - SMA(TP,n)) / (0.015 * MeanDeviation(TP,n))
  v = clip(CCI/200, -0.999, 0.999) to map approx [-100,100] → [-0.5,0.5]
  inv_fisher = (exp(2*v) - 1) / (exp(2*v) + 1)

Deterministic rule:
- Buy when inv_fisher crosses above 0
- Sell when inv_fisher crosses below 0
"""

import pandas as pd
import numpy as np
from typing import Dict
import sys, os

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)
from strategies.base_strategy import BaseStrategy

try:
    sys.path.insert(0, os.path.join(project_root, 'src'))
    from utils.vectorized_helpers import crossover, crossunder
except Exception:
    def crossover(a,b): return (a>b) & (a.shift(1)<=b)
    def crossunder(a,b): return (a<b) & (a.shift(1)>=b)


class Strategy054InverseFisherCCI(BaseStrategy):
    def __init__(self, parameters: Dict = None):
        params = {
            'length': 20,
            'signal_threshold': 0.6,
        }
        if parameters:
            params.update(parameters)
        super().__init__('Strategy_054_InverseFisher_CCI', params)

    @staticmethod
    def _cci(high: pd.Series, low: pd.Series, close: pd.Series, n: int) -> pd.Series:
        tp = (high + low + close) / 3.0
        sma_tp = tp.rolling(n, min_periods=1).mean()
        md = (tp - sma_tp).abs().rolling(n, min_periods=1).mean()
        denom = (0.015 * md).replace(0, np.nan)
        cci = (tp - sma_tp) / denom
        return cci.fillna(0)

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        for c in ('high','low','close'):
            if c not in data.columns and 'price' in data.columns:
                data[c]=data['price']
        n = int(self.parameters['length'])
        cci = self._cci(data['high'], data['low'], data['close'], n)
        v = (cci / 200.0).clip(-0.999, 0.999)
        inv = (np.exp(2*v) - 1) / (np.exp(2*v) + 1)
        data['cci'] = cci
        data['inv_fisher_cci'] = pd.Series(inv, index=data.index)
        return data

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        buy = crossover(data['inv_fisher_cci'], 0.0)
        sell = crossunder(data['inv_fisher_cci'], 0.0)
        data['buy_signal'] = buy
        data['sell_signal'] = sell
        st = data['inv_fisher_cci'].abs().clip(0,1)
        st = st.where(buy|sell, 0.0)
        thr = float(self.parameters['signal_threshold'])
        st[(buy|sell) & (st < thr)] = thr
        data['signal_strength'] = st.clip(0,1)
        return data

