#!/usr/bin/env python3
"""
Strategy 066: Correlation Coefficient + MA (adapted)

Pine reference: pine_scripts/066_correlation_coefficient_ma.pine
- Original computes rolling correlation between current symbol and another symbol
  via security(), then plots correlation and its SMA with guide lines at ±0.2.

Adaptation for single-instrument pipeline (no external series):
- Use rolling correlation of the instrument's returns with its own lagged returns (lag=1 by default),
  which is the standard lag-1 autocorrelation over the window.
- Plot-equivalent series: corr and its moving average.

Deterministic rule (consistent with ±0.2 guides):
- Buy when corr crosses above +upper (default +0.2)
- Sell when corr crosses below -lower (default -0.2)
Signal strength scales with distance beyond the respective threshold.
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
    from utils.vectorized_helpers import crossover, crossunder, calculate_signal_strength
except Exception:
    def crossover(a,b): return (a>b) & (a.shift(1)<= (b.shift(1) if isinstance(b, pd.Series) else b))
    def crossunder(a,b): return (a<b) & (a.shift(1)>= (b.shift(1) if isinstance(b, pd.Series) else b))
    def calculate_signal_strength(fs,weights=None):
        df = pd.concat(fs,axis=1); return df.mean(axis=1).clip(0,1)


class Strategy066CorrelationCoefficientMA(BaseStrategy):
    def __init__(self, parameters: Dict = None):
        params = {
            'length_correlation': 20,
            'length_ma': 10,
            'lag': 1,
            'upper': 0.2,
            'lower': -0.2,
            'signal_threshold': 0.6,
        }
        if parameters:
            params.update(parameters)
        super().__init__('Strategy_066_CorrelationCoefficient_MA', params)

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        if 'close' not in data.columns and 'price' in data.columns:
            data['close'] = data['price']
        # Returns and lagged returns
        r = data['close'].pct_change()
        lag = int(self.parameters['lag'])
        r_lag = r.shift(lag)
        # Rolling correlation over window
        n = int(self.parameters['length_correlation'])
        corr = r.rolling(n, min_periods=3).corr(r_lag)
        ma = corr.rolling(int(self.parameters['length_ma']), min_periods=1).mean()
        data['cc_corr'] = corr.fillna(0)
        data['cc_ma'] = ma.fillna(0)
        return data

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        up = float(self.parameters['upper'])
        lo = float(self.parameters['lower'])
        c = data['cc_corr']
        buy = crossover(c, up)
        sell = crossunder(c, lo)
        data['buy_signal'] = buy
        data['sell_signal'] = sell
        # Strength proportional to how far correlation is beyond thresholds
        st = pd.Series(0.0, index=data.index)
        st[buy] = ((c - up) / (1.0 - up)).clip(0,1)[buy]
        st[sell] = ((-lo - (-c)) / (1.0 - (-lo))).clip(0,1)[sell]  # distance below -0.2
        thr = float(self.parameters['signal_threshold'])
        st[(buy|sell) & (st < thr)] = thr
        st[~(buy|sell)] = 0.0
        data['signal_strength'] = st.clip(0,1)
        return data
