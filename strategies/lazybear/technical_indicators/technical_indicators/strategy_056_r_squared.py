#!/usr/bin/env python3
"""
Strategy 056: R-Squared (Chande's R2)

Pine source not available; implement rolling coefficient of determination R^2
for linear regression of close vs time index over window n.

Deterministic rule:
- Consider trending when R2 crosses above threshold (e.g., 0.5) → buy
- Consider non-trending when R2 crosses below threshold → sell
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
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


class Strategy056RSquared(BaseStrategy):
    def __init__(self, parameters: Dict = None):
        params = {
            'length': 20,
            'threshold': 0.5,
            'signal_threshold': 0.6,
        }
        if parameters:
            params.update(parameters)
        super().__init__('Strategy_056_RSquared', params)

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        if 'close' not in data.columns and 'price' in data.columns:
            data['close']=data['price']
        n = int(self.parameters['length'])
        x = pd.Series(np.arange(len(data)) + 1, index=data.index, dtype=float)
        # Rolling components
        sum_x = x.rolling(n, min_periods=1).sum()
        sum_x2 = (x*x).rolling(n, min_periods=1).sum()
        sum_y = data['close'].rolling(n, min_periods=1).sum()
        sum_y2 = (data['close']*data['close']).rolling(n, min_periods=1).sum()
        sum_xy = (x*data['close']).rolling(n, min_periods=1).sum()
        nn = float(n)
        # Pearson r formula over rolling window
        num = (nn*sum_xy - sum_x*sum_y)
        den = np.sqrt((nn*sum_x2 - sum_x**2) * (nn*sum_y2 - sum_y**2))
        r = (num / den).replace([np.inf,-np.inf], np.nan).fillna(0)
        r2 = (r*r).clip(0,1)
        data['r_squared'] = r2
        return data

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        th = float(self.parameters['threshold'])
        buy = crossover(data['r_squared'], th)
        sell = crossunder(data['r_squared'], th)
        data['buy_signal'] = buy
        data['sell_signal'] = sell
        st = data['r_squared'].where(buy|sell, 0.0)
        st = st.clip(0,1)
        thr = float(self.parameters['signal_threshold'])
        st[(buy|sell) & (st < thr)] = thr
        data['signal_strength'] = st.clip(0,1)
        return data

    @classmethod
    def parameter_space(cls) -> Dict[str, Tuple[float, float]]:
        """Define parameter bounds for optimization"""
        return {
            'length': (5, 50),  # Reasonable window size
            'threshold': (0.1, 0.9),  # R-squared threshold must be reasonable (not 0 or 1)
            'signal_threshold': (0.0, 0.95),  # Signal threshold can't be 1.0
        }

