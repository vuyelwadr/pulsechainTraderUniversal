#!/usr/bin/env python3
"""
Strategy 052: Hurst Bands

Pine reference: pine_scripts/052_hurst_bands.pine
- Centered moving average CMA built from displaced price (hl2 shifted by displacement)
- If displaced price is NA (early bars), CMA extrapolates: CMA[t] = 2*CMA[t-1]-CMA[t-2]
- Bands are percentages of CMA: Inner/Outer/Extreme

Deterministic trading rule:
- Buy when close crosses above UpperOuterBand after being below on prior bar
- Sell when close crosses below LowerOuterBand after being above on prior bar
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
    def crossover(a,b): return (a>b) & (a.shift(1)<=b.shift(1))
    def crossunder(a,b): return (a<b) & (a.shift(1)>=b.shift(1))
    def calculate_signal_strength(fs,weights=None):
        df=pd.concat(fs,axis=1); return df.mean(axis=1).clip(0,1)


class Strategy052HurstBands(BaseStrategy):
    def __init__(self, parameters: Dict = None):
        params = {
            'length': 10,           # displacement length in Pine inputs
            'inner_pct': 1.6,
            'outer_pct': 2.6,
            'extreme_pct': 4.2,
            'use_extreme': False,
            'signal_threshold': 0.6,
        }
        if parameters:
            params.update(parameters)
        super().__init__('Strategy_052_HurstBands', params)

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        for c in ('high','low','close'):
            if c not in data.columns and 'price' in data.columns:
                data[c] = data['price']
        price = (data['high'] + data['low']) / 2.0  # hl2
        length = int(self.parameters['length'])
        displacement = int((length / 2.0) + 1)

        dprice = price.shift(displacement)
        # CMA: SMA of displaced price over abs(length)
        cma = dprice.rolling(abs(length), min_periods=1).mean()
        # Extrapolate where dprice is NA using CMA[t] = 2*CMA[t-1] - CMA[t-2]
        # Build iteratively to avoid look-ahead
        cma_vals = cma.to_numpy().copy()
        for i in range(len(cma_vals)):
            if np.isnan(dprice.iloc[i]):
                if i >= 2 and not np.isnan(cma_vals[i-1]) and not np.isnan(cma_vals[i-2]):
                    cma_vals[i] = 2*cma_vals[i-1] - cma_vals[i-2]
                elif i >= 1 and not np.isnan(cma_vals[i-1]):
                    cma_vals[i] = cma_vals[i-1]
        cma = pd.Series(cma_vals, index=data.index)

        inner = cma * float(self.parameters['inner_pct']) / 100.0
        outer = cma * float(self.parameters['outer_pct']) / 100.0
        extreme = cma * float(self.parameters['extreme_pct']) / 100.0

        data['hb_center'] = cma
        data['hb_up_outer'] = cma + outer
        data['hb_dn_outer'] = cma - outer
        data['hb_up_inner'] = cma + inner
        data['hb_dn_inner'] = cma - inner
        data['hb_up_ext'] = cma + extreme
        data['hb_dn_ext'] = cma - extreme
        return data

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        # Use outer bands for signals
        buy = crossover(data['close'], data['hb_up_outer']) & (data['close'].shift(1) <= data['hb_up_outer'].shift(1))
        sell = crossunder(data['close'], data['hb_dn_outer']) & (data['close'].shift(1) >= data['hb_dn_outer'].shift(1))
        data['buy_signal'] = buy
        data['sell_signal'] = sell

        width = (data['hb_up_outer'] - data['hb_dn_outer']).replace(0,np.nan)
        dist = pd.Series(0.0, index=data.index)
        dist[buy] = ((data['close'] - data['hb_up_outer']) / width).clip(0,1)[buy]
        dist[sell] = ((data['hb_dn_outer'] - data['close']) / width).clip(0,1)[sell]
        strength = calculate_signal_strength([dist.fillna(0)],[1.0])
        thr = float(self.parameters['signal_threshold'])
        strength[(buy|sell) & (strength < thr)] = thr
        strength[~(buy|sell)] = 0.0
        data['signal_strength'] = strength.clip(0,1)
        return data

