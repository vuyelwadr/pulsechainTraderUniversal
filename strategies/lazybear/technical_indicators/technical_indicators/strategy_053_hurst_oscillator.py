#!/usr/bin/env python3
"""
Strategy 053: Hurst Oscillator

Pine source is unavailable; consistent with Hurst Bands (Strategy 052),
we define a normalized oscillator measuring deviation from CMA relative to
OuterBand amplitude:
  hurst_osc = (close - CMA) / (outer_pct * CMA)
This approximates multiples of the outer band distance.

Deterministic trading rule:
- Buy on crossing above +1.0 (outside upper outer band)
- Sell on crossing below -1.0 (outside lower outer band)
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
    def crossover(a,b): return (a>b) & (a.shift(1)<=b)
    def crossunder(a,b): return (a<b) & (a.shift(1)>=b)
    def calculate_signal_strength(fs,weights=None):
        df=pd.concat(fs,axis=1); return df.mean(axis=1).clip(0,1)


class Strategy053HurstOscillator(BaseStrategy):
    def __init__(self, parameters: Dict = None):
        params = {
            'length': 10,
            'outer_pct': 2.6,
            'signal_threshold': 0.6,
        }
        if parameters:
            params.update(parameters)
        super().__init__('Strategy_053_HurstOscillator', params)

    def _cma(self, price: pd.Series, length: int) -> pd.Series:
        displacement = int((length/2.0) + 1)
        dprice = price.shift(displacement)
        cma = dprice.rolling(abs(length), min_periods=1).mean()
        arr = cma.to_numpy().copy()
        for i in range(len(arr)):
            if np.isnan(dprice.iloc[i]):
                if i >= 2 and not np.isnan(arr[i-1]) and not np.isnan(arr[i-2]):
                    arr[i] = 2*arr[i-1] - arr[i-2]
                elif i >= 1 and not np.isnan(arr[i-1]):
                    arr[i] = arr[i-1]
        return pd.Series(arr, index=price.index)

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        for c in ('high','low','close'):
            if c not in data.columns and 'price' in data.columns:
                data[c]=data['price']
        price = (data['high'] + data['low'])/2.0
        length = int(self.parameters['length'])
        cma = self._cma(price, length)
        outer = float(self.parameters['outer_pct'])/100.0
        denom = (outer * cma).replace(0,np.nan)
        osc = ((data['close'] - cma) / denom).replace([np.inf,-np.inf], np.nan).fillna(0)
        data['hurst_cma'] = cma
        data['hurst_osc'] = osc
        return data

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        buy = crossover(data['hurst_osc'], 1.0)
        sell = crossunder(data['hurst_osc'], -1.0)
        data['buy_signal'] = buy
        data['sell_signal'] = sell
        st = data['hurst_osc'].abs().clip(0,1)
        st = st.where(buy|sell, 0.0)
        thr = float(self.parameters['signal_threshold'])
        st[(buy|sell) & (st < thr)] = thr
        data['signal_strength'] = st.clip(0,1)
        return data

