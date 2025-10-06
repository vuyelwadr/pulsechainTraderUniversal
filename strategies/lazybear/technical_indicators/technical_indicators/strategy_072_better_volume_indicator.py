#!/usr/bin/env python3
"""
Strategy 072: Better Volume Indicator (simplified)

Pine source in repo is truncated. Implement a simplified Better Volume classifier
using volume and true range conditions, then derive minimal signals:
- Classify bars as climax_up, climax_down, churn, low_vol vs rolling baselines
- Buy when (climax_up or churn) and close crosses above SMA(close, length)
- Sell when (climax_down) and close crosses below SMA(close, length)

This preserves intent (volume-based momentum/churn) without speculative smoothing.
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


class Strategy072BetterVolume(BaseStrategy):
    def __init__(self, parameters: Dict = None):
        params = {
            'length': 8,
            'vol_mult': 1.5,
            'range_mult': 1.2,
            'signal_threshold': 0.6,
        }
        if parameters:
            params.update(parameters)
        super().__init__('Strategy_072_BetterVolume', params)

    def _sma(self, s: pd.Series, n: int) -> pd.Series:
        return s.rolling(max(1,int(n)), min_periods=1).mean()

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        for c in ('open','high','low','close','volume'):
            if c not in data.columns:
                if c=='volume':
                    data[c]=0
                elif 'price' in data.columns:
                    data[c]=data['price']
        n = int(self.parameters['length'])
        rng = (data['high'] - data['low']).abs()
        vol_ma = self._sma(data['volume'], n)
        rng_ma = self._sma(rng, n)
        vol_hi = data['volume'] > (vol_ma * float(self.parameters['vol_mult']))
        rng_hi = rng > (rng_ma * float(self.parameters['range_mult']))
        rng_lo = rng < (rng_ma * 0.7)
        up_bar = data['close'] >= data['open']
        down_bar = ~up_bar
        data['climax_up'] = vol_hi & rng_hi & up_bar
        data['climax_down'] = vol_hi & rng_hi & down_bar
        data['churn'] = vol_hi & rng_lo
        data['low_vol'] = data['volume'] < (vol_ma * 0.7)
        data['bvi_sma'] = self._sma(data['close'], n)
        return data

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        buy_trig = (data['climax_up'] | data['churn'])
        sell_trig = data['climax_down']
        buy = crossover(data['close'], data['bvi_sma']) & buy_trig
        sell = crossunder(data['close'], data['bvi_sma']) & sell_trig
        data['buy_signal'] = buy
        data['sell_signal'] = sell
        # Strength based on volume surge and range condition
        vol_ma = data['volume'].rolling(int(self.parameters['length']), min_periods=1).mean().replace(0,np.nan)
        vol_ratio = (data['volume'] / vol_ma).fillna(0).clip(0,3) / 3.0
        st = vol_ratio.where(buy|sell, 0.0)
        thr = float(self.parameters['signal_threshold'])
        st[(buy|sell) & (st < thr)] = thr
        data['signal_strength'] = st.clip(0,1)
        return data

