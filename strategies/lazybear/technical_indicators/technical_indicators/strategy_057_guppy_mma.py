#!/usr/bin/env python3
"""
Strategy 057: Guppy Multiple Moving Averages (GMMA)

Pine reference: pine_scripts/057_guppy_mma.pine
- Short EMAs: 3,5,8,10,12,15
- Long EMAs: 30,35,40,45,50,60

Deterministic rule:
- Buy when short bundle entirely crosses above long bundle (first bar all short > all long)
- Sell when short bundle entirely crosses below long bundle
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
    from utils.vectorized_helpers import calculate_signal_strength
except Exception:
    def calculate_signal_strength(fs,weights=None):
        df=pd.concat(fs,axis=1); return df.mean(axis=1).clip(0,1)


class Strategy057GuppyMMA(BaseStrategy):
    def __init__(self, parameters: Dict = None):
        params = {
            'short_periods': [3,5,8,10,12,15],
            'long_periods': [30,35,40,45,50,60],
            'signal_threshold': 0.6,
        }
        if parameters:
            params.update(parameters)
        super().__init__('Strategy_057_GuppyMMA', params)

    @staticmethod
    def _ema(s: pd.Series, n: int) -> pd.Series:
        return s.ewm(span=max(1,int(n)), adjust=False, min_periods=1).mean()

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        if 'close' not in data.columns and 'price' in data.columns:
            data['close']=data['price']
        shorts = []
        longs = []
        for n in self.parameters['short_periods']:
            data[f'ema_s_{n}'] = self._ema(data['close'], int(n))
            shorts.append(f'ema_s_{n}')
        for n in self.parameters['long_periods']:
            data[f'ema_l_{n}'] = self._ema(data['close'], int(n))
            longs.append(f'ema_l_{n}')
        data['gmma_short_min'] = data[shorts].min(axis=1)
        data['gmma_short_max'] = data[shorts].max(axis=1)
        data['gmma_long_min']  = data[longs].min(axis=1)
        data['gmma_long_max']  = data[longs].max(axis=1)
        return data

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        # Bundle alignment
        up_align = (data['gmma_short_min'] > data['gmma_long_max'])
        dn_align = (data['gmma_short_max'] < data['gmma_long_min'])
        buy = up_align & (~up_align.shift(1).fillna(False))
        sell = dn_align & (~dn_align.shift(1).fillna(False))
        data['buy_signal'] = buy
        data['sell_signal'] = sell
        # Strength by separation normalized by price
        sep = (data['gmma_short_min'] - data['gmma_long_max']).where(up_align, 0)
        sep = sep.where(~sell, (data['gmma_long_min'] - data['gmma_short_max']))
        norm = (sep.abs() / data['close'].replace(0,np.nan)).fillna(0).clip(0,1)
        st = norm.where(buy|sell, 0.0)
        thr = float(self.parameters['signal_threshold'])
        st[(buy|sell) & (st < thr)] = thr
        data['signal_strength'] = st.clip(0,1)
        return data

