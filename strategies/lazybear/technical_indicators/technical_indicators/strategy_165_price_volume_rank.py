#!/usr/bin/env python3
"""
Strategy 165: Price Volume Rank

Pine reference: pine_scripts/165_price_volume_rank.pine

Signals: buy when PVR transitions into bull region (>3); sell when transitions into bear region (<2).
"""

import pandas as pd
import numpy as np
from typing import Dict
import os, sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)
from strategies.base_strategy import BaseStrategy


class Strategy165PriceVolumeRank(BaseStrategy):
    def __init__(self, parameters: Dict=None):
        params = {
            'showMA': False,
            'ma1': 5,
            'ma2': 10,
            'signal_threshold': 0.6,
        }
        if parameters:
            params.update(parameters)
        super().__init__('Strategy_165_PriceVolumeRank', params)

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        for c in ('close','volume'):
            if c not in data:
                data[c] = 0 if c=='volume' else data.get('close', pd.Series(index=data.index)).fillna(method='ffill')
        c = data['close']; v = data['volume']
        pvr = pd.Series(index=data.index, dtype=float)
        up = (c > c.shift(1))
        vd = (v > v.shift(1))
        pvr[up & vd] = 1
        pvr[up & ~vd] = 2
        pvr[~up & ~vd] = 3
        pvr[~up & vd] = 4
        data['pvr'] = pvr.fillna(method='ffill').fillna(2.5)
        return data

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        bull = data['pvr'] > 3.0
        bear = data['pvr'] < 2.0
        buy = bull & (~bull.shift(1).fillna(False))
        sell = bear & (~bear.shift(1).fillna(False))
        data['buy_signal'] = buy
        data['sell_signal'] = sell
        st = pd.Series(0.0, index=data.index)
        st[buy | sell] = float(self.parameters['signal_threshold'])
        data['signal_strength'] = st
        return data

