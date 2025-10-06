#!/usr/bin/env python3
"""
Strategy 123: Bulkowski NR7/NR4 Pattern Identifier

Pine reference: pine_scripts/123_bulkowsky_nr7_nr4.pine

Signals (minimal):
- On NR7 bar, buy if close > prior close; sell if close < prior close.
This provides deterministic outputs for backtesting while preserving indicator intent.
"""

import pandas as pd
import numpy as np
from typing import Dict
import os, sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)
from strategies.base_strategy import BaseStrategy


class Strategy123BulkowskyNR7NR4(BaseStrategy):
    def __init__(self, parameters: Dict=None):
        params = {
            'signal_threshold': 0.6,
        }
        if parameters:
            params.update(parameters)
        super().__init__('Strategy_123_Bulkowsky_NR7_NR4', params)

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        for c in ('high','low','close'):
            if c not in data:
                data[c] = data.get('close', pd.Series(index=data.index)).fillna(method='ffill')
        rng = (data['high'] - data['low']).fillna(0)
        nr7 = (rng < rng.shift(1)) & (rng < rng.shift(2)) & (rng < rng.shift(3)) & (rng < rng.shift(4)) & (rng < rng.shift(5)) & (rng < rng.shift(6))
        nr4 = (rng < rng.shift(1)) & (rng < rng.shift(2)) & (rng < rng.shift(3))
        data['nr7'] = nr7
        data['nr4'] = nr4
        return data

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        buy = data['nr7'] & (data['close'] > data['close'].shift(1))
        sell = data['nr7'] & (data['close'] < data['close'].shift(1))
        data['buy_signal'] = buy.fillna(False)
        data['sell_signal'] = sell.fillna(False)
        st = pd.Series(0.0, index=data.index)
        st[buy | sell] = float(self.parameters['signal_threshold'])
        data['signal_strength'] = st
        return data

