#!/usr/bin/env python3
"""
Strategy 046: Market Facilitation Index (color-coded states)

Pine reference: pine_scripts/046_market_facilitation_index.pine
Classifies bars into Green/Fade/Fake/Squat using ROC of (high-low)/volume and volume itself.

Signals: buy on transition into Green; sell on transition into Red (Squat) or Fade.
"""

import pandas as pd
import numpy as np
from typing import Dict
import os, sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)
from strategies.base_strategy import BaseStrategy


class Strategy046MarketFacilitationIndex(BaseStrategy):
    def __init__(self, parameters: Dict=None):
        params = {
            'signal_threshold': 0.6,
        }
        if parameters:
            params.update(parameters)
        super().__init__('Strategy_046_MarketFacilitationIndex', params)

    @staticmethod
    def _roc(s: pd.Series) -> pd.Series:
        return s.diff()

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        for c in ('high','low','close','volume'):
            if c not in data:
                data[c] = 0 if c=='volume' else data.get('close', pd.Series(index=data.index)).fillna(method='ffill')
        hl_over_v = (data['high'] - data['low']) / data['volume'].replace(0, np.nan)
        r_hl = self._roc(hl_over_v).fillna(0)
        r_v = self._roc(data['volume']).fillna(0)
        data['mfi_green'] = (r_hl > 0) & (r_v > 0)
        data['mfi_fade'] = (r_hl < 0) & (r_v < 0)
        data['mfi_fake'] = (r_hl > 0) & (r_v < 0)
        data['mfi_squat'] = (r_hl < 0) & (r_v > 0)
        return data

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        green = data['mfi_green']
        redlike = data['mfi_squat'] | data['mfi_fade']
        buy = green & (~green.shift(1).fillna(False))
        sell = redlike & (~redlike.shift(1).fillna(False))
        data['buy_signal'] = buy
        data['sell_signal'] = sell
        st = pd.Series(0.0, index=data.index)
        st[buy | sell] = float(self.parameters['signal_threshold'])
        data['signal_strength'] = st
        return data

