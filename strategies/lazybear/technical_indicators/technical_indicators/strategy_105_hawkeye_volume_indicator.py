#!/usr/bin/env python3
"""
Strategy 105: HawkEye Volume Indicator (bar color classification)

Pine reference: pine_scripts/105_hawkeye_volume.pine

Signals: buy on transition into Green; sell on transition into Red. Gray/Yellow treated as neutral.
"""

import pandas as pd
import numpy as np
from typing import Dict
import os, sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)
from strategies.base_strategy import BaseStrategy


class Strategy105HawkEyeVolumeIndicator(BaseStrategy):
    def __init__(self, parameters: Dict=None):
        params = {
            'length': 200,
            'divisor': 3.6,
            'signal_threshold': 0.6,
        }
        if parameters:
            params.update(parameters)
        super().__init__('Strategy_105_HawkEyeVolumeIndicator', params)

    @staticmethod
    def _sma(s: pd.Series, n: int) -> pd.Series:
        return s.rolling(max(1,int(n)), min_periods=1).mean()

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        for c in ('high','low','close','volume'):
            if c not in data:
                data[c] = 0 if c=='volume' else data.get('close', pd.Series(index=data.index)).fillna(method='ffill')
        n = int(self.parameters['length'])
        divisor = float(self.parameters['divisor'])
        range_ = data['high'] - data['low']
        range_avg = self._sma(range_, n)
        volumeA = self._sma(data['volume'], n)
        high1 = data['high'].shift(1); low1 = data['low'].shift(1); mid1 = ((data['high'] + data['low'])/2.0).shift(1)
        u1 = mid1 + (high1 - low1) / divisor
        d1 = mid1 - (high1 - low1) / divisor

        r_enabled1 = (range_ > range_avg) & (data['close'] < d1) & (data['volume'] > volumeA)
        r_enabled2 = data['close'] < mid1
        r_enabled = r_enabled1 | r_enabled2

        g_enabled1 = data['close'] > mid1
        g_enabled2 = (range_ > range_avg) & (data['close'] > u1) & (data['volume'] > volumeA)
        g_enabled3 = (data['high'] > high1) & (range_ < (range_avg/1.5)) & (data['volume'] < volumeA)
        g_enabled4 = (data['low'] < low1) & (range_ < (range_avg/1.5)) & (data['volume'] > volumeA)
        g_enabled = g_enabled1 | g_enabled2 | g_enabled3 | g_enabled4

        data['he_green'] = g_enabled
        data['he_red'] = r_enabled
        return data

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        green = data['he_green']
        red = data['he_red']
        buy = green & (~green.shift(1).fillna(False))
        sell = red & (~red.shift(1).fillna(False))
        data['buy_signal'] = buy
        data['sell_signal'] = sell
        st = pd.Series(0.0, index=data.index)
        st[buy | sell] = float(self.parameters['signal_threshold'])
        data['signal_strength'] = st
        return data

