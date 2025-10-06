#!/usr/bin/env python3
"""
Strategy 128: Colored Volume Bars

Pine reference: pine_scripts/128_colored_volume_bars.pine

Signals: buy on transition into green/blue class, sell on transition into red/orange class.
"""

import pandas as pd
import numpy as np
from typing import Dict
import os, sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)
from strategies.base_strategy import BaseStrategy


class Strategy128ColoredVolumeBars(BaseStrategy):
    def __init__(self, parameters: Dict=None):
        params = {
            'lookback': 10,
            'showMA': False,
            'lengthMA': 20,
            'signal_threshold': 0.6,
        }
        if parameters:
            params.update(parameters)
        super().__init__('Strategy_128_ColoredVolumeBars', params)

    @staticmethod
    def _sma(s: pd.Series, n: int) -> pd.Series:
        return s.rolling(max(1,int(n)), min_periods=1).mean()

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        for c in ('close','volume'):
            if c not in data:
                data[c] = 0 if c=='volume' else data.get('price', pd.Series(index=data.index)).fillna(method='ffill')
        lb = int(self.parameters['lookback'])
        p2 = data['close']
        v2 = data['volume']
        p1 = p2.shift(lb)
        v1 = v2.shift(lb)
        green = (p2>p1) & (v2>v1)
        blue  = (p2>p1) & (v2<v1)
        orange = (p2<p1) & (v2<v1)
        red   = (p2<p1) & (v2>v1)
        data['cvb_green'] = green
        data['cvb_blue'] = blue
        data['cvb_orange'] = orange
        data['cvb_red'] = red
        data['cvb_ma'] = self._sma(v2, int(self.parameters['lengthMA'])) if bool(self.parameters.get('showMA', False)) else 0
        return data

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        buy_mask = (data['cvb_green'] | data['cvb_blue'])
        sell_mask = (data['cvb_red'] | data['cvb_orange'])
        buy = buy_mask & (~buy_mask.shift(1).fillna(False))
        sell = sell_mask & (~sell_mask.shift(1).fillna(False))
        data['buy_signal'] = buy
        data['sell_signal'] = sell
        st = pd.Series(0.0, index=data.index)
        st[buy | sell] = float(self.parameters['signal_threshold'])
        data['signal_strength'] = st
        return data

