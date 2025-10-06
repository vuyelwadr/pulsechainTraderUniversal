#!/usr/bin/env python3
"""
Strategy 166: LBR PaintBars (signalized)

Pine reference: pine_scripts/166_lbr_paintbars_updated.pine

Signals: buy on transition into uvf (close>b1 and close>b2); sell on transition
into lvf (close<b1 and close<b2). Bands computed per Pine.
"""

import pandas as pd
import numpy as np
from typing import Dict
import os, sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)
from strategies.base_strategy import BaseStrategy


class Strategy166LBRPaintbars(BaseStrategy):
    def __init__(self, parameters: Dict=None):
        params = {
            'lbperiod': 16,
            'atrperiod': 9,
            'mult': 2.5,
            'signal_threshold': 0.6,
        }
        if parameters:
            params.update(parameters)
        super().__init__('Strategy_166_LBR_Paintbars', params)

    @staticmethod
    def _atr(h: pd.Series, l: pd.Series, c: pd.Series, n: int) -> pd.Series:
        pc = c.shift(1)
        tr = pd.concat([(h-l), (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
        return tr.rolling(max(1,int(n)), min_periods=1).mean()

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        for c in ('high','low','close'):
            if c not in data:
                data[c] = data.get('close', pd.Series(index=data.index)).fillna(method='ffill')
        lb = int(self.parameters['lbperiod'])
        ap = int(self.parameters['atrperiod'])
        mult = float(self.parameters['mult'])
        aatr = mult * self._atr(data['high'], data['low'], data['close'], ap).rolling(ap, min_periods=1).mean()
        b1 = data['low'].rolling(lb, min_periods=1).min() + aatr
        b2 = data['high'].rolling(lb, min_periods=1).max() - aatr
        data['lbr_b1'] = b1
        data['lbr_b2'] = b2
        data['lbr_uvf'] = (data['close'] > b1) & (data['close'] > b2)
        data['lbr_lvf'] = (data['close'] < b1) & (data['close'] < b2)
        return data

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        uvf = data['lbr_uvf']
        lvf = data['lbr_lvf']
        buy = uvf & (~uvf.shift(1).fillna(False))
        sell = lvf & (~lvf.shift(1).fillna(False))
        data['buy_signal'] = buy
        data['sell_signal'] = sell
        st = pd.Series(0.0, index=data.index)
        st[buy | sell] = float(self.parameters['signal_threshold'])
        data['signal_strength'] = st
        return data

