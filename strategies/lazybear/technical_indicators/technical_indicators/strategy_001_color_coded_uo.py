#!/usr/bin/env python3
"""
Strategy 001: Color coded Ultimate Oscillator (UO)

TradingView URL: https://www.tradingview.com/v/CDJHwbyx/
Type: momentum/oscillator

Indicator parity:
- UO = 100 * (4*avg7 + 2*avg14 + avg28)/7 where
  avgN = sum(bp, N) / sum(tr, N), bp = close - min(low, close[1]),
  tr = max(high, close[1]) - min(low, close[1])
- lengthSlope parameter used to determine rising/falling state as in Pine.

Signals (derived for trading):
- Buy when UO crosses above mid_level (default 50) and is rising over
  `lengthSlope` bars.
- Sell when UO crosses below mid_level and is falling over `lengthSlope` bars.
This mirrors the color-coded intent while providing actionable signals for the
bot. No look-ahead bias.
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
    from utils.vectorized_helpers import crossover, crossunder, calculate_signal_strength, apply_position_constraints
except Exception:
    def crossover(a,b): return (a>b) & (a.shift(1) <= b.shift(1))
    def crossunder(a,b): return (a<b) & (a.shift(1) >= b.shift(1))
    def calculate_signal_strength(fs,weights=None):
        import pandas as pd
        df=pd.concat(fs,axis=1); return df.mean(axis=1).clip(0,1)
    def apply_position_constraints(b,s,allow_short=False): return b,s


class Strategy001ColorCodedUO(BaseStrategy):
    def __init__(self, parameters: Dict=None):
        params = {
            'length7': 7,
            'length14': 14,
            'length28': 28,
            'lengthSlope': 1,     # Pine default used for rising/falling
            'mid_level': 50.0,    # Trading level for signals
            'signal_threshold': 0.6,
        }
        if parameters:
            params.update(parameters)
        super().__init__('Strategy_001_ColorCodedUO', params)

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        # Ensure required columns
        for c in ('open','high','low','close'):
            if c not in data.columns and 'price' in data.columns:
                data[c] = data['price']

        close = data['close']
        high = data['high']
        low = data['low']

        prev_close = close.shift(1)
        high_ = pd.concat([high, prev_close], axis=1).max(axis=1)
        low_ = pd.concat([low, prev_close], axis=1).min(axis=1)
        bp = close - low_
        tr = (high_ - low_).replace(0, np.nan)

        def avg_ratio(n: int) -> pd.Series:
            return (bp.rolling(n, min_periods=1).sum() / tr.rolling(n, min_periods=1).sum()).fillna(0)

        l7 = int(self.parameters['length7'])
        l14 = int(self.parameters['length14'])
        l28 = int(self.parameters['length28'])

        avg7 = avg_ratio(l7)
        avg14 = avg_ratio(l14)
        avg28 = avg_ratio(l28)
        uo = 100.0 * (4*avg7 + 2*avg14 + avg28) / 7.0

        data['uo'] = uo

        # rising/falling over lengthSlope (as Pine color-coding)
        k = int(self.parameters['lengthSlope'])
        if k < 1:
            k = 1
        data['uo_rising'] = data['uo'] > data['uo'].shift(k)
        data['uo_falling'] = data['uo'] < data['uo'].shift(k)

        return data

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        p = self.parameters
        data['buy_signal'] = False
        data['sell_signal'] = False
        data['signal_strength'] = 0.0

        mid = float(p['mid_level'])
        rising = data['uo_rising']
        falling = data['uo_falling']

        buy = crossover(data['uo'], mid) & rising
        sell = crossunder(data['uo'], mid) & falling

        data['buy_signal'] = buy
        data['sell_signal'] = sell

        # Strength combines distance from mid and slope magnitude
        dist = (data['uo'] - mid).abs() / 50.0
        slope = (data['uo'] - data['uo'].shift(int(max(1, p['lengthSlope'])))).abs() / 50.0
        data['signal_strength'] = calculate_signal_strength([
            dist.clip(0,1).fillna(0), slope.clip(0,1).fillna(0)
        ], weights=[0.6, 0.4])

        # Ensure non-zero strength on actual signals to allow execution
        data.loc[buy | sell, 'signal_strength'] = data.loc[buy | sell, 'signal_strength'].clip(lower=0.6)

        # Apply threshold
        weak = data['signal_strength'] < p['signal_threshold']
        data.loc[weak, ['buy_signal','sell_signal']] = False

        return data
