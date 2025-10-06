#!/usr/bin/env python3
"""
Strategy 178: Range Identifier v2

Pine reference: pine_scripts/178_range_identifier_v2.pine

Signals: buy when price crosses above EMA(length=34); sell when price crosses below.
"""

import pandas as pd
import numpy as np
from typing import Dict
import os, sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)
from strategies.base_strategy import BaseStrategy

try:
    sys.path.insert(0, os.path.join(project_root, 'src'))
    from utils.vectorized_helpers import crossover, crossunder
except Exception:
    def crossover(a,b): return (a>b) & (a.shift(1)<=b)
    def crossunder(a,b): return (a<b) & (a.shift(1)>=b)


class Strategy178RangeIdentifier(BaseStrategy):
    def __init__(self, parameters: Dict=None):
        params = {
            'ema_length': 34,
            'signal_threshold': 0.6,
        }
        if parameters:
            params.update(parameters)
        super().__init__('Strategy_178_RangeIdentifier', params)

    @staticmethod
    def _ema(s: pd.Series, n: int) -> pd.Series:
        return s.ewm(span=max(1,int(n)), adjust=False, min_periods=1).mean()

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        for c in ('high','low','close'):
            if c not in data:
                data[c]=data.get('close', pd.Series(index=data.index)).fillna(method='ffill')
        data['ri_ema'] = self._ema(data['close'], int(self.parameters['ema_length']))
        # Optionally compute up/down bands (not used for signals here)
        up = pd.Series(index=data.index, dtype=float)
        down = pd.Series(index=data.index, dtype=float)
        for i in range(len(data)):
            if i == 0:
                up.iloc[i] = data['high'].iloc[i]
                down.iloc[i] = data['low'].iloc[i]
            else:
                if (data['close'].iloc[i] < up.iloc[i-1]) and (data['close'].iloc[i] > down.iloc[i-1]):
                    up.iloc[i] = up.iloc[i-1]
                    down.iloc[i] = down.iloc[i-1]
                else:
                    up.iloc[i] = data['high'].iloc[i]
                    down.iloc[i] = data['low'].iloc[i]
        data['ri_up'] = up
        data['ri_down'] = down
        return data

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        buy = crossover(data['close'], data['ri_ema'])
        sell = crossunder(data['close'], data['ri_ema'])
        data['buy_signal'] = buy
        data['sell_signal'] = sell
        st = (data['close'] - data['ri_ema']).abs() / (data['close'].rolling(50, min_periods=10).std(ddof=0).replace(0,np.nan))
        st = st.clip(0,1).fillna(0).where(buy|sell, 0.0)
        thr=float(self.parameters['signal_threshold'])
        st[(buy|sell) & (st < thr)] = thr
        data['signal_strength'] = st
        return data

