#!/usr/bin/env python3
"""
Strategy 107: Trend Trigger Factor (TTF)

Pine reference: pine_scripts/107_trend_trigger_factor.pine

Signals: buy on crossing above sell threshold (st), sell on crossing below buy threshold (bt)
Default bt=+100, st=-100.
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


class Strategy107TrendTriggerFactor(BaseStrategy):
    def __init__(self, parameters: Dict=None):
        params = {
            'length': 15,
            'bt': 100.0,
            'st': -100.0,
            'signal_threshold': 0.6,
        }
        if parameters:
            params.update(parameters)
        super().__init__('Strategy_107_TrendTriggerFactor', params)

    @staticmethod
    def _calc_ttf(high: pd.Series, low: pd.Series, periods: int) -> pd.Series:
        periods = max(1,int(periods))
        bp = high.rolling(periods, min_periods=1).max() - low.shift(periods).rolling(periods, min_periods=1).min()
        sp = high.shift(periods).rolling(periods, min_periods=1).max() - low.rolling(periods, min_periods=1).min()
        denom = 0.5 * (bp + sp)
        return 100.0 * (bp - sp) / denom.replace(0, np.nan)

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        for c in ('high','low'):
            if c not in data:
                data[c] = data.get('close', pd.Series(index=data.index)).fillna(method='ffill')
        length = int(self.parameters['length'])
        data['ttf'] = self._calc_ttf(data['high'], data['low'], length).replace([np.inf,-np.inf], np.nan).fillna(0)
        return data

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        bt = float(self.parameters['bt'])
        stv = float(self.parameters['st'])
        buy = crossover(data['ttf'], stv)
        sell = crossunder(data['ttf'], bt)
        data['buy_signal'] = buy
        data['sell_signal'] = sell
        st = (data['ttf'].abs() / (data['ttf'].rolling(50, min_periods=10).std(ddof=0).replace(0,np.nan))).clip(0,1).fillna(0)
        st = st.where(buy|sell, 0.0)
        thr=float(self.parameters['signal_threshold'])
        st[(buy|sell) & (st < thr)] = thr
        data['signal_strength'] = st
        return data

