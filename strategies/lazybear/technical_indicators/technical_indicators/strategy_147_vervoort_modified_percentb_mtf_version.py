#!/usr/bin/env python3
"""
Strategy 147: Vervoort Modified %b (MTF version simplified to current timeframe)

Pine reference: pine_scripts/147_vervoort_modified_percentb_mtf.pine
Implements HA-based TEMA cascade and modified %b calculation. MTF inputs
are ignored here (current timeframe only) to keep pipeline local.

Signals: buy when %b crosses above dynamic upper band; sell when crosses below
dynamic lower band.
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


class Strategy147VervoortModifiedPercentBMTF(BaseStrategy):
    def __init__(self, parameters: Dict=None):
        params = {
            'length': 18,
            'temaLength': 8,
            'stdevHigh': 1.6,
            'stdevLow': 1.6,
            'stdevLength': 200,
            'signal_threshold': 0.6,
        }
        if parameters:
            params.update(parameters)
        super().__init__('Strategy_147_Vervoort_Modified_PercentB_MTF', params)

    @staticmethod
    def _ema(s: pd.Series, n: int) -> pd.Series:
        return s.ewm(span=max(1,int(n)), adjust=False, min_periods=1).mean()

    def _tema(self, s: pd.Series, n: int) -> pd.Series:
        e1 = self._ema(s, n)
        e2 = self._ema(e1, n)
        e3 = self._ema(e2, n)
        return 3*(e1 - e2) + e3

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        for c in ('open','high','low','close'):
            if c not in data:
                data[c]=data.get('close', pd.Series(index=data.index)).fillna(method='ffill')
        p=self.parameters
        ohlc4 = data[['open','high','low','close']].mean(axis=1)
        ha_open = (ohlc4.shift(1) + ohlc4.shift(1)) / 2.0
        ha_c = (ohlc4 + ha_open + data[['high', 'low']].max(axis=1) + data[['high','low']].min(axis=1)) / 4.0
        tma1 = self._tema(ha_c, int(p['temaLength']))
        tma2 = self._tema(tma1, int(p['temaLength']))
        zlha = tma1 + (tma1 - tma2)
        zt = self._tema(zlha, int(p['temaLength']))
        std = zt.rolling(int(p['length']), min_periods=1).std(ddof=0)
        w = zt.rolling(int(p['length']), min_periods=1).apply(lambda x: pd.Series(x).ewm(span=int(p['length']), adjust=False).mean().iloc[-1], raw=False)
        percb = ((zt + 2*std) - w) / (4*std.replace(0,np.nan)) * 100.0
        data['percb'] = percb.replace([np.inf,-np.inf], np.nan).fillna(0)
        sh = float(p['stdevHigh']); sl=float(p['stdevLow']); slen=int(p['stdevLength'])
        rolling_sd = data['percb'].rolling(slen, min_periods=10).std(ddof=0)
        data['ub'] = 50.0 + sh*rolling_sd
        data['lb'] = 50.0 - sl*rolling_sd
        return data

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        buy = crossover(data['percb'], data['ub'])
        sell = crossunder(data['percb'], data['lb'])
        data['buy_signal'] = buy
        data['sell_signal'] = sell
        st = (data['percb'] - 50.0).abs() / (data['percb'].rolling(50, min_periods=10).std(ddof=0).replace(0,np.nan))
        st = st.clip(0,1).fillna(0).where(buy|sell, 0.0)
        thr=float(self.parameters['signal_threshold'])
        st[(buy|sell) & (st < thr)] = thr
        data['signal_strength'] = st
        return data

