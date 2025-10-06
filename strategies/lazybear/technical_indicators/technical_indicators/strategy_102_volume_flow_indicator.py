#!/usr/bin/env python3
"""
Strategy 102: Volume Flow Indicator (VFI)

Pine reference: pine_scripts/102_volume_flow.pine

Signals: buy on VFI crossing above its EMA; sell on crossing below.
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


class Strategy102VolumeFlowIndicator(BaseStrategy):
    def __init__(self, parameters: Dict=None):
        params = {
            'length': 130,
            'coef': 0.2,
            'vcoef': 2.5,
            'signalLength': 5,
            'smoothVFI': False,
            'signal_threshold': 0.6,
        }
        if parameters:
            params.update(parameters)
        super().__init__('Strategy_102_VolumeFlowIndicator', params)

    @staticmethod
    def _sma(s: pd.Series, n: int) -> pd.Series:
        return s.rolling(max(1,int(n)), min_periods=1).mean()

    @staticmethod
    def _ema(s: pd.Series, n: int) -> pd.Series:
        return s.ewm(span=max(1,int(n)), adjust=False, min_periods=1).mean()

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        for c in ('high','low','close','volume'):
            if c not in data:
                data[c] = 0 if c=='volume' else data.get('close', pd.Series(index=data.index)).fillna(method='ffill')
        length = int(self.parameters['length'])
        coef = float(self.parameters['coef'])
        vcoef = float(self.parameters['vcoef'])
        signal_len = int(self.parameters['signalLength'])
        smooth = bool(self.parameters.get('smoothVFI', False))

        typical = (data['high'] + data['low'] + data['close']) / 3.0
        inter = (typical.apply(np.log) - typical.shift(1).apply(lambda x: np.log(x) if pd.notna(x) and x>0 else np.nan))
        vinter = inter.rolling(30, min_periods=1).std(ddof=0)
        cutoff = coef * vinter * data['close']
        vave = self._sma(data['volume'], length).shift(1)
        vmax = vave * vcoef
        vc = np.where(data['volume'] < vmax, data['volume'], vmax)
        vc = pd.Series(vc, index=data.index)
        mf = typical - typical.shift(1)
        vcp = np.where(mf > cutoff, vc, np.where(mf < -cutoff, -vc, 0))
        vcp = pd.Series(vcp, index=data.index)
        base = vcp.rolling(length, min_periods=1).sum() / vave.replace(0, np.nan)
        vfi = self._sma(base, 3) if smooth else base
        vfi = vfi.replace([np.inf,-np.inf], np.nan).fillna(0)
        vfima = self._ema(vfi, signal_len)
        data['vfi'] = vfi
        data['vfima'] = vfima
        data['vfi_diff'] = vfi - vfima
        return data

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        buy = crossover(data['vfi'], data['vfima'])
        sell = crossunder(data['vfi'], data['vfima'])
        data['buy_signal'] = buy
        data['sell_signal'] = sell
        st = (data['vfi_diff'].abs() / (data['vfi'].rolling(50, min_periods=10).std(ddof=0).replace(0,np.nan))).clip(0,1).fillna(0)
        st = st.where(buy|sell, 0.0)
        thr=float(self.parameters['signal_threshold'])
        st[(buy|sell) & (st < thr)] = thr
        data['signal_strength'] = st
        return data

