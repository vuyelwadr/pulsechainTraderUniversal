#!/usr/bin/env python3
"""
Strategy 071: Volume Price Confirmation Indicator (VPCI)

Pine reference: pine_scripts/071_vpci_lazybear.pine
vpci = (VWMA(close, long) - SMA(close, long)) * (VWMA(close, short)/SMA(close, short)) * (SMA(volume, short)/SMA(volume, long))
Signals on zero-line crosses; optional MA overlay exists in Pine but not required for signals here.
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
    from utils.vectorized_helpers import crossover, crossunder
except Exception:
    def crossover(a,b): return (a>b) & (a.shift(1)<=b)
    def crossunder(a,b): return (a<b) & (a.shift(1)>=b)


class Strategy071VPCI(BaseStrategy):
    def __init__(self, parameters: Dict = None):
        params = {
            'shortTerm': 5,
            'longTerm': 20,
            'lengthMA': 8,
            'signal_threshold': 0.6,
        }
        if parameters:
            params.update(parameters)
        super().__init__('Strategy_071_VolumePriceConfirmationIndex', params)

    def _sma(self, s: pd.Series, n: int) -> pd.Series:
        return s.rolling(max(1,int(n)), min_periods=1).mean()

    def _vwma(self, price: pd.Series, volume: pd.Series, n: int) -> pd.Series:
        n = max(1,int(n))
        pv = (price * volume).rolling(n, min_periods=1).sum()
        v = volume.rolling(n, min_periods=1).sum().replace(0,np.nan)
        return (pv / v).fillna(method='ffill').fillna(0)

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        for c in ('close','volume'):
            if c not in data.columns:
                if c=='close' and 'price' in data.columns:
                    data['close']=data['price']
                else:
                    data['volume']=0
        st = int(self.parameters['shortTerm'])
        lt = int(self.parameters['longTerm'])
        vpci = (
            (self._vwma(data['close'], data['volume'], lt) - self._sma(data['close'], lt)) *
            (self._vwma(data['close'], data['volume'], st) / self._sma(data['close'], st).replace(0,np.nan)) *
            (self._sma(data['volume'], st) / self._sma(data['volume'], lt).replace(0,np.nan))
        )
        data['vpci'] = vpci.replace([np.inf,-np.inf], np.nan).fillna(0)
        data['vpci_ma'] = self._sma(data['vpci'], int(self.parameters['lengthMA']))
        return data

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        buy = crossover(data['vpci'], 0.0)
        sell = crossunder(data['vpci'], 0.0)
        data['buy_signal'] = buy
        data['sell_signal'] = sell
        # Strength by standardized vpci vs its MA
        dev = (data['vpci'] - data['vpci_ma']).abs()
        norm = dev / (data['vpci'].rolling(50, min_periods=10).std(ddof=0).replace(0,np.nan))
        st = norm.fillna(0).clip(0,1)
        st = st.where(buy|sell, 0.0)
        thr = float(self.parameters['signal_threshold'])
        st[(buy|sell) & (st < thr)] = thr
        data['signal_strength'] = st
        return data

