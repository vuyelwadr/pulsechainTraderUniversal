#!/usr/bin/env python3
"""
Strategy 090: Time and Money Chart Channel (approximation)

Channel around an EMA centerline with band width scaled by volatility (ATR)
and relative volume (volume / SMA(volume)). Signals on price breaking bands.
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


class Strategy090TimeAndMoneyChartChannel(BaseStrategy):
    def __init__(self, parameters: Dict=None):
        params = {
            'ema_len': 20,
            'atr_len': 14,
            'vol_len': 20,
            'base_mult': 1.0,
            'vol_sensitivity': 1.0,
            'signal_threshold': 0.6,
        }
        if parameters:
            params.update(parameters)
        super().__init__('Strategy_090_Time_and_Money_Chart_Channel', params)

    @staticmethod
    def _ema(s: pd.Series, n: int) -> pd.Series:
        return s.ewm(span=max(1,int(n)), adjust=False, min_periods=1).mean()

    @staticmethod
    def _atr(h: pd.Series, l: pd.Series, c: pd.Series, n: int) -> pd.Series:
        pc=c.shift(1)
        tr=pd.concat([(h-l),(h-pc).abs(),(l-pc).abs()],axis=1).max(axis=1)
        return tr.rolling(max(1,int(n)),min_periods=1).mean()

    @staticmethod
    def _sma(s: pd.Series, n: int) -> pd.Series:
        return s.rolling(max(1,int(n)), min_periods=1).mean()

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        for c in ('high','low','close','volume'):
            if c not in data:
                data[c]=0 if c=='volume' else data.get('close', pd.Series(index=data.index)).fillna(method='ffill')
        ema = self._ema(data['close'], int(self.parameters['ema_len']))
        atr = self._atr(data['high'], data['low'], data['close'], int(self.parameters['atr_len']))
        vratio = (data['volume'] / self._sma(data['volume'], int(self.parameters['vol_len'])).replace(0,np.nan)).fillna(1.0)
        width = float(self.parameters['base_mult']) * atr * (1.0 + float(self.parameters['vol_sensitivity']) * (vratio - 1.0))
        width = width.clip(lower=atr*0.5)
        data['tmc_mid'] = ema
        data['tmc_up'] = ema + width
        data['tmc_dn'] = ema - width
        return data

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        buy = crossover(data['close'], data['tmc_up'])
        sell = crossunder(data['close'], data['tmc_dn'])
        data['buy_signal'] = buy
        data['sell_signal'] = sell
        bw = (data['tmc_up'] - data['tmc_dn']).replace(0,np.nan)
        st = (data['close'] - data['tmc_mid']).abs() / bw
        st = st.clip(0,1).fillna(0).where(buy|sell, 0.0)
        thr=float(self.parameters['signal_threshold'])
        st[(buy|sell) & (st<thr)] = thr
        data['signal_strength'] = st
        return data

