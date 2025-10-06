#!/usr/bin/env python3
"""
Strategy 121: RSI with Ehlers Roofing Filter

Applies a high-pass then low-pass (Super Smoother) roofing filter to price, then
computes RSI on the filtered series. Signals on RSI crossing midline.
"""

import pandas as pd
import numpy as np
from typing import Dict
import os, sys, math

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)
from strategies.base_strategy import BaseStrategy

try:
    sys.path.insert(0, os.path.join(project_root, 'src'))
    from utils.vectorized_helpers import crossover, crossunder
except Exception:
    def crossover(a,b): return (a>b) & (a.shift(1)<=b)
    def crossunder(a,b): return (a<b) & (a.shift(1)>=b)


class Strategy121RSIWithEhlersRoofingFilter(BaseStrategy):
    def __init__(self, parameters: Dict=None):
        params = {
            'hp_period': 48,
            'lp_period': 10,
            'rsi_length': 14,
            'signal_threshold': 0.6,
        }
        if parameters:
            params.update(parameters)
        super().__init__('Strategy_121_RSI_with_Ehlers_Roofing_Filter', params)

    def _super_smoother(self, x: pd.Series, period: int) -> pd.Series:
        n=max(1,int(period))
        a1 = math.exp(-math.sqrt(2.0) * math.pi / n)
        b1 = 2.0 * a1 * math.cos(math.sqrt(2.0) * math.pi / n)
        c2 = b1
        c3 = -a1*a1
        c1 = 1.0 - c2 - c3
        y = pd.Series(index=x.index, dtype=float)
        x1 = x.shift(1).fillna(x.iloc[0])
        y_prev1 = 0.0
        y_prev2 = 0.0
        for i, idx in enumerate(x.index):
            yi = c1 * (x.loc[idx] + x1.loc[idx]) * 0.5 + c2 * y_prev1 + c3 * y_prev2
            y.loc[idx] = yi
            y_prev2 = y_prev1
            y_prev1 = yi
        return y

    def _high_pass(self, x: pd.Series, period: int) -> pd.Series:
        # Simple 2nd-order high-pass derived from Super Smoother complement
        ss = self._super_smoother(x, period)
        return x - ss

    @staticmethod
    def _rsi(s: pd.Series, n: int) -> pd.Series:
        n=max(1,int(n))
        d=s.diff()
        up=d.clip(lower=0); dn=-d.clip(upper=0)
        ru=up.ewm(alpha=1.0/n, adjust=False, min_periods=1).mean()
        rd=dn.ewm(alpha=1.0/n, adjust=False, min_periods=1).mean()
        rs=ru/rd.replace(0,np.nan)
        return (100 - 100/(1+rs)).fillna(50)

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        if 'close' not in data and 'price' in data:
            data['close']=data['price']
        hp = self._high_pass(data['close'].astype(float), int(self.parameters['hp_period']))
        rf = self._super_smoother(hp, int(self.parameters['lp_period']))
        data['rf_rsi'] = self._rsi(rf, int(self.parameters['rsi_length']))
        return data

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        buy = crossover(data['rf_rsi'], 50.0)
        sell = crossunder(data['rf_rsi'], 50.0)
        data['buy_signal'] = buy
        data['sell_signal'] = sell
        st = (data['rf_rsi'] - 50.0).abs() / 50.0
        st = st.clip(0,1).fillna(0).where(buy|sell, 0.0)
        thr=float(self.parameters['signal_threshold'])
        st[(buy|sell) & (st < thr)] = thr
        data['signal_strength'] = st
        return data

