#!/usr/bin/env python3
"""
Strategy 061: Constance Brown Composite Index

Pine source not available; implement a composite consistent with description:
Combine RSI, Stochastic %K, and normalized momentum into a composite and
smooth it.

Signals:
- Buy when composite moves up through 30 level (recovering from oversold)
- Sell when composite moves down through 70 level (falling from overbought)
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
    from utils.vectorized_helpers import pine_rma, crossover, crossunder
except Exception:
    def pine_rma(s,n): return s.ewm(alpha=1/max(1,int(n)), adjust=False, min_periods=1).mean()
    def crossover(a,b): return (a>b) & (a.shift(1)<=b)
    def crossunder(a,b): return (a<b) & (a.shift(1)>=b)


class Strategy061ConstanceBrownCompositeIndex(BaseStrategy):
    def __init__(self, parameters: Dict = None):
        params = {
            'rsi_period': 14,
            'stoch_period': 14,
            'momentum_period': 10,
            'smooth': 5,
            'signal_threshold': 0.6,
        }
        if parameters:
            params.update(parameters)
        super().__init__('Strategy_061_CB_Composite', params)

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        for c in ('high','low','close'):
            if c not in data.columns and 'price' in data.columns:
                data[c]=data['price']
        # RSI via Pine RMA
        n = int(self.parameters['rsi_period'])
        delta = data['close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_g = pine_rma(gain, n)
        avg_l = pine_rma(loss, n)
        rs = avg_g / avg_l.replace(0,np.nan)
        rsi = (100 - 100/(1+rs)).fillna(50)

        # Stochastic %K
        k_n = int(self.parameters['stoch_period'])
        ll = data['low'].rolling(k_n, min_periods=1).min()
        hh = data['high'].rolling(k_n, min_periods=1).max()
        stoch_k = ((data['close'] - ll) / (hh - ll).replace(0,np.nan) * 100).fillna(50)

        # Momentum normalized
        m_n = int(self.parameters['momentum_period'])
        mom = data['close'] - data['close'].shift(m_n)
        mom_norm = ((mom / data['close'].replace(0,np.nan)) * 1000).clip(-100,100) + 50

        comp = (rsi + stoch_k + mom_norm) / 3.0
        smooth = int(self.parameters['smooth'])
        comp_s = comp.ewm(span=max(1,smooth), adjust=False, min_periods=1).mean()

        data['cb_comp'] = comp
        data['cb_comp_s'] = comp_s
        return data

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        buy = crossover(data['cb_comp_s'], 30.0)
        sell = crossunder(data['cb_comp_s'], 70.0)
        data['buy_signal'] = buy
        data['sell_signal'] = sell
        st = (data['cb_comp_s'] - 50.0).abs() / 50.0
        st = st.clip(0,1).where(buy|sell, 0.0)
        thr = float(self.parameters['signal_threshold'])
        st[(buy|sell) & (st < thr)] = thr
        data['signal_strength'] = st.clip(0,1)
        return data

