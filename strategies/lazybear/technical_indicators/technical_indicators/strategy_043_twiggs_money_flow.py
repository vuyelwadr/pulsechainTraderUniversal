#!/usr/bin/env python3
"""
Strategy 043: Twiggs Money Flow

Pine reference: pine_scripts/043_twiggs_money_flow.pine
Implements TMF using Wilder-style recursive moving average (WiMA).

Signals: buy on TMF crossing above 0, sell on crossing below 0.
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


class Strategy043TwiggsMoneyFlow(BaseStrategy):
    def __init__(self, parameters: Dict=None):
        params = {
            'length': 21,
            'signal_threshold': 0.6,
        }
        if parameters:
            params.update(parameters)
        super().__init__('Strategy_043_TwiggsMoneyFlow', params)

    @staticmethod
    def _wima(s: pd.Series, n: int) -> pd.Series:
        n = max(1,int(n))
        alpha = 1.0 / n
        return s.ewm(alpha=alpha, adjust=False, min_periods=1).mean()

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        for c in ('high','low','close','volume'):
            if c not in data.columns:
                data[c] = 0 if c=='volume' else data.get('close', data.get('price', pd.Series(index=data.index))).fillna(method='ffill')
        length = int(self.parameters['length'])

        tr_h = data[['close','high']].shift(1)
        tr_h = pd.DataFrame({'a':data['close'].shift(1),'b':data['high']}).max(axis=1)
        tr_l = pd.DataFrame({'a':data['close'].shift(1),'b':data['low']}).min(axis=1)
        tr_c = (tr_h - tr_l).replace(0, np.nan)
        adv = data['volume'] * ((data['close'] - tr_l) - (tr_h - data['close'])) / tr_c
        adv = adv.replace([np.inf,-np.inf], np.nan).fillna(0)
        wmV = self._wima(data['volume'], length)
        wmA = self._wima(adv, length)
        tmf = (wmA / wmV.replace(0,np.nan)).replace([np.inf,-np.inf], np.nan).fillna(0)
        data['tmf'] = tmf
        return data

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        buy = crossover(data['tmf'], 0.0)
        sell = crossunder(data['tmf'], 0.0)
        data['buy_signal'] = buy
        data['sell_signal'] = sell
        st = data['tmf'].abs() / (data['tmf'].rolling(50, min_periods=10).std(ddof=0).replace(0,np.nan))
        st = st.clip(0,1).fillna(0)
        st = st.where(buy|sell, 0.0)
        thr = float(self.parameters['signal_threshold'])
        st[(buy|sell) & (st < thr)] = thr
        data['signal_strength'] = st
        return data

