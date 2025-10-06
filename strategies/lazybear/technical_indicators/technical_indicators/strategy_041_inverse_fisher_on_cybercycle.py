#!/usr/bin/env python3
"""
Strategy 041: Inverse Fisher on CyberCycle

Computes Ehlers Cyber Cycle (per Strategy191 structure) and applies inverse
Fisher transform to enhance turning points. Signals on transformed cycle
crossing 0.
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


class Strategy041InverseFisherCyberCycle(BaseStrategy):
    def __init__(self, parameters: Dict=None):
        params = {
            'alpha': 0.07,
            'signal_threshold': 0.6,
        }
        if parameters:
            params.update(parameters)
        super().__init__('Strategy_041_InverseFisher_CyberCycle', params)

    def _cyber_cycle(self, src: pd.Series, alpha: float) -> pd.Series:
        smooth = (src + 2*src.shift(1) + 2*src.shift(2) + src.shift(3)) / 6.0
        cycle = pd.Series(index=src.index, dtype=float)
        for i in range(len(src)):
            if i < 7:
                cycle.iloc[i] = (src.iloc[i] - 2*src.shift(1).fillna(src.iloc[i]).iloc[i] + src.shift(2).fillna(src.iloc[i]).iloc[i]) / 4.0
            else:
                cycle.iloc[i] = (1-0.5*alpha)*(1-0.5*alpha)*(smooth.iloc[i]-2*smooth.shift(1).iloc[i]+smooth.shift(2).iloc[i]) + 2*(1-alpha)*cycle.iloc[i-1] - (1-alpha)*(1-alpha)*cycle.iloc[i-2]
        return cycle.fillna(0)

    @staticmethod
    def _invfisher(x: pd.Series) -> pd.Series:
        # normalize to roughly [-1,1] domain, then apply inverse Fisher
        xn = x - x.rolling(20, min_periods=5).mean()
        sd = x.rolling(20, min_periods=5).std(ddof=0).replace(0,np.nan)
        z = (xn / (2*sd)).clip(-2,2).fillna(0)
        return (np.exp(2*z) - 1) / (np.exp(2*z) + 1)

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        if 'high' in data and 'low' in data:
            src = (data['high'] + data['low'])/2.0
        else:
            if 'close' not in data and 'price' in data:
                data['close']=data['price']
            src = data['close']
        alpha = float(self.parameters['alpha'])
        cc = self._cyber_cycle(src.astype(float), alpha)
        data['if_cc'] = self._invfisher(cc)
        return data

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        buy = crossover(data['if_cc'], 0.0)
        sell = crossunder(data['if_cc'], 0.0)
        data['buy_signal'] = buy
        data['sell_signal'] = sell
        st = data['if_cc'].abs().clip(0,1).fillna(0)
        st = st.where(buy|sell, 0.0)
        thr=float(self.parameters['signal_threshold'])
        st[(buy|sell) & (st < thr)] = thr
        data['signal_strength'] = st
        return data

