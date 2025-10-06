#!/usr/bin/env python3
"""
Strategy 067: Elder's Force Index (EFI)

Pine reference: pine_scripts/067_elders_force_index.pine
efi = SMA(change(close) * volume, length); s = SMA(efi, lengthMA)
Signals on zero-line cross of smoothed EFI.
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


class Strategy067EldersForceIndex(BaseStrategy):
    def __init__(self, parameters: Dict = None):
        params = {
            'length': 13,
            'lengthMA': 8,
            'signal_threshold': 0.6,
        }
        if parameters:
            params.update(parameters)
        super().__init__('Strategy_067_EldersForceIndex', params)

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        for c in ('close','volume'):
            if c not in data.columns:
                if c=='close' and 'price' in data.columns:
                    data['close']=data['price']
                else:
                    data['volume']=0
        efi_raw = (data['close'].diff() * data['volume'])
        efi = efi_raw.rolling(int(self.parameters['length']), min_periods=1).mean()
        s = efi.rolling(int(self.parameters['lengthMA']), min_periods=1).mean()
        data['efi'] = efi
        data['efi_s'] = s
        return data

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        buy = crossover(data['efi_s'], 0.0)
        sell = crossunder(data['efi_s'], 0.0)
        data['buy_signal'] = buy
        data['sell_signal'] = sell
        st = data['efi_s'].abs()
        denom = data['efi_s'].rolling(50, min_periods=10).std().replace(0,np.nan)
        st = (st / denom).fillna(0).clip(0,1)
        st = st.where(buy|sell, 0.0)
        thr = float(self.parameters['signal_threshold'])
        st[(buy|sell) & (st < thr)] = thr
        data['signal_strength'] = st
        return data

