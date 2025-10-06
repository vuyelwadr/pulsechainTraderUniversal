#!/usr/bin/env python3
"""
Strategy 143: Short-term Volume And Price Oscillator (SVAPO)

Pine reference: pine_scripts/143_short_term_vol_price.pine

Signals: buy when SVAPO crosses above +devH*stdev(SVAPO, stdevper);
sell when SVAPO crosses below -devL*stdev(SVAPO, stdevper).
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


class Strategy143ShortTermVolumePriceOscillator(BaseStrategy):
    def __init__(self, parameters: Dict=None):
        params = {
            'length': 8,
            'cutoff': 1.0,
            'devH': 1.5,
            'devL': 1.3,
            'stdevper': 100,
            'signal_threshold': 0.6,
        }
        if parameters:
            params.update(parameters)
        super().__init__('Strategy_143_SVAPO', params)

    @staticmethod
    def _ema(s: pd.Series, n: int) -> pd.Series:
        return s.ewm(span=max(1,int(n)), adjust=False, min_periods=1).mean()

    def _tema(self, s: pd.Series, n: int) -> pd.Series:
        e1 = self._ema(s, n)
        e2 = self._ema(e1, n)
        e3 = self._ema(e2, n)
        return 3*(e1 - e2) + e3

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        for c in ('open','high','low','close','volume'):
            if c not in data:
                data[c] = 0 if c=='volume' else data.get('close', pd.Series(index=data.index)).fillna(method='ffill')
        p = self.parameters
        ohlc4 = data[['open','high','low','close']].mean(axis=1)
        ha_open = (ohlc4.shift(1) + ohlc4.shift(1)) / 2.0
        ha_close = (ohlc4 + ha_open + data[['high', 'low']].max(axis=1) + data[['high','low']].min(axis=1)) / 4.0
        vola = self._tema(ha_close, int(p['length'])) * 100.0
        vb = self._tema(data['volume'], int(p['length']))
        vave = vb.rolling(int(p['length']), min_periods=1).mean()
        vmax = vb.rolling(int(p['length']*3), min_periods=1).max()
        vmin = vb.rolling(int(p['length']*3), min_periods=1).min()
        vola2 = self._tema(vb, int(p['length']))
        nr = (vola2 - vave) / (vmax - vmin).replace(0, np.nan) * 100.0
        pct = (ha_close / ha_close.shift(1) - 1.0).abs() * 100.0
        r1 = (pct >= float(p['cutoff'])) | (pct.shift(1).fillna(False))
        svapo = pd.Series(0.0, index=data.index)
        svapo[r1] = nr[r1]
        svapo[~r1] = svapo.shift(1)[~r1]
        data['svapo'] = svapo.fillna(0)
        stdev = data['svapo'].rolling(int(p['stdevper']), min_periods=10).std(ddof=0)
        data['svapo_up'] = float(p['devH']) * stdev
        data['svapo_dn'] = -float(p['devL']) * stdev
        return data

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        buy = crossover(data['svapo'], data['svapo_up'])
        sell = crossunder(data['svapo'], data['svapo_dn'])
        data['buy_signal'] = buy
        data['sell_signal'] = sell
        st = (data['svapo'].abs() / (data['svapo'].rolling(50, min_periods=10).std(ddof=0).replace(0,np.nan))).clip(0,1).fillna(0)
        st = st.where(buy|sell, 0.0)
        thr=float(self.parameters['signal_threshold'])
        st[(buy|sell) & (st < thr)] = thr
        data['signal_strength'] = st
        return data

