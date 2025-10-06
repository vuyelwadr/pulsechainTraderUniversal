#!/usr/bin/env python3
"""
Strategy 068: Traders Dynamic Index (TDI)

Pine partial reference: pine_scripts/068_traders_dynamic_index_lazybear_partial.pine
Implements common TDI components:
- RSI of close (lengthrsi)
- RSI Price Line (RSI smoothed by EMA lengthrsipl)
- Trade Signal Line (EMA of price line, lengthtradesl)
- Volatility Bands on RSI (SMA lengthband with stdev bands)

Signals:
- Buy when price line crosses above signal line and RSI above its middle band
- Sell when price line crosses below signal line and RSI below middle band
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


class Strategy068TradersDynamicIndex(BaseStrategy):
    def __init__(self, parameters: Dict = None):
        params = {
            'lengthrsi': 13,
            'lengthband': 34,
            'lengthrsipl': 2,
            'lengthtradesl': 7,
            'bb_mult': 2.0,
            'signal_threshold': 0.6,
        }
        if parameters:
            params.update(parameters)
        super().__init__('Strategy_068_TDI', params)

    @staticmethod
    def _ema(s: pd.Series, n: int) -> pd.Series:
        return s.ewm(span=max(1,int(n)), adjust=False, min_periods=1).mean()

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        if 'close' not in data.columns and 'price' in data.columns:
            data['close']=data['price']
        # RSI via Pine RMA
        n = int(self.parameters['lengthrsi'])
        delta = data['close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_g = pine_rma(gain, n)
        avg_l = pine_rma(loss, n)
        rs = avg_g / avg_l.replace(0,np.nan)
        rsi = (100 - 100/(1+rs)).fillna(50)
        # Price line and Trade signal line
        rsi_price = self._ema(rsi, int(self.parameters['lengthrsipl']))
        trade_sig = self._ema(rsi_price, int(self.parameters['lengthtradesl']))
        # Bands
        basis = rsi.rolling(int(self.parameters['lengthband']), min_periods=1).mean()
        dev = rsi.rolling(int(self.parameters['lengthband']), min_periods=1).std(ddof=0) * float(self.parameters['bb_mult'])
        upper = basis + dev
        lower = basis - dev
        data['rsi'] = rsi
        data['tdi_price'] = rsi_price
        data['tdi_signal'] = trade_sig
        data['tdi_basis'] = basis
        data['tdi_upper'] = upper
        data['tdi_lower'] = lower
        return data

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        price_above_sig = crossover(data['tdi_price'], data['tdi_signal'])
        price_below_sig = crossunder(data['tdi_price'], data['tdi_signal'])
        buy = price_above_sig & (data['rsi'] > data['tdi_basis'])
        sell = price_below_sig & (data['rsi'] < data['tdi_basis'])
        data['buy_signal'] = buy
        data['sell_signal'] = sell
        # Strength by distance between price and signal normalized to band width
        width = (data['tdi_upper'] - data['tdi_lower']).replace(0,np.nan)
        gap = (data['tdi_price'] - data['tdi_signal']).abs() / width
        st = gap.clip(0,1).where(buy|sell, 0.0)
        thr = float(self.parameters['signal_threshold'])
        st[(buy|sell) & (st < thr)] = thr
        data['signal_strength'] = st.clip(0,1)
        return data

