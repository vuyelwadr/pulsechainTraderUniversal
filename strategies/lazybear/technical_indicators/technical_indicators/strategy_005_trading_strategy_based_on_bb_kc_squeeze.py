#!/usr/bin/env python3
"""
Strategy 005: Trading Strategy based on BB/KC Squeeze

TradingView URL: https://www.tradingview.com/v/x9r2dOhI/
Type: squeeze/momentum

Derived trading rule:
- Squeeze ON when BB width < KC width; OFF otherwise.
- Use a simple momentum proxy (close - SMA) to determine direction.
- Buy when squeeze turns ON and momentum crosses > 0; Sell when squeeze turns ON and momentum crosses < 0.
"""

import pandas as pd
import numpy as np
from typing import Dict
import sys, os

project_root=os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)
from strategies.base_strategy import BaseStrategy

try:
    sys.path.insert(0, os.path.join(project_root,'src'))
    from utils.vectorized_helpers import crossover, crossunder, calculate_signal_strength
except Exception:
    def crossover(a,b): return (a>b) & (a.shift(1)<=b.shift(1))
    def crossunder(a,b): return (a<b) & (a.shift(1)>=b.shift(1))
    def calculate_signal_strength(fs,weights=None):
        import pandas as pd
        df=pd.concat(fs,axis=1); return df.mean(axis=1).clip(0,1)


class Strategy005BollingerKeltnerSqueeze(BaseStrategy):
    def __init__(self, parameters: Dict=None):
        params={
            'bb_length':20,'bb_mult':2.0,
            'kc_length':20,'kc_mult':1.5,
            'mom_length':12,
            'signal_threshold':0.6,
        }
        if parameters: params.update(parameters)
        super().__init__('Strategy_005_BB_KC_Squeeze', params)

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        if 'close' not in data.columns and 'price' in data.columns:
            data['close']=data['price']
        c=data['close']
        # Bollinger Bands
        bb_len=int(self.parameters['bb_length']); bb_mult=float(self.parameters['bb_mult'])
        sma=c.rolling(bb_len, min_periods=1).mean()
        std=c.rolling(bb_len, min_periods=1).std(ddof=0)
        data['bb_up']=sma+bb_mult*std; data['bb_dn']=sma-bb_mult*std
        # Keltner (use EMA approx for mid)
        kc_len=int(self.parameters['kc_length']); kc_mult=float(self.parameters['kc_mult'])
        mid=c.ewm(span=kc_len, adjust=False, min_periods=1).mean()
        tr=pd.concat([(data['high']-data['low']), (data['high']-c.shift(1)).abs(), (data['low']-c.shift(1)).abs()],axis=1).max(axis=1)
        atr=tr.rolling(kc_len, min_periods=1).mean()
        data['kc_up']=mid+kc_mult*atr; data['kc_dn']=mid-kc_mult*atr
        # Widths
        data['bb_width']=data['bb_up']-data['bb_dn']
        data['kc_width']=data['kc_up']-data['kc_dn']
        # Momentum proxy
        mlen=int(self.parameters['mom_length'])
        data['mom']=c - c.rolling(mlen, min_periods=1).mean()
        return data

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        squeeze_on = data['bb_width'] < data['kc_width']
        # Trigger only on transition to squeeze
        squeeze_trigger = (~squeeze_on.shift(1).fillna(False)) & squeeze_on
        buy = squeeze_trigger & crossover(data['mom'], 0)
        sell = squeeze_trigger & crossunder(data['mom'], 0)
        data['buy_signal']=buy
        data['sell_signal']=sell
        # Strength: normalized momentum magnitude relative to std
        mom_std = data['mom'].rolling(50, min_periods=1).std(ddof=0) + 1e-9
        st = (data['mom'].abs() / mom_std).clip(0,1)
        st = st.mask(~(buy|sell), 0.0).mask((buy|sell)&(st<self.parameters['signal_threshold']), self.parameters['signal_threshold'])
        data['signal_strength']=st.fillna(0.0)
        return data

