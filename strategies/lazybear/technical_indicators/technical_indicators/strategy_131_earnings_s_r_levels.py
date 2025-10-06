#!/usr/bin/env python3
"""
Strategy 131: Earnings S/R Levels (Single‑Instrument Proxy)

Original uses external earnings calendar to place S/R at earnings events.
Proxy: Detect local "event" bars using volume and move spikes, then set an
S/R level based on immediate surrounding bars, and carry it forward.

Signals: price crosses the active S/R level.
"""

from typing import Dict
import pandas as pd
import numpy as np
import os, sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)
from strategies.base_strategy import BaseStrategy

try:
    sys.path.insert(0, os.path.join(project_root,'src'))
    from utils.vectorized_helpers import crossover, crossunder
except Exception:
    def crossover(a,b): return (a>b) & (a.shift(1)<=b)
    def crossunder(a,b): return (a<b) & (a.shift(1)>=b)


class Strategy131EarningsSRLevelsProxy(BaseStrategy):
    def __init__(self, parameters: Dict=None):
        params={'vol_len':20,'vol_thr':1.8,'ret_len':1,'ret_thr':0.03,'mode':1,'signal_threshold':0.5}
        if parameters: params.update(parameters)
        super().__init__('Strategy_131_Earnings_SR_Proxy', params)

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        df=data.copy()
        for col in ('high','low','close'):
            if col not in df and 'price' in df and col=='close':
                df['close']=df['price']
        if 'volume' not in df:
            df['volume']=0.0
        vol_ma = df['volume'].astype(float).ewm(span=max(1,int(self.parameters['vol_len'])), adjust=False).mean().replace(0,np.nan)
        rv = (df['volume'] / vol_ma).fillna(0.0)
        ret = df['close'].astype(float).pct_change(int(self.parameters['ret_len'])).abs().fillna(0.0)
        event = (rv > float(self.parameters['vol_thr'])) | (ret > float(self.parameters['ret_thr']))

        # Compute SR level when event occurs
        # mode1: avg(low[-1], high[0]); mode2: avg(HL2[-1], HL2[0], close[-1])
        low_prev = df['low'].shift(1)
        high_now = df['high']
        hl2_prev = (df['high'].shift(1) + df['low'].shift(1))/2.0
        hl2_now = (df['high'] + df['low'])/2.0
        close_prev = df['close'].shift(1)
        ehl2_mode1 = np.where(event, (low_prev + high_now)/2.0, np.nan)
        ehl2_mode2 = np.where(event, (hl2_prev + hl2_now + close_prev)/3.0, np.nan)
        mode=int(self.parameters['mode'])
        sr = pd.Series(np.where(mode==1, ehl2_mode1, ehl2_mode2), index=df.index)
        # Defer SR activation by 1 bar to avoid lookahead
        sr = sr.shift(1)
        df['earn_sr'] = sr.ffill()
        return df

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df=data.copy()
        valid = df['earn_sr'].notna()
        buy = valid & crossover(df['close'], df['earn_sr'])
        sell = valid & crossunder(df['close'], df['earn_sr'])
        df['buy_signal']=buy
        df['sell_signal']=sell
        strength=((df['close']-df['earn_sr']).abs() / (df['close'].rolling(50, min_periods=10).std(ddof=0).replace(0,np.nan))).clip(0,1).fillna(0.0)
        thr=float(self.parameters['signal_threshold'])
        strength[(buy|sell) & (strength<thr)] = thr
        df['signal_strength']=strength.where(buy|sell, 0.0)
        return df

