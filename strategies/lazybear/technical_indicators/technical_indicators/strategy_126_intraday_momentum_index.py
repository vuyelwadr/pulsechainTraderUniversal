#!/usr/bin/env python3
"""
Strategy 126: Intraday Momentum Index (IMI)

Definition (canonical):
  For window N:
    up = sum( close-open for bars where close>open )
    dn = sum( open-close for bars where close<open )
    IMI = 100 * up / (up + dn)

We implement a vectorized approximation using separated positive/negative parts
and rolling sums. Signals when IMI crosses overbought/oversold thresholds.
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


class Strategy126IntradayMomentumIndex(BaseStrategy):
    def __init__(self, parameters: Dict=None):
        params={'length':14, 'obLevel':70.0, 'osLevel':30.0, 'signal_threshold':0.5}
        if parameters: params.update(parameters)
        super().__init__('Strategy_126_Intraday_Momentum_Index', params)

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        df=data.copy()
        # Ensure OHLC
        if 'close' not in df and 'price' in df: df['close']=df['price']
        for col in ('open','high','low'):
            if col not in df: df[col]=df.get('close', pd.Series(index=df.index)).ffill()
        n=int(self.parameters['length'])
        diff = (df['close'].astype(float) - df['open'].astype(float))
        pos = diff.clip(lower=0.0)
        neg = (-diff).clip(lower=0.0)
        up = pos.rolling(n, min_periods=1).sum()
        dn = neg.rolling(n, min_periods=1).sum()
        denom = (up + dn).replace(0, np.nan)
        imi = 100.0 * (up / denom)
        df['imi'] = imi.fillna(50.0)
        return df

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df=data.copy()
        ob=float(self.parameters['obLevel']); os_=float(self.parameters['osLevel'])
        buy=crossover(df['imi'], os_)
        sell=crossunder(df['imi'], ob)
        df['buy_signal']=buy
        df['sell_signal']=sell
        # Strength: distance from 50 normalized by 50
        strength=((df['imi']-50.0).abs()/50.0).clip(0,1).fillna(0.0)
        thr=float(self.parameters['signal_threshold'])
        strength[(buy|sell) & (strength<thr)]=thr
        df['signal_strength']=strength.where(buy|sell, 0.0)
        return df

