#!/usr/bin/env python3
"""
Strategy 025: Forecast Oscillator (Chande)

TradingView URL: https://www.tradingview.com/v/CMSQGuGP/
Type: momentum/forecast

Description:
Forecast Oscillator = 100 * (Close - TSF(period)) / Close, where TSF is the
Time Series Forecast (rolling linear regression projection). Signals on zero
cross with optional trend filter.
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
    from utils.vectorized_helpers import crossover, crossunder, calculate_signal_strength, apply_position_constraints
except Exception:
    def crossover(a,b): return (a>b)&(a.shift(1)<=b.shift(1))
    def crossunder(a,b): return (a<b)&(a.shift(1)>=b.shift(1))
    def calculate_signal_strength(fs,weights=None):
        df=pd.concat(fs,axis=1); return df.mean(axis=1).clip(0,1)
    def apply_position_constraints(b,s,allow_short=False): return b,s


class Strategy025ForecastOscillator(BaseStrategy):
    def __init__(self, parameters: Dict=None):
        params={'period':14,'trend_ma':50,'signal_threshold':0.6}
        if parameters: params.update(parameters)
        super().__init__('Strategy_025_ForecastOscillator', params)

    def _tsf(self, s: pd.Series, n: int) -> pd.Series:
        # Rolling linear regression to last index within window
        idx = np.arange(n)
        def proj(x):
            if np.isnan(x).any():
                x = pd.Series(x).ffill().bfill().values
            y = x
            X = np.vstack([idx, np.ones(n)]).T
            # slope, intercept via least squares
            m, c = np.linalg.lstsq(X, y, rcond=None)[0]
            # forecast at last point (n-1)
            return m*(n-1)+c
        return s.rolling(n, min_periods=n).apply(proj, raw=True)

    def calculate_indicators(self, data:pd.DataFrame)->pd.DataFrame:
        if 'close' not in data.columns and 'price' in data.columns:
            data['close']=data['price']
        tsf=self._tsf(data['close'], self.parameters['period'])
        data['tsf']=tsf
        data['fo']=100.0*(data['close']-data['tsf'])/data['close']
        data['trend_sma']=data['close'].rolling(self.parameters['trend_ma'],min_periods=1).mean()
        return data

    def generate_signals(self, data:pd.DataFrame)->pd.DataFrame:
        p=self.parameters
        data['buy_signal']=False; data['sell_signal']=False; data['signal_strength']=0.0
        up=crossover(data['fo'],0) & (data['close']>data['trend_sma'])
        dn=crossunder(data['fo'],0) & (data['close']<data['trend_sma'])
        data['buy_signal']=up; data['sell_signal']=dn
        mag=(data['fo'].abs()/data['fo'].abs().rolling(50,min_periods=1).max().replace(0,np.nan)).fillna(0).clip(0,1)
        data['signal_strength']=calculate_signal_strength([mag],[1.0])
        weak=data['signal_strength']<p['signal_threshold']
        data.loc[weak,['buy_signal','sell_signal']]=False
        return data
