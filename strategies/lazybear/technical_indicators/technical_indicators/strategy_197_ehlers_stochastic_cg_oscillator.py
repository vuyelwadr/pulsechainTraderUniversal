#!/usr/bin/env python3
"""
Strategy 197: Ehlers Stochastic CG Oscillator (Proxy)

Build on Ehlers COG (Strategy 193). Compute stochastic normalization of COG
over a lookback window and smooth to get %K and %D.

Signals: %K crosses %D; optional zero/midline cross.
"""

from typing import Dict
import pandas as pd
import numpy as np
import os, sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)
from strategies.base_strategy import BaseStrategy

try:
    sys.path.insert(0, os.path.join(project_root, 'src'))
    from utils.vectorized_helpers import crossover, crossunder, highest, lowest
except Exception:
    def crossover(a,b): return (a>b) & (a.shift(1)<=b)
    def crossunder(a,b): return (a<b) & (a.shift(1)>=b)
    def highest(s,n): return s.rolling(n, min_periods=1).max()
    def lowest(s,n): return s.rolling(n, min_periods=1).min()


class Strategy197EhlersStochasticCGProxy(BaseStrategy):
    def __init__(self, parameters: Dict=None):
        params={'cog_len':10,'stoch_len':20,'smooth_k':3,'smooth_d':3,'signal_threshold':0.5}
        if parameters: params.update(parameters)
        super().__init__('Strategy_197_Ehlers_Stochastic_CG_Proxy', params)

    def _cog(self, src: pd.Series, length: int) -> pd.Series:
        n=max(1,int(length))
        def _f(window):
            w = window.astype(float)
            s = w.sum()
            if s == 0:
                return 0.0
            idx = np.arange(1, len(w)+1, dtype=float)
            return np.dot(w[::-1], idx) / s
        return src.rolling(n, min_periods=1).apply(_f, raw=True)

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        df=data.copy()
        if 'close' not in df and 'price' in df: df['close']=df['price']
        src = (df['high']+df['low'])/2.0 if 'high' in df and 'low' in df else df['close']
        cog = self._cog(src.astype(float), int(self.parameters['cog_len']))
        # Detrend cog to oscillate around 0
        cog = (cog - cog.rolling(int(self.parameters['cog_len']), min_periods=1).mean()).fillna(0.0)
        lb=int(self.parameters['stoch_len'])
        hi = highest(cog, lb)
        lo = lowest(cog, lb)
        width = (hi - lo).replace(0, np.nan)
        k = ((cog - lo) / width).clip(0,1).fillna(0.5)
        k_s = k.ewm(span=max(1,int(self.parameters['smooth_k'])), adjust=False).mean()
        d_s = k_s.ewm(span=max(1,int(self.parameters['smooth_d'])), adjust=False).mean()
        df['stoch_cg_k']=k_s
        df['stoch_cg_d']=d_s
        return df

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df=data.copy()
        buy=crossover(df['stoch_cg_k'], df['stoch_cg_d'])
        sell=crossunder(df['stoch_cg_k'], df['stoch_cg_d'])
        df['buy_signal']=buy; df['sell_signal']=sell
        strength=(df['stoch_cg_k']-df['stoch_cg_d']).abs().clip(0,1).fillna(0.0)
        thr=float(self.parameters['signal_threshold'])
        strength[(buy|sell) & (strength<thr)]=thr
        df['signal_strength']=strength.where(buy|sell, 0.0)
        return df

