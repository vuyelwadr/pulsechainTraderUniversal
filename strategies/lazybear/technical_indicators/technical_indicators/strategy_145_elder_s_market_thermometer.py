#!/usr/bin/env python3
"""
Strategy 145: Elder’s Market Thermometer (Proxy)

Proxy: Normalize True Range by ATR to gauge "heat". High values imply
volatile (hot) market conditions.

Formula:
  TR = max(high-low, abs(high-close[1]), abs(low-close[1]))
  ATR = EMA(TR, atr_len)
  EMT = TR / ATR

Signals: EMT crosses above hot_threshold (sell/exit) or below cold_threshold (buy).
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


class Strategy145EldersMarketThermometerProxy(BaseStrategy):
    def __init__(self, parameters: Dict=None):
        params={'atr_len':14,'hot_threshold':1.5,'cold_threshold':0.8,'signal_threshold':0.4}
        if parameters: params.update(parameters)
        super().__init__('Strategy_145_Elders_Market_Thermometer_Proxy', params)

    @staticmethod
    def _tr(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        prev_close = close.shift(1)
        tr1 = (high - low).abs()
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        df=data.copy()
        for col in ('high','low','close'):
            if col not in df and 'price' in df and col=='close':
                df['close']=df['price']
        tr = self._tr(df['high'].astype(float), df['low'].astype(float), df['close'].astype(float))
        atr = tr.ewm(span=max(1,int(self.parameters['atr_len'])), adjust=False).mean().replace(0,np.nan)
        df['emt'] = (tr / atr).fillna(0.0)
        return df

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df=data.copy()
        hot=float(self.parameters['hot_threshold']); cold=float(self.parameters['cold_threshold'])
        buy = crossunder(df['emt'], cold)
        sell = crossover(df['emt'], hot)
        df['buy_signal']=buy
        df['sell_signal']=sell
        # Strength relative to band width
        width=max(1e-6, hot-cold)
        base=np.where(buy, cold, np.where(sell, hot, 0.0))
        dist=(df['emt']-base).abs()/width
        strength=pd.Series(dist, index=df.index).clip(0,1).fillna(0.0)
        thr=float(self.parameters['signal_threshold'])
        strength[(buy|sell) & (strength<thr)]=thr
        df['signal_strength']=strength.where(buy|sell, 0.0)
        return df

