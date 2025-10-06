#!/usr/bin/env python3
"""
Strategy 030: Adaptive RSI

TradingView URL: https://www.tradingview.com/v/wZGOIz9r/
Type: adaptive/momentum

Description:
RSI with period adapted to recent volatility (ATR proxy via std of returns).
Period is clamped between min/max. Signals on RSI cross 30/70 with trend filter.
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
    from utils.vectorized_helpers import crossover, crossunder, calculate_signal_strength, apply_position_constraints, pine_rma
except Exception:
    def crossover(a,b): return (a>b)&(a.shift(1)<=b.shift(1))
    def crossunder(a,b): return (a<b)&(a.shift(1)>=b.shift(1))
    def calculate_signal_strength(fs,weights=None):
        df=pd.concat(fs,axis=1); return df.mean(axis=1).clip(0,1)
    def apply_position_constraints(b,s,allow_short=False): return b,s
    def pine_rma(s, n): return s.ewm(alpha=1/n, adjust=False, min_periods=1).mean()


class Strategy030AdaptiveRSI(BaseStrategy):
    def __init__(self, parameters: Dict=None):
        params={'min_period':7,'max_period':28,'vol_window':20,'overbought':70,'oversold':30,'signal_threshold':0.6,'trend_ma':50}
        if parameters: params.update(parameters)
        super().__init__('Strategy_030_AdaptiveRSI', params)

    def calculate_indicators(self, data:pd.DataFrame)->pd.DataFrame:
        if 'close' not in data.columns and 'price' in data.columns:
            data['close']=data['price']
        # Volatility proxy
        ret=data['close'].pct_change()
        vol=ret.rolling(self.parameters['vol_window'],min_periods=1).std(ddof=0)
        # Map vol to period range inversely (higher vol -> shorter period)
        v=(vol/vol.rolling(self.parameters['vol_window'],min_periods=1).max()).fillna(0)
        dyn_period=(self.parameters['max_period']-(self.parameters['max_period']-self.parameters['min_period'])*v).round().clip(self.parameters['min_period'], self.parameters['max_period'])
        # Compute RSI with rolling dynamic period (approximate by recomputing per common period set)
        # Precompute RSIs for each possible period
        periods=range(self.parameters['min_period'], self.parameters['max_period']+1)
        rsi_map={}
        delta=data['close'].diff(); gain=delta.clip(lower=0); loss=-delta.clip(upper=0)
        for n in periods:
            rs=pine_rma(gain, n)/pine_rma(loss, n).replace(0,np.nan)
            rsi=100-(100/(1+rs))
            rsi_map[n]=rsi
        # Select per-row
        rsi_series=pd.Series(index=data.index, dtype=float)
        for i,(idx,n) in enumerate(zip(data.index, dyn_period.astype(int))):
            rsi_series.iloc[i]=rsi_map[int(n)].iloc[i]
        data['rsi_adapt']=rsi_series.fillna(50)
        data['trend_sma']=data['close'].rolling(self.parameters['trend_ma'],min_periods=1).mean()
        return data

    def generate_signals(self, data:pd.DataFrame)->pd.DataFrame:
        p=self.parameters
        data['buy_signal']=False; data['sell_signal']=False; data['signal_strength']=0.0
        up=crossover(data['rsi_adapt'], p['oversold']) & (data['close']>data['trend_sma'])
        dn=crossunder(data['rsi_adapt'], p['overbought']) & (data['close']<data['trend_sma'])
        data['buy_signal']=up; data['sell_signal']=dn
        # Strength by distance from midline (50) and vol factor
        dist=(data['rsi_adapt']-50).abs()/50
        vol=(data['close'].pct_change().rolling(20,min_periods=1).std(ddof=0)/0.05).clip(0,1)
        data['signal_strength']=calculate_signal_strength([dist,vol],[0.7,0.3])
        weak=data['signal_strength']<p['signal_threshold']
        data.loc[weak,['buy_signal','sell_signal']]=False
        return data

