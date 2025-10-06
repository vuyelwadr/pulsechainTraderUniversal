#!/usr/bin/env python3
"""
Strategy 015: 8x MA with Cross-highlights

TradingView URL: https://www.tradingview.com/v/np0rmAMB/
Type: trend/momentum

Description:
Compute 8 moving averages and highlight crossovers. Signals:
 - Buy when short MA crosses above long MA and majority of MAs are aligned up
 - Sell when short MA crosses below long MA and alignment down

This is a faithful, vectorized approximation suitable for tuning.
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
    from utils.vectorized_helpers import crossover, crossunder, calculate_signal_strength, apply_position_constraints
except Exception:
    def crossover(a,b): return (a>b)&(a.shift(1)<=b.shift(1))
    def crossunder(a,b): return (a<b)&(a.shift(1)>=b.shift(1))
    def calculate_signal_strength(fs, weights=None):
        df=pd.concat(fs,axis=1); return df.mean(axis=1).clip(0,1)
    def apply_position_constraints(b,s,allow_short=False): return b,s


class Strategy015EightMA(BaseStrategy):
    """8-EMA stack with majority alignment + crossover triggers."""

    def __init__(self, parameters: Dict = None):
        default_params = {
            'periods': [5, 8, 13, 21, 34, 55, 89, 144],
            'fast_idx': 0,
            'slow_idx': 5,
            'signal_threshold': 0.6,
            'alignment_min': 5,  # minimum count of MAs trending up/down
        }
        if parameters:
            default_params.update(parameters)
        super().__init__("Strategy_015_EightMA", default_params)

    @staticmethod
    def _ema(s: pd.Series, n: int) -> pd.Series:
        return s.ewm(span=n, adjust=False, min_periods=1).mean()

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        if 'close' not in data.columns and 'price' in data.columns:
            data['close']=data['price']
        periods=self.parameters['periods']
        for i,p in enumerate(periods):
            data[f'ema_{p}']=self._ema(data['close'], p)
        return data

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        p=self.parameters
        data['buy_signal']=False; data['sell_signal']=False; data['signal_strength']=0.0
        fast=data[f'ema_{p["periods"][p["fast_idx"]]}']
        slow=data[f'ema_{p["periods"][p["slow_idx"]]}']
        cu=crossover(fast, slow)
        cd=crossunder(fast, slow)
        # Alignment
        ma_cols=[f'ema_{pp}' for pp in p['periods']]
        aligned_up=(data[ma_cols].diff()>0).sum(axis=1)>=p['alignment_min']
        aligned_dn=(data[ma_cols].diff()<0).sum(axis=1)>=p['alignment_min']
        data['buy_signal']=(cu & aligned_up)
        data['sell_signal']=(cd & aligned_dn)
        # Strength
        slope=(fast - fast.shift(3)).abs()
        div=(fast - slow).abs()
        slope_norm=(slope/(data['close'].rolling(20,min_periods=1).std(ddof=0)+1e-9)).clip(0,1)
        div_norm=(div/(data['close'].rolling(50,min_periods=1).std(ddof=0)+1e-9)).clip(0,1)
        data['signal_strength']=calculate_signal_strength([slope_norm,div_norm],[0.5,0.5])
        weak=data['signal_strength']<self.parameters['signal_threshold']
        data.loc[weak,['buy_signal','sell_signal']]=False
        data['buy_signal'],data['sell_signal']=apply_position_constraints(data['buy_signal'],data['sell_signal'])
        return data

