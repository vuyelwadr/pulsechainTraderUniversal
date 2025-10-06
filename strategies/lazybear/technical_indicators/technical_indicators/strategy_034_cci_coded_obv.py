#!/usr/bin/env python3
"""
Strategy 034: CCI coded OBV

TradingView URL: https://www.tradingview.com/v/D8ld7sgR/
Type: volume/momentum

Description:
On-Balance Volume (OBV) modulated by CCI state. Signals when CCI crosses
thresholds and OBV trend aligns.
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


class Strategy034CCICodedOBV(BaseStrategy):
    def __init__(self, parameters: Dict=None):
        params={'cci_period':20,'obv_ma':20,'upper':100,'lower':-100,'signal_threshold':0.6}
        if parameters: params.update(parameters)
        super().__init__('Strategy_034_CCI_Coded_OBV', params)

    def calculate_indicators(self, data:pd.DataFrame)->pd.DataFrame:
        for c in ('high','low','close','volume'):
            if c not in data.columns and 'price' in data.columns:
                if c=='volume': data[c]=0
                else: data[c]=data['price']
        # CCI
        tp=(data['high']+data['low']+data['close'])/3.0
        sma=tp.rolling(self.parameters['cci_period'],min_periods=1).mean()
        md=(tp-sma).abs().rolling(self.parameters['cci_period'],min_periods=1).mean()
        data['cci']=(tp-sma)/(0.015*md.replace(0,np.nan))
        # OBV
        direction=np.sign(data['close'].diff()).fillna(0)
        data['obv']=(direction*data['volume']).cumsum()
        data['obv_ma']=data['obv'].rolling(self.parameters['obv_ma'],min_periods=1).mean()
        return data

    def generate_signals(self, data:pd.DataFrame)->pd.DataFrame:
        p=self.parameters
        data['buy_signal']=False; data['sell_signal']=False; data['signal_strength']=0.0
        # CCI crosses
        cci_up=crossover(data['cci'], p['lower'])
        cci_dn=crossunder(data['cci'], p['upper'])
        # OBV trend
        obv_up=data['obv']>data['obv_ma']
        obv_dn=data['obv']<data['obv_ma']
        data['buy_signal']=cci_up & obv_up
        data['sell_signal']=cci_dn & obv_dn
        # Strength: normalized CCI distance and OBV slope
        cci_mag=(data['cci'].abs()/300.0).clip(0,1)
        obv_slope=((data['obv']-data['obv'].shift(5)).abs()/(data['obv'].abs().rolling(100,min_periods=1).max()+1e-9)).clip(0,1)
        data['signal_strength']=calculate_signal_strength([cci_mag, obv_slope],[0.6,0.4])
        weak=data['signal_strength']<p['signal_threshold']
        data.loc[weak,['buy_signal','sell_signal']]=False
        data['buy_signal'],data['sell_signal']=apply_position_constraints(data['buy_signal'],data['sell_signal'])
        return data

