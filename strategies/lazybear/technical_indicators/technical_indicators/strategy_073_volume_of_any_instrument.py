#!/usr/bin/env python3
"""
Strategy 073: Volume of any Instrument (adapted)

Original concept references other instruments' volume. We adapt to the tracked token's volume by
normalizing volume against its own moving average and using threshold crosses.
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
    from utils.vectorized_helpers import crossover, crossunder
except Exception:
    def crossover(a,b): return (a>b)&(a.shift(1)<=b)
    def crossunder(a,b): return (a<b)&(a.shift(1)>=b)


class Strategy073VolumeAnyInstrument(BaseStrategy):
    def __init__(self, parameters: Dict=None):
        params={'length':20,'high_th':1.5,'low_th':0.5,'signal_threshold':0.6}
        if parameters: params.update(parameters)
        super().__init__('Strategy_073_VolumeAnyInstrument', params)
    def calculate_indicators(self, data:pd.DataFrame)->pd.DataFrame:
        if 'volume' not in data.columns: data['volume']=0
        n=int(self.parameters['length'])
        vol_ma=data['volume'].rolling(n, min_periods=1).mean().replace(0,np.nan)
        data['vol_ratio']=(data['volume']/vol_ma).replace([np.inf,-np.inf],np.nan).fillna(0)
        return data
    def generate_signals(self, data:pd.DataFrame)->pd.DataFrame:
        hi=float(self.parameters['high_th']); lo=float(self.parameters['low_th'])
        buy=crossover(data['vol_ratio'], hi)
        sell=crossunder(data['vol_ratio'], lo)
        data['buy_signal']=buy; data['sell_signal']=sell
        st=(data['vol_ratio']-1).abs()/2.0
        st=st.clip(0,1).where(buy|sell,0.0)
        thr=float(self.parameters['signal_threshold']); st[(buy|sell)&(st<thr)]=thr
        data['signal_strength']=st
        return data
