#!/usr/bin/env python3
import pandas as pd
import numpy as np
from typing import Dict
import sys, os

project_root=os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)
from strategies.base_strategy import BaseStrategy
try:
    sys.path.insert(0, os.path.join(project_root,'src'))
    from utils.vectorized_helpers import crossover, crossunder, calculate_signal_strength, pine_rma
except Exception:
    def crossover(a,b): return (a>b) & (a.shift(1)<=b.shift(1))
    def crossunder(a,b): return (a<b) & (a.shift(1)>=b.shift(1))
    def calculate_signal_strength(fs,weights=None):
        import pandas as pd
        df=pd.concat(fs,axis=1); return df.mean(axis=1).clip(0,1)
    def pine_rma(s,n): return s.ewm(alpha=1/max(1,int(n)), adjust=False, min_periods=1).mean()


class Strategy039InverseFisherRSI(BaseStrategy):
    def __init__(self, parameters: Dict=None):
        params={'rsi_period':14,'signal_threshold':0.6}
        if parameters: params.update(parameters)
        super().__init__('Strategy_039_InverseFisher_RSI', params)

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        if 'close' not in data.columns and 'price' in data.columns:
            data['close']=data['price']
        # RSI via Pine RMA
        delta=data['close'].diff()
        gain=delta.clip(lower=0); loss=-delta.clip(upper=0)
        avg_g=pine_rma(gain, int(self.parameters['rsi_period']))
        avg_l=pine_rma(loss, int(self.parameters['rsi_period']))
        rs=avg_g/avg_l.replace(0,np.nan)
        rsi=100 - 100/(1+rs)
        # Normalize to -1..1 then transform: inv_fisher(x)= (exp(2x)-1)/(exp(2x)+1)
        v = (rsi - 50)/50.0
        t = (np.exp(2*v)-1)/(np.exp(2*v)+1)
        data['if_rsi']=pd.Series(t, index=data.index)
        return data

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        # Cross 0 as signal
        buy=crossover(data['if_rsi'], 0)
        sell=crossunder(data['if_rsi'], 0)
        data['buy_signal']=buy
        data['sell_signal']=sell
        st=(data['if_rsi'].abs()).clip(0,1)
        st=st.mask(~(buy|sell),0.0).mask((buy|sell)&(st<self.parameters['signal_threshold']), self.parameters['signal_threshold'])
        data['signal_strength']=st
        return data

