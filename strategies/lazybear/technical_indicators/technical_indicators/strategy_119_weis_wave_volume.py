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
    from utils.vectorized_helpers import crossover, crossunder
except Exception:
    def crossover(a,b): return (a>b)&(a.shift(1)<=b)
    def crossunder(a,b): return (a<b)&(a.shift(1)>=b)

class Strategy119WeisWaveVolume(BaseStrategy):
    def __init__(self, parameters: Dict=None):
        params={'trend_len':2,'signal_threshold':0.6}
        if parameters: params.update(parameters)
        super().__init__('Strategy_119_WeisWaveVolume', params)
    def calculate_indicators(self, data:pd.DataFrame)->pd.DataFrame:
        for c in ('open','high','low','close','volume'):
            if c not in data.columns:
                if c=='volume': data[c]=0
                elif 'price' in data.columns: data[c]=data['price']
        mov = np.where(data['close']>data['close'].shift(1),1, np.where(data['close']<data['close'].shift(1),-1,0))
        mov = pd.Series(mov,index=data.index)
        # Simple wave flip on direction change sustained by trend_len
        tlen=int(self.parameters['trend_len'])
        rising = data['close'] > data['close'].shift(tlen)
        falling = data['close'] < data['close'].shift(tlen)
        trend = np.where(rising,1, np.where(falling,-1,0))
        trend = pd.Series(trend,index=data.index).replace(0, np.nan).ffill().fillna(0)
        wave = trend
        vol = pd.Series(0.0, index=data.index)
        vol = wave.groupby((wave != wave.shift()).cumsum()).apply(lambda g: data.loc[g.index,'volume'].cumsum())
        up = np.where(wave==1, vol, 0.0)
        dn = np.where(wave==-1, vol, 0.0)
        data['wwv_up']=pd.Series(up,index=data.index)
        data['wwv_dn']=pd.Series(dn,index=data.index)
        data['wwv_diff']=data['wwv_up'] - data['wwv_dn']
        return data
    def generate_signals(self, data:pd.DataFrame)->pd.DataFrame:
        buy=crossover(data['wwv_diff'], 0.0)
        sell=crossunder(data['wwv_diff'], 0.0)
        data['buy_signal']=buy; data['sell_signal']=sell
        st=(data['wwv_diff'].abs() / (data['wwv_diff'].rolling(50, min_periods=10).std(ddof=0).replace(0,np.nan))).fillna(0).clip(0,1)
        st=st.where(buy|sell,0.0)
        thr=float(self.parameters['signal_threshold']); st[(buy|sell)&(st<thr)]=thr
        data['signal_strength']=st
        return data

