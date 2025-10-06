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

class Strategy115RSIBands(BaseStrategy):
    def __init__(self, parameters: Dict=None):
        params={'obLevel':70,'osLevel':30,'length':14,'signal_threshold':0.6}
        if parameters: params.update(parameters)
        super().__init__('Strategy_115_RSI_Bands', params)
    @staticmethod
    def _ema(s:pd.Series,n:int)->pd.Series:
        return s.ewm(span=max(1,int(n)), adjust=False, min_periods=1).mean()
    def calculate_indicators(self, data:pd.DataFrame)->pd.DataFrame:
        if 'close' not in data.columns and 'price' in data.columns: data['close']=data['price']
        ep=int(2*int(self.parameters['length'])-1)
        auc=self._ema((data['close']-data['close'].shift(1)).clip(lower=0), ep)
        adc=self._ema((data['close'].shift(1)-data['close']).clip(lower=0), ep)
        ob=float(self.parameters['obLevel']); os_=float(self.parameters['osLevel']); L=int(self.parameters['length'])
        x1=(L-1)*(adc*ob/(100-ob) - auc)
        ub=np.where(x1>=0, data['close']+x1, data['close'] + x1*(100-ob)/ob)
        x2=(L-1)*(adc*os_/(100-os_) - auc)
        lb=np.where(x2>=0, data['close']+x2, data['close'] + x2*(100-os_)/os_)
        data['rsi_bands_ub']=pd.Series(ub,index=data.index)
        data['rsi_bands_lb']=pd.Series(lb,index=data.index)
        data['rsi_bands_mid']=(data['rsi_bands_ub']+data['rsi_bands_lb'])/2.0
        return data
    def generate_signals(self, data:pd.DataFrame)->pd.DataFrame:
        buy=crossover(data['close'], data['rsi_bands_ub'])
        sell=crossunder(data['close'], data['rsi_bands_lb'])
        data['buy_signal']=buy; data['sell_signal']=sell
        width=(data['rsi_bands_ub']-data['rsi_bands_lb']).replace(0,np.nan)
        st=pd.Series(0.0,index=data.index)
        st[buy]=((data['close']-data['rsi_bands_ub'])/width).clip(0,1)[buy]
        st[sell]=((data['rsi_bands_lb']-data['close'])/width).clip(0,1)[sell]
        thr=float(self.parameters['signal_threshold']); st[(buy|sell)&(st<thr)]=thr
        st[~(buy|sell)]=0.0
        data['signal_strength']=st
        return data

