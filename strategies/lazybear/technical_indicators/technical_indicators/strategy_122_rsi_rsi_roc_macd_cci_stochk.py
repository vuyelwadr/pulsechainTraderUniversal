#!/usr/bin/env python3
"""
Strategy 122: RSI++ (RSI+ROC+MACD+CCI+StochK)

Pine reference: pine_scripts/122_rsi_plus_plus.pine

Signals: buy when composite crosses above 0; sell when crosses below 0.
"""

import pandas as pd
import numpy as np
from typing import Dict
import os, sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)
from strategies.base_strategy import BaseStrategy

try:
    sys.path.insert(0, os.path.join(project_root, 'src'))
    from utils.vectorized_helpers import crossover, crossunder
except Exception:
    def crossover(a,b): return (a>b) & (a.shift(1)<=b)
    def crossunder(a,b): return (a<b) & (a.shift(1)>=b)


class Strategy122RSIPlusPlus(BaseStrategy):
    def __init__(self, parameters: Dict=None):
        params = {
            'lengthRSI': 25,
            'lengthROC': 25,
            'fastMACDLength': 10,
            'slowMACDLength': 21,
            'lengthCCI': 50,
            'lengthStoch': 14,
            'smoothK': 3,
            'signal_threshold': 0.6,
        }
        if parameters:
            params.update(parameters)
        super().__init__('Strategy_122_RSI_Plus_Plus', params)

    @staticmethod
    def _ema(s: pd.Series, n: int) -> pd.Series:
        return s.ewm(span=max(1,int(n)), adjust=False, min_periods=1).mean()

    @staticmethod
    def _sma(s: pd.Series, n: int) -> pd.Series:
        return s.rolling(max(1,int(n)), min_periods=1).mean()

    @staticmethod
    def _rsi(s: pd.Series, n: int) -> pd.Series:
        n = max(1,int(n))
        delta = s.diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        roll_up = up.ewm(alpha=1.0/n, adjust=False, min_periods=1).mean()
        roll_down = down.ewm(alpha=1.0/n, adjust=False, min_periods=1).mean()
        rs = roll_up / roll_down.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)

    @staticmethod
    def _roc(s: pd.Series, n: int) -> pd.Series:
        return s.pct_change(periods=max(1,int(n))).fillna(0)

    @staticmethod
    def _cci(src: pd.Series, high: pd.Series, low: pd.Series, n: int) -> pd.Series:
        tp = (high + low + src) / 3.0
        sma_tp = tp.rolling(max(1,int(n)), min_periods=1).mean()
        md = (tp - sma_tp).abs().rolling(max(1,int(n)), min_periods=1).mean()
        return (tp - sma_tp) / (0.015 * md.replace(0, np.nan))

    @staticmethod
    def _stoch(close: pd.Series, high: pd.Series, low: pd.Series, n: int) -> pd.Series:
        hh = high.rolling(max(1,int(n)), min_periods=1).max()
        ll = low.rolling(max(1,int(n)), min_periods=1).min()
        return 100.0 * (close - ll) / (hh - ll).replace(0, np.nan)

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        for c in ('high','low','close'):
            if c not in data:
                data[c] = data.get('close', pd.Series(index=data.index)).fillna(method='ffill')
        p = self.parameters
        rsi = self._rsi(data['close'], p['lengthRSI'])
        roc = self._roc(data['close'], p['lengthROC'])
        macd = self._ema(data['close'], p['fastMACDLength']) - self._ema(data['close'], p['slowMACDLength'])
        cci = self._cci(data['close'], data['high'], data['low'], p['lengthCCI'])
        stochk = self._sma(self._stoch(data['close'], data['high'], data['low'], p['lengthStoch']), p['smoothK'])
        data['acc'] = (rsi + roc + macd + cci + stochk) / 5.0
        return data

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        buy = crossover(data['acc'], 0.0)
        sell = crossunder(data['acc'], 0.0)
        data['buy_signal'] = buy
        data['sell_signal'] = sell
        st = (data['acc'].abs() / (data['acc'].rolling(50, min_periods=10).std(ddof=0).replace(0,np.nan))).clip(0,1).fillna(0)
        st = st.where(buy|sell, 0.0)
        thr=float(self.parameters['signal_threshold'])
        st[(buy|sell) & (st < thr)] = thr
        data['signal_strength'] = st
        return data

