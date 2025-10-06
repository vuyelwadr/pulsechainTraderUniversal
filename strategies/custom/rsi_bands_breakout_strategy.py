"""
RSI Bands Breakout Strategy

Breakout with RSI band context; trend filter by EMA.
"""
from typing import Dict
import numpy as np
import pandas as pd
from strategies.base_strategy import BaseStrategy


def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=max(2, int(n)), adjust=False).mean()


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    dn = -delta.clip(upper=0.0)
    au = up.rolling(period).mean()
    ad = dn.rolling(period).mean().replace(0, np.nan)
    rs = au / ad
    return (100 - 100/(1+rs)).fillna(50.0)


class RSIBandsBreakoutStrategy(BaseStrategy):
    def __init__(self, parameters: Dict = None):
        p = {
            'ema_len': 100,
            'rsi_period': 14,
            'rsi_band_low': 40,
            'rsi_band_high': 60,
            'break_confirm': 5,
            'signal_threshold': 0.5,
            'timeframe_minutes': 60,
        }
        if parameters:
            p.update(parameters)
        super().__init__('RSIBandsBreakoutStrategy', p)

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        d = data.copy()
        price = d.get('price', d.get('close', d['close']))
        d['ema'] = _ema(price, int(self.parameters['ema_len']))
        d['rsi'] = _rsi(price, int(self.parameters['rsi_period']))
        d['hi'] = d['high'].rolling(int(self.parameters['break_confirm'])).max()
        d['lo'] = d['low'].rolling(int(self.parameters['break_confirm'])).min()
        return d

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        d = self.calculate_indicators(data)
        price = d.get('price', d.get('close', d['close']))
        rsi = d['rsi']
        buy = (price > d['hi']) & (rsi >= float(self.parameters['rsi_band_high'])) & (price > d['ema'])
        sell = (price < d['lo']) & (rsi <= float(self.parameters['rsi_band_low'])) & (price < d['ema'])
        dist = (price - d['ema']).abs() / (price.abs() + 1e-9)
        st = dist.clip(0, 1.0)
        thr = float(self.parameters['signal_threshold'])
        st = st.where(buy | sell, 0.0)
        st[(buy | sell) & (st < thr)] = thr
        d['buy_signal'] = buy.fillna(False)
        d['sell_signal'] = sell.fillna(False)
        d['signal_strength'] = st.fillna(0.0)
        return d

