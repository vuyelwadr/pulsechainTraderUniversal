"""
RSI Bands PZI Gated Strategy

RSI band context gated by Price Zone Index and VMA slope to allow trades only
in supportive regimes.
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


class RSIBandsPZIGatedStrategy(BaseStrategy):
    def __init__(self, parameters: Dict = None):
        p = {
            'rsi_period': 14,
            'rsi_buy': 35,
            'rsi_sell': 65,
            'band_len': 50,
            'ema_len': 30,
            'signal_threshold': 0.5,
            'timeframe_minutes': 60,
        }
        if parameters:
            p.update(parameters)
        super().__init__('RSIBandsPZIGatedStrategy', p)

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        d = data.copy()
        price = d.get('price', d.get('close', d['close']))
        d['rsi'] = _rsi(price, int(self.parameters['rsi_period']))
        n = int(self.parameters['band_len'])
        ll = price.rolling(n).min(); hh = price.rolling(n).max()
        d['pzi'] = ((price - ll) / (hh - ll).replace(0, np.nan)).clip(0, 1.0).fillna(0.5)
        d['vma'] = _ema(price, int(self.parameters['ema_len']))
        d['vma_slope'] = d['vma'].diff()
        return d

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        d = self.calculate_indicators(data)
        rsi = d['rsi']; pzi = d['pzi']; slope = d['vma_slope']
        buy_gate = (pzi < 0.6) & (slope > 0)
        sell_gate = (pzi > 0.4) & (slope < 0)
        buy = (rsi <= float(self.parameters['rsi_buy'])) & buy_gate
        sell = (rsi >= float(self.parameters['rsi_sell'])) & sell_gate
        st = (1 - (pzi - 0.5).abs() * 2).clip(0, 1.0)
        thr = float(self.parameters['signal_threshold'])
        st = st.where(buy | sell, 0.0)
        st[(buy | sell) & (st < thr)] = thr
        d['buy_signal'] = buy.fillna(False)
        d['sell_signal'] = sell.fillna(False)
        d['signal_strength'] = st.fillna(0.0)
        return d

