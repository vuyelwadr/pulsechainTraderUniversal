"""
Time And Money Chart Channel Strategy (TMCC)

Approximates a channel using EMA(mid) ± k*ATR. Buys near lower channel,
sells near upper; includes trend gate.
"""
from typing import Dict
import numpy as np
import pandas as pd
from strategies.base_strategy import BaseStrategy


def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=max(2, int(n)), adjust=False).mean()


def _atr(df: pd.DataFrame, n: int) -> pd.Series:
    h, l, c = df['high'], df['low'], df['close']
    tr = np.maximum(h - l, np.maximum((h - c.shift(1)).abs(), (l - c.shift(1)).abs()))
    return pd.Series(tr).rolling(max(2, int(n))).mean()


class TimeAndMoneyChartChannel(BaseStrategy):
    def __init__(self, parameters: Dict = None):
        p = {
            'ema_len': 34,
            'atr_len': 14,
            'mul': 1.6,
            'trend_len': 100,
            'signal_threshold': 0.5,
            'timeframe_minutes': 60,
        }
        if parameters:
            p.update(parameters)
        super().__init__('TimeAndMoneyChartChannel', p)

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        d = data.copy()
        price = d.get('price', d.get('close', d['close']))
        mid = _ema(price, int(self.parameters['ema_len']))
        atr = _atr(d.assign(close=price), int(self.parameters['atr_len']))
        up = mid + float(self.parameters['mul']) * atr
        dn = mid - float(self.parameters['mul']) * atr
        d['tmcc_mid'] = mid; d['tmcc_up'] = up; d['tmcc_dn'] = dn
        d['trend'] = _ema(price, int(self.parameters['trend_len']))
        return d

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        d = self.calculate_indicators(data)
        price = d.get('price', d.get('close', d['close']))
        # Mean reversion in channel, trend gate reduces opposing signals
        buy_raw = price <= d['tmcc_dn']
        sell_raw = price >= d['tmcc_up']
        bull = price > d['trend']
        bear = price < d['trend']
        buy = buy_raw & (~bear)
        sell = sell_raw & (~bull)
        # Strength by band distance
        band_span = (d['tmcc_up'] - d['tmcc_dn']).replace(0, np.nan)
        pos = ((price - d['tmcc_dn']) / band_span).clip(0, 1.0)
        st = (1.0 - (pos - 0.5).abs() * 2).clip(0, 1.0)
        thr = float(self.parameters['signal_threshold'])
        st = st.where(buy | sell, 0.0)
        st[(buy | sell) & (st < thr)] = thr
        d['buy_signal'] = buy.fillna(False)
        d['sell_signal'] = sell.fillna(False)
        d['signal_strength'] = st.fillna(0.0)
        return d

