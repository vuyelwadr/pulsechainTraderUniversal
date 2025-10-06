"""
TMCC + CBCI Hybrid

Combines Time & Money Chart Channel mean-reversion with Constance Brown
Composite Index confirmation.
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


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean().replace(0, np.nan)
    rs = avg_gain / avg_loss
    return (100 - 100/(1+rs)).fillna(50.0)


class TMCCCBCIHybridStrategy(BaseStrategy):
    def __init__(self, parameters: Dict = None):
        p = {
            'ema_len': 34,
            'atr_len': 14,
            'mul': 1.5,
            'rsi_period': 14,
            'cb_low': 30,
            'cb_high': 70,
            'signal_threshold': 0.5,
            'timeframe_minutes': 60,
        }
        if parameters:
            p.update(parameters)
        super().__init__('TMCCCBCIHybridStrategy', p)

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        d = data.copy()
        price = d.get('price', d.get('close', d['close']))
        mid = _ema(price, int(self.parameters['ema_len']))
        atr = _atr(d.assign(close=price), int(self.parameters['atr_len']))
        up = mid + float(self.parameters['mul']) * atr
        dn = mid - float(self.parameters['mul']) * atr
        rsi = _rsi(price, int(self.parameters['rsi_period']))
        d['mid'] = mid; d['up'] = up; d['dn'] = dn; d['rsi'] = rsi
        # simple CBCI proxy as RSI smoothing (to avoid heavy computation)
        d['cbci'] = (rsi + ((d['close'] - d['close'].rolling(10).min()) / (d['close'].rolling(10).max() - d['close'].rolling(10).min()).replace(0,np.nan) * 100).fillna(50.0)) / 2.0
        return d

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        d = self.calculate_indicators(data)
        price = d.get('price', d.get('close', d['close']))
        # TMCC mean-reversion gated by CBCI direction
        buy = (price <= d['dn']) & (d['cbci'].diff() > 0) & (d['rsi'] < float(self.parameters['cb_low']))
        sell = (price >= d['up']) & (d['cbci'].diff() < 0) & (d['rsi'] > float(self.parameters['cb_high']))
        span = (d['up'] - d['dn']).replace(0, np.nan)
        pos = ((price - d['dn']) / span).clip(0, 1.0)
        st = (1 - (pos - 0.5).abs() * 2).clip(0, 1.0)
        thr = float(self.parameters['signal_threshold'])
        st = st.where(buy | sell, 0.0)
        st[(buy | sell) & (st < thr)] = thr
        d['buy_signal'] = buy.fillna(False)
        d['sell_signal'] = sell.fillna(False)
        d['signal_strength'] = st.fillna(0.0)
        return d

