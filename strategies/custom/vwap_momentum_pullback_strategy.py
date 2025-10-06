"""
VWAP Momentum Pullback Strategy (VMP)

Trade in the direction of trend + VWAP support with relative volume
confirmation; enter on pullbacks and exit on reversion or ATR stops.
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


class VWAPMomentumPullbackStrategy(BaseStrategy):
    def __init__(self, parameters: Dict = None):
        p = {
            'ema_trend': 100,
            'ema_entry': 20,
            'atr_len': 14,
            'atr_mult': 2.0,
            'rv_len': 20,           # relative volume window
            'rv_gate': 1.05,        # require vol_today >= rv_gate*rv_mean
            'pullback_z': 0.5,      # pullback z threshold toward ema_entry
            'signal_threshold': 0.4,
            'timeframe_minutes': 60,
        }
        if parameters:
            p.update(parameters)
        super().__init__('VWAPMomentumPullbackStrategy', p)

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        price = df.get('price', df.get('close', df['close']))
        p = self.parameters
        df['ema_trend'] = _ema(price, int(p['ema_trend']))
        df['ema_entry'] = _ema(price, int(p['ema_entry']))
        df['atr'] = _atr(df.assign(close=price), int(p['atr_len']))
        # rolling vwap approximation (price*vol cumulative / vol cumulative)
        vol = df.get('volume', pd.Series(0, index=df.index))
        pv = price * vol
        c_pv = pv.cumsum().replace(0, np.nan)
        c_v = vol.cumsum().replace(0, np.nan)
        df['vwap'] = (c_pv / c_v).fillna(price)
        # relative volume
        rv_len = max(2, int(p['rv_len']))
        df['rv'] = (vol / (vol.rolling(rv_len).mean().replace(0, np.nan))).fillna(1.0)
        # pullback z to entry EMA
        diff = price - df['ema_entry']
        sd = diff.rolling(40).std().replace(0, np.nan)
        df['pb_z'] = (diff / sd).fillna(0.0)
        return df

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = self.calculate_indicators(data)
        p = self.parameters
        price = df.get('price', df.get('close', df['close']))
        # trend + vwap alignment
        trend_up = (price > df['ema_trend']) & (price > df['vwap'])
        # relative volume confirmation
        rv_ok = df['rv'] >= float(p['rv_gate'])
        # pullback to entry EMA
        z = df['pb_z']
        allow_entry = trend_up & rv_ok & (z <= -float(p['pullback_z']))
        # exits: reversion above entry EMA (i.e., rebound) or ATR stop
        exit = (price < df['vwap']) | (price <= (price - float(p['atr_mult']) * df['atr']))
        df['buy_signal'] = allow_entry.fillna(False)
        df['sell_signal'] = exit.fillna(False)
        # strength: blend of pullback depth and relative volume
        z_n = (-z / max(float(p['pullback_z']), 1e-6)).clip(0.0, 1.0)
        rv_n = (df['rv'] / (float(p['rv_gate']) + 1e-9)).clip(0.0, 1.0)
        strength = (0.6 * z_n + 0.4 * rv_n).clip(0.0, 1.0)
        thr = float(p['signal_threshold'])
        strength = np.where(df['buy_signal'] | df['sell_signal'], np.maximum(strength, thr), 0.0)
        df['signal_strength'] = pd.Series(strength, index=df.index).fillna(0.0)
        return df

