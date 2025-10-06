"""
Volatility Breakout with KAMA Filter (VBK)

Breakout over Donchian channel highs with a KAMA direction filter and
optional volatility‑contraction precondition. Emits buy/sell and signal_strength.
"""
from typing import Dict
import numpy as np
import pandas as pd
from strategies.base_strategy import BaseStrategy


def _hhv(x: pd.Series, n: int) -> pd.Series:
    return x.rolling(max(1, int(n))).max()


def _llv(x: pd.Series, n: int) -> pd.Series:
    return x.rolling(max(1, int(n))).min()


def _atr(df: pd.DataFrame, n: int) -> pd.Series:
    h, l, c = df['high'], df['low'], df['close']
    tr = np.maximum(h - l, np.maximum((h - c.shift(1)).abs(), (l - c.shift(1)).abs()))
    return pd.Series(tr).rolling(max(2, int(n))).mean()


def _kama(price: pd.Series, er_length: int = 10, fast: int = 2, slow: int = 30) -> pd.Series:
    change = (price - price.shift(er_length)).abs()
    vol = price.diff().abs().rolling(er_length).sum().replace(0, np.nan)
    er = (change / vol).clip(0.0, 1.0).fillna(0.0)
    sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
    kama = price.astype(float).copy()
    for i in range(1, len(price)):
        s = sc.iat[i] if not np.isnan(sc.iat[i]) else (2/(slow+1))**2
        kama.iat[i] = kama.iat[i-1] + s * (price.iat[i] - kama.iat[i-1])
    return kama


class VolatilityBreakoutKAMAStrategy(BaseStrategy):
    def __init__(self, parameters: Dict = None):
        p = {
            'breakout_len': 40,
            'confirm_len': 5,
            'atr_len': 14,
            'atr_mult': 2.8,
            'kama_er': 10,
            'kama_fast': 2,
            'kama_slow': 30,
            'pre_vol_window': 20,
            'pre_vol_ratio': 0.65,
            'signal_threshold': 0.4,
            'timeframe_minutes': 60,
        }
        if parameters:
            p.update(parameters)
        super().__init__('VolatilityBreakoutKAMAStrategy', p)

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        p = self.parameters
        df['donch_hi'] = _hhv(df['high'], int(p['breakout_len']))
        df['donch_lo'] = _llv(df['low'], int(p['breakout_len']))
        df['donch_hi_confirm'] = _hhv(df['high'], int(p['confirm_len']))
        df['atr'] = _atr(df, int(p['atr_len']))
        price = df.get('price', df.get('close', df['close']))
        df['kama'] = _kama(price, int(p['kama_er']), int(p['kama_fast']), int(p['kama_slow']))
        df['kama_slope'] = df['kama'].diff()
        # volatility contraction: recent ATR <= ratio * prior ATR
        atr_now = df['atr']
        atr_prev = df['atr'].shift(int(p['pre_vol_window']))
        df['vol_contract_ok'] = (atr_now <= float(p['pre_vol_ratio']) * atr_prev)
        return df

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = self.calculate_indicators(data)
        p = self.parameters
        close = df.get('price', df.get('close', df['close']))
        breakout = (close >= df['donch_hi']) & (close >= df['donch_hi_confirm'])
        kama_filter = (close > df['kama']) & (df['kama_slope'] > 0)
        entry = (breakout & kama_filter & df['vol_contract_ok']).fillna(False)
        exit = (close <= df['kama']) | (close <= (close - float(p['atr_mult']) * df['atr']))

        df['buy_signal'] = entry
        df['sell_signal'] = exit.fillna(False)

        dist = (close - df['donch_hi']).clip(lower=0.0)
        dist_n = (dist / (close.abs() + 1e-9)).clip(0, 1.0)
        kama_slope_n = (df['kama_slope'] / (close.abs() * 0.005 + 1e-9)).clip(0, 1.0)
        strength = (0.7 * dist_n + 0.3 * kama_slope_n).clip(0.0, 1.0)
        thr = float(p['signal_threshold'])
        strength = np.where(df['buy_signal'] | df['sell_signal'], np.maximum(strength, thr), 0.0)
        df['signal_strength'] = pd.Series(strength, index=df.index).fillna(0.0)
        return df

