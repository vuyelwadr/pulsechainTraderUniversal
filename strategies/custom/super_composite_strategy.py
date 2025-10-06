"""
Super Composite Strategy

Blends multiple normalized components: trend (EMA slope), band position
(Keltner proxy), RSI position, and KAMA slope. Produces a composite score
mapped to buy/sell thresholds.
"""
from typing import Dict, Tuple
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
    up = delta.clip(lower=0.0)
    dn = -delta.clip(upper=0.0)
    au = up.rolling(period).mean()
    ad = dn.rolling(period).mean().replace(0, np.nan)
    rs = au / ad
    return (100 - 100/(1+rs)).fillna(50.0)


class SuperCompositeStrategy(BaseStrategy):
    def __init__(self, parameters: Dict = None):
        p = {
            'ema_len': 50,
            'keltner_len': 34,
            'keltner_mult': 1.5,
            'rsi_period': 14,
            'signal_threshold': 0.55,
            'timeframe_minutes': 60,
        }
        if parameters:
            p.update(parameters)
        super().__init__('SuperCompositeStrategy', p)

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        d = data.copy()
        price = d.get('price', d.get('close', d['close']))
        ema = _ema(price, int(self.parameters['ema_len']))
        slope = ema.diff()
        atr = _atr(d.assign(close=price), int(self.parameters['keltner_len']))
        mid = ema
        up = mid + float(self.parameters['keltner_mult']) * atr
        dn = mid - float(self.parameters['keltner_mult']) * atr
        rsi = _rsi(price, int(self.parameters['rsi_period']))
        d['ema'] = ema; d['slope'] = slope; d['kel_up'] = up; d['kel_dn'] = dn; d['rsi'] = rsi
        return d

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        d = self.calculate_indicators(data)
        price = d.get('price', d.get('close', d['close']))
        # Normalize components
        # Trend: positive slope good
        tr = np.tanh(d['slope'] / (price.abs() * 0.005 + 1e-9))
        # Band position: closer to lower band → buy; upper band → sell
        span = (d['kel_up'] - d['kel_dn']).replace(0, np.nan)
        pos = ((price - d['kel_dn']) / span).clip(0, 1.0)
        pos_comp = 1 - (pos - 0.5).abs() * 2
        # RSI centered at 50
        rsi_comp = 1 - (d['rsi'] - 50).abs() / 50.0
        # Composite
        S = (0.4 * tr + 0.3 * pos_comp + 0.3 * rsi_comp).clip(0, 1.0)
        buy = S > float(self.parameters['signal_threshold'])
        sell = S < (1.0 - float(self.parameters['signal_threshold']))
        st = (S - 0.5).abs() * 2
        d['buy_signal'] = buy
        d['sell_signal'] = sell
        d['signal_strength'] = st.clip(0, 1.0).fillna(0.0)
        return d

    @classmethod
    def parameter_space(cls) -> Dict[str, Tuple[float, float]]:
        """Define parameter bounds for optimization"""
        return {
            'ema_len': (5, 50),  # EMA period
            'keltner_len': (5, 50),  # Keltner channel period
            'keltner_mult': (0.5, 3.0),  # Keltner multiplier
            'rsi_period': (5, 30),  # RSI period
            'signal_threshold': (0.0, 0.95),  # Signal threshold can't be 1.0
        }

