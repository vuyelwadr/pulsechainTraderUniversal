"""
Constance Brown Composite Index (CBCI) Strategy

Composite of RSI, Stochastic %K, and normalized momentum. Signals when the
composite crosses recovery/fade thresholds.
"""
from typing import Dict
import numpy as np
import pandas as pd
from strategies.base_strategy import BaseStrategy


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean().replace(0, np.nan)
    rs = avg_gain / avg_loss
    return (100 - 100/(1+rs)).fillna(50.0)


class ConstanceBrownCompositeIndex(BaseStrategy):
    def __init__(self, parameters: Dict = None):
        p = {
            'rsi_period': 14,
            'stoch_period': 14,
            'momentum_period': 10,
            'smooth': 5,
            'low_level': 30.0,
            'high_level': 70.0,
            'signal_threshold': 0.6,
            'timeframe_minutes': 60,
        }
        if parameters:
            p.update(parameters)
        super().__init__('ConstanceBrownCompositeIndex', p)

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        d = data.copy()
        price = d.get('price', d.get('close', d['close']))
        rsi = _rsi(price, int(self.parameters['rsi_period']))
        n = int(self.parameters['stoch_period'])
        ll = d['low'].rolling(n, min_periods=1).min()
        hh = d['high'].rolling(n, min_periods=1).max()
        stoch_k = ((d['close'] - ll) / (hh - ll).replace(0, np.nan) * 100).fillna(50.0)
        m = int(self.parameters['momentum_period'])
        mom = (d['close'] - d['close'].shift(m)) / d['close'].replace(0, np.nan)
        mom_norm = (mom * 1000).clip(-100, 100) + 50
        comp = (rsi + stoch_k + mom_norm) / 3.0
        comp_s = comp.rolling(max(1, int(self.parameters['smooth']))).mean().bfill().fillna(comp)
        d['cbci'] = comp_s
        return d

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        d = self.calculate_indicators(data)
        lo = float(self.parameters['low_level'])
        hi = float(self.parameters['high_level'])
        cb = d['cbci']
        # Crosses of composite through levels
        buy = (cb > lo) & (cb.shift(1) <= lo)
        sell = (cb < hi) & (cb.shift(1) >= hi)
        # Strength based on distance into the band
        st = np.where(buy, np.clip((cb - lo) / max(1.0, (50-lo)), 0.0, 1.0), 0.0)
        st = np.where(sell, np.clip((hi - cb) / max(1.0, (hi-50)), 0.0, 1.0), st)
        thr = float(self.parameters['signal_threshold'])
        st = np.where(buy | sell, np.maximum(st, thr), 0.0)
        d['buy_signal'] = buy.fillna(False)
        d['sell_signal'] = sell.fillna(False)
        d['signal_strength'] = pd.Series(st, index=d.index).fillna(0.0)
        return d
