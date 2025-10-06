"""
Trend Trigger Factor (PB variant)

Simplified TTF-like momentum of EMA differences with thresholds; emits
buy/sell triggers when momentum crosses zero bands.
"""
from typing import Dict
import numpy as np
import pandas as pd
from strategies.base_strategy import BaseStrategy


def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=max(2, int(n)), adjust=False).mean()


class TrendTriggerFactorPB(BaseStrategy):
    def __init__(self, parameters: Dict = None):
        p = {
            'ema_fast': 20,
            'ema_slow': 50,
            'band': 0.0,
            'signal_threshold': 0.5,
            'timeframe_minutes': 60,
        }
        if parameters:
            p.update(parameters)
        super().__init__('TrendTriggerFactorPB', p)

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        d = data.copy()
        price = d.get('price', d.get('close', d['close']))
        ef = _ema(price, int(self.parameters['ema_fast']))
        es = _ema(price, int(self.parameters['ema_slow']))
        mom = (ef - es) / price.replace(0, np.nan)
        d['ttf_m'] = mom
        return d

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        d = self.calculate_indicators(data)
        m = d['ttf_m']
        band = float(self.parameters['band'])
        buy = (m > band) & (m.shift(1) <= band)
        sell = (m < -band) & (m.shift(1) >= -band)
        st = np.where(buy, np.clip((m - band) / (abs(band) + 1e-6), 0.0, 1.0), 0.0)
        st = np.where(sell, np.clip(((-band) - m) / (abs(band) + 1e-6), 0.0, 1.0), st)
        thr = float(self.parameters['signal_threshold'])
        st = np.where(buy | sell, np.maximum(st, thr), 0.0)
        d['buy_signal'] = buy.fillna(False)
        d['sell_signal'] = sell.fillna(False)
        d['signal_strength'] = pd.Series(st, index=d.index).fillna(0.0)
        return d

