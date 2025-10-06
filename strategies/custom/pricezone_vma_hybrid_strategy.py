"""
PriceZone VMA Hybrid

Uses a price zone index (PZI: normalized location within a rolling band)
and VMA slope (approximated by EMA slope) to gate signals.
"""
from typing import Dict
import numpy as np
import pandas as pd
from strategies.base_strategy import BaseStrategy


def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=max(2, int(n)), adjust=False).mean()


class PriceZoneVMAHybridStrategy(BaseStrategy):
    def __init__(self, parameters: Dict = None):
        p = {
            'band_len': 50,
            'ema_len': 30,
            'pzi_upper': 0.8,
            'pzi_lower': 0.2,
            'signal_threshold': 0.5,
            'timeframe_minutes': 60,
        }
        if parameters:
            p.update(parameters)
        super().__init__('PriceZoneVMAHybridStrategy', p)

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        d = data.copy()
        price = d.get('price', d.get('close', d['close']))
        n = int(self.parameters['band_len'])
        ll = price.rolling(n).min()
        hh = price.rolling(n).max()
        pzi = ((price - ll) / (hh - ll).replace(0, np.nan)).clip(0, 1.0).fillna(0.5)
        vma = _ema(price, int(self.parameters['ema_len']))
        d['pzi'] = pzi; d['vma'] = vma; d['vma_slope'] = vma.diff()
        return d

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        d = self.calculate_indicators(data)
        pzi = d['pzi']; slope = d['vma_slope']
        buy = (pzi <= float(self.parameters['pzi_lower'])) & (slope > 0)
        sell = (pzi >= float(self.parameters['pzi_upper'])) & (slope < 0)
        st = (1 - (pzi - 0.5).abs() * 2).clip(0, 1.0)
        thr = float(self.parameters['signal_threshold'])
        st = st.where(buy | sell, 0.0)
        st[(buy | sell) & (st < thr)] = thr
        d['buy_signal'] = buy.fillna(False)
        d['sell_signal'] = sell.fillna(False)
        d['signal_strength'] = st.fillna(0.0)
        return d

