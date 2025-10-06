#!/usr/bin/env python3
"""
Strategy 142: NYSE A/D line (Single‑Instrument Proxy)

Proxy rationale (no external breadth feeds allowed):
- Use the instrument's own up/down day information as an internal breadth proxy.
- Define adv/decl proxy as sign of close-to-close return magnitude.
- Build an A/D line as cumulative sum of that series and smooth it.
- Signals: A/D line crosses its signal EMA.

Inputs: OHLCV for a single instrument. No synthetic price data.
"""

from typing import Dict
import pandas as pd
import numpy as np
import os, sys

# Repo root for BaseStrategy import
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)
from strategies.base_strategy import BaseStrategy

try:
    sys.path.insert(0, os.path.join(project_root, 'src'))
    from utils.vectorized_helpers import crossover, crossunder
except Exception:
    def crossover(a,b): return (a>b) & (a.shift(1)<=b)
    def crossunder(a,b): return (a<b) & (a.shift(1)>=b)


class Strategy142NYSEADLineProxy(BaseStrategy):
    """Single‑instrument A/D line proxy with EMA signal cross entries."""

    def __init__(self, parameters: Dict = None):
        params = {
            'use_pct_change': True,  # use percent change magnitude as adv/decl weight
            'scale': 100.0,          # scale for pct_change based proxy
            'ema_len': 10,           # smoothing for A/D line
            'signal_len': 20,        # signal EMA length
            'min_strength': 0.2,     # floor for emitted signal strength on crosses
        }
        if parameters:
            params.update(parameters)
        super().__init__('Strategy_142_NYSE_AD_Line_Proxy', params)

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        if 'close' not in df and 'price' in df:
            df['close'] = df['price']
        close = df['close'].astype(float)

        # Adv/Decl proxy: signed magnitude of return (or just sign)
        if self.parameters['use_pct_change']:
            advdecl = (close.pct_change().fillna(0.0) * float(self.parameters['scale']))
        else:
            advdecl = np.sign(close.diff().fillna(0.0))

        df['ad_proxy'] = advdecl.cumsum()
        # EMA smoothing
        ema_len = int(self.parameters['ema_len'])
        signal_len = int(self.parameters['signal_len'])
        df['ad_ema'] = df['ad_proxy'].ewm(span=max(1, ema_len), adjust=False).mean()
        df['ad_sig'] = df['ad_proxy'].ewm(span=max(1, signal_len), adjust=False).mean()
        return df

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        buy = crossover(df['ad_proxy'], df['ad_sig'])
        sell = crossunder(df['ad_proxy'], df['ad_sig'])
        df['buy_signal'] = buy
        df['sell_signal'] = sell
        # Strength scaled by distance between line and signal, normalized
        dist = (df['ad_proxy'] - df['ad_sig']).abs()
        # Normalize by rolling volatility of the proxy to keep [0,1] reasonably bounded
        denom = df['ad_proxy'].diff().abs().rolling(50, min_periods=1).mean().replace(0, np.nan)
        strength = (dist / denom).clip(0, 1).fillna(0.0)
        # Ensure minimum floor on actual cross events
        min_st = float(self.parameters['min_strength'])
        strength[(buy | sell) & (strength < min_st)] = min_st
        df['signal_strength'] = strength.where(buy | sell, 0.0)
        return df
