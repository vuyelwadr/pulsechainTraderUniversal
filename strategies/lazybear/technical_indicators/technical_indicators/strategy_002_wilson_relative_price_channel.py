#!/usr/bin/env python3
"""
Strategy 002: Wilson Relative Price Channel

TradingView URL: https://www.tradingview.com/v/w0VlTHHV/
Type: channel/volatility (RSI-derived envelope)

Pine parity summary (from pine_scripts/002_wilson_relative_price_channel.pine):
RSI = rsi(close, periods)
OB = ema(RSI - overbought, smoothing)
OS = ema(RSI - oversold, smoothing)
NZU = ema(RSI - upperNeutralZone, smoothing)
NZL = ema(RSI - lowerNeutralZone, smoothing)

Plotted bands in Pine map to price envelopes here as:
 upper1 = close - (close * OB/100)
 lower1 = close - (close * OS/100)
 upper2 = close - (close * NZU/100)
 lower2 = close - (close * NZL/100)

Signals (derived for trading):
 - Buy when close crosses above upper2 (leaving neutral zone upward)
 - Sell when close crosses below lower2 (leaving neutral zone downward)
We clamp signal strength and guard against look-ahead.
"""

import pandas as pd
import numpy as np
from typing import Dict
import sys, os

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)
from strategies.base_strategy import BaseStrategy

try:
    sys.path.insert(0, os.path.join(project_root, 'src'))
    from utils.vectorized_helpers import crossover, crossunder, calculate_signal_strength
except Exception:
    def crossover(a,b): return (a>b) & (a.shift(1) <= b.shift(1))
    def crossunder(a,b): return (a<b) & (a.shift(1) >= b.shift(1))
    def calculate_signal_strength(fs,weights=None):
        import pandas as pd
        df=pd.concat(fs,axis=1); return df.mean(axis=1).clip(0,1)


class Strategy002WilsonRelativePriceChannel(BaseStrategy):
    def __init__(self, parameters: Dict=None):
        params = {
            'periods': 34,
            'smoothing': 1,
            'overbought': 70.0,
            'oversold': 30.0,
            'upperNeutralZone': 55.0,
            'lowerNeutralZone': 45.0,
            'signal_threshold': 0.6,
        }
        if parameters:
            params.update(parameters)
        super().__init__('Strategy_002_WilsonRelativePriceChannel', params)

    @staticmethod
    def _ema(s: pd.Series, n: int) -> pd.Series:
        return s.ewm(span=max(1,int(n)), adjust=False, min_periods=1).mean()

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        if 'close' not in data.columns and 'price' in data.columns:
            data['close'] = data['price']

        rsi_period = int(self.parameters['periods'])
        smooth = int(self.parameters['smoothing'])

        # RSI (Wilder RMA-style approximation via EMA on gains/losses)
        delta = data['close'].diff()
        gain = delta.clip(lower=0.0)
        loss = -delta.clip(upper=0.0)
        avg_gain = self._ema(gain, rsi_period)
        avg_loss = self._ema(loss, rsi_period)
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - 100/(1+rs)

        ob = self._ema(rsi - float(self.parameters['overbought']), smooth)
        os_ = self._ema(rsi - float(self.parameters['oversold']), smooth)
        nzu = self._ema(rsi - float(self.parameters['upperNeutralZone']), smooth)
        nzl = self._ema(rsi - float(self.parameters['lowerNeutralZone']), smooth)

        close = data['close']
        data['wrpc_upper1'] = close - (close * ob/100.0)
        data['wrpc_lower1'] = close - (close * os_/100.0)
        data['wrpc_upper2'] = close - (close * nzu/100.0)
        data['wrpc_lower2'] = close - (close * nzl/100.0)
        data['rsi'] = rsi
        return data

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        data['buy_signal'] = crossover(data['close'], data['wrpc_upper2'])
        data['sell_signal'] = crossunder(data['close'], data['wrpc_lower2'])

        dist_up = ((data['close'] - data['wrpc_upper2']).abs() / (data['close'].abs()+1e-9)).clip(0,1)
        dist_dn = ((data['wrpc_lower2'] - data['close']).abs() / (data['close'].abs()+1e-9)).clip(0,1)
        strength = calculate_signal_strength([
            pd.Series(0.0, index=data.index).where(~data['buy_signal'], dist_up),
            pd.Series(0.0, index=data.index).where(~data['sell_signal'], dist_dn)
        ])
        # Clamp a floor for actual signals so engine executes
        floor = 0.6
        strength = strength.mask(~(data['buy_signal']|data['sell_signal']), 0.0)
        strength = strength.mask((data['buy_signal']|data['sell_signal']) & (strength<floor), floor)
        data['signal_strength'] = strength.fillna(0.0)
        return data

