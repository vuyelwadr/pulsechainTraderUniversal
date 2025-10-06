"""
Adaptive Grid KAMA (from refined pack 301)

Anchors grid to KAMA with ATR spacing and EMA trend gate.
"""
from typing import Dict
import numpy as np
import pandas as pd
from strategies.base_strategy import BaseStrategy


def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=max(1, int(n)), adjust=False, min_periods=1).mean()


def _atr(h: pd.Series, l: pd.Series, c: pd.Series, n: int) -> pd.Series:
    pc = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(span=max(1, int(n)), adjust=False, min_periods=1).mean()


def _kama(price: pd.Series, er_len=10, fast=2, slow=30) -> pd.Series:
    price = price.astype(float)
    fast_sc = 2 / (fast + 1.0)
    slow_sc = 2 / (slow + 1.0)
    out = price.copy()
    out.iloc[: max(1, er_len)] = price.iloc[: max(1, er_len)].expanding().mean()
    for i in range(max(1, er_len), len(price)):
        change = abs(price.iloc[i] - price.iloc[i - er_len])
        vol = price.iloc[i - er_len + 1 : i + 1].diff().abs().sum()
        er = 0.0 if vol == 0 else change / vol
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        out.iloc[i] = out.iloc[i - 1] + sc * (price.iloc[i] - out.iloc[i - 1])
    return out


class AdaptiveGridKAMAStrategy(BaseStrategy):
    def __init__(self, parameters: Dict = None):
        params = dict(
            kama_fast=2,
            kama_slow=30,
            kama_er_len=10,
            ema_trend_len=200,
            atr_len=14,
            grid_atr_mult=0.75,
            max_levels_per_bar=3,
            trend_gate_strength=0.6,
            signal_threshold=0.6,
            timeframe_minutes=60,
        )
        if parameters:
            params.update(parameters)
        super().__init__('AdaptiveGridKAMAStrategy', params)

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        d = data.copy()
        if 'close' not in d and 'price' in d:
            d['close'] = d['price']
        for c in ('high', 'low', 'close'):
            if c not in d:
                d[c] = d.get('close', pd.Series(index=d.index)).fillna(method='ffill')
        d['atr'] = _atr(d['high'], d['low'], d['close'], int(self.parameters['atr_len']))
        d['kama'] = _kama(d['close'], int(self.parameters['kama_er_len']), int(self.parameters['kama_fast']), int(self.parameters['kama_slow']))
        d['ema_trend'] = _ema(d['close'], int(self.parameters['ema_trend_len']))
        spacing = float(self.parameters['grid_atr_mult']) * d['atr'].replace(0, np.nan)
        spacing = spacing.replace(0, np.nan).ffill().bfill()
        d['grid_center'] = d['kama']
        d['grid_idx'] = np.floor((d['close'] - d['grid_center']) / spacing)
        return d

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        d = data.copy()
        if 'grid_idx' not in d or 'kama' not in d:
            d = self.calculate_indicators(d)
        delta_idx = d['grid_idx'].diff().fillna(0)
        kama_slope = d['kama'].diff()
        bullish = (d['close'] > d['ema_trend']) & (kama_slope > 0)
        bearish = (d['close'] < d['ema_trend']) & (kama_slope < 0)
        raw_buy = delta_idx <= -1
        raw_sell = delta_idx >= 1
        gate = float(self.parameters['trend_gate_strength'])
        buy = raw_buy & (~bearish | (gate < 0.5))
        sell = raw_sell & (~bullish | (gate < 0.5))
        crossed = np.where(delta_idx < 0, -delta_idx, delta_idx)
        crossed = np.clip(crossed, 0, int(self.parameters['max_levels_per_bar']))
        spacing = float(self.parameters['grid_atr_mult']) * d['atr'].replace(0, np.nan)
        spacing = spacing.replace(0, np.nan).ffill().bfill()
        frac = ((d['close'] - d['grid_center']) / spacing) - d['grid_idx']
        dist_edge = np.minimum(frac, 1 - frac).abs().fillna(0)
        strength = (crossed / max(1, int(self.parameters['max_levels_per_bar']))) * 0.7 + (1 - dist_edge) * 0.3
        thr = float(self.parameters['signal_threshold'])
        strength = strength.clip(0, 1).where(buy | sell, 0.0)
        strength[(buy | sell) & (strength < thr)] = thr
        d['buy_signal'] = buy
        d['sell_signal'] = sell
        d['signal_strength'] = strength.astype(float).fillna(0.0)
        return d

