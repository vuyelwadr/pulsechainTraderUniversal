"""Enhanced cost-aware SMA reversion strategy with adaptive exits.

This variant keeps the core deep-dip entry of ``CSMARevertStrategy`` but adds
regime- and gain-aware exit management so large rebounds are held for longer
while still respecting catastrophic risk controls. The implementation mirrors
the real swap-cost model expectations (no synthetic pricing assumptions) and is
designed to plug into the existing ``BaseStrategy`` evaluation pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

from .base_strategy import BaseStrategy


def _sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=window).mean()


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=max(span, 1), adjust=False).mean()


def _atr(series: pd.DataFrame, period: int) -> pd.Series:
    high = series['high'] if 'high' in series else series['price']
    low = series['low'] if 'low' in series else series['price']
    close = series['price'] if 'price' in series else series['close']
    prev_close = close.shift(1)

    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return tr.rolling(period, min_periods=period).mean()


def _rsi(series: pd.Series, window: int) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    avg_gain = up.rolling(window, min_periods=window).mean()
    avg_loss = down.rolling(window, min_periods=window).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    return 100 - (100 / (1 + rs))


@dataclass
class _State:
    in_position: bool = False
    entry_price: float = 0.0
    entry_sma: float = 0.0
    peak_price: float = 0.0
    trail_armed: bool = False
    bars_in_trade: int = 0


class CSMARevertDynamicCodex1Strategy(BaseStrategy):
    """Adaptive version of the C-SMA reversion strategy with dynamic exits."""

    def __init__(self, parameters: Dict | None = None):
        defaults = {
            'n_sma': 576,  # ~2 days of 5m bars
            'entry_drop': 0.24,
            'rsi_period': 14,
            'rsi_max': 32.0,
            'ema_fast_period': 288,  # bars (~1 day)
            'ema_slow_period': 1152,  # bars (~4 days)
            'atr_period': 288,
            'atr_entry_ratio_min': 0.10,
            'exit_sma_buffer': 0.06,
            'exit_rsi_min': 52.0,
            'trail_activation_gain': 0.35,
            'trail_pct': 0.18,
            'panic_stop_pct': 0.48,
            'max_hold_bars': 5760,  # ~20 days
            'timeframe_minutes': 5,
        }
        if parameters:
            defaults.update(parameters)
        super().__init__('CSMARevertDynamicCodex1Strategy', defaults)

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        if not self.validate_data(data):
            return data

        df = data.copy()
        price = df['price'] if 'price' in df else df['close']
        df['price'] = price

        n_sma = int(self.parameters['n_sma'])
        ema_fast_period = max(1, int(self.parameters['ema_fast_period']))
        ema_slow_period = max(1, int(self.parameters['ema_slow_period']))
        atr_period = int(self.parameters['atr_period'])
        rsi_period = int(self.parameters['rsi_period'])

        df['sma'] = _sma(price, n_sma)
        df['ema_fast'] = _ema(price, ema_fast_period)
        df['ema_slow'] = _ema(price, ema_slow_period)
        df['ema_fast_slope'] = df['ema_fast'].diff()
        df['atr'] = _atr(df, atr_period)
        df['atr_ratio'] = (df['atr'] / price).fillna(0.0)
        df['rsi'] = _rsi(price, rsi_period)
        return df

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        if 'sma' not in df.columns:
            df = self.calculate_indicators(df)

        df['buy_signal'] = False
        df['sell_signal'] = False
        df['signal_strength'] = 0.0

        price = df['price'].to_numpy(float)
        sma = df['sma'].to_numpy(float)
        rsi = df['rsi'].to_numpy(float)
        ema_fast = df['ema_fast'].to_numpy(float)
        ema_slow = df['ema_slow'].to_numpy(float)
        ema_fast_slope = df['ema_fast_slope'].to_numpy(float)
        atr_ratio = df['atr_ratio'].to_numpy(float)

        entry_drop = float(self.parameters['entry_drop'])
        rsi_max = float(self.parameters['rsi_max'])
        atr_entry_ratio_min = float(self.parameters['atr_entry_ratio_min'])
        exit_sma_buffer = float(self.parameters['exit_sma_buffer'])
        exit_rsi_min = float(self.parameters['exit_rsi_min'])
        trail_activation_gain = float(self.parameters['trail_activation_gain'])
        trail_pct = float(self.parameters['trail_pct'])
        panic_stop_pct = float(self.parameters['panic_stop_pct'])
        max_hold_bars = int(self.parameters['max_hold_bars'])

        state = _State()

        for i in range(len(df)):
            px = price[i]
            sm = sma[i]
            rs = rsi[i]
            atr_ratio_i = atr_ratio[i]

            if np.isnan(px) or np.isnan(sm) or np.isnan(rs):
                state.bars_in_trade += int(state.in_position)
                continue

            if not state.in_position:
                entry_band = sm * (1.0 - entry_drop)
                atr_ok = atr_ratio_i >= atr_entry_ratio_min
                if px <= entry_band and rs <= rsi_max and atr_ok:
                    df.iat[i, df.columns.get_loc('buy_signal')] = True
                    df.iat[i, df.columns.get_loc('signal_strength')] = 1.0
                    state = _State(
                        in_position=True,
                        entry_price=px,
                        entry_sma=sm,
                        peak_price=px,
                        trail_armed=False,
                        bars_in_trade=0,
                    )
            else:
                state.bars_in_trade += 1
                state.peak_price = max(state.peak_price, px)

                if not state.trail_armed and state.peak_price >= state.entry_price * (1.0 + trail_activation_gain):
                    state.trail_armed = True

                ema_regime_up = (not np.isnan(ema_fast[i])) and (not np.isnan(ema_slow[i])) and ema_fast[i] >= ema_slow[i]
                ema_accelerating = ema_fast_slope[i] > 0

                exit_reasons = []

                if state.entry_price > 0 and px <= state.entry_price * (1.0 - panic_stop_pct):
                    exit_reasons.append('panic')

                if state.trail_armed and state.peak_price > 0:
                    if px <= state.peak_price * (1.0 - trail_pct):
                        exit_reasons.append('trailing')

                if px >= sm * (1.0 + exit_sma_buffer) and rs >= exit_rsi_min:
                    if not ema_regime_up or not ema_accelerating:
                        exit_reasons.append('sma_exit')

                if state.bars_in_trade >= max_hold_bars:
                    exit_reasons.append('time')

                if exit_reasons:
                    df.iat[i, df.columns.get_loc('sell_signal')] = True
                    df.iat[i, df.columns.get_loc('signal_strength')] = 1.0
                    state = _State()

        return df

