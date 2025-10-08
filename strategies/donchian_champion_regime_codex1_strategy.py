"""Regime-gated Donchian breakout with dynamic trailing stop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

from .base_strategy import BaseStrategy


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=max(1, span), adjust=False).mean()


def _atr_ratio(df: pd.DataFrame, period: int) -> pd.Series:
    high = df['high'] if 'high' in df.columns else df['price']
    low = df['low'] if 'low' in df.columns else df['price']
    close = df['price'] if 'price' in df.columns else df['close']
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(period, min_periods=period).mean()
    return (atr / close).fillna(0.0)


@dataclass
class _TradeState:
    in_position: bool = False
    peak_price: float = 0.0
    entry_price: float = 0.0
    cooldown: int = 0


class DonchianChampionRegimeCodex1Strategy(BaseStrategy):
    """Donchian breakout with EMA regime gating and volatility-aware trailing."""

    def __init__(self, parameters: Dict | None = None):
        defaults = {
            'entry_days': 11.0,
            'exit_days': 2.0,
            'ema_exit_days': 3.0,
            'regime_fast_days': 1.0,
            'regime_slow_days': 5.0,
            'min_regime_slope': 0.0008,
            'exit_regime_slope': -0.0010,
            'macro_days': 14.0,
            'macro_min_slope': 0.0008,
            'macro_exit_slope': -0.0006,
            'macro_buffer_pct': 0.020,
            'atr_days': 1.0,
            'trail_base': 0.18,
            'trail_k': 0.45,
            'trail_min': 0.12,
            'trail_max': 0.32,
            'max_atr_ratio': 0.18,
            'cooldown_hours': 24.0,
            'post_loss_cooldown_hours': 240.0,
            'timeframe_minutes': 5,
        }
        if parameters:
            defaults.update(parameters)
        super().__init__('DonchianChampionRegimeCodex1Strategy', defaults)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _window_bars(self, days: float) -> int:
        timeframe = float(self.parameters.get('timeframe_minutes', 5))
        bars = int(round(days * 24 * 60 / timeframe))
        return max(1, bars)

    def _hours_to_bars(self, hours: float) -> int:
        timeframe = float(self.parameters.get('timeframe_minutes', 5))
        bars = int(round(hours * 60 / timeframe))
        return max(1, bars)

    # ------------------------------------------------------------------
    # Indicator pipeline
    # ------------------------------------------------------------------
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        if not self.validate_data(data):
            return data

        df = data.copy()
        price = df['price'] if 'price' in df.columns else df['close']
        df['price'] = price

        entry_bars = self._window_bars(float(self.parameters['entry_days']))
        exit_bars = self._window_bars(float(self.parameters['exit_days']))
        ema_exit_bars = self._window_bars(float(self.parameters['ema_exit_days']))
        fast_bars = self._window_bars(float(self.parameters['regime_fast_days']))
        slow_bars = self._window_bars(float(self.parameters['regime_slow_days']))
        macro_bars = self._window_bars(float(self.parameters['macro_days']))
        atr_bars = self._window_bars(float(self.parameters['atr_days']))

        slope_bars = max(1, fast_bars)
        macro_slope_bars = max(1, self._window_bars(float(self.parameters['macro_days']) / 2))

        df['donchian_high'] = price.rolling(entry_bars, min_periods=entry_bars).max().shift(1)
        df['donchian_low'] = price.rolling(exit_bars, min_periods=exit_bars).min().shift(1)
        df['ema_exit'] = _ema(price, ema_exit_bars)
        df['ema_fast'] = _ema(price, fast_bars)
        df['ema_slow'] = _ema(price, slow_bars)
        df['ema_fast_slope'] = df['ema_fast'].diff(slope_bars) / df['ema_fast'].shift(slope_bars)
        df['ema_macro'] = _ema(price, macro_bars)
        df['ema_macro_slope'] = df['ema_macro'].diff(macro_slope_bars) / df['ema_macro'].shift(macro_slope_bars)
        df['atr_ratio'] = _atr_ratio(df, atr_bars)
        df['atr_smooth'] = df['atr_ratio'].rolling(slope_bars, min_periods=slope_bars).mean()

        return df

    # ------------------------------------------------------------------
    # Signal logic
    # ------------------------------------------------------------------
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        if 'donchian_high' not in df.columns:
            df = self.calculate_indicators(df)

        df['buy_signal'] = False
        df['sell_signal'] = False
        df['signal_strength'] = 0.0

        price = df['price'].to_numpy(float)
        highs = df['donchian_high'].to_numpy(float)
        lows = df['donchian_low'].to_numpy(float)
        ema_exit = df['ema_exit'].to_numpy(float)
        ema_fast = df['ema_fast'].to_numpy(float)
        ema_slow = df['ema_slow'].to_numpy(float)
        fast_slope = df['ema_fast_slope'].to_numpy(float)
        ema_macro = df['ema_macro'].to_numpy(float)
        macro_slope = df['ema_macro_slope'].to_numpy(float)
        atr_ratio = df['atr_ratio'].to_numpy(float)
        atr_smooth = df['atr_smooth'].to_numpy(float)

        min_regime_slope = float(self.parameters['min_regime_slope'])
        exit_regime_slope = float(self.parameters['exit_regime_slope'])
        macro_min_slope = float(self.parameters['macro_min_slope'])
        macro_exit_slope = float(self.parameters['macro_exit_slope'])
        macro_buffer = float(self.parameters['macro_buffer_pct'])
        max_atr_ratio = float(self.parameters['max_atr_ratio'])

        trail_base = float(self.parameters['trail_base'])
        trail_k = float(self.parameters['trail_k'])
        trail_min = float(self.parameters['trail_min'])
        trail_max = float(self.parameters['trail_max'])

        cooldown_bars = self._hours_to_bars(float(self.parameters['cooldown_hours']))
        loss_cooldown_bars = self._hours_to_bars(float(self.parameters['post_loss_cooldown_hours']))

        state = _TradeState(False, 0.0, 0.0, 0)

        for i in range(len(df)):
            px = price[i]
            hi = highs[i]
            lo = lows[i]
            ema_e = ema_exit[i]
            fast = ema_fast[i]
            slow = ema_slow[i]
            fast_sl = fast_slope[i]
            macro = ema_macro[i]
            macro_sl = macro_slope[i]
            vol = atr_ratio[i]
            vol_avg = atr_smooth[i]

            if np.isnan(px) or np.isnan(fast) or np.isnan(slow):
                if state.cooldown > 0:
                    state.cooldown -= 1
                continue

            regime_ok = fast > slow and (np.isnan(fast_sl) or fast_sl >= min_regime_slope)
            macro_ok = (
                not np.isnan(macro)
                and px >= macro * (1.0 + macro_buffer)
                and (np.isnan(macro_sl) or macro_sl >= macro_min_slope)
            )
            vol_ok = np.isnan(vol_avg) or vol_avg <= max_atr_ratio

            if not state.in_position:
                if state.cooldown > 0:
                    state.cooldown -= 1
                    continue

                if (
                    not np.isnan(hi)
                    and px >= hi
                    and regime_ok
                    and macro_ok
                    and vol_ok
                ):
                    df.iat[i, df.columns.get_loc('buy_signal')] = True
                    df.iat[i, df.columns.get_loc('signal_strength')] = 1.0
                    state.in_position = True
                    state.peak_price = px
                    state.entry_price = px
            else:
                state.peak_price = max(state.peak_price, px)
                dyn_dd = trail_base + trail_k * (vol if not np.isnan(vol) else 0.0)
                dyn_dd = min(max(dyn_dd, trail_min), trail_max)

                trail_stop = state.peak_price > 0 and px <= state.peak_price * (1.0 - dyn_dd)
                exit_donchian = (
                    not np.isnan(lo)
                    and px <= lo
                    and not np.isnan(ema_e)
                    and px < ema_e
                )
                regime_break = fast <= slow or (not np.isnan(fast_sl) and fast_sl <= exit_regime_slope)
                macro_break = (
                    np.isnan(macro)
                    or px <= macro * (1.0 - macro_buffer)
                    or (not np.isnan(macro_sl) and macro_sl <= macro_exit_slope)
                )
                vol_spike = not np.isnan(vol) and vol > max_atr_ratio * 1.3

                if trail_stop or exit_donchian or regime_break or macro_break or vol_spike:
                    df.iat[i, df.columns.get_loc('sell_signal')] = True
                    df.iat[i, df.columns.get_loc('signal_strength')] = 1.0
                    loss_exit = state.entry_price > 0 and px < state.entry_price
                    state.in_position = False
                    state.peak_price = 0.0
                    state.entry_price = 0.0
                    state.cooldown = loss_cooldown_bars if loss_exit else cooldown_bars

            if not state.in_position and state.cooldown > 0 and not df.iat[i, df.columns.get_loc('sell_signal')]:
                state.cooldown = max(state.cooldown - 1, 0)

        return df

