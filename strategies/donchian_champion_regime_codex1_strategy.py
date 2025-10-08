"""Regime-gated Donchian breakout variant with adaptive drawdown control."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from .donchian_champion_strategy import DonchianChampionDynamicStrategy, _ema, _atr_ratio


class DonchianChampionRegimeCodex1Strategy(DonchianChampionDynamicStrategy):
    """Donchian breakout that only trades when multi-timeframe trend regime agrees.

    This subclass builds on the Champion dynamic drawdown variant but layers in
    three additional protections designed around the real swap-cost model:

    1. **Trend confirmation** across 1-day, 3-day and 10-day EMAs with an
       explicit slope check. Entries require the shorter EMAs to sit above the
       longer baselines and the 1-day EMA slope to be positive.
    2. **Macro rate-of-change and drawdown guards** to avoid opening new
       breakouts while HEX is still in a structural downtrend. This slashes the
       choppy trading that previously bled swap costs in bear phases.
    3. **Cooldown and recovery filters** to ensure re-entries only happen after
       a meaningful rebound, preventing immediate churn around failed breakouts.

    The defaults were tuned offline with the real `swap_cost_cache.json` buckets
    and aim to deliver high full-period returns while remaining positive over
    the recent 3-month and 1-month windows.
    """

    def __init__(self, parameters: Dict | None = None):
        defaults = {
            'entry_days': 11.0,
            'exit_days': 2.0,
            'ema_exit_days': 3.0,
            'dd_base': 0.14,
            'dd_k': 0.55,
            'gain_weight': 0.12,
            'dd_min': 0.11,
            'dd_max': 0.42,
            'atr_days': 1.0,
            'entry_buffer_frac': 0.0012,
            'regime_fast_days': 1.0,
            'regime_slow_days': 3.0,
            'regime_trend_days': 10.0,
            'regime_fast_margin': 0.001,
            'regime_trend_margin': 0.0,
            'slope_days': 1.0,
            'slope_min': 0.001,
            'roc_days': 21.0,
            'roc_min': 0.07,
            'drawdown_lookback_days': 30.0,
            'drawdown_min': -0.3,
            'max_atr_ratio': 0.3,
            'cooldown_days': 1.5,
            'post_exit_recovery': 0.08,
            'regime_exit_margin': -0.002,
            'trend_break_exit_margin': -0.01,
            'trend_slope_days': 3.0,
            'trend_slope_min': 0.012,
            'price_trend_margin': 0.01,
            'warmup_days': 60.0,
            'timeframe_minutes': 5,
        }
        if parameters:
            defaults.update(parameters)
        super().__init__(defaults)
        # Ensure a unique strategy name for reporting.
        self.name = 'DonchianChampionRegimeCodex1Strategy'

    # Helper to convert day-based inputs to bar counts.
    def _window_bars(self, days: float) -> int:
        timeframe = float(self.parameters.get('timeframe_minutes', 5))
        bars = int(round(days * 24 * 60 / timeframe))
        return max(1, bars)

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        df = super().calculate_indicators(data)
        if df.empty:
            return df

        price = df['price'] if 'price' in df.columns else df['close']
        df['price'] = price

        fast_span = self._window_bars(float(self.parameters['regime_fast_days']))
        slow_span = self._window_bars(float(self.parameters['regime_slow_days']))
        trend_span = self._window_bars(float(self.parameters['regime_trend_days']))
        slope_span = self._window_bars(float(self.parameters['slope_days']))
        roc_span = self._window_bars(float(self.parameters['roc_days']))
        drawdown_span = self._window_bars(float(self.parameters['drawdown_lookback_days']))
        trend_slope_span = self._window_bars(float(self.parameters['trend_slope_days']))

        df['ema_fast_regime'] = _ema(price, fast_span)
        df['ema_slow_regime'] = _ema(price, slow_span)
        df['ema_trend_regime'] = _ema(price, trend_span)

        df['ema_fast_slope'] = df['ema_fast_regime'] / df['ema_fast_regime'].shift(slope_span) - 1.0
        df['roc_long'] = price / price.shift(roc_span) - 1.0
        df['ema_trend_slope'] = df['ema_trend_regime'] / df['ema_trend_regime'].shift(trend_slope_span) - 1.0

        rolling_peak = price.rolling(drawdown_span, min_periods=1).max()
        df['drawdown_from_peak'] = price / rolling_peak - 1.0

        # Carry over ATR ratio from parent for volatility gating. Ensure column exists.
        if 'atr_ratio' not in df.columns:
            atr_period = self._window_bars(float(self.parameters['atr_days']))
            df['atr_ratio'] = _atr_ratio(df, atr_period)

        return df

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        if 'ema_fast_regime' not in df.columns:
            df = self.calculate_indicators(df)
        if df.empty:
            return df

        df['buy_signal'] = False
        df['sell_signal'] = False
        df['signal_strength'] = 0.0

        price = df['price'].to_numpy(float)
        highs = df['donchian_high'].to_numpy(float) if 'donchian_high' in df.columns else np.full(len(df), np.nan)
        lows = df['donchian_low'].to_numpy(float) if 'donchian_low' in df.columns else np.full(len(df), np.nan)
        ema_exit = df['ema_exit'].to_numpy(float) if 'ema_exit' in df.columns else np.full(len(df), np.nan)
        atr_ratio = df['atr_ratio'].to_numpy(float)

        ema_fast = df['ema_fast_regime'].to_numpy(float)
        ema_slow = df['ema_slow_regime'].to_numpy(float)
        ema_trend = df['ema_trend_regime'].to_numpy(float)
        ema_slope = df['ema_fast_slope'].to_numpy(float)
        roc_long = df['roc_long'].to_numpy(float)
        dd_series = df['drawdown_from_peak'].to_numpy(float)
        trend_slope = df['ema_trend_slope'].to_numpy(float)

        dd_base = float(self.parameters['dd_base'])
        dd_k = float(self.parameters['dd_k'])
        gain_weight = float(self.parameters.get('gain_weight', 0.0))
        dd_min = float(self.parameters['dd_min'])
        dd_max = float(self.parameters['dd_max'])
        entry_buffer = float(self.parameters.get('entry_buffer_frac', 0.0))

        fast_margin = float(self.parameters['regime_fast_margin'])
        trend_margin = float(self.parameters['regime_trend_margin'])
        slope_min = float(self.parameters['slope_min'])
        roc_min = float(self.parameters['roc_min'])
        drawdown_min = float(self.parameters['drawdown_min'])
        max_atr = float(self.parameters['max_atr_ratio'])
        cooldown_bars = self._window_bars(float(self.parameters['cooldown_days']))
        post_exit_recovery = float(self.parameters['post_exit_recovery'])
        regime_exit_margin = float(self.parameters['regime_exit_margin'])
        trend_break_exit_margin = float(self.parameters['trend_break_exit_margin'])
        trend_slope_min = float(self.parameters['trend_slope_min'])
        price_trend_margin = float(self.parameters['price_trend_margin'])
        warmup_bars = self._window_bars(float(self.parameters['warmup_days']))

        in_position = False
        peak_price = 0.0
        entry_price = 0.0
        last_exit_idx = -cooldown_bars
        last_exit_price = np.nan

        for i in range(len(df)):
            px = price[i]
            hi = highs[i]
            lo = lows[i]
            ema_val = ema_exit[i]
            atr_val = atr_ratio[i]
            ema_fast_val = ema_fast[i]
            ema_slow_val = ema_slow[i]
            ema_trend_val = ema_trend[i]
            slope_val = ema_slope[i]
            roc_val = roc_long[i]
            dd_val = dd_series[i]
            trend_slope_val = trend_slope[i]

            if np.isnan(px) or np.isnan(hi):
                continue
            if i < warmup_bars:
                continue

            if not in_position:
                if i - last_exit_idx < cooldown_bars:
                    continue
                if not np.isnan(last_exit_price) and px < last_exit_price * (1.0 + post_exit_recovery):
                    continue

                cond_regime = (
                    not np.isnan(ema_fast_val)
                    and not np.isnan(ema_slow_val)
                    and not np.isnan(ema_trend_val)
                    and ema_fast_val > ema_slow_val * (1.0 + fast_margin)
                    and ema_slow_val > ema_trend_val * (1.0 + trend_margin)
                )
                cond_slope = not np.isnan(slope_val) and slope_val >= slope_min
                cond_roc = not np.isnan(roc_val) and roc_val >= roc_min
                cond_drawdown = not np.isnan(dd_val) and dd_val >= drawdown_min
                cond_atr = np.isnan(atr_val) or atr_val <= max_atr
                cond_trend_slope = (
                    not np.isnan(trend_slope_val) and trend_slope_val >= trend_slope_min
                )
                cond_price_trend = (
                    not np.isnan(ema_trend_val) and px >= ema_trend_val * (1.0 + price_trend_margin)
                )

                if (
                    cond_regime
                    and cond_slope
                    and cond_roc
                    and cond_drawdown
                    and cond_atr
                    and cond_trend_slope
                    and cond_price_trend
                ):
                    if px >= hi * (1.0 + entry_buffer):
                        df.iat[i, df.columns.get_loc('buy_signal')] = True
                        df.iat[i, df.columns.get_loc('signal_strength')] = 1.0
                        in_position = True
                        peak_price = px
                        entry_price = px
            else:
                peak_price = max(peak_price, px)

                atr_adj = atr_ratio[i] if not np.isnan(atr_ratio[i]) else 0.0
                dd_dyn = dd_base + dd_k * atr_adj
                if entry_price > 0 and peak_price > entry_price and gain_weight > 0:
                    gain = (peak_price / entry_price) - 1.0
                    if gain > 0:
                        dd_dyn += gain_weight * gain
                dd_dyn = min(max(dd_dyn, dd_min), dd_max)

                exit_donchian = (
                    (not np.isnan(lo))
                    and px <= lo
                    and (not np.isnan(ema_val))
                    and px < ema_val
                )
                exit_drawdown = peak_price > 0 and px <= peak_price * (1.0 - dd_dyn)

                regime_break = False
                if not np.isnan(ema_fast_val) and not np.isnan(ema_slow_val):
                    if ema_fast_val <= ema_slow_val * (1.0 + regime_exit_margin):
                        regime_break = True
                if not np.isnan(ema_slow_val) and not np.isnan(ema_trend_val):
                    if ema_slow_val <= ema_trend_val * (1.0 + trend_break_exit_margin):
                        regime_break = True

                if exit_donchian or exit_drawdown or regime_break:
                    df.iat[i, df.columns.get_loc('sell_signal')] = True
                    df.iat[i, df.columns.get_loc('signal_strength')] = 1.0
                    in_position = False
                    peak_price = 0.0
                    entry_price = 0.0
                    last_exit_idx = i
                    last_exit_price = px

        return df
