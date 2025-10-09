"""Cost-aware SMA Reversion strategy adapted for BaseStrategy interface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

from .base_strategy import BaseStrategy
from .indicator_utils import compute_adx, compute_multi_timeframe_adx, compute_atr


def _sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=window).mean()


def _rsi(series: pd.Series, window: int) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    avg_gain = up.rolling(window, min_periods=window).mean()
    avg_loss = down.rolling(window, min_periods=window).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    rsi = 100 - (100 / (1 + rs))
    return rsi


class CSMARevertStrategy(BaseStrategy):
    """SMA reversion with RSI filter and configurable entry/exit bands."""

    def __init__(self, parameters: Dict | None = None):
        defaults = {
            'n_sma': 576,          # ~2 days on 5m bars
            'entry_drop': 0.25,    # 25% below SMA to enter
            'exit_up': 0.048,      # tuned exit overshoot ~=4.8%
            'rsi_period': 14,
            'rsi_max': 32.0,
            'atr_period': 48,      # ~4 hours of 5m bars
            'atr_mult': 1.8,       # trailing stop multiple
            'atr_floor_pct': 0.003,  # 0.3% price floor for ATR
            'timeframe_minutes': 5,
            'adx_period': 14,
            'adx_htf_period': 14,
            'adx_htf_minutes': 60,
            'adx_range_threshold': 22.0,
            'time_stop_bars': 12,
        }
        if parameters:
            defaults.update(parameters)
        super().__init__('CSMARevertStrategy', defaults)

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        if not self.validate_data(data):
            return data

        df = data.copy()
        price = df['price'] if 'price' in df else df['close']
        df['price'] = price
        n_sma = int(self.parameters['n_sma'])
        rsi_period = int(self.parameters['rsi_period'])
        atr_period = int(self.parameters.get('atr_period', 48))
        atr_floor_pct = float(self.parameters.get('atr_floor_pct', 0.0))

        df['sma'] = _sma(price, n_sma)
        df['rsi'] = _rsi(price, rsi_period)
        high = df['high'] if 'high' in df.columns else price
        low = df['low'] if 'low' in df.columns else price
        close = df['close'] if 'close' in df.columns else price
        atr_series = compute_atr(high, low, close, atr_period).ffill().fillna(0.0)
        if atr_floor_pct > 0:
            atr_series = atr_series.clip(lower=price * atr_floor_pct)
        df['atr_abs'] = atr_series

        adx_df = compute_adx(high, low, close, int(self.parameters['adx_period']))
        df['adx'] = adx_df['adx']
        df['adx_htf'] = compute_multi_timeframe_adx(
            df[['timestamp', 'open', 'high', 'low', 'close']].copy(),
            period=int(self.parameters['adx_htf_period']),
            timeframe_minutes=int(self.parameters['adx_htf_minutes']),
        )
        adx_threshold = float(self.parameters['adx_range_threshold'])
        df['is_range'] = (df['adx'] <= adx_threshold) & (df['adx_htf'] <= adx_threshold)

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
        atr_abs = df.get('atr_abs', pd.Series(0.0, index=df.index)).to_numpy(float)
        range_mask = df.get('is_range', pd.Series(True, index=df.index)).to_numpy(bool)

        entry_drop = float(self.parameters['entry_drop'])
        exit_up = float(self.parameters['exit_up'])
        rsi_max = float(self.parameters['rsi_max'])
        atr_mult = float(self.parameters.get('atr_mult', 1.8))
        time_stop_bars = int(self.parameters.get('time_stop_bars', 0))

        in_position = False
        trail_stop = None
        peak_price = None
        bars_in_trade = 0
        for i in range(len(df)):
            px = price[i]
            sm = sma[i]
            rs = rsi[i]

            if np.isnan(sm) or np.isnan(rs):
                continue

            if not in_position:
                if not range_mask[i]:
                    continue
                if px <= sm * (1.0 - entry_drop) and rs <= rsi_max:
                    df.iat[i, df.columns.get_loc('buy_signal')] = True
                    df.iat[i, df.columns.get_loc('signal_strength')] = 1.0
                    in_position = True
                    trail_stop = None
                    peak_price = px
                    bars_in_trade = 0
            else:
                bars_in_trade += 1
                # Update trailing stop with latest ATR reading and peak price
                if peak_price is None or px > peak_price:
                    peak_price = px
                if atr_abs[i] > 0 and px >= sm:
                    candidate = peak_price - atr_mult * atr_abs[i]
                    candidate = min(px, candidate)
                    candidate = max(0.0, candidate)
                    if trail_stop is None:
                        trail_stop = candidate
                    else:
                        trail_stop = max(trail_stop, candidate)
                # Profit target exit
                if px >= sm * (1.0 + exit_up):
                    df.iat[i, df.columns.get_loc('sell_signal')] = True
                    df.iat[i, df.columns.get_loc('signal_strength')] = 1.0
                    in_position = False
                    trail_stop = None
                    peak_price = None
                # Protective trailing stop
                elif trail_stop is not None and px <= trail_stop:
                    df.iat[i, df.columns.get_loc('sell_signal')] = True
                    df.iat[i, df.columns.get_loc('signal_strength')] = 0.75
                    in_position = False
                    trail_stop = None
                    peak_price = None
                    bars_in_trade = 0
                elif time_stop_bars > 0 and bars_in_trade >= time_stop_bars:
                    df.iat[i, df.columns.get_loc('sell_signal')] = True
                    df.iat[i, df.columns.get_loc('signal_strength')] = 0.5
                    in_position = False
                    trail_stop = None
                    peak_price = None
                    bars_in_trade = 0

        return df
