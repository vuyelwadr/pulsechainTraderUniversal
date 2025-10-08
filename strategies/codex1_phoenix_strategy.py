"""Codex1 Phoenix Strategy: capture extreme crash recoveries."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from .base_strategy import BaseStrategy


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def _rsi(series: pd.Series, window: int) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    avg_gain = up.rolling(window, min_periods=window).mean()
    avg_loss = down.rolling(window, min_periods=window).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    rsi = 100 - (100 / (1 + rs))
    return rsi


class Codex1PhoenixStrategy(BaseStrategy):
    """Extreme drawdown recovery trend strategy."""

    def __init__(self, parameters: Dict | None = None):
        defaults = {
            'dd_lookback': 8640,
            'dd_threshold': -0.85,
            'rsi_period': 14,
            'rsi_max': 32.0,
            'momentum_span': 12,
            'ema_fast': 288,
            'ema_slow': 1728,
            'profit_activation': 2.50,
            'trailing_pct': 0.45,
            'hard_stop': 0.35,
            'max_hold_bars': 11520,
            'cooldown_bars': 576,
            'timeframe_minutes': 5,
        }
        if parameters:
            defaults.update(parameters)
        super().__init__('Codex1PhoenixStrategy', defaults)

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        if not self.validate_data(data):
            return data

        df = data.copy()
        price = df['price'] if 'price' in df else df['close']
        df['price'] = price

        dd_lookback = int(self.parameters['dd_lookback'])
        momentum_span = int(self.parameters['momentum_span'])
        ema_fast_span = int(self.parameters['ema_fast'])
        ema_slow_span = int(self.parameters['ema_slow'])
        rsi_period = int(self.parameters['rsi_period'])

        df['rolling_max'] = price.rolling(dd_lookback, min_periods=dd_lookback).max()
        df['drawdown'] = price / df['rolling_max'] - 1.0
        df['momentum_ema'] = _ema(price, max(momentum_span, 5))
        df['ema_fast'] = _ema(price, ema_fast_span)
        df['ema_slow'] = _ema(price, ema_slow_span)
        df['rsi'] = _rsi(price, rsi_period)
        return df

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        if 'drawdown' not in df.columns:
            df = self.calculate_indicators(df)

        df['buy_signal'] = False
        df['sell_signal'] = False
        df['signal_strength'] = 0.0

        price = df['price'].to_numpy(float)
        drawdown = df['drawdown'].to_numpy(float)
        momentum = df['momentum_ema'].to_numpy(float)
        ema_fast = df['ema_fast'].to_numpy(float)
        ema_slow = df['ema_slow'].to_numpy(float)
        rsi = df['rsi'].to_numpy(float)

        dd_threshold = float(self.parameters['dd_threshold'])
        rsi_max = float(self.parameters['rsi_max'])
        profit_activation = float(self.parameters['profit_activation'])
        trailing_pct = float(self.parameters['trailing_pct'])
        hard_stop = float(self.parameters['hard_stop'])
        max_hold_bars = int(self.parameters['max_hold_bars'])
        cooldown_bars = int(self.parameters['cooldown_bars'])

        in_position = False
        entry_price = 0.0
        highest_price = 0.0
        entry_index = -1
        trailing_active = False
        last_exit_index = -cooldown_bars

        buy_col = df.columns.get_loc('buy_signal')
        sell_col = df.columns.get_loc('sell_signal')
        strength_col = df.columns.get_loc('signal_strength')

        for i in range(len(df)):
            px = price[i]
            dd = drawdown[i]
            mom = momentum[i]
            ef = ema_fast[i]
            es = ema_slow[i]
            rs = rsi[i]

            if np.isnan(px) or np.isnan(dd) or np.isnan(mom) or np.isnan(ef) or np.isnan(es) or np.isnan(rs):
                continue

            if not in_position:
                if i - last_exit_index < cooldown_bars:
                    continue

                if (
                    dd <= dd_threshold
                    and rs <= rsi_max
                    and px >= mom
                    and ef <= es * 1.05
                ):
                    df.iat[i, buy_col] = True
                    df.iat[i, strength_col] = 1.0
                    in_position = True
                    entry_price = px
                    highest_price = px
                    entry_index = i
                    trailing_active = False
            else:
                highest_price = max(highest_price, px)

                if px <= entry_price * (1.0 - hard_stop):
                    df.iat[i, sell_col] = True
                    df.iat[i, strength_col] = 1.0
                    in_position = False
                    trailing_active = False
                    last_exit_index = i
                    continue

                if not trailing_active and px >= entry_price * (1.0 + profit_activation):
                    trailing_active = True

                if trailing_active and px <= highest_price * (1.0 - trailing_pct):
                    df.iat[i, sell_col] = True
                    df.iat[i, strength_col] = 1.0
                    in_position = False
                    trailing_active = False
                    last_exit_index = i
                    continue

                if trailing_active and ef < es:
                    df.iat[i, sell_col] = True
                    df.iat[i, strength_col] = 1.0
                    in_position = False
                    trailing_active = False
                    last_exit_index = i
                    continue

                if max_hold_bars > 0 and i - entry_index >= max_hold_bars:
                    df.iat[i, sell_col] = True
                    df.iat[i, strength_col] = 1.0
                    in_position = False
                    trailing_active = False
                    last_exit_index = i

        return df
