"""Cost-aware variant of the CSMA reversion play with stricter crash gating."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

from .base_strategy import BaseStrategy


def _sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=window).mean()


def _compute_rsi(series: pd.Series, window: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    alpha = 1.0 / max(window, 1)
    avg_gain = gain.ewm(alpha=alpha, adjust=False).mean()
    avg_loss = loss.ewm(alpha=alpha, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


def _compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
    prev_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(window).mean()


@dataclass
class _TradeState:
    in_position: bool = False
    entry_index: int = -1
    peak_price: float = 0.0
    last_exit_index: int = -100000


class CSMARevertPro1Strategy(BaseStrategy):
    """Deep crash mean-revert tuned for high notional buckets via reduced churn."""

    def __init__(self, parameters: Dict | None = None):
        defaults = {
            'n_sma': 576,
            'entry_drop': 0.30,
            'exit_up': 0.07,
            'rsi_period': 21,
            'rsi_max': 35.0,
            'rsi_exit': 65.0,
            'atr_window': 288,
            'atr_min_norm': 0.015,
            'crash_window': 2880,
            'crash_drawdown': -0.65,
            'trail_pct': 0.20,
            'cooldown_bars': 1440,
            'min_hold_bars': 288,
            'timeframe_minutes': 5,
        }
        if parameters:
            defaults.update(parameters)
        super().__init__('CSMARevertPro1Strategy', defaults)

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        if not self.validate_data(data):
            return data

        df = data.copy()
        price = df['price'] if 'price' in df else df['close']
        df['price'] = price

        n_sma = int(self.parameters['n_sma'])
        rsi_period = int(self.parameters['rsi_period'])
        atr_window = int(self.parameters['atr_window'])
        crash_window = int(self.parameters['crash_window'])

        df['sma'] = _sma(price, n_sma)
        df['rsi'] = _compute_rsi(price, rsi_period)

        high = df['high'] if 'high' in df else price
        low = df['low'] if 'low' in df else price
        df['atr'] = _compute_atr(high, low, price, atr_window)
        df['atr_norm'] = (df['atr'] / price).replace([np.inf, -np.inf], np.nan)

        df['rolling_max'] = price.rolling(crash_window).max()
        df['drawdown'] = price / df['rolling_max'] - 1.0

        return df

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        if 'sma' not in df.columns:
            df = self.calculate_indicators(df)

        df['buy_signal'] = False
        df['sell_signal'] = False
        df['signal_strength'] = 0.0

        params = self.parameters
        entry_drop = float(params['entry_drop'])
        exit_up = float(params['exit_up'])
        rsi_max = float(params['rsi_max'])
        rsi_exit = float(params['rsi_exit'])
        atr_min = float(params['atr_min_norm'])
        crash_drawdown = float(params['crash_drawdown'])
        trail_pct = float(params['trail_pct'])
        cooldown = int(params['cooldown_bars'])
        min_hold = int(params['min_hold_bars'])

        price = df['price'].to_numpy()
        sma = df['sma'].to_numpy()
        rsi = df['rsi'].to_numpy()
        atr_norm = df['atr_norm'].to_numpy()
        drawdown = df['drawdown'].to_numpy()

        state = _TradeState()

        for i in range(len(df)):
            px = price[i]
            sm = sma[i]
            rs = rsi[i]
            atr_i = atr_norm[i]
            dd = drawdown[i]

            if not np.isfinite(px) or np.isnan(sm) or np.isnan(rs):
                continue

            if not state.in_position:
                if i - state.last_exit_index < cooldown:
                    continue

                if atr_i < atr_min or np.isnan(atr_i) or np.isnan(dd):
                    continue

                if px <= sm * (1.0 - entry_drop) and rs <= rsi_max and dd <= crash_drawdown:
                    df.iat[i, df.columns.get_loc('buy_signal')] = True
                    df.iat[i, df.columns.get_loc('signal_strength')] = 1.0
                    state.in_position = True
                    state.entry_index = i
                    state.peak_price = px
            else:
                state.peak_price = max(state.peak_price, px)
                hold = i - state.entry_index
                exit_price = sm * (1.0 + exit_up) if not np.isnan(sm) else np.inf

                trail_hit = state.peak_price > 0 and px <= state.peak_price * (1.0 - trail_pct)
                target_hit = px >= exit_price
                rsi_relief = rs >= rsi_exit

                if hold >= min_hold and (target_hit or rsi_relief or trail_hit):
                    df.iat[i, df.columns.get_loc('sell_signal')] = True
                    df.iat[i, df.columns.get_loc('signal_strength')] = 1.0
                    state.in_position = False
                    state.entry_index = -1
                    state.peak_price = 0.0
                    state.last_exit_index = i

        return df

