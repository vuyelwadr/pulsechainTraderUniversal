"""Deep Pullback Re-entry Strategy.

Waits for a 75%+ drawdown from the 120-day high and then only re-enters once
price begins recovering with positive short-term momentum. Designed to trade
extremely infrequently while capturing post-crash rebounds despite high swap
costs.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from .base_strategy import BaseStrategy


class DeepPullbackReentryStrategy(BaseStrategy):
    """Buy deep capitulation recoveries, exit when recovery matures or fails."""

    def __init__(self, parameters: Dict | None = None):
        defaults = {
            'lookback_high': 34560,   # 120 days of 5m bars
            'entry_drawdown': -0.75,  # -75%
            'exit_drawdown': -0.35,   # exit once drawdown shrinks inside -35%
            'momentum_window': 288,   # 1 day momentum check
            'ema_span': 288,          # 1 day EMA for recovery confirmation
            'stop_loss_pct': 0.25,    # fail-safe below entry
            'trail_drawdown_pct': 0.4,
            'min_hold_bars': 288,     # at least 1 day before exits allowed
            'timeframe_minutes': 5,
        }
        if parameters:
            defaults.update(parameters)
        super().__init__('DeepPullbackReentryStrategy', defaults)

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        if not self.validate_data(data):
            return data

        df = data.copy()
        price = df['close'] if 'close' in df else df['price']
        df['price'] = price

        lookback = int(self.parameters['lookback_high'])
        df['rolling_max'] = price.rolling(lookback).max()
        df['drawdown'] = price / df['rolling_max'] - 1.0
        momentum_window = int(self.parameters['momentum_window'])
        df['momentum'] = price.pct_change(momentum_window)
        ema_span = int(self.parameters['ema_span'])
        df['ema'] = price.ewm(span=ema_span, adjust=False).mean()

        return df

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        if 'drawdown' not in df.columns:
            df = self.calculate_indicators(df)

        df['buy_signal'] = False
        df['sell_signal'] = False
        df['signal_strength'] = 0.0

        entry_dd = float(self.parameters['entry_drawdown'])
        exit_dd = float(self.parameters['exit_drawdown'])
        stop_loss_pct = float(self.parameters['stop_loss_pct'])
        trail_pct = float(self.parameters['trail_drawdown_pct'])
        min_hold = int(self.parameters['min_hold_bars'])

        drawdown = df['drawdown'].to_numpy()
        momentum = df['momentum'].to_numpy()
        ema = df['ema'].to_numpy()
        price = df['price'].to_numpy()

        buy_flags = np.zeros(len(df), dtype=bool)
        sell_flags = np.zeros(len(df), dtype=bool)
        strength = np.zeros(len(df))

        in_position = False
        entry_price = 0.0
        peak_price = 0.0
        entry_index = -1

        for i in range(len(df)):
            if np.isnan(drawdown[i]) or np.isnan(momentum[i]) or np.isnan(ema[i]):
                continue

            if not in_position:
                qualifies = (
                    drawdown[i] <= entry_dd
                    and momentum[i] > 0
                    and price[i] > ema[i]
                )
                if qualifies:
                    buy_flags[i] = True
                    strength[i] = min(1.0, abs(drawdown[i] / entry_dd))
                    in_position = True
                    entry_price = price[i]
                    peak_price = price[i]
                    entry_index = i
            else:
                peak_price = max(peak_price, price[i])
                hold_duration = i - entry_index
                recovered = drawdown[i] >= exit_dd
                stop_hit = price[i] <= entry_price * (1.0 - stop_loss_pct)
                trail_hit = peak_price > 0 and price[i] <= peak_price * (1.0 - trail_pct)
                if hold_duration >= min_hold and (recovered or stop_hit or trail_hit):
                    sell_flags[i] = True
                    strength[i] = 1.0
                    in_position = False
                    entry_price = 0.0
                    peak_price = 0.0
                    entry_index = -1

        df['buy_signal'] = buy_flags
        df['sell_signal'] = sell_flags
        df['signal_strength'] = strength

        return df

