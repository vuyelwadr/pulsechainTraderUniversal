"""Ultra selective multi-week breakout strategy."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from .base_strategy import BaseStrategy


class MultiWeekBreakoutUltraStrategy(BaseStrategy):
    """Breakouts on 8-week highs with strict volume confirmation and wide stops."""

    def __init__(self, parameters: Dict | None = None):
        defaults = {
            'lookback_breakout': 16128,  # ~8 weeks of 5m bars
            'confirmation_window': 1152,  # 4-day confirmation
            'exit_lookback': 1152,
            'volume_window': 1152,
            'volume_multiplier': 1.25,
            'trail_drawdown_pct': 0.28,
            'min_hold_bars': 2304,  # 8 days
            'timeframe_minutes': 5,
        }
        if parameters:
            defaults.update(parameters)
        super().__init__('MultiWeekBreakoutUltraStrategy', defaults)

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        if not self.validate_data(data):
            return data

        df = data.copy()
        price = df['close'] if 'close' in df else df['price']
        df['price'] = price

        breakout_window = int(self.parameters['lookback_breakout'])
        confirm_window = int(self.parameters['confirmation_window'])
        exit_window = int(self.parameters['exit_lookback'])
        volume_window = int(self.parameters['volume_window'])

        df['breakout_high'] = price.rolling(breakout_window).max().shift(1)
        df['confirm_high'] = price.rolling(confirm_window).max().shift(1)
        df['exit_low'] = price.rolling(exit_window).min().shift(1)
        df['volume_ma'] = df['volume'].rolling(volume_window).mean()

        return df

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        if 'breakout_high' not in df.columns:
            df = self.calculate_indicators(df)

        df['buy_signal'] = False
        df['sell_signal'] = False
        df['signal_strength'] = 0.0

        vol_multiplier = float(self.parameters['volume_multiplier'])
        trail_pct = float(self.parameters['trail_drawdown_pct'])
        min_hold = int(self.parameters['min_hold_bars'])

        price = df['price'].to_numpy()
        breakout_high = df['breakout_high'].to_numpy()
        confirm_high = df['confirm_high'].to_numpy()
        exit_low = df['exit_low'].to_numpy()
        volume = df['volume'].to_numpy()
        volume_ma = df['volume_ma'].to_numpy()

        buy_flags = np.zeros(len(df), dtype=bool)
        sell_flags = np.zeros(len(df), dtype=bool)
        strength = np.zeros(len(df))

        in_position = False
        entry_index = -1
        peak_price = 0.0

        for i in range(len(df)):
            if np.isnan(breakout_high[i]) or np.isnan(confirm_high[i]):
                continue

            vol_ok = True
            if not np.isnan(volume_ma[i]) and volume_ma[i] > 0:
                vol_ok = volume[i] >= volume_ma[i] * vol_multiplier

            if not in_position:
                breakout = price[i] > breakout_high[i] and price[i] > confirm_high[i]
                if breakout and vol_ok:
                    buy_flags[i] = True
                    strength[i] = 0.8
                    in_position = True
                    entry_index = i
                    peak_price = price[i]
            else:
                peak_price = max(peak_price, price[i])
                hold_duration = i - entry_index
                exit_trigger = False
                if hold_duration >= min_hold and not np.isnan(exit_low[i]):
                    exit_trigger = price[i] < exit_low[i]
                trail_hit = peak_price > 0 and price[i] <= peak_price * (1.0 - trail_pct)

                if exit_trigger or trail_hit:
                    sell_flags[i] = True
                    strength[i] = 1.0
                    in_position = False
                    entry_index = -1
                    peak_price = 0.0

        df['buy_signal'] = buy_flags
        df['sell_signal'] = sell_flags
        df['signal_strength'] = strength

        return df

