"""Tight trend-following breakout strategy with regime confirmation."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from .base_strategy import BaseStrategy


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=max(1, span), adjust=False).mean()


class TightTrendFollowStrategy(BaseStrategy):
    """Detects strong uptrends and rides breakouts with a trailing stop."""

    def __init__(self, parameters: Dict | None = None):
        defaults = {
            'timeframe_minutes': 5,
            'ema_fast_days': 1.0,
            'ema_mid_days': 3.0,
            'ema_slow_days': 10.0,
            'entry_days': 12.0,
            'exit_days': 2.0,
            'trailing_drawdown_pct': 0.25,
        }
        if parameters:
            defaults.update(parameters)
        super().__init__('TightTrendFollowStrategy', defaults)

    def _bars(self, days: float) -> int:
        timeframe = float(self.parameters.get('timeframe_minutes', 5))
        return max(1, int(round(days * 24 * 60 / timeframe)))

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        if not self.validate_data(data):
            return data

        df = data.copy()
        price = df['price'] if 'price' in df else df['close']
        df['price'] = price

        ema_fast = self._bars(float(self.parameters['ema_fast_days']))
        ema_mid = self._bars(float(self.parameters['ema_mid_days']))
        ema_slow = self._bars(float(self.parameters['ema_slow_days']))

        df['ema_fast'] = _ema(price, ema_fast)
        df['ema_mid'] = _ema(price, ema_mid)
        df['ema_slow'] = _ema(price, ema_slow)
        df['ema_fast_slope'] = df['ema_fast'] - df['ema_fast'].shift(1)

        entry_bars = self._bars(float(self.parameters['entry_days']))
        exit_bars = self._bars(float(self.parameters['exit_days']))
        df['donchian_high'] = price.rolling(entry_bars, min_periods=entry_bars).max().shift(1)
        df['donchian_low'] = price.rolling(exit_bars, min_periods=exit_bars).min().shift(1)

        return df

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        if 'donchian_high' not in df.columns:
            df = self.calculate_indicators(df)

        df['buy_signal'] = False
        df['sell_signal'] = False
        df['signal_strength'] = 0.0

        price = df['price'].to_numpy(float)
        hi = df['donchian_high'].to_numpy(float)
        lo = df['donchian_low'].to_numpy(float)
        ema_fast = df['ema_fast'].to_numpy(float)
        ema_mid = df['ema_mid'].to_numpy(float)
        ema_slow = df['ema_slow'].to_numpy(float)
        ema_slope = df['ema_fast_slope'].to_numpy(float)
        trail_pct = float(self.parameters['trailing_drawdown_pct'])

        in_position = False
        peak_price = 0.0

        for i in range(len(df)):
            px = price[i]
            if np.isnan(px):
                continue

            fast = ema_fast[i]
            mid = ema_mid[i]
            slow = ema_slow[i]
            slope = ema_slope[i]
            uptrend = (
                not np.isnan(fast)
                and not np.isnan(mid)
                and not np.isnan(slow)
                and px > fast > mid > slow
                and slope is not np.nan
                and slope > 0
            )

            if not in_position:
                if uptrend and not np.isnan(hi[i]) and px >= hi[i]:
                    df.iat[i, df.columns.get_loc('buy_signal')] = True
                    df.iat[i, df.columns.get_loc('signal_strength')] = 1.0
                    in_position = True
                    peak_price = px
            else:
                peak_price = max(peak_price, px)
                exit_break = (not np.isnan(lo[i])) and px <= lo[i]
                exit_regime = not uptrend
                exit_trail = trail_pct > 0 and peak_price > 0 and px <= peak_price * (1.0 - trail_pct)

                if exit_break or exit_regime or exit_trail:
                    df.iat[i, df.columns.get_loc('sell_signal')] = True
                    df.iat[i, df.columns.get_loc('signal_strength')] = 1.0
                    in_position = False
                    peak_price = 0.0

        return df

