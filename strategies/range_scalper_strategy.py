"""
Range Scalper Strategy

Trades mean-reversion inside low-ADX ranges using StochRSI triggers with tight
ATR stops and time-based exit."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
import talib

from .base_strategy import BaseStrategy
from .indicator_utils import compute_adx, compute_multi_timeframe_adx, compute_atr


class RangeScalperStrategy(BaseStrategy):
    """Mean-reversion scalper active only during low-ADX ranges."""

    def __init__(self, parameters: Dict | None = None):
        defaults = {
            'stoch_period': 14,
            'stoch_k': 3,
            'stoch_d': 3,
            'adx_period': 14,
            'adx_htf_period': 14,
            'adx_htf_minutes': 60,
            'adx_range_threshold': 20.0,
            'atr_period': 14,
            'atr_stop_mult': 1.2,
            'atr_floor_pct': 0.001,
            'time_stop_bars': 12,
        }
        if parameters:
            defaults.update(parameters)
        super().__init__('RangeScalperStrategy', defaults)

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()

        close = df['close']
        high = df['high']
        low = df['low']

        stoch_k, stoch_d = talib.STOCHRSI(
            close,
            timeperiod=int(self.parameters['stoch_period']),
            fastk_period=int(self.parameters['stoch_k']),
            fastd_period=int(self.parameters['stoch_d']),
            fastd_matype=0,
        )
        df['stoch_k'] = stoch_k * 100
        df['stoch_d'] = stoch_d * 100

        adx_df = compute_adx(high, low, close, int(self.parameters['adx_period']))
        df['adx'] = adx_df['adx']
        df['adx_htf'] = compute_multi_timeframe_adx(
            df[['timestamp', 'open', 'high', 'low', 'close']].copy(),
            period=int(self.parameters['adx_htf_period']),
            timeframe_minutes=int(self.parameters['adx_htf_minutes']),
        )
        adx_thresh = float(self.parameters['adx_range_threshold'])
        df['is_range'] = (df['adx'] <= adx_thresh) & (df['adx_htf'] <= adx_thresh)

        atr_period = int(self.parameters['atr_period'])
        atr_series = compute_atr(high, low, close, atr_period)
        atr_floor_pct = float(self.parameters['atr_floor_pct'])
        if atr_floor_pct > 0:
            atr_series = atr_series.clip(lower=close * atr_floor_pct)
        df['atr'] = atr_series

        return df

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        if 'stoch_k' not in df.columns:
            df = self.calculate_indicators(df)

        df['buy_signal'] = False
        df['sell_signal'] = False
        df['signal_strength'] = 0.0

        stoch = df['stoch_k'].to_numpy(float)
        close = df['close'].to_numpy(float)
        atr = df.get('atr', pd.Series(np.nan, index=df.index)).to_numpy(float)
        range_mask = df.get('is_range', pd.Series(True, index=df.index)).to_numpy(bool)

        atr_mult = float(self.parameters['atr_stop_mult'])
        time_stop = int(self.parameters.get('time_stop_bars', 0))

        in_position = False
        entry_price = 0.0
        bars_in_trade = 0

        for i in range(len(df)):
            if not in_position:
                if not range_mask[i]:
                    continue
                prev_k = stoch[i - 1] if i > 0 else 50.0
                if prev_k < 20 and stoch[i] > 20:
                    df.iat[i, df.columns.get_loc('buy_signal')] = True
                    df.iat[i, df.columns.get_loc('signal_strength')] = min(1.0, (50 - stoch[i]) / 30.0)
                    in_position = True
                    entry_price = close[i]
                    bars_in_trade = 0
            else:
                bars_in_trade += 1
                stop_price = entry_price - atr_mult * atr[i] if not np.isnan(atr[i]) else entry_price * 0.99
                target_price = entry_price + atr_mult * atr[i] * 1.5 if not np.isnan(atr[i]) else entry_price * 1.015
                prev_k = stoch[i - 1] if i > 0 else 50.0
                sell = False
                strength = 0.6

                if prev_k > 80 and stoch[i] < 80:
                    sell = True
                    strength = 1.0
                elif close[i] <= stop_price:
                    sell = True
                    strength = 0.5
                elif close[i] >= target_price:
                    sell = True
                    strength = 0.8
                elif time_stop > 0 and bars_in_trade >= time_stop:
                    sell = True
                    strength = 0.5
                elif not range_mask[i]:
                    sell = True
                    strength = 0.4

                if sell:
                    df.iat[i, df.columns.get_loc('sell_signal')] = True
                    df.iat[i, df.columns.get_loc('signal_strength')] = strength
                    in_position = False
                    entry_price = 0.0
                    bars_in_trade = 0

        return df

    @classmethod
    def parameter_space(cls) -> Dict[str, Tuple[float, float]]:
        return {
            'stoch_period': (10, 28),
            'stoch_k': (3, 6),
            'stoch_d': (3, 6),
            'adx_period': (10, 28),
            'adx_htf_period': (10, 28),
            'adx_htf_minutes': (30, 240),
            'adx_range_threshold': (15.0, 28.0),
            'atr_period': (10, 28),
            'atr_stop_mult': (0.8, 2.0),
            'atr_floor_pct': (0.0, 0.01),
            'time_stop_bars': (6, 24),
        }
