"""
Volatility Breakout Strategy

Detects Bollinger Band squeezes and enters on volatility expansion, using
ATR-based trailing exits to capture explosive moves while controlling risk.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
import talib

from .base_strategy import BaseStrategy
from .indicator_utils import compute_atr, compute_adx


class VolatilityBreakoutStrategy(BaseStrategy):
    """Bollinger squeeze breakout with ATR trailing exits."""

    def __init__(self, parameters: Dict | None = None):
        defaults = {
            'bb_period': 20,
            'bb_std': 2.0,
            'squeeze_lookback': 40,
            'ema_trend_period': 50,
            'adx_period': 14,
            'adx_threshold': 20.0,
            'atr_period': 14,
            'atr_floor_pct': 0.002,
            'chandelier_period': 22,
            'chandelier_atr_mult': 3.0,
        }
        if parameters:
            defaults.update(parameters)
        super().__init__('VolatilityBreakoutStrategy', defaults)

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()

        close = df['close']
        upper, middle, lower = talib.BBANDS(
            close,
            timeperiod=int(self.parameters['bb_period']),
            nbdevup=float(self.parameters['bb_std']),
            nbdevdn=float(self.parameters['bb_std']),
        )
        df['bb_upper'] = upper
        df['bb_middle'] = middle
        df['bb_lower'] = lower
        df['bb_width'] = (upper - lower) / (middle + 1e-12)
        squeeze_lb = int(self.parameters['squeeze_lookback'])
        df['is_squeezed'] = df['bb_width'] <= df['bb_width'].rolling(squeeze_lb, min_periods=squeeze_lb).min()

        ema_period = int(self.parameters['ema_trend_period'])
        df['ema_trend'] = talib.EMA(close, timeperiod=ema_period)

        high = df['high']
        low = df['low']
        atr = compute_atr(high, low, close, int(self.parameters['atr_period']))
        atr_floor_pct = float(self.parameters['atr_floor_pct'])
        if atr_floor_pct > 0:
            atr = atr.clip(lower=close * atr_floor_pct)
        df['atr'] = atr
        df['atr_pct'] = atr / close.replace(0, np.nan)

        chandelier_period = int(self.parameters['chandelier_period'])
        chandelier_mult = float(self.parameters['chandelier_atr_mult'])
        rolling_high = high.rolling(chandelier_period, min_periods=1).max()
        df['chandelier_long'] = rolling_high - chandelier_mult * atr

        adx_df = compute_adx(high, low, close, int(self.parameters['adx_period']))
        df['adx'] = adx_df['adx']
        df['trend_ready'] = df['adx'] >= float(self.parameters['adx_threshold'])

        return df

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        if 'bb_upper' not in df.columns:
            df = self.calculate_indicators(df)

        squeeze_prev = df['is_squeezed'].shift(1).fillna(False)
        breakout_long = (
            squeeze_prev
            & df['trend_ready']
            & (df['close'] > df['bb_upper'].shift(1))
            & (df['close'] > df['ema_trend'])
        )

        df['buy_signal'] = breakout_long
        sell_cross = (df['close'] < df['bb_middle']) & (df['close'].shift(1) >= df['bb_middle'])
        chandelier_stop = df['close'] <= df.get('chandelier_long', df['close'])
        df['sell_signal'] = sell_cross | chandelier_stop

        strength = np.clip((df['close'] - df['bb_upper']) / (df['bb_upper'] + 1e-12), 0.0, 1.0)
        df['signal_strength'] = np.where(df['buy_signal'], strength, 0.0)
        df.loc[df['sell_signal'], 'signal_strength'] = np.maximum(
            df.loc[df['sell_signal'], 'signal_strength'], 0.6
        )
        return df

    @classmethod
    def parameter_space(cls) -> Dict[str, Tuple[float, float]]:
        return {
            'bb_period': (10, 40),
            'bb_std': (1.5, 3.5),
            'squeeze_lookback': (20, 80),
            'ema_trend_period': (30, 120),
            'adx_period': (10, 28),
            'adx_threshold': (18.0, 30.0),
            'atr_period': (10, 28),
            'atr_floor_pct': (0.0, 0.01),
            'chandelier_period': (14, 40),
            'chandelier_atr_mult': (2.0, 4.0),
        }
