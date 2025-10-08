"""VOST Pullback Accumulator Strategy.

Trades in the direction of the VOST regime but waits for momentum resets
confirmed by RSI and volume exhaustion. Designed to reinstate long exposure
after shake-outs without fighting primary trends.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from .base_strategy import BaseStrategy
from .vost_indicator import VOSTResult, compute_vost


class VOSTPullbackAccumulatorStrategy(BaseStrategy):
    """Accumulates on controlled pullbacks within bullish VOST regimes."""

    def __init__(self, parameters: Dict | None = None):
        defaults = {
            'atr_period': 14,
            'base_multiplier': 2.2,
            'vol_period': 60,
            'vol_smooth': 300,
            'vol_ratio_floor': 0.7,
            'vol_ratio_cap': 2.3,
            'multiplier_power': 1.0,
            'ema_fast': 72,
            'ema_slow': 288,
            'pullback_to_line_pct': 0.02,
            'max_drawdown_from_high': 0.25,
            'rsi_period': 14,
            'rsi_pullback_level': 48,
            'take_profit_pct': 0.15,
            'stop_loss_pct': 0.055,
            'cooldown_bars': 36,
            'timeframe_minutes': 5,
        }
        if parameters:
            defaults.update(parameters)
        super().__init__('VOSTPullbackAccumulatorStrategy', defaults)

        self._cooldown = 0

    @staticmethod
    def _ema(series: pd.Series, span: int) -> pd.Series:
        return series.ewm(span=max(2, span), adjust=False).mean()

    @staticmethod
    def _rsi(series: pd.Series, period: int) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        if not self.validate_data(data):
            return data

        df = data.copy()

        vost: VOSTResult | None = compute_vost(
            df,
            atr_period=int(self.parameters['atr_period']),
            base_multiplier=float(self.parameters['base_multiplier']),
            vol_period=int(self.parameters['vol_period']),
            vol_smooth=int(self.parameters['vol_smooth']),
            vol_ratio_floor=float(self.parameters['vol_ratio_floor']),
            vol_ratio_cap=float(self.parameters['vol_ratio_cap']),
            multiplier_power=float(self.parameters['multiplier_power']),
        )
        if vost is None:
            return df

        df['vost_line'] = vost.line
        df['vost_trend'] = vost.trend
        df['vost_vol_ratio'] = vost.vol_ratio

        price = df['close']
        df['ema_fast'] = self._ema(price, int(self.parameters['ema_fast']))
        df['ema_slow'] = self._ema(price, int(self.parameters['ema_slow']))
        df['ema_regime'] = df['ema_fast'] > df['ema_slow']

        df['rsi'] = self._rsi(price, int(self.parameters['rsi_period']))

        df['drawdown_from_high'] = 1.0 - (price / price.cummax()).clip(upper=1.0)

        df['atr'] = df['close'].rolling(14).std().bfill()

        return df

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        if 'vost_trend' not in df.columns:
            df = self.calculate_indicators(df)

        df['buy_signal'] = False
        df['sell_signal'] = False
        df['signal_strength'] = 0.0

        if 'vost_trend' not in df.columns:
            return df

        pullback_pct = float(self.parameters['pullback_to_line_pct'])
        max_drawdown = float(self.parameters['max_drawdown_from_high'])
        rsi_level = float(self.parameters['rsi_pullback_level'])
        take_profit_pct = float(self.parameters['take_profit_pct'])
        stop_loss_pct = float(self.parameters['stop_loss_pct'])
        cooldown_bars = int(self.parameters['cooldown_bars'])

        vost_line = df['vost_line']
        trend = df['vost_trend']
        price = df['close']

        pullback_to_line = (price >= vost_line * (1.0 - pullback_pct)) & (price <= vost_line * (1.0 + pullback_pct))
        controlled_drawdown = df['drawdown_from_high'] <= max_drawdown
        rsi_reset = df['rsi'] <= rsi_level
        ema_regime = df['ema_regime']
        vol_control = df['vost_vol_ratio'] <= float(self.parameters['vol_ratio_cap'])

        buy_flags = np.zeros(len(df), dtype=bool)
        sell_flags = np.zeros(len(df), dtype=bool)
        strengths = np.zeros(len(df))

        cooldown_counter = 0
        in_position = False
        entry_price = 0.0
        peak_price = 0.0

        for i in range(len(df)):
            if cooldown_counter > 0:
                cooldown_counter -= 1

            qualifies = (
                (trend.iat[i] > 0)
                and ema_regime.iat[i]
                and pullback_to_line.iat[i]
                and controlled_drawdown.iat[i]
                and rsi_reset.iat[i]
                and vol_control.iat[i]
            )

            if qualifies and not in_position and cooldown_counter == 0:
                buy_flags[i] = True
                in_position = True
                entry_price = price.iat[i]
                peak_price = entry_price
                depth = abs(price.iat[i] - vost_line.iat[i]) / max(price.iat[i], 1e-12)
                strengths[i] = np.clip(1.0 - depth / max(pullback_pct, 1e-4), 0.0, 1.0)
                continue

            if in_position:
                peak_price = max(peak_price, price.iat[i])
                exit_due_to_trend = trend.iat[i] < 0
                exit_due_to_stop = price.iat[i] <= vost_line.iat[i] * (1.0 - stop_loss_pct)
                exit_due_to_profit = peak_price > 0 and price.iat[i] >= peak_price * (1.0 + take_profit_pct)

                if exit_due_to_trend or exit_due_to_stop or exit_due_to_profit:
                    sell_flags[i] = True
                    strengths[i] = 1.0
                    in_position = False
                    entry_price = 0.0
                    peak_price = 0.0
                    cooldown_counter = cooldown_bars

        df['buy_signal'] = buy_flags
        df['sell_signal'] = sell_flags
        df['signal_strength'] = strengths

        # Persist cooldown state so subsequent calls reflect current timer.
        self._cooldown = cooldown_counter

        return df
