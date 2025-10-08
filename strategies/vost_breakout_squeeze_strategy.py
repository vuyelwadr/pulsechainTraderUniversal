"""VOST Breakout Squeeze Strategy.

Identifies volatility compression phases using Bollinger-width percentiles and
waits for a VOST-confirmed upside breakout with volume confirmation. The goal
is to capture explosive legs while the adaptive VOST line manages downside.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from .base_strategy import BaseStrategy
from .vost_indicator import VOSTResult, compute_vost


class VOSTBreakoutSqueezeStrategy(BaseStrategy):
    """Volatility breakout aligned with VOST trend and volume expansion."""

    def __init__(self, parameters: Dict | None = None):
        defaults = {
            'atr_period': 14,
            'base_multiplier': 2.1,
            'vol_period': 72,
            'vol_smooth': 288,
            'vol_ratio_floor': 0.6,
            'vol_ratio_cap': 2.6,
            'multiplier_power': 1.05,
            'bb_period': 48,
            'compression_window': 288,
            'compression_quantile': 0.35,
            'breakout_window': 96,
            'volume_ma_window': 288,
            'volume_multiplier': 1.5,
            'exit_trail_pct': 0.12,
            'stop_loss_pct': 0.06,
            'max_hold_bars': 288,
            'timeframe_minutes': 5,
        }
        if parameters:
            defaults.update(parameters)
        super().__init__('VOSTBreakoutSqueezeStrategy', defaults)

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

        bb_period = max(10, int(self.parameters['bb_period']))
        rolling_mean = df['close'].rolling(bb_period).mean()
        rolling_std = df['close'].rolling(bb_period).std(ddof=0)
        df['bb_upper'] = rolling_mean + 2 * rolling_std
        df['bb_lower'] = rolling_mean - 2 * rolling_std
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['close']

        comp_window = max(bb_period, int(self.parameters['compression_window']))
        quantile = float(self.parameters['compression_quantile'])
        df['compression_threshold'] = df['bb_width'].rolling(comp_window).quantile(quantile)
        df['is_compressed'] = df['bb_width'] <= df['compression_threshold']

        breakout_window = max(10, int(self.parameters['breakout_window']))
        df['breakout_level'] = df['close'].rolling(breakout_window).max().shift(1)

        vol_window = max(12, int(self.parameters['volume_ma_window']))
        df['volume_ma'] = df['volume'].rolling(vol_window).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']

        df['max_close_since_entry'] = df['close'].cummax()

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

        volume_multiplier = float(self.parameters['volume_multiplier'])
        stop_loss_pct = float(self.parameters['stop_loss_pct'])
        exit_trail_pct = float(self.parameters['exit_trail_pct'])
        max_hold = int(self.parameters['max_hold_bars'])

        trend = df['vost_trend']
        close = df['close']

        compression = df['is_compressed'].fillna(False)
        breakout = (close > df['breakout_level']) & (close.shift(1) <= df['breakout_level'])
        volume_confirm = df['volume_ratio'] >= volume_multiplier
        bullish_regime = trend > 0

        df['buy_signal'] = compression & breakout & volume_confirm & bullish_regime

        vost_line = df['vost_line']
        time_in_trade = np.zeros(len(df), dtype=int)
        in_position = False
        last_entry_index = None
        peak_since_entry = 0.0

        for i in range(len(df)):
            if df['buy_signal'].iat[i]:
                in_position = True
                last_entry_index = i
                peak_since_entry = close.iat[i]
                time_in_trade[i] = 0
                continue

            if in_position:
                if last_entry_index is not None:
                    time_in_trade[i] = i - last_entry_index
                    peak_since_entry = max(peak_since_entry, close.iat[i])

                exit_reasons = [
                    close.iat[i] <= vost_line.iat[i] * (1 - stop_loss_pct),
                    close.iat[i] <= peak_since_entry * (1 - exit_trail_pct),
                    trend.iat[i] < 0,
                    (last_entry_index is not None and (i - last_entry_index) >= max_hold),
                ]
                if any(exit_reasons):
                    df.at[df.index[i], 'sell_signal'] = True
                    in_position = False
                    last_entry_index = None
                    peak_since_entry = 0.0

        strength = np.zeros(len(df))
        idx_buy = np.where(df['buy_signal'])[0]
        strength[idx_buy] = np.clip(
            0.4
            + 0.3 * np.clip(df['volume_ratio'].to_numpy()[idx_buy] / volume_multiplier, 0, 2)
            + 0.3 * np.clip(1.0 - df['bb_width'].to_numpy()[idx_buy] / df['compression_threshold'].to_numpy()[idx_buy], 0, 1),
            0.0,
            1.0,
        )
        strength[np.where(df['sell_signal'])[0]] = 1.0
        df['signal_strength'] = strength

        return df
