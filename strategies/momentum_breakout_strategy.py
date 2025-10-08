"""
Momentum Breakout Strategy for PDAI-DAI Trading

This strategy identifies momentum breakouts and rides the resulting moves.
Optimized for trending markets with strong momentum phases.
"""
import pandas as pd
import numpy as np
from typing import Dict, List
import logging

from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

class MomentumBreakoutStrategy(BaseStrategy):
    """
    Momentum Breakout Strategy

    Identifies breakouts from consolidation periods and trades in the direction
    of the breakout with momentum confirmation.

    Parameters:
        consolidation_period (int): Period to identify consolidation (default: 20)
        breakout_threshold (float): Minimum % move for breakout (default: 0.03)
        momentum_period (int): Period for momentum calculation (default: 10)
        volume_breakout_multiplier (float): Volume required for breakout (default: 1.5)
        min_breakout_strength (float): Minimum breakout strength (default: 0.6)
        exit_profit_target (float): Profit target % (default: 0.08)
        exit_stop_loss (float): Stop loss % (default: 0.04)
        rsi_filter_enabled (bool): Use RSI filter (default: True)
        rsi_breakout_level (int): RSI level for momentum confirmation (default: 60)
    """

    def __init__(self, parameters: Dict = None):
        default_params = {
            'consolidation_period': 20,
            'breakout_threshold': 0.03,
            'momentum_period': 10,
            'volume_breakout_multiplier': 1.5,
            'min_breakout_strength': 0.6,
            'exit_profit_target': 0.08,
            'exit_stop_loss': 0.04,
            'rsi_filter_enabled': True,
            'rsi_breakout_level': 60,
            'timeframe_minutes': 5
        }

        if parameters:
            default_params.update(parameters)

        super().__init__("MomentumBreakoutStrategy", default_params)

        # Breakout state tracking
        self.consolidation_high = 0.0
        self.consolidation_low = 0.0
        self.breakout_direction = 0  # 1 for up, -1 for down, 0 for none
        self.breakout_price = 0.0
        self.entry_price = 0.0

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate momentum breakout indicators"""
        if not self.validate_data(data):
            return data

        df = data.copy()

        # Ensure we have enough data
        min_periods = max(self.parameters['consolidation_period'], 50)
        if len(df) < min_periods:
            logger.warning(f"Not enough data for momentum breakout. Need {min_periods}, got {len(df)}")
            return df

        consolidation_period = self.parameters['consolidation_period']

        # Identify consolidation periods (low volatility ranges)
        df['range_pct'] = (
            df['price'].rolling(window=consolidation_period).max() -
            df['price'].rolling(window=consolidation_period).min()
        ) / df['price'].rolling(window=consolidation_period).mean()

        df['is_consolidating'] = df['range_pct'] < 0.05  # Less than 5% range

        # Consolidation boundaries
        df['consolidation_high'] = df['price'].rolling(window=consolidation_period).max()
        df['consolidation_low'] = df['price'].rolling(window=consolidation_period).min()
        df['consolidation_mid'] = (df['consolidation_high'] + df['consolidation_low']) / 2

        # Momentum indicators
        momentum_period = self.parameters['momentum_period']
        df['momentum'] = df['price'] / df['price'].shift(momentum_period) - 1
        df['momentum_acceleration'] = df['momentum'] - df['momentum'].shift(1)

        # RSI for momentum confirmation
        df['rsi'] = self._calculate_rsi(df['price'], 14)

        # Volume indicators
        if 'volume' in df.columns:
            df['volume_ma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma']
            df['volume_breakout'] = df['volume_ratio'] >= self.parameters['volume_breakout_multiplier']
        else:
            df['volume_ratio'] = 1.0
            df['volume_breakout'] = True

        # Price position within consolidation
        df['position_in_range'] = (
            (df['price'] - df['consolidation_low']) /
            (df['consolidation_high'] - df['consolidation_low'])
        ).fillna(0.5)

        # Breakout detection
        df['breakout_up'] = (
            (df['price'] > df['consolidation_high'].shift(1)) &
            (df['price'] / df['consolidation_high'].shift(1) - 1 >= self.parameters['breakout_threshold'])
        )

        df['breakout_down'] = (
            (df['price'] < df['consolidation_low'].shift(1)) &
            (df['consolidation_low'].shift(1) / df['price'] - 1 >= self.parameters['breakout_threshold'])
        )

        # Breakout strength
        df['breakout_strength'] = 0.0
        for i in range(len(df)):
            if df.iloc[i]['breakout_up']:
                strength = self._calculate_breakout_strength(df.iloc[i], 'up')
                df.iloc[i, df.columns.get_loc('breakout_strength')] = strength
            elif df.iloc[i]['breakout_down']:
                strength = self._calculate_breakout_strength(df.iloc[i], 'down')
                df.iloc[i, df.columns.get_loc('breakout_strength')] = strength

        # Continuation signals (for holding positions)
        df['continuation_up'] = (
            df['momentum'] > 0.02  # Positive momentum
        ) & (
            df['momentum_acceleration'] > 0  # Accelerating
        )

        df['continuation_down'] = (
            df['momentum'] < -0.02  # Negative momentum
        ) & (
            df['momentum_acceleration'] < 0  # Accelerating down
        )

        return df

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)

    def _calculate_breakout_strength(self, row: pd.Series, direction: str) -> float:
        """Calculate breakout strength score"""
        strength = 0.0

        # Volume confirmation (30% weight)
        if row['volume_breakout']:
            strength += 0.3

        # Momentum confirmation (25% weight)
        momentum_score = min(1.0, abs(row['momentum']) / 0.05)  # Normalize to 5%
        strength += momentum_score * 0.25

        # RSI confirmation (20% weight)
        if direction == 'up' and row['rsi'] >= self.parameters['rsi_breakout_level']:
            rsi_score = min(1.0, (row['rsi'] - 50) / 50)  # Higher RSI = stronger up breakout
            strength += rsi_score * 0.2
        elif direction == 'down' and row['rsi'] <= (100 - self.parameters['rsi_breakout_level']):
            rsi_score = min(1.0, (50 - row['rsi']) / 50)  # Lower RSI = stronger down breakout
            strength += rsi_score * 0.2

        # Consolidation tightness (15% weight) - tighter consolidation = stronger breakout
        range_score = max(0, 1.0 - row['range_pct'] / 0.05)  # Better if range < 5%
        strength += range_score * 0.15

        # Position in range (10% weight) - breakouts from middle are stronger
        position_score = 1.0 - abs(row['position_in_range'] - 0.5) * 2  # Best at 0.5
        strength += position_score * 0.1

        return min(1.0, strength)

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate buy/sell signals based on momentum breakout logic"""
        df = data.copy()

        if 'breakout_strength' not in df.columns:
            df = self.calculate_indicators(df)

        # Initialize signal columns
        df['buy_signal'] = False
        df['sell_signal'] = False
        df['signal_strength'] = 0.0

        min_strength = self.parameters['min_breakout_strength']
        rsi_filter = self.parameters['rsi_filter_enabled']
        rsi_level = self.parameters['rsi_breakout_level']

        # Process each data point
        for i in range(len(df)):
            breakout_strength = df.iloc[i]['breakout_strength']
            rsi = df.iloc[i]['rsi']

            # Check breakout signals
            if breakout_strength >= min_strength:
                # RSI filter if enabled
                rsi_ok = True
                if rsi_filter:
                    if df.iloc[i]['breakout_up'] and rsi < rsi_level:
                        rsi_ok = False
                    elif df.iloc[i]['breakout_down'] and rsi > (100 - rsi_level):
                        rsi_ok = False

                if rsi_ok:
                    if df.iloc[i]['breakout_up']:
                        df.iloc[i, df.columns.get_loc('buy_signal')] = True
                        df.iloc[i, df.columns.get_loc('signal_strength')] = breakout_strength
                    elif df.iloc[i]['breakout_down']:
                        df.iloc[i, df.columns.get_loc('sell_signal')] = True
                        df.iloc[i, df.columns.get_loc('signal_strength')] = breakout_strength

        return df

    def get_breakout_info(self) -> Dict:
        """Get current breakout analysis information"""
        return {
            'consolidation_high': self.consolidation_high,
            'consolidation_low': self.consolidation_low,
            'breakout_direction': self.breakout_direction,
            'breakout_price': self.breakout_price,
            'entry_price': self.entry_price,
            'parameters': self.parameters
        }