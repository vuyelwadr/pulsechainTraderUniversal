"""
Adaptive Trend-Aware Strategy for PDAI-DAI Trading

This strategy adapts between trend-following and range-trading based on market conditions.
Uses trend-following in strong trends and grid trading only during consolidation periods.
"""
import pandas as pd
import numpy as np
from typing import Dict, List
import logging

from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

class AdaptiveTrendAwareStrategy(BaseStrategy):
    """
    Adaptive Trend-Aware Strategy

    Dynamically switches between trend-following and range-trading strategies
    based on current market conditions. Optimized for markets with both trends and consolidations.

    Parameters:
        trend_strength_threshold (float): Minimum strength for trend mode (default: 0.025)
        consolidation_threshold (float): Maximum range % for consolidation mode (default: 0.03)
        fast_ma_period (int): Fast MA for trend detection (default: 20)
        slow_ma_period (int): Slow MA for trend detection (default: 50)
        grid_levels (int): Number of grid levels when in range mode (default: 5)
        grid_spacing_pct (float): Grid spacing percentage (default: 0.015)
        pullback_entry_threshold (float): Pullback % for trend entries (default: 0.03)
        breakout_confirmation_periods (int): Periods to confirm breakout (default: 3)
        min_trend_duration (int): Minimum trend duration (default: 15)
        volume_filter_enabled (bool): Use volume confirmation (default: True)
    """

    def __init__(self, parameters: Dict = None):
        default_params = {
            'trend_strength_threshold': 0.025,
            'consolidation_threshold': 0.03,
            'fast_ma_period': 20,
            'slow_ma_period': 50,
            'grid_levels': 5,
            'grid_spacing_pct': 0.015,
            'pullback_entry_threshold': 0.03,
            'breakout_confirmation_periods': 3,
            'min_trend_duration': 15,
            'volume_filter_enabled': True,
            'timeframe_minutes': 5
        }

        if parameters:
            default_params.update(parameters)

        super().__init__("AdaptiveTrendAwareStrategy", default_params)

        # Market regime tracking
        self.current_regime = 'unknown'  # 'trend_up', 'trend_down', 'consolidation'
        self.regime_start_price = 0.0
        self.regime_duration = 0

        # Grid state for consolidation periods
        self.grid_center = 0.0
        self.grid_levels = []

        # Trend state for trending periods
        self.trend_direction = 0  # 1 for up, -1 for down
        self.trend_high = 0.0
        self.trend_low = 0.0

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate adaptive indicators for regime detection and strategy selection"""
        if not self.validate_data(data):
            return data

        df = data.copy()

        # Ensure we have enough data
        min_periods = max(self.parameters['slow_ma_period'], 60)
        if len(df) < min_periods:
            logger.warning(f"Not enough data for adaptive strategy. Need {min_periods}, got {len(df)}")
            return df

        # Core trend indicators
        fast_period = self.parameters['fast_ma_period']
        slow_period = self.parameters['slow_ma_period']

        df['fast_ma'] = df['price'].rolling(window=fast_period).mean()
        df['slow_ma'] = df['price'].rolling(window=slow_period).mean()
        df['ma_diff'] = df['fast_ma'] - df['slow_ma']
        df['ma_diff_pct'] = df['ma_diff'] / df['slow_ma']
        df['trend_strength'] = abs(df['ma_diff_pct'])

        # Market regime detection
        consolidation_window = 30
        df['price_range_pct'] = (
            df['price'].rolling(window=consolidation_window).max() -
            df['price'].rolling(window=consolidation_window).min()
        ) / df['price'].rolling(window=consolidation_window).mean()

        df['is_consolidating'] = df['price_range_pct'] < self.parameters['consolidation_threshold']
        df['is_trending_up'] = (
            (df['trend_strength'] >= self.parameters['trend_strength_threshold']) &
            (df['fast_ma'] > df['slow_ma'])
        )
        df['is_trending_down'] = (
            (df['trend_strength'] >= self.parameters['trend_strength_threshold']) &
            (df['fast_ma'] < df['slow_ma'])
        )

        # Regime classification
        df['regime'] = 'consolidation'
        df.loc[df['is_trending_up'], 'regime'] = 'trend_up'
        df.loc[df['is_trending_down'], 'regime'] = 'trend_down'

        # Trend duration tracking
        df['trend_duration'] = 0
        current_duration = 0
        current_regime = 'consolidation'

        for i in range(len(df)):
            regime = df.iloc[i]['regime']
            if regime == current_regime:
                current_duration += 1
            else:
                current_duration = 1
                current_regime = regime
            df.iloc[i, df.columns.get_loc('trend_duration')] = current_duration

        # Volume indicators
        if 'volume' in df.columns:
            df['volume_ma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma']
            df['high_volume'] = df['volume_ratio'] > 1.2
        else:
            df['volume_ratio'] = 1.0
            df['high_volume'] = True

        # RSI for timing
        df['rsi'] = self._calculate_rsi(df['price'], 14)

        # Pullback detection for trend mode
        df['pullback_pct'] = 0.0
        for i in range(len(df)):
            if df.iloc[i]['regime'] == 'trend_up' and df.iloc[i]['trend_duration'] >= self.parameters['min_trend_duration']:
                # In uptrend, measure pullback from recent high
                lookback = min(20, i)
                recent_high = df['price'].iloc[max(0, i-lookback):i+1].max()
                df.iloc[i, df.columns.get_loc('pullback_pct')] = (recent_high - df.iloc[i]['price']) / recent_high
            elif df.iloc[i]['regime'] == 'trend_down' and df.iloc[i]['trend_duration'] >= self.parameters['min_trend_duration']:
                # In downtrend, measure pullback from recent low
                lookback = min(20, i)
                recent_low = df['price'].iloc[max(0, i-lookback):i+1].min()
                df.iloc[i, df.columns.get_loc('pullback_pct')] = (df.iloc[i]['price'] - recent_low) / recent_low

        # Grid setup for consolidation periods
        self._setup_adaptive_grid(df)

        # Breakout detection for regime changes
        df['breakout_up'] = self._detect_breakout(df, 'up')
        df['breakout_down'] = self._detect_breakout(df, 'down')

        return df

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)

    def _detect_breakout(self, df: pd.DataFrame, direction: str) -> pd.Series:
        """Detect breakouts from consolidation"""
        breakout = pd.Series(False, index=df.index)
        confirmation_periods = self.parameters['breakout_confirmation_periods']

        for i in range(confirmation_periods, len(df)):
            if df.iloc[i]['regime'] == 'consolidation':
                # Check if price broke out and stayed out for confirmation periods
                if direction == 'up':
                    breakout_level = df['price'].iloc[i-confirmation_periods:i].max()
                    if df.iloc[i]['price'] > breakout_level * 1.02:  # 2% breakout
                        breakout.iloc[i] = True
                else:  # down
                    breakout_level = df['price'].iloc[i-confirmation_periods:i].min()
                    if df.iloc[i]['price'] < breakout_level * 0.98:  # 2% breakdown
                        breakout.iloc[i] = True

        return breakout

    def _setup_adaptive_grid(self, df: pd.DataFrame):
        """Setup grid levels for consolidation periods"""
        grid_levels_col = []
        grid_center_col = []

        for i in range(len(df)):
            if df.iloc[i]['regime'] == 'consolidation':
                # Use current price as center during consolidation
                center = df.iloc[i]['price']
                grid_center_col.append(center)

                # Create grid levels around center
                levels = []
                spacing = self.parameters['grid_spacing_pct']
                num_levels = self.parameters['grid_levels']

                for level in range(-num_levels, num_levels + 1):
                    if level != 0:  # Skip center
                        level_price = center * (1 + level * spacing)
                        levels.append({
                            'price': level_price,
                            'type': 'buy' if level < 0 else 'sell',
                            'distance': abs(level)
                        })

                grid_levels_col.append(levels)
            else:
                grid_center_col.append(0.0)
                grid_levels_col.append([])

        df['grid_center'] = grid_center_col
        df['grid_levels'] = grid_levels_col

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate signals based on current market regime"""
        df = data.copy()

        if 'regime' not in df.columns:
            df = self.calculate_indicators(df)

        # Initialize signal columns
        df['buy_signal'] = False
        df['sell_signal'] = False
        df['signal_strength'] = 0.0

        # Process each data point
        for i in range(len(df)):
            regime = df.iloc[i]['regime']
            trend_duration = df.iloc[i]['trend_duration']

            if regime == 'trend_up' and trend_duration >= self.parameters['min_trend_duration']:
                # Trend-following mode: buy on pullbacks
                self._generate_trend_signals(df, i, 'up')
            elif regime == 'trend_down' and trend_duration >= self.parameters['min_trend_duration']:
                # Trend-following mode: sell on pullbacks
                self._generate_trend_signals(df, i, 'down')
            elif regime == 'consolidation':
                # Range-trading mode: grid trading
                self._generate_grid_signals(df, i)

        return df

    def _generate_trend_signals(self, df: pd.DataFrame, i: int, direction: str):
        """Generate trend-following signals"""
        pullback_pct = df.iloc[i]['pullback_pct']
        pullback_threshold = self.parameters['pullback_entry_threshold']
        volume_ok = not self.parameters['volume_filter_enabled'] or df.iloc[i]['high_volume']
        rsi = df.iloc[i]['rsi']

        if pullback_pct >= pullback_threshold and volume_ok:
            # Additional RSI filter
            rsi_ok = True
            if direction == 'up' and rsi > 75:  # Too overbought for uptrend entry
                rsi_ok = False
            elif direction == 'down' and rsi < 25:  # Too oversold for downtrend entry
                rsi_ok = False

            if rsi_ok:
                strength = self._calculate_trend_signal_strength(df.iloc[i], direction)
                if direction == 'up':
                    df.iloc[i, df.columns.get_loc('buy_signal')] = True
                else:
                    df.iloc[i, df.columns.get_loc('sell_signal')] = True
                df.iloc[i, df.columns.get_loc('signal_strength')] = strength

    def _generate_grid_signals(self, df: pd.DataFrame, i: int):
        """Generate grid trading signals during consolidation"""
        current_price = df.iloc[i]['price']
        grid_levels = df.iloc[i]['grid_levels']

        if not grid_levels:
            return

        # Find nearest grid level
        nearest_level = None
        min_distance = float('inf')

        for level in grid_levels:
            distance = abs(current_price - level['price']) / current_price
            if distance < min_distance:
                min_distance = distance
                nearest_level = level

        # Signal if close to a grid level
        if nearest_level and min_distance <= 0.005:  # Within 0.5%
            strength = self._calculate_grid_signal_strength(df.iloc[i], nearest_level)
            if nearest_level['type'] == 'buy':
                df.iloc[i, df.columns.get_loc('buy_signal')] = True
            else:
                df.iloc[i, df.columns.get_loc('sell_signal')] = True
            df.iloc[i, df.columns.get_loc('signal_strength')] = strength

    def _calculate_trend_signal_strength(self, row: pd.Series, direction: str) -> float:
        """Calculate signal strength for trend entries"""
        strength = 0.0

        # Trend strength (40% weight)
        trend_component = min(1.0, row['trend_strength'] / 0.05)
        strength += trend_component * 0.4

        # Pullback depth (25% weight) - optimal pullback depth
        pullback_pct = row['pullback_pct']
        if pullback_pct <= 0.08:  # Max 8% pullback
            pullback_component = pullback_pct / 0.08
        else:
            pullback_component = 0.0
        strength += pullback_component * 0.25

        # Volume confirmation (20% weight)
        if row['high_volume']:
            strength += 0.2

        # RSI positioning (15% weight)
        rsi = row['rsi']
        if direction == 'up':
            rsi_component = 1.0 - abs(rsi - 45) / 45  # Best around 45 for uptrend entries
        else:
            rsi_component = 1.0 - abs(rsi - 55) / 55  # Best around 55 for downtrend entries
        strength += rsi_component * 0.15

        return min(1.0, strength)

    def _calculate_grid_signal_strength(self, row: pd.Series, level: Dict) -> float:
        """Calculate signal strength for grid entries"""
        strength = 0.0

        # Distance from level (40% weight) - closer = stronger
        distance_pct = abs(row['price'] - level['price']) / row['price']
        distance_component = max(0, 1.0 - distance_pct / 0.005)  # Max 0.5% distance
        strength += distance_component * 0.4

        # Volume confirmation (30% weight)
        if row['high_volume']:
            strength += 0.3

        # Level distance from center (20% weight) - closer to center = stronger
        level_distance = level['distance']
        level_component = max(0, 1.0 - level_distance / 5.0)  # Max 5 levels out
        strength += level_component * 0.2

        # RSI filter (10% weight)
        rsi = row['rsi']
        rsi_component = 1.0 - abs(rsi - 50) / 50  # Best at neutral RSI
        strength += rsi_component * 0.1

        return min(1.0, strength)

    def get_adaptive_info(self) -> Dict:
        """Get current adaptive strategy information"""
        return {
            'current_regime': self.current_regime,
            'regime_start_price': self.regime_start_price,
            'regime_duration': self.regime_duration,
            'trend_direction': self.trend_direction,
            'trend_high': self.trend_high,
            'trend_low': self.trend_low,
            'grid_center': self.grid_center,
            'num_grid_levels': len(self.grid_levels),
            'parameters': self.parameters
        }