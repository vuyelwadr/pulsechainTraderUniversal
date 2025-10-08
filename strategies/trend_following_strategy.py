"""
Trend Following Strategy for PDAI-DAI Trading

This strategy identifies and follows strong trends, entering on pullbacks
and exiting on trend exhaustion signals. Optimized for trending markets.
"""
import pandas as pd
import numpy as np
from typing import Dict, List
import logging

from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

class TrendFollowingStrategy(BaseStrategy):
    """
    Trend Following Strategy

    Identifies strong trends and trades in the direction of the trend,
    entering on pullbacks and exiting on trend exhaustion.

    Parameters:
        fast_ma_period (int): Fast moving average period (default: 20)
        slow_ma_period (int): Slow moving average period (default: 50)
        trend_strength_threshold (float): Minimum trend strength to trade (default: 0.02)
        pullback_threshold (float): Maximum pullback % to enter (default: 0.05)
        exit_threshold (float): Profit target % or stop loss % (default: 0.10)
        min_trend_duration (int): Minimum trend duration in periods (default: 10)
        volume_confirmation (bool): Require volume confirmation (default: True)
        rsi_overbought (int): RSI overbought level (default: 70)
        rsi_oversold (int): RSI oversold level (default: 30)
    """

    def __init__(self, parameters: Dict = None):
        default_params = {
            'fast_ma_period': 20,
            'slow_ma_period': 50,
            'trend_strength_threshold': 0.02,
            'pullback_threshold': 0.05,
            'exit_threshold': 0.10,
            'min_trend_duration': 10,
            'volume_confirmation': True,
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            'timeframe_minutes': 5
        }

        if parameters:
            default_params.update(parameters)

        super().__init__("TrendFollowingStrategy", default_params)

        # Trend state tracking
        self.current_trend = 'sideways'  # 'uptrend', 'downtrend', 'sideways'
        self.trend_start_price = 0.0
        self.trend_duration = 0
        self.entry_price = 0.0
        self.stop_loss = 0.0
        self.take_profit = 0.0

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate trend following indicators"""
        if not self.validate_data(data):
            return data

        df = data.copy()

        # Ensure we have enough data
        min_periods = max(self.parameters['slow_ma_period'], 50)
        if len(df) < min_periods:
            logger.warning(f"Not enough data for trend following. Need {min_periods}, got {len(df)}")
            return df

        # Moving averages
        fast_period = self.parameters['fast_ma_period']
        slow_period = self.parameters['slow_ma_period']

        df['fast_ma'] = df['price'].rolling(window=fast_period).mean()
        df['slow_ma'] = df['price'].rolling(window=slow_period).mean()

        # Trend direction and strength
        df['ma_diff'] = df['fast_ma'] - df['slow_ma']
        df['ma_diff_pct'] = df['ma_diff'] / df['slow_ma']

        # Trend strength (absolute value of MA difference)
        df['trend_strength'] = abs(df['ma_diff_pct'])

        # Trend direction
        df['uptrend'] = df['fast_ma'] > df['slow_ma']
        df['downtrend'] = df['fast_ma'] < df['slow_ma']

        # Trend duration counter
        df['trend_duration'] = 0
        current_duration = 0
        current_trend = 'sideways'

        for i in range(len(df)):
            if df.iloc[i]['uptrend']:
                if current_trend == 'uptrend':
                    current_duration += 1
                else:
                    current_duration = 1
                    current_trend = 'uptrend'
            elif df.iloc[i]['downtrend']:
                if current_trend == 'downtrend':
                    current_duration += 1
                else:
                    current_duration = 1
                    current_trend = 'downtrend'
            else:
                current_duration = 0
                current_trend = 'sideways'

            df.iloc[i, df.columns.get_loc('trend_duration')] = current_duration

        # RSI for overbought/oversold conditions
        df['rsi'] = self._calculate_rsi(df['price'], 14)

        # Volume indicators
        if 'volume' in df.columns:
            df['volume_ma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma']
        else:
            df['volume_ratio'] = 1.0

        # Pullback detection
        df['pullback_pct'] = 0.0
        for i in range(len(df)):
            if df.iloc[i]['uptrend'] and df.iloc[i]['trend_duration'] >= self.parameters['min_trend_duration']:
                # In uptrend, pullback is current price below recent high
                lookback = min(20, i)
                recent_high = df['price'].iloc[i-lookback:i+1].max()
                df.iloc[i, df.columns.get_loc('pullback_pct')] = (recent_high - df.iloc[i]['price']) / recent_high
            elif df.iloc[i]['downtrend'] and df.iloc[i]['trend_duration'] >= self.parameters['min_trend_duration']:
                # In downtrend, pullback is current price above recent low
                lookback = min(20, i)
                recent_low = df['price'].iloc[i-lookback:i+1].min()
                df.iloc[i, df.columns.get_loc('pullback_pct')] = (df.iloc[i]['price'] - recent_low) / recent_low

        # Support and resistance levels
        df['support_level'] = df['price'].rolling(window=20).min()
        df['resistance_level'] = df['price'].rolling(window=20).max()

        return df

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate buy/sell signals based on trend following logic"""
        df = data.copy()

        if 'trend_strength' not in df.columns:
            df = self.calculate_indicators(df)

        # Initialize signal columns
        df['buy_signal'] = False
        df['sell_signal'] = False
        df['signal_strength'] = 0.0

        trend_threshold = self.parameters['trend_strength_threshold']
        pullback_threshold = self.parameters['pullback_threshold']
        exit_threshold = self.parameters['exit_threshold']
        min_duration = self.parameters['min_trend_duration']
        volume_confirmation = self.parameters['volume_confirmation']

        # Process each data point
        for i in range(len(df)):
            current_price = df.iloc[i]['price']
            trend_strength = df.iloc[i]['trend_strength']
            trend_duration = df.iloc[i]['trend_duration']
            pullback_pct = df.iloc[i]['pullback_pct']
            rsi = df.iloc[i]['rsi']
            volume_ratio = df.iloc[i]['volume_ratio']

            # Check for trend-following entry signals
            if trend_strength >= trend_threshold and trend_duration >= min_duration:
                # Volume confirmation if required
                if not volume_confirmation or volume_ratio >= 1.0:

                    # Uptrend entry: buy on pullbacks
                    if df.iloc[i]['uptrend'] and pullback_pct >= pullback_threshold:
                        # Additional filters
                        if rsi <= self.parameters['rsi_overbought']:  # Not overbought
                            strength = self._calculate_entry_strength(
                                trend_strength, trend_duration, pullback_pct, rsi
                            )
                            df.iloc[i, df.columns.get_loc('buy_signal')] = True
                            df.iloc[i, df.columns.get_loc('signal_strength')] = strength

                    # Downtrend entry: sell on pullbacks (shorts)
                    elif df.iloc[i]['downtrend'] and pullback_pct >= pullback_threshold:
                        # Additional filters
                        if rsi >= self.parameters['rsi_oversold']:  # Not oversold
                            strength = self._calculate_entry_strength(
                                trend_strength, trend_duration, pullback_pct, rsi
                            )
                            df.iloc[i, df.columns.get_loc('sell_signal')] = True
                            df.iloc[i, df.columns.get_loc('signal_strength')] = strength

            # Check for exit signals (trend exhaustion or profit targets)
            # This would be handled by the trading engine with position management

        return df

    def _calculate_entry_strength(self, trend_strength: float, trend_duration: int,
                                pullback_pct: float, rsi: float) -> float:
        """Calculate signal strength for trend following entry"""
        # Trend strength component (stronger trend = higher strength)
        trend_component = min(1.0, trend_strength / 0.05)  # Normalize to 5% threshold

        # Trend duration component (longer trend = higher strength)
        duration_component = min(1.0, trend_duration / 50.0)  # Max at 50 periods

        # Pullback component (deeper pullback = higher strength, but not too deep)
        if pullback_pct <= 0.15:  # Max 15% pullback
            pullback_component = pullback_pct / 0.15
        else:
            pullback_component = 0.0  # Too deep, not a pullback anymore

        # RSI component (moderate RSI = higher strength)
        rsi_component = 1.0 - abs(rsi - 50) / 50  # Best at RSI 50

        # Combine components
        total_strength = (
            trend_component * 0.4 +
            duration_component * 0.2 +
            pullback_component * 0.25 +
            rsi_component * 0.15
        )

        return min(1.0, max(0.0, total_strength))

    def get_trend_info(self) -> Dict:
        """Get current trend analysis information"""
        return {
            'current_trend': self.current_trend,
            'trend_start_price': self.trend_start_price,
            'trend_duration': self.trend_duration,
            'entry_price': self.entry_price,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'parameters': self.parameters
        }