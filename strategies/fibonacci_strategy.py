"""
Fibonacci Retracement Strategy for PulseChain Trading Bot

This strategy uses Fibonacci levels for entry/exit points in trending markets.
Provides support/resistance based mean reversion with trend confirmation.
"""
import pandas as pd
import numpy as np
from typing import Dict
import logging

from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

class FibonacciStrategy(BaseStrategy):
    """
    Fibonacci Retracement Strategy
    
    Uses Fibonacci levels for entry/exit points in trending markets.
    Enters on retracements to key Fibonacci levels with trend confirmation.
    
    Parameters:
        lookback_period (int): Period for swing high/low detection (default: 20)
        trend_ma_fast (int): Fast MA for trend identification (default: 50)
        trend_ma_slow (int): Slow MA for trend identification (default: 200)
        fib_levels (list): Fibonacci retracement levels (default: [0.236, 0.382, 0.5, 0.618, 0.786])
        min_strength (float): Minimum signal strength (default: 0.6)
        confluence_threshold (float): Price threshold for level confluence (default: 0.02)
        timeframe_minutes (int): Timeframe for analysis (default: 5)
    """
    
    def __init__(self, parameters: Dict = None):
        default_params = {
            'lookback_period': 20,
            'trend_ma_fast': 50,
            'trend_ma_slow': 200,
            'fib_levels': [0.236, 0.382, 0.5, 0.618, 0.786],
            'min_strength': 0.6,
            'confluence_threshold': 0.02,
            'timeframe_minutes': 5
        }
        
        if parameters:
            default_params.update(parameters)
            
        super().__init__("FibonacciStrategy", default_params)
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Fibonacci levels and related indicators"""
        if not self.validate_data(data):
            return data
        
        df = data.copy()
        lookback = self.parameters['lookback_period']
        fast_ma = self.parameters['trend_ma_fast']
        slow_ma = self.parameters['trend_ma_slow']
        fib_levels = self.parameters['fib_levels']
        
        # Ensure we have enough data
        min_required = max(lookback, slow_ma) + 10
        if len(df) < min_required:
            logger.warning(f"Not enough data for Fibonacci calculation ({min_required} required)")
            return df
        
        # Calculate trend indicators
        df['ma_fast'] = df['price'].rolling(window=fast_ma).mean()
        df['ma_slow'] = df['price'].rolling(window=slow_ma).mean()
        
        # Determine trend direction
        df['uptrend'] = df['ma_fast'] > df['ma_slow']
        df['downtrend'] = df['ma_fast'] < df['ma_slow']
        df['trend_strength'] = abs(df['ma_fast'] - df['ma_slow']) / df['price']
        
        # Find swing highs and lows
        df['swing_high'] = df['high'].rolling(window=lookback, center=True).max() == df['high']
        df['swing_low'] = df['low'].rolling(window=lookback, center=True).min() == df['low']
        
        # Initialize Fibonacci level columns
        for level in fib_levels:
            df[f'fib_{int(level*1000)}'] = np.nan
        
        df['fib_high'] = np.nan
        df['fib_low'] = np.nan
        df['current_swing_range'] = np.nan
        
        # Calculate Fibonacci levels for each point
        for i in range(lookback, len(df) - lookback):
            # Look back for recent swing high and low
            recent_data = df.iloc[max(0, i-50):i+1]
            
            # Find most recent swing high and low
            swing_highs = recent_data[recent_data['swing_high']].copy()
            swing_lows = recent_data[recent_data['swing_low']].copy()
            
            if len(swing_highs) > 0 and len(swing_lows) > 0:
                # Get most recent swing points
                latest_high = swing_highs.iloc[-1]
                latest_low = swing_lows.iloc[-1]
                
                high_price = latest_high['high']
                low_price = latest_low['low']
                
                # Determine if we're in uptrend or downtrend for Fibonacci direction
                if df.iloc[i]['uptrend']:
                    # In uptrend, measure retracement from recent high to low
                    # Fibonacci levels are potential support areas
                    range_val = high_price - low_price
                    base_price = high_price
                    
                    for level in fib_levels:
                        fib_price = base_price - (range_val * level)
                        df.iloc[i, df.columns.get_loc(f'fib_{int(level*1000)}')] = fib_price
                        
                elif df.iloc[i]['downtrend']:
                    # In downtrend, measure retracement from recent low to high
                    # Fibonacci levels are potential resistance areas
                    range_val = high_price - low_price
                    base_price = low_price
                    
                    for level in fib_levels:
                        fib_price = base_price + (range_val * level)
                        df.iloc[i, df.columns.get_loc(f'fib_{int(level*1000)}')] = fib_price
                
                df.iloc[i, df.columns.get_loc('fib_high')] = high_price
                df.iloc[i, df.columns.get_loc('fib_low')] = low_price
                df.iloc[i, df.columns.get_loc('current_swing_range')] = range_val
        
        # Forward fill Fibonacci levels
        fib_columns = [f'fib_{int(level*1000)}' for level in fib_levels]
        df[fib_columns + ['fib_high', 'fib_low', 'current_swing_range']] = df[fib_columns + ['fib_high', 'fib_low', 'current_swing_range']].fillna(method='ffill')
        
        # Calculate distance to each Fibonacci level
        confluence_threshold = self.parameters['confluence_threshold']
        
        for level in fib_levels:
            level_col = f'fib_{int(level*1000)}'
            distance_col = f'distance_to_fib_{int(level*1000)}'
            at_level_col = f'at_fib_{int(level*1000)}'
            
            df[distance_col] = abs(df['price'] - df[level_col]) / df['price']
            df[at_level_col] = df[distance_col] < confluence_threshold
        
        # Find nearest Fibonacci level
        distance_cols = [f'distance_to_fib_{int(level*1000)}' for level in fib_levels]
        df['nearest_fib_distance'] = df[distance_cols].min(axis=1)
        df['at_any_fib_level'] = df['nearest_fib_distance'] < confluence_threshold
        
        # Calculate momentum indicators
        df['price_momentum'] = df['price'].pct_change(5)
        df['volume_surge'] = False  # Will be True if volume data available
        if 'volume' in df.columns:
            df['volume_ma'] = df['volume'].rolling(window=20).mean()
            df['volume_surge'] = df['volume'] > (df['volume_ma'] * 1.5)
        
        # Pullback detection
        df['pullback_in_uptrend'] = (
            df['uptrend'] & 
            (df['price'] < df['price'].rolling(window=5).max()) &
            (df['price_momentum'] < 0)
        )
        
        df['pullback_in_downtrend'] = (
            df['downtrend'] & 
            (df['price'] > df['price'].rolling(window=5).min()) &
            (df['price_momentum'] > 0)
        )
        
        # Support/Resistance strength at Fibonacci levels
        df['fib_support_strength'] = 0.0
        df['fib_resistance_strength'] = 0.0
        
        # Calculate how many times price has bounced from current levels (simplified)
        for i in range(20, len(df)):
            if df.iloc[i]['at_any_fib_level']:
                # Look back for price interactions with similar levels
                recent_window = df.iloc[i-20:i]
                bounces = (recent_window['at_any_fib_level']).sum()
                
                if df.iloc[i]['uptrend']:
                    df.iloc[i, df.columns.get_loc('fib_support_strength')] = min(bounces / 5.0, 1.0)
                else:
                    df.iloc[i, df.columns.get_loc('fib_resistance_strength')] = min(bounces / 5.0, 1.0)
        
        return df
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate buy/sell signals based on Fibonacci analysis"""
        df = data.copy()
        
        if 'fib_382' not in df.columns:
            df = self.calculate_indicators(df)
        
        # Initialize signal columns
        df['buy_signal'] = False
        df['sell_signal'] = False
        df['signal_strength'] = 0.0
        
        min_strength = self.parameters['min_strength']
        fib_levels = self.parameters['fib_levels']
        
        # Buy signals - Fibonacci support in uptrend
        buy_conditions = (
            df['uptrend'] &  # Overall uptrend
            df['pullback_in_uptrend'] &  # Currently pulling back
            df['at_any_fib_level'] &  # At a Fibonacci level
            (df['price_momentum'] > -0.02) &  # Momentum slowing/reversing
            (df['trend_strength'] > 0.01) &  # Strong enough trend
            (df['fib_support_strength'] > 0.2)  # Fibonacci level has shown support
        )
        
        # Enhanced buy signals for key Fibonacci levels (38.2%, 50%, 61.8%)
        key_fib_buy = (
            df['uptrend'] &
            df['pullback_in_uptrend'] &
            (df['at_fib_382'] | df['at_fib_500'] | df['at_fib_618']) &
            df['volume_surge'] &  # Volume confirmation
            (df['price_momentum'] > -0.01)  # Momentum stabilizing
        )
        
        # Sell signals - Fibonacci resistance in downtrend
        sell_conditions = (
            df['downtrend'] &  # Overall downtrend
            df['pullback_in_downtrend'] &  # Currently bouncing up
            df['at_any_fib_level'] &  # At a Fibonacci level
            (df['price_momentum'] < 0.02) &  # Momentum slowing/reversing
            (df['trend_strength'] > 0.01) &  # Strong enough trend
            (df['fib_resistance_strength'] > 0.2)  # Fibonacci level has shown resistance
        )
        
        # Enhanced sell signals for key Fibonacci levels
        key_fib_sell = (
            df['downtrend'] &
            df['pullback_in_downtrend'] &
            (df['at_fib_382'] | df['at_fib_500'] | df['at_fib_618']) &
            df['volume_surge'] &  # Volume confirmation
            (df['price_momentum'] < 0.01)  # Momentum stabilizing
        )
        
        # Combine conditions
        df['buy_signal'] = buy_conditions | key_fib_buy
        df['sell_signal'] = sell_conditions | key_fib_sell
        
        # Calculate signal strength
        # Base strength from Fibonacci level importance
        fib_strength = 0.0
        
        # Key Fibonacci levels get higher strength
        key_fib_strength = np.where(
            df['at_fib_382'] | df['at_fib_618'],
            0.8,  # Golden ratio levels
            np.where(
                df['at_fib_500'],
                0.7,  # 50% retracement
                np.where(
                    df['at_fib_236'] | df['at_fib_786'],
                    0.6,  # Other levels
                    0.3  # Generic level
                )
            )
        )
        
        # Trend strength component
        trend_strength_component = np.minimum(df['trend_strength'] * 50, 1.0)
        
        # Support/Resistance strength
        sr_strength = np.maximum(df['fib_support_strength'], df['fib_resistance_strength'])
        
        # Momentum reversal strength
        momentum_reversal = np.where(
            df['buy_signal'],
            np.maximum(0, -df['price_momentum'] * 20),  # Strength from momentum slowing
            np.where(
                df['sell_signal'],
                np.maximum(0, df['price_momentum'] * 20),
                0
            )
        )
        momentum_reversal = np.minimum(momentum_reversal, 1.0)
        
        # Volume confirmation strength
        volume_strength = np.where(df['volume_surge'], 0.3, 0.0)
        
        # Distance strength (closer to level = stronger)
        distance_strength = np.maximum(0, 1.0 - (df['nearest_fib_distance'] * 50))
        
        # Combine strength components
        df['signal_strength'] = np.where(
            df['buy_signal'] | df['sell_signal'],
            np.minimum(
                (key_fib_strength * 0.3 +
                 trend_strength_component * 0.25 +
                 sr_strength * 0.2 +
                 momentum_reversal * 0.15 +
                 distance_strength * 0.05 +
                 volume_strength * 0.05),
                1.0
            ),
            0.0
        )
        
        # Apply minimum strength filter
        df.loc[df['signal_strength'] < min_strength, 'buy_signal'] = False
        df.loc[df['signal_strength'] < min_strength, 'sell_signal'] = False
        df.loc[(~df['buy_signal']) & (~df['sell_signal']), 'signal_strength'] = 0.0
        
        return df