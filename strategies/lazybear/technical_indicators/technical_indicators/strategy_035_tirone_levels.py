#!/usr/bin/env python3
"""
Strategy 035: Tirone Levels

LazyBear Strategy Number: 011 (Original)
Strategy ID: 035 (Canonical)
LazyBear Name: Tirone Levels
Type: support/resistance/levels
TradingView URL: https://www.tradingview.com/script/ZdbzUf9B-Indicator-Tirone-Levels/

Description:
Implements Tirone Levels indicator which calculates dynamic support and resistance levels
based on highest high and lowest low over a specified period. Uses both Midpoint Method
and Mean Method as described by LazyBear. The levels act as potential reversal points
for trading decisions.

Tirone Levels provide horizontal support and resistance lines that adapt to price action,
helping traders identify potential entry and exit points based on calculated levels.
"""

import pandas as pd
import numpy as np
import sys
import os

# Add parent directories to path for imports
repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, repo_root)
from strategies.base_strategy import BaseStrategy


class Strategy035TironeLevels(BaseStrategy):
    """
    Strategy 035: Tirone Levels
    
    Calculates dynamic support and resistance levels using both Midpoint Method
    and Mean Method based on highest high and lowest low over specified periods.
    Generates buy/sell signals when price interacts with these levels.
    """

    def __init__(self, parameters: dict = None):
        """
        Initialize strategy with Tirone Levels parameters
        
        Args:
            parameters: Dictionary with strategy parameters
        """
        # Default parameters matching LazyBear implementation
        default_params = {
            'length': 20,  # Period for calculating highest/lowest (Pine default)
            # Pine defaults: method_mp=false (off), method_mm=true (on)
            'use_midpoint_method': False,   # Midpoint method off by default
            'use_mean_method': True,        # Mean method on by default  
            'signal_threshold': 0.6,        # Minimum signal strength to trade
            'level_tolerance': 0.002,       # Tolerance for level touches (0.2%)
        }
        
        # Merge with provided parameters
        if parameters:
            default_params.update(parameters)
        
        # Initialize parent class
        super().__init__(
            name="Strategy_035_TironeLevels",
            parameters=default_params
        )

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Tirone Levels using both Midpoint and Mean methods
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with Tirone level columns
        """
        # Ensure we have required columns
        if 'close' not in data.columns and 'price' in data.columns:
            data['close'] = data['price']
        
        # Ensure we have OHLC data
        if 'open' not in data.columns:
            data['open'] = data['close'].shift(1).fillna(data['close'])
        if 'high' not in data.columns:
            data['high'] = data['close']
        if 'low' not in data.columns:
            data['low'] = data['close']
        
        # Extract parameters
        length = self.parameters['length']
        
        # Calculate rolling highest high and lowest low
        data['highest_high'] = data['high'].rolling(window=length, min_periods=1).max()
        data['lowest_low'] = data['low'].rolling(window=length, min_periods=1).min()
        
        # Calculate range
        data['hl_range'] = data['highest_high'] - data['lowest_low']
        
        # Midpoint Method Levels (1/3-2/3 method)
        if self.parameters['use_midpoint_method']:
            # Top Level: hh - ((hh-ll)/3)
            data['midpoint_top'] = data['highest_high'] - (data['hl_range'] / 3)
            
            # Center Level: ll + ((hh-ll)/2) 
            data['midpoint_center'] = data['lowest_low'] + (data['hl_range'] / 2)
            
            # Bottom Level: ll + ((hh-ll)/3)
            data['midpoint_bottom'] = data['lowest_low'] + (data['hl_range'] / 3)
        
        # Mean Method Levels (5-level system)
        if self.parameters['use_mean_method']:
            # Adjusted Mean: (hh + ll + close) / 3
            data['adjusted_mean'] = (data['highest_high'] + data['lowest_low'] + data['close']) / 3
            
            # Extreme High: am + (hh-ll)
            data['mean_extreme_high'] = data['adjusted_mean'] + data['hl_range']
            
            # Extreme Low: am - (hh-ll)  
            data['mean_extreme_low'] = data['adjusted_mean'] - data['hl_range']
            
            # Resistance High: 2*am - ll
            data['mean_resistance_high'] = 2 * data['adjusted_mean'] - data['lowest_low']
            
            # Resistance Low: 2*am - hh
            data['mean_resistance_low'] = 2 * data['adjusted_mean'] - data['highest_high']
        
        # Store indicators for debugging
        level_columns = ['highest_high', 'lowest_low']
        if self.parameters['use_midpoint_method']:
            level_columns.extend(['midpoint_top', 'midpoint_center', 'midpoint_bottom'])
        if self.parameters['use_mean_method']:
            level_columns.extend(['adjusted_mean', 'mean_extreme_high', 'mean_extreme_low', 
                                 'mean_resistance_high', 'mean_resistance_low'])
        
        self.indicators = data[level_columns].copy()
        
        return data

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate buy/sell signals based on price interaction with Tirone Levels
        
        CRITICAL: Apply look-ahead bias prevention using .shift(1) for stateful logic.
        
        Args:
            data: DataFrame with price and indicator data
            
        Returns:
            DataFrame with buy_signal, sell_signal, and signal_strength columns
        """
        # Initialize signal columns
        data['buy_signal'] = False
        data['sell_signal'] = False
        data['signal_strength'] = 0.0
        
        # Extract parameters
        tolerance = self.parameters['level_tolerance']
        threshold = self.parameters['signal_threshold']
        
        # Collect all support and resistance levels
        support_levels = []
        resistance_levels = []
        
        if self.parameters['use_midpoint_method']:
            support_levels.extend(['midpoint_bottom', 'midpoint_center'])
            resistance_levels.extend(['midpoint_center', 'midpoint_top'])
        
        if self.parameters['use_mean_method']:
            support_levels.extend(['mean_extreme_low', 'mean_resistance_low', 'adjusted_mean'])
            resistance_levels.extend(['adjusted_mean', 'mean_resistance_high', 'mean_extreme_high'])
        
        # Initialize signal strength components
        buy_strength = pd.Series(0.0, index=data.index)
        sell_strength = pd.Series(0.0, index=data.index)
        
        # Check for bounces off support levels (buy signals)
        for support_col in support_levels:
            if support_col in data.columns:
                support_level = data[support_col]
                
                # Price touching support from above (potential bounce)
                price_near_support = np.abs(data['close'] - support_level) <= (support_level * tolerance)
                price_above_support = data['close'] > support_level
                
                # Look for recent low near support followed by price recovery
                recent_low_at_support = (
                    data['low'].shift(1) <= support_level * (1 + tolerance)
                ) & (data['close'] > data['low'].shift(1))
                
                # Combine conditions for support bounce
                support_bounce = (price_near_support & price_above_support) | recent_low_at_support
                
                # Add to buy strength (weight by level importance)
                level_weight = 1.0 if 'center' in support_col or 'adjusted_mean' in support_col else 0.8
                buy_strength += support_bounce.astype(float) * level_weight
        
        # Check for rejections at resistance levels (sell signals)  
        for resistance_col in resistance_levels:
            if resistance_col in data.columns:
                resistance_level = data[resistance_col]
                
                # Price touching resistance from below (potential rejection)
                price_near_resistance = np.abs(data['close'] - resistance_level) <= (resistance_level * tolerance)
                price_below_resistance = data['close'] < resistance_level
                
                # Look for recent high near resistance followed by price decline
                recent_high_at_resistance = (
                    data['high'].shift(1) >= resistance_level * (1 - tolerance)
                ) & (data['close'] < data['high'].shift(1))
                
                # Combine conditions for resistance rejection
                resistance_rejection = (price_near_resistance & price_below_resistance) | recent_high_at_resistance
                
                # Add to sell strength (weight by level importance)
                level_weight = 1.0 if 'center' in resistance_col or 'adjusted_mean' in resistance_col else 0.8
                sell_strength += resistance_rejection.astype(float) * level_weight
        
        # Normalize signal strengths (clip to 0-1 range)
        max_levels = max(len(support_levels), len(resistance_levels))
        if max_levels > 0:
            buy_strength = (buy_strength / max_levels).clip(0, 1)
            sell_strength = (sell_strength / max_levels).clip(0, 1)
        
        # Generate signals with momentum confirmation
        # Add momentum filter: only buy on upward momentum, sell on downward momentum
        momentum = data['close'] - data['close'].shift(3)  # 3-bar momentum
        
        # Buy signals: support bounce + upward momentum
        strong_buy = (buy_strength >= threshold) & (momentum > 0)
        data.loc[strong_buy, 'buy_signal'] = True
        data.loc[strong_buy, 'signal_strength'] = buy_strength[strong_buy]
        
        # Sell signals: resistance rejection + downward momentum  
        strong_sell = (sell_strength >= threshold) & (momentum < 0)
        data.loc[strong_sell, 'sell_signal'] = True
        data.loc[strong_sell, 'signal_strength'] = sell_strength[strong_sell]
        
        # --- State Management for Look-ahead Bias Prevention ---
        # Create unified signal for position tracking
        raw_signal = np.select(
            [data['buy_signal'], data['sell_signal']],
            [1, -1],  # +1 for buy, -1 for sell
            default=0
        )
        
        # Calculate position state: ffill non-zero signals to track current position
        position = pd.Series(raw_signal).replace(0, np.nan).ffill().fillna(0)
        
        # Apply state-based filtering using previous bar to avoid look-ahead bias
        is_flat = (position.shift(1) == 0)
        is_long = (position.shift(1) == 1) 
        is_short = (position.shift(1) == -1)
        
        # Refine signals based on position state
        # Only allow buy if flat, only allow sell if long (or flat for short strategies)
        data.loc[~(data['buy_signal'] & (is_flat | is_short)), 'buy_signal'] = False
        data.loc[~(data['sell_signal'] & (is_long | is_flat)), 'sell_signal'] = False
        
        # Reset signal strength where signals were filtered out
        data.loc[~(data['buy_signal'] | data['sell_signal']), 'signal_strength'] = 0.0
        
        # Store signals for debugging
        self.signals = data[['buy_signal', 'sell_signal', 'signal_strength']].copy()
        
        return data


# Test function for smoke testing
if __name__ == "__main__":
    # Create sample data with realistic OHLCV structure
    import datetime
    
    print("🔍 Testing Strategy 035: Tirone Levels")
    
    # Generate sample data with trending and ranging periods
    np.random.seed(42)  # For reproducible results
    dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='5min')
    n_points = len(dates)
    
    # Create price movements with trends and reversals (good for level testing)
    base_price = 100
    trend = np.linspace(0, 10, n_points)  # Upward trend
    noise = np.random.randn(n_points).cumsum() * 0.3
    cyclical = 5 * np.sin(np.linspace(0, 4*np.pi, n_points))  # Cyclical pattern
    
    price_series = base_price + trend + noise + cyclical
    
    sample_data = pd.DataFrame({
        'timestamp': dates,
        'open': price_series + np.random.randn(n_points) * 0.1,
        'high': price_series + np.abs(np.random.randn(n_points)) * 0.3 + 0.2,
        'low': price_series - np.abs(np.random.randn(n_points)) * 0.3 - 0.2,
        'close': price_series,
        'volume': np.random.randint(1000, 10000, n_points)
    })
    
    # Ensure OHLC consistency
    sample_data['high'] = sample_data[['open', 'close', 'high']].max(axis=1)
    sample_data['low'] = sample_data[['open', 'close', 'low']].min(axis=1)
    sample_data['price'] = sample_data['close']
    
    print(f"Sample data shape: {sample_data.shape}")
    print(f"Price range: {sample_data['close'].min():.2f} - {sample_data['close'].max():.2f}")
    
    try:
        # Test strategy initialization
        strategy = Strategy011TironeLevels()
        print("✅ Strategy initialization successful")
        
        # Test indicator calculation
        data_with_indicators = strategy.calculate_indicators(sample_data.copy())
        print(f"✅ Indicator calculation successful")
        
        # Show some level calculations
        latest_data = data_with_indicators.iloc[-1]
        print(f"📊 Latest Tirone Levels:")
        if 'midpoint_top' in latest_data:
            print(f"   Midpoint Top: {latest_data['midpoint_top']:.2f}")
            print(f"   Midpoint Center: {latest_data['midpoint_center']:.2f}")  
            print(f"   Midpoint Bottom: {latest_data['midpoint_bottom']:.2f}")
        if 'adjusted_mean' in latest_data:
            print(f"   Adjusted Mean: {latest_data['adjusted_mean']:.2f}")
            print(f"   Mean Extreme High: {latest_data['mean_extreme_high']:.2f}")
            print(f"   Mean Extreme Low: {latest_data['mean_extreme_low']:.2f}")
        
        # Test signal generation
        data_with_signals = strategy.generate_signals(data_with_indicators)
        print("✅ Signal generation successful")
        
        # Print summary statistics
        buy_signals = data_with_signals['buy_signal'].sum()
        sell_signals = data_with_signals['sell_signal'].sum()
        avg_strength = data_with_signals['signal_strength'].mean()
        max_strength = data_with_signals['signal_strength'].max()
        
        print(f"📊 Signal Summary:")
        print(f"   Buy signals: {buy_signals}")
        print(f"   Sell signals: {sell_signals}")
        print(f"   Average signal strength: {avg_strength:.3f}")
        print(f"   Maximum signal strength: {max_strength:.3f}")
        
        # Check level spread
        if 'midpoint_top' in data_with_indicators.columns:
            avg_spread = (data_with_indicators['midpoint_top'] - data_with_indicators['midpoint_bottom']).mean()
            print(f"   Average level spread: {avg_spread:.2f}")
        
        if buy_signals > 0 or sell_signals > 0:
            print(f"✅ Strategy generated {buy_signals + sell_signals} total signals - Ready for deployment")
        else:
            print("ℹ️  No signals in test data - May need parameter adjustment or longer test period")
            
        # Test different parameters
        print("\n🔧 Testing with different parameters...")
        alt_params = {'length': 10, 'signal_threshold': 0.4}
        strategy_alt = Strategy011TironeLevels(parameters=alt_params)
        data_alt = strategy_alt.calculate_indicators(sample_data.copy())
        data_alt = strategy_alt.generate_signals(data_alt)
        
        alt_buy = data_alt['buy_signal'].sum()
        alt_sell = data_alt['sell_signal'].sum()
        print(f"   With length=10, threshold=0.4: {alt_buy} buy, {alt_sell} sell signals")
            
    except Exception as e:
        print(f"❌ Strategy test failed with error: {e}")
        import traceback
        traceback.print_exc()
