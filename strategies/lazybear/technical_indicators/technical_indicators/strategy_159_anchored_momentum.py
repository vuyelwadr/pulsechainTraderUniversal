#!/usr/bin/env python3
"""
Strategy 159: Anchored Momentum

LazyBear's Anchored Momentum indicator captures "relative momentum" by using a 
Simple Moving Average (SMA) as a reference point instead of previous close values.
This creates a momentum oscillator that shows how price momentum compares to the
average price over a given period.

Formula: amom = 100*((smoothed_close/SMA(close, period)) - 1)

The indicator provides three key components:
1. Momentum Line - The main anchored momentum calculation
2. Signal Line - EMA smoothing of the momentum line  
3. Zero Line - Reference point for bullish/bearish momentum

TradingView URL: https://www.tradingview.com/v/TBTFDWDq/
Type: momentum/oscillator
"""

import pandas as pd
import numpy as np
import talib
from typing import Dict
import sys
import os

# Add parent directories to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)

# Change to current working directory for import
current_dir = os.getcwd()
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from strategies.base_strategy import BaseStrategy

# Also try importing vectorized helpers if available
try:
    sys.path.insert(0, os.path.join(project_root, 'src'))
    from utils.vectorized_helpers import (
        crossover, crossunder, apply_position_constraints,
        calculate_signal_strength
    )
    VECTORIZED_HELPERS_AVAILABLE = True
except ImportError:
    VECTORIZED_HELPERS_AVAILABLE = False
    
    # Define simple crossover functions if not available
    def crossover(series1, series2):
        """Simple crossover detection"""
        return (series1 > series2) & (series1.shift(1) <= series2.shift(1))
    
    def crossunder(series1, series2):
        """Simple crossunder detection"""
        return (series1 < series2) & (series1.shift(1) >= series2.shift(1))
    
    def apply_position_constraints(buy_signals, sell_signals, allow_short=False):
        """Simple position constraint application"""
        return buy_signals, sell_signals
    
    def calculate_signal_strength(factors, weights=None):
        """Simple signal strength calculation"""
        if not factors:
            return pd.Series(0.0, index=factors[0].index if factors else [])
        df = pd.concat(factors, axis=1)
        return df.mean(axis=1)


class Strategy159AnchoredMomentum(BaseStrategy):
    """
    Strategy 159: Anchored Momentum
    
    Strategy Number: 159
    LazyBear Name: Anchored Momentum
    Type: momentum/oscillator
    
    Description:
    Anchored Momentum is a modified momentum indicator that captures "relative momentum"
    by using a Simple Moving Average as the reference point instead of previous close
    values. This approach provides a more stable momentum reading that shows how
    current price momentum compares to the average momentum over the specified period.
    
    Key Features:
    - Uses SMA as momentum anchor point for stability
    - Optional EMA smoothing of close prices before calculation
    - Signal line provides crossover trading opportunities
    - Zero line acts as bullish/bearish momentum divider
    - Configurable periods for momentum and signal calculations
    """
    
    def __init__(self, parameters: Dict = None):
        """
        Initialize Anchored Momentum strategy
        
        Default parameters based on LazyBear's original implementation
        """
        # Define default parameters matching TradingView specification
        default_params = {
            'momentum_period': 10,          # Period for SMA anchor (default: 10)
            'signal_period': 8,             # Period for signal line smoothing (default: 8)
            'smooth_momentum': False,       # Use EMA smoothing of close (default: FALSE)
            'smoothing_period': 7,          # Period for close price smoothing (default: 7)
            'signal_threshold': 0.6,        # Minimum signal strength to trade
            'overbought_threshold': 5.0,    # Upper threshold for strong momentum
            'oversold_threshold': -5.0,     # Lower threshold for strong momentum
            'use_zero_line_filter': True,   # Filter signals based on zero line position
            'min_momentum_strength': 1.0,   # Minimum momentum absolute value for signals
        }
        
        # Merge with provided parameters
        if parameters:
            default_params.update(parameters)
        
        # Initialize parent class
        super().__init__(
            name="Strategy_159_AnchoredMomentum",
            parameters=default_params
        )
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Anchored Momentum and related indicators
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with Anchored Momentum indicators
        """
        # Ensure we have required columns
        if 'close' not in data.columns and 'price' in data.columns:
            data['close'] = data['price']
        
        # Ensure we have all required data
        if 'close' not in data.columns:
            raise ValueError("Data must contain 'close' or 'price' column")
        
        # Extract parameters
        momentum_period = self.parameters['momentum_period']
        signal_period = self.parameters['signal_period']
        smooth_momentum = self.parameters['smooth_momentum']
        smoothing_period = self.parameters['smoothing_period']
        
        # Step 1: Prepare close prices (with optional smoothing)
        if smooth_momentum:
            # Use EMA smoothing of close prices as specified in TradingView
            data['smoothed_close'] = talib.EMA(data['close'].values, timeperiod=smoothing_period)
        else:
            # Use raw close prices
            data['smoothed_close'] = data['close']
        
        # Step 2: Calculate the SMA anchor
        data['sma_anchor'] = talib.SMA(data['close'].values, timeperiod=momentum_period)
        
        # Step 3: Calculate Anchored Momentum
        # Formula: amom = 100 * ((smoothed_close / SMA(close, period)) - 1)
        data['anchored_momentum_raw'] = 100 * ((data['smoothed_close'] / data['sma_anchor']) - 1)
        
        # Handle division by zero or NaN values
        data['anchored_momentum_raw'] = data['anchored_momentum_raw'].fillna(0)
        
        # Replace infinite values with 0
        data['anchored_momentum_raw'] = data['anchored_momentum_raw'].replace([np.inf, -np.inf], 0)
        
        # Cap extreme values to prevent unrealistic momentum readings
        # This prevents division by very small SMA values from creating huge momentum values
        momentum_cap = 100.0  # Cap at +/- 100%
        data['anchored_momentum_raw'] = data['anchored_momentum_raw'].clip(-momentum_cap, momentum_cap)
        
        # Step 4: Apply the momentum line (this is our main indicator)
        data['anchored_momentum'] = data['anchored_momentum_raw']
        
        # Step 5: Calculate signal line (EMA of momentum)
        data['momentum_signal'] = talib.EMA(data['anchored_momentum'].values, timeperiod=signal_period)
        
        # Fill NaN values in signal line
        data['momentum_signal'] = data['momentum_signal'].fillna(0)
        
        # Step 6: Calculate zero line (always 0 for reference)
        data['zero_line'] = 0.0
        
        # Step 7: Calculate momentum histogram (momentum - signal)
        data['momentum_histogram'] = data['anchored_momentum'] - data['momentum_signal']
        
        # Step 8: Additional trend indicators for filtering
        # Short-term trend
        data['sma_short'] = talib.SMA(data['close'].values, timeperiod=20)
        
        # Price momentum for confirmation
        data['price_momentum'] = data['close'] - data['close'].shift(5)
        
        # Store indicators for debugging
        self.indicators = data[[
            'smoothed_close', 'sma_anchor', 'anchored_momentum_raw', 
            'anchored_momentum', 'momentum_signal', 'momentum_histogram'
        ]].copy()
        
        return data
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate buy/sell signals based on Anchored Momentum crossovers and levels
        
        Args:
            data: DataFrame with price and momentum data
            
        Returns:
            DataFrame with buy_signal, sell_signal, and signal_strength columns
        """
        # Initialize signal columns
        data['buy_signal'] = False
        data['sell_signal'] = False
        data['signal_strength'] = 0.0
        
        # Extract parameters
        threshold = self.parameters['signal_threshold']
        overbought = self.parameters['overbought_threshold']
        oversold = self.parameters['oversold_threshold']
        use_zero_filter = self.parameters['use_zero_line_filter']
        min_strength = self.parameters['min_momentum_strength']
        
        # Ensure we have enough data
        min_periods = max(self.parameters['momentum_period'], self.parameters['signal_period'])
        if len(data) < min_periods:
            return data
        
        # Primary signal conditions
        
        # Buy signals: Momentum crosses above signal line
        momentum_cross_up = crossover(data['anchored_momentum'], data['momentum_signal'])
        
        # Sell signals: Momentum crosses below signal line  
        momentum_cross_down = crossunder(data['anchored_momentum'], data['momentum_signal'])
        
        # Additional signal conditions
        buy_conditions = [momentum_cross_up]
        sell_conditions = [momentum_cross_down]
        
        # Filter 1: Momentum strength (avoid weak signals near zero)
        momentum_strong_positive = data['anchored_momentum'] > min_strength
        momentum_strong_negative = data['anchored_momentum'] < -min_strength
        
        buy_conditions.append(momentum_strong_positive)
        sell_conditions.append(momentum_strong_negative)
        
        # Filter 2: Zero line filter (optional)
        if use_zero_filter:
            # Buy signals stronger when momentum is above zero (bullish territory)
            momentum_bullish = data['anchored_momentum'] > 0
            # Sell signals stronger when momentum is below zero (bearish territory)
            momentum_bearish = data['anchored_momentum'] < 0
            
            buy_conditions.append(momentum_bullish)
            sell_conditions.append(momentum_bearish)
        
        # Filter 3: Trend confirmation using price vs SMA
        if 'sma_short' in data.columns:
            # Buy when price is above short-term average (uptrend)
            uptrend = data['close'] > data['sma_short']
            # Sell when price is below short-term average (downtrend)
            downtrend = data['close'] < data['sma_short']
            
            buy_conditions.append(uptrend)
            sell_conditions.append(downtrend)
        
        # Combine conditions
        data['buy_signal'] = pd.concat(buy_conditions, axis=1).all(axis=1)
        data['sell_signal'] = pd.concat(sell_conditions, axis=1).all(axis=1)
        
        # Calculate signal strength (0-1 scale)
        strength_factors = []
        
        # Factor 1: Momentum-signal divergence (how strong the crossover is)
        momentum_divergence = np.abs(data['anchored_momentum'] - data['momentum_signal'])
        # Normalize to 0-1 scale (assume max divergence of 10 for normalization)
        divergence_strength = (momentum_divergence / 10.0).clip(0, 1)
        strength_factors.append(divergence_strength)
        
        # Factor 2: Absolute momentum level (extreme values = stronger signals)
        momentum_extreme = pd.Series(0.0, index=data.index)
        momentum_abs = np.abs(data['anchored_momentum'])
        
        # Scale momentum strength
        momentum_extreme[momentum_abs > 15] = 1.0      # Very extreme
        momentum_extreme[(momentum_abs > 10) & (momentum_abs <= 15)] = 0.8   # Strong
        momentum_extreme[(momentum_abs > 5) & (momentum_abs <= 10)] = 0.6    # Moderate  
        momentum_extreme[(momentum_abs > 2) & (momentum_abs <= 5)] = 0.4     # Weak
        momentum_extreme[momentum_abs <= 2] = 0.2      # Very weak
        
        strength_factors.append(momentum_extreme)
        
        # Factor 3: Coming from extreme levels bonus
        extreme_bonus = pd.Series(0.0, index=data.index)
        
        # Buy signals coming from oversold get bonus
        coming_from_oversold = data['anchored_momentum'].shift(1) < oversold
        extreme_bonus[coming_from_oversold & data['buy_signal']] = 0.3
        
        # Sell signals coming from overbought get bonus
        coming_from_overbought = data['anchored_momentum'].shift(1) > overbought  
        extreme_bonus[coming_from_overbought & data['sell_signal']] = 0.3
        
        strength_factors.append(extreme_bonus)
        
        # Factor 4: Histogram direction confirmation
        histogram_confirmation = pd.Series(0.0, index=data.index)
        
        # Positive histogram supports buy signals
        positive_histogram = data['momentum_histogram'] > 0
        histogram_confirmation[positive_histogram & data['buy_signal']] = 0.2
        
        # Negative histogram supports sell signals
        negative_histogram = data['momentum_histogram'] < 0
        histogram_confirmation[negative_histogram & data['sell_signal']] = 0.2
        
        strength_factors.append(histogram_confirmation)
        
        # Factor 5: Price momentum confirmation
        if 'price_momentum' in data.columns:
            price_momentum_confirmation = pd.Series(0.0, index=data.index)
            
            # Positive price momentum supports buy signals
            positive_price_momentum = data['price_momentum'] > 0
            price_momentum_confirmation[positive_price_momentum & data['buy_signal']] = 0.15
            
            # Negative price momentum supports sell signals
            negative_price_momentum = data['price_momentum'] < 0
            price_momentum_confirmation[negative_price_momentum & data['sell_signal']] = 0.15
            
            strength_factors.append(price_momentum_confirmation)
        
        # Combine strength factors
        data['signal_strength'] = calculate_signal_strength(
            strength_factors,
            weights=None  # Equal weights
        )
        
        # Ensure signal strength has no NaN values
        data['signal_strength'] = data['signal_strength'].fillna(0.0)
        
        # Ensure signal strength is in valid range [0, 1]
        data['signal_strength'] = data['signal_strength'].clip(0.0, 1.0)
        
        # Apply minimum threshold - filter out weak signals
        weak_signals = data['signal_strength'] < threshold
        data.loc[weak_signals, 'buy_signal'] = False
        data.loc[weak_signals, 'sell_signal'] = False
        
        # Prevent look-ahead bias
        data['buy_signal'] = data['buy_signal'] & (data['anchored_momentum'].shift(1).notna())
        data['sell_signal'] = data['sell_signal'] & (data['anchored_momentum'].shift(1).notna())
        
        # Apply position constraints (avoid simultaneous buy/sell, etc.)
        data['buy_signal'], data['sell_signal'] = apply_position_constraints(
            data['buy_signal'],
            data['sell_signal'],
            allow_short=False
        )
        
        # Store signals for debugging
        self.signals = data[['buy_signal', 'sell_signal', 'signal_strength']].copy()
        
        return data
    
    def validate_signals(self, data: pd.DataFrame) -> bool:
        """
        Validate that signals are properly formed
        
        Returns:
            True if signals are valid, False otherwise
        """
        # Check for required columns
        required = ['buy_signal', 'sell_signal', 'signal_strength']
        if not all(col in data.columns for col in required):
            return False
        
        # Check for NaN values in signal columns
        if data[required].isna().any().any():
            return False
        
        # Check for simultaneous buy and sell signals
        simultaneous = (data['buy_signal'] & data['sell_signal']).any()
        if simultaneous:
            return False
        
        # Check signal strength is in valid range
        if (data['signal_strength'] < 0).any() or (data['signal_strength'] > 1).any():
            return False
        
        # Check that we have momentum indicators
        momentum_cols = ['anchored_momentum', 'momentum_signal']
        if not all(col in data.columns for col in momentum_cols):
            return False
        
        # Check for reasonable momentum values (not too extreme)
        if np.abs(data['anchored_momentum']).max() > 1000:  # Sanity check
            return False
        
        return True


if __name__ == "__main__":
    """Test the strategy with sample data"""
    import warnings
    warnings.filterwarnings('ignore')
    
    # Create sample data with realistic OHLC structure
    np.random.seed(42)  # For reproducible results
    dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='5min')
    n_points = len(dates)
    
    # Generate trending price data to test anchored momentum
    base_price = 100
    trend = np.linspace(0, 20, n_points)  # Upward trend
    noise = np.random.normal(0, 2, n_points).cumsum()
    prices = base_price + trend + noise
    
    # Create OHLC from price series
    sample_data = pd.DataFrame({
        'timestamp': dates,
        'close': prices,
        'volume': np.random.randint(1000, 10000, n_points)
    })
    
    # Generate realistic OHLC data
    sample_data['open'] = sample_data['close'].shift(1).fillna(sample_data['close'])
    
    # Add some noise to create high/low
    noise = np.random.normal(0, 0.5, n_points)
    sample_data['high'] = sample_data['close'] + np.abs(noise)
    sample_data['low'] = sample_data['close'] - np.abs(noise)
    
    # Ensure OHLC relationships are maintained
    sample_data['high'] = np.maximum(sample_data['high'], 
                                   np.maximum(sample_data['open'], sample_data['close']))
    sample_data['low'] = np.minimum(sample_data['low'], 
                                  np.minimum(sample_data['open'], sample_data['close']))
    
    sample_data['price'] = sample_data['close']
    
    print(f"Testing Strategy 159: Anchored Momentum")
    print(f"Sample data shape: {sample_data.shape}")
    print(f"Date range: {sample_data['timestamp'].min()} to {sample_data['timestamp'].max()}")
    print(f"Price range: {sample_data['close'].min():.2f} to {sample_data['close'].max():.2f}")
    
    # Test strategy
    try:
        strategy = Strategy159AnchoredMomentum()
        print(f"Strategy initialized: {strategy.name}")
        print(f"Parameters: {strategy.parameters}")
        
        # Calculate indicators
        data_with_indicators = strategy.calculate_indicators(sample_data.copy())
        print(f"✅ Indicators calculated successfully")
        print(f"Momentum columns added: {[col for col in data_with_indicators.columns if 'momentum' in col.lower() or 'anchor' in col.lower()]}")
        
        # Generate signals
        data_with_signals = strategy.generate_signals(data_with_indicators)
        print(f"✅ Signals generated successfully")
        
        # Debug validation before checking
        print(f"Debug validation:")
        print(f"  Required columns present: {all(col in data_with_signals.columns for col in ['buy_signal', 'sell_signal', 'signal_strength'])}")
        print(f"  Signal strength range: {data_with_signals['signal_strength'].min():.3f} to {data_with_signals['signal_strength'].max():.3f}")
        print(f"  NaN values in signals: {data_with_signals[['buy_signal', 'sell_signal', 'signal_strength']].isna().sum().sum()}")
        print(f"  Simultaneous buy/sell: {(data_with_signals['buy_signal'] & data_with_signals['sell_signal']).sum()}")
        print(f"  Momentum columns present: {all(col in data_with_signals.columns for col in ['anchored_momentum', 'momentum_signal'])}")
        print(f"  Max momentum value: {np.abs(data_with_signals['anchored_momentum']).max():.2f}")
        
        # Validate signals
        if strategy.validate_signals(data_with_signals):
            print("✅ Signal validation passed")
            
            # Show results
            buy_count = data_with_signals['buy_signal'].sum()
            sell_count = data_with_signals['sell_signal'].sum()
            avg_strength = data_with_signals['signal_strength'].mean()
            
            print(f"Buy signals: {buy_count}")
            print(f"Sell signals: {sell_count}")  
            print(f"Average signal strength: {avg_strength:.3f}")
            
            # Show sample of Anchored Momentum values
            momentum_stats = data_with_signals['anchored_momentum'].describe()
            print(f"Anchored Momentum statistics:")
            print(f"  Mean: {momentum_stats['mean']:.4f}")
            print(f"  Std: {momentum_stats['std']:.4f}")  
            print(f"  Min: {momentum_stats['min']:.4f}")
            print(f"  Max: {momentum_stats['max']:.4f}")
            
            # Show some example values
            print(f"\nExample momentum values (last 5 points):")
            last_5 = data_with_signals[['anchored_momentum', 'momentum_signal', 'signal_strength', 'buy_signal', 'sell_signal']].tail(5)
            for idx, row in last_5.iterrows():
                print(f"  Momentum: {row['anchored_momentum']:6.2f}, Signal: {row['momentum_signal']:6.2f}, Strength: {row['signal_strength']:.3f}, Buy: {row['buy_signal']}, Sell: {row['sell_signal']}")
            
            # Check for any extreme values
            if np.abs(data_with_signals['anchored_momentum']).max() > 100:
                print("⚠️  Warning: Some momentum values are quite large (>100)")
            
            print("✅ Strategy 159 Anchored Momentum implementation completed successfully!")
            
        else:
            print("❌ Signal validation failed")
            
    except Exception as e:
        print(f"❌ Error testing strategy: {e}")
        import traceback
        traceback.print_exc()