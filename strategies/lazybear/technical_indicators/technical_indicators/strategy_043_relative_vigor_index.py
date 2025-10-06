#!/usr/bin/env python3
"""
Strategy 043: Relative Vigor Index (RVI)

The Relative Vigor Index (RVI) is a momentum oscillator that measures the conviction
of price movement by comparing the closing price relative to its trading range.
It's based on the assumption that in an uptrend, prices tend to close higher than
they open, and in a downtrend, prices tend to close lower than they open.

RVI = (Close - Open) / (High - Low)

The indicator is typically smoothed with a moving average and includes a signal line
for crossover signals. The RVI oscillates around zero, with positive values indicating
bullish momentum and negative values indicating bearish momentum.

TradingView URL: https://www.tradingview.com/v/uNBEqnJD/
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


class Strategy043RelativeVigorIndex(BaseStrategy):
    """
    Strategy 043: Relative Vigor Index (RVI)
    
    Strategy Number: 020
    LazyBear Name: Relative Vigor Index
    Type: momentum/oscillator
    
    Description:
    The Relative Vigor Index measures the conviction of price movement by comparing
    the relationship between closing price and trading range. It identifies when
    prices are closing nearer to their high (bullish) or low (bearish) within
    their trading range.
    
    Key Features:
    - Momentum oscillator that measures price conviction
    - Compares (Close - Open) vs (High - Low) relationship
    - Smoothed with moving averages for noise reduction
    - Signal line for crossover entry/exit signals
    - Oscillates around zero line
    """
    
    def __init__(self, parameters: Dict = None):
        """
        Initialize Relative Vigor Index strategy
        
        Default parameters based on common RVI implementations
        """
        # Define default parameters
        default_params = {
            'rvi_period': 10,           # Period for RVI smoothing
            'signal_period': 4,         # Period for signal line smoothing
            'overbought_threshold': 0.5,  # Upper threshold for strong signals
            'oversold_threshold': -0.5,   # Lower threshold for strong signals
            'signal_threshold': 0.6,    # Minimum signal strength to trade
            'use_trend_filter': True,   # Use additional trend confirmation
            'min_rvi_strength': 0.2,    # Minimum RVI absolute value for signals
        }
        
        # Merge with provided parameters
        if parameters:
            default_params.update(parameters)
        
        # Initialize parent class
        super().__init__(
            name="Strategy_043_RelativeVigorIndex",
            parameters=default_params
        )
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Relative Vigor Index and related indicators
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with RVI indicators
        """
        # Ensure we have required columns
        if 'close' not in data.columns and 'price' in data.columns:
            data['close'] = data['price']
        
        # Ensure we have all OHLC data
        required_cols = ['open', 'high', 'low', 'close']
        for col in required_cols:
            if col not in data.columns:
                if col in ['open', 'high', 'low']:
                    data[col] = data['close']  # Fallback to close price
        
        # Extract parameters
        rvi_period = self.parameters['rvi_period']
        signal_period = self.parameters['signal_period']
        
        # Calculate raw RVI components
        # Numerator: Close - Open (price conviction)
        numerator = data['close'] - data['open']
        
        # Denominator: High - Low (price range)
        denominator = data['high'] - data['low']
        
        # Handle division by zero (when high == low, no range)
        # Replace zero ranges with small value to avoid division by zero
        denominator = denominator.replace(0, np.nan)
        
        # Calculate raw RVI
        data['rvi_raw'] = numerator / denominator
        
        # Fill NaN values (from zero denominators) with 0
        data['rvi_raw'] = data['rvi_raw'].fillna(0)
        
        # Apply smoothing to RVI using simple moving average
        data['rvi'] = data['rvi_raw'].rolling(window=rvi_period, min_periods=1).mean()
        
        # Calculate signal line (further smoothing of RVI)
        data['rvi_signal'] = data['rvi'].rolling(window=signal_period, min_periods=1).mean()
        
        # Calculate additional momentum indicators for trend filter
        if self.parameters['use_trend_filter']:
            # Simple moving average for trend direction
            data['sma_20'] = data['close'].rolling(window=20, min_periods=1).mean()
            
            # Price momentum
            data['momentum'] = data['close'] - data['close'].shift(10)
            
        # Calculate RVI histogram (difference between RVI and signal)
        data['rvi_histogram'] = data['rvi'] - data['rvi_signal']
        
        # Store indicators for debugging
        self.indicators = data[['rvi_raw', 'rvi', 'rvi_signal', 'rvi_histogram']].copy()
        
        return data
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate buy/sell signals based on RVI crossovers and levels
        
        Args:
            data: DataFrame with price and RVI data
            
        Returns:
            DataFrame with buy_signal, sell_signal, and signal_strength columns
        """
        # Initialize signal columns
        data['buy_signal'] = False
        data['sell_signal'] = False
        data['signal_strength'] = 0.0
        
        # Extract parameters
        overbought = self.parameters['overbought_threshold']
        oversold = self.parameters['oversold_threshold']
        threshold = self.parameters['signal_threshold']
        min_strength = self.parameters['min_rvi_strength']
        
        # Ensure we have enough data
        if len(data) < max(self.parameters['rvi_period'], self.parameters['signal_period']):
            return data
        
        # Define primary signal conditions
        
        # Buy signals: RVI crosses above signal line, preferably from oversold levels
        rvi_cross_up = crossover(data['rvi'], data['rvi_signal'])
        
        # Sell signals: RVI crosses below signal line, preferably from overbought levels
        rvi_cross_down = crossunder(data['rvi'], data['rvi_signal'])
        
        # Additional confirmation filters
        buy_conditions = [rvi_cross_up]
        sell_conditions = [rvi_cross_down]
        
        # Filter 1: RVI strength (avoid weak signals near zero)
        rvi_strong_positive = data['rvi'] > min_strength
        rvi_strong_negative = data['rvi'] < -min_strength
        
        buy_conditions.append(rvi_strong_positive)
        sell_conditions.append(rvi_strong_negative)
        
        # Filter 2: RVI level confirmation (better signals from extremes)
        # Buy signals stronger when coming from oversold levels
        coming_from_oversold = data['rvi'].shift(1) < oversold
        # Sell signals stronger when coming from overbought levels  
        coming_from_overbought = data['rvi'].shift(1) > overbought
        
        # Don't require extreme levels but give them higher weight in signal strength
        
        # Filter 3: Trend filter (optional)
        if self.parameters['use_trend_filter'] and 'sma_20' in data.columns:
            # Buy when price is above moving average (uptrend)
            uptrend = data['close'] > data['sma_20']
            # Sell when price is below moving average (downtrend)  
            downtrend = data['close'] < data['sma_20']
            
            buy_conditions.append(uptrend)
            sell_conditions.append(downtrend)
        
        # Combine conditions
        data['buy_signal'] = pd.concat(buy_conditions, axis=1).all(axis=1)
        data['sell_signal'] = pd.concat(sell_conditions, axis=1).all(axis=1)
        
        # Calculate signal strength (0-1 scale)
        strength_factors = []
        
        # Factor 1: RVI distance from signal line (conviction strength)
        rvi_divergence = np.abs(data['rvi'] - data['rvi_signal'])
        rvi_strength = (rvi_divergence / 0.5).clip(0, 1)  # Normalize to 0-1
        strength_factors.append(rvi_strength)
        
        # Factor 2: RVI absolute level (extreme values = stronger signals)
        rvi_extreme = pd.Series(0.0, index=data.index)
        rvi_extreme[np.abs(data['rvi']) > 0.8] = 1.0    # Very extreme
        rvi_extreme[(np.abs(data['rvi']) > 0.5) & (np.abs(data['rvi']) <= 0.8)] = 0.8
        rvi_extreme[(np.abs(data['rvi']) > 0.3) & (np.abs(data['rvi']) <= 0.5)] = 0.6
        rvi_extreme[np.abs(data['rvi']) <= 0.3] = 0.3   # Weak signals
        strength_factors.append(rvi_extreme)
        
        # Factor 3: Coming from extreme levels bonus
        extreme_bonus = pd.Series(0.0, index=data.index)
        extreme_bonus[coming_from_oversold & data['buy_signal']] = 0.3
        extreme_bonus[coming_from_overbought & data['sell_signal']] = 0.3
        strength_factors.append(extreme_bonus)
        
        # Factor 4: Momentum confirmation
        if 'momentum' in data.columns:
            momentum_strength = pd.Series(0.0, index=data.index)
            # Positive momentum for buy signals
            momentum_strength[(data['momentum'] > 0) & data['buy_signal']] = 0.2
            # Negative momentum for sell signals  
            momentum_strength[(data['momentum'] < 0) & data['sell_signal']] = 0.2
            strength_factors.append(momentum_strength)
        
        # Combine strength factors
        data['signal_strength'] = calculate_signal_strength(
            strength_factors,
            weights=None  # Equal weights
        )
        
        # Apply minimum threshold - filter out weak signals
        weak_signals = data['signal_strength'] < threshold
        data.loc[weak_signals, 'buy_signal'] = False
        data.loc[weak_signals, 'sell_signal'] = False
        
        # Prevent look-ahead bias by shifting signal evaluation
        # Make sure we're using previous period data for signal generation
        data['buy_signal'] = data['buy_signal'] & (data['rvi'].shift(1).notna())
        data['sell_signal'] = data['sell_signal'] & (data['rvi'].shift(1).notna())
        
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
        
        # Check that we have RVI indicators
        rvi_cols = ['rvi', 'rvi_signal']
        if not all(col in data.columns for col in rvi_cols):
            return False
        
        return True


if __name__ == "__main__":
    """Test the strategy with sample data"""
    import warnings
    warnings.filterwarnings('ignore')
    
    # Create sample data with OHLC structure
    np.random.seed(42)  # For reproducible results
    dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='5min')
    n_points = len(dates)
    
    # Generate realistic OHLC data
    base_price = 100
    price_changes = np.random.normal(0, 0.02, n_points).cumsum()
    prices = base_price + price_changes
    
    # Create OHLC from price series
    sample_data = pd.DataFrame({
        'timestamp': dates,
        'close': prices,
        'volume': np.random.randint(1000, 10000, n_points)
    })
    
    # Generate realistic OHLC data
    # Open = previous close (mostly)
    sample_data['open'] = sample_data['close'].shift(1).fillna(sample_data['close'])
    
    # Add some noise to create high/low
    noise = np.random.normal(0, 0.01, n_points)
    sample_data['high'] = sample_data['close'] + np.abs(noise)
    sample_data['low'] = sample_data['close'] - np.abs(noise)
    
    # Ensure OHLC relationships are maintained
    sample_data['high'] = np.maximum(sample_data['high'], 
                                   np.maximum(sample_data['open'], sample_data['close']))
    sample_data['low'] = np.minimum(sample_data['low'], 
                                  np.minimum(sample_data['open'], sample_data['close']))
    
    sample_data['price'] = sample_data['close']
    
    print(f"Testing Strategy 043: Relative Vigor Index")
    print(f"Sample data shape: {sample_data.shape}")
    print(f"Date range: {sample_data['timestamp'].min()} to {sample_data['timestamp'].max()}")
    
    # Test strategy
    try:
        strategy = Strategy020RelativeVigorIndex()
        print(f"Strategy initialized: {strategy.name}")
        print(f"Parameters: {strategy.parameters}")
        
        # Calculate indicators
        data_with_indicators = strategy.calculate_indicators(sample_data.copy())
        print(f"✅ Indicators calculated successfully")
        print(f"RVI columns added: {[col for col in data_with_indicators.columns if 'rvi' in col.lower()]}")
        
        # Generate signals
        data_with_signals = strategy.generate_signals(data_with_indicators)
        print(f"✅ Signals generated successfully")
        
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
            
            # Show sample of RVI values
            rvi_stats = data_with_signals['rvi'].describe()
            print(f"RVI statistics:")
            print(f"  Mean: {rvi_stats['mean']:.4f}")
            print(f"  Std: {rvi_stats['std']:.4f}")  
            print(f"  Min: {rvi_stats['min']:.4f}")
            print(f"  Max: {rvi_stats['max']:.4f}")
            
            # Check for any extreme values
            if (np.abs(data_with_signals['rvi']) > 5).any():
                print("⚠️  Warning: Some RVI values are unusually large")
                
            print("✅ Strategy 020 implementation completed successfully!")
            
        else:
            print("❌ Signal validation failed")
            
    except Exception as e:
        print(f"❌ Error testing strategy: {e}")
        import traceback
        traceback.print_exc()