#!/usr/bin/env python3
"""
Strategy 039: MACD Leader [LazyBear]

This strategy implements the MACD Leader indicator originally created by LazyBear
for TradingView. The MACD Leader uses a "zero-lag" technique to reduce the lag
inherent in traditional MACD calculations, making it more responsive to price changes.

The key innovation is adding a component of the price/MA difference back to the
moving averages, which helps the indicator "lead" the regular MACD, especially
when significant trend changes are about to occur.
"""

import pandas as pd
import numpy as np
import talib
from typing import Dict
import sys
import os

# Add parent directories to path for imports
repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, repo_root)
from strategies.base_strategy import BaseStrategy

# Inline helper functions to avoid import dependency
def crossover(series1: pd.Series, series2) -> pd.Series:
    """Detect when series1 crosses above series2."""
    if isinstance(series2, (int, float)):
        series2 = pd.Series(series2, index=series1.index)
    return (series1 > series2) & (series1.shift(1) <= series2.shift(1))

def crossunder(series1: pd.Series, series2) -> pd.Series:
    """Detect when series1 crosses below series2.""" 
    if isinstance(series2, (int, float)):
        series2 = pd.Series(series2, index=series1.index)
    return (series1 < series2) & (series1.shift(1) >= series2.shift(1))

def pine_ema(series: pd.Series, period: int) -> pd.Series:
    """Calculate EMA using pandas (equivalent to Pine Script's ema function)."""
    return series.ewm(span=period, adjust=False).mean()

def calculate_signal_strength(factors: list, weights: list = None) -> pd.Series:
    """Calculate weighted signal strength from multiple factors."""
    if not factors:
        return pd.Series(0.0, index=factors[0].index if factors else [])
    
    if weights is None:
        weights = [1.0 / len(factors)] * len(factors)
    
    # Normalize weights
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]
    
    # Combine factors
    result = pd.Series(0.0, index=factors[0].index)
    for factor, weight in zip(factors, weights):
        result += factor * weight
    
    return result.clip(0, 1)

def apply_position_constraints(buy_signals: pd.Series, sell_signals: pd.Series, 
                             allow_short: bool = False) -> tuple:
    """Apply position constraints to prevent invalid signal combinations."""
    # Track position state
    position = pd.Series(0, index=buy_signals.index)  # 0=flat, 1=long, -1=short
    
    for i in range(1, len(position)):
        prev_pos = position.iloc[i-1]
        
        if buy_signals.iloc[i] and prev_pos <= 0:
            position.iloc[i] = 1
        elif sell_signals.iloc[i] and prev_pos >= 0:
            position.iloc[i] = -1 if allow_short else 0
        else:
            position.iloc[i] = prev_pos
    
    # Only allow signals that change position
    valid_buys = buy_signals & (position.shift(1).fillna(0) <= 0)
    valid_sells = sell_signals & (position.shift(1).fillna(0) >= 0)
    
    return valid_buys, valid_sells


class Strategy039MacdLeader(BaseStrategy):
    """
    MACD Leader [LazyBear] Strategy Implementation
    
    Strategy Number: 015
    LazyBear Name: MACD Leader
    Type: momentum/trend
    TradingView URL: https://www.tradingview.com/script/y9HCZoQi-MACD-Leader-LazyBear/
    
    Description:
    The MACD Leader uses a zero-lag technique to reduce the traditional MACD's lag.
    It calculates modified EMAs by adding back a component of the price difference,
    making the indicator more responsive to trend changes. Signals are generated
    from crossovers of the MACD Leader line and its signal line.
    """
    
    def __init__(self, parameters: Dict = None):
        """
        Initialize MACD Leader strategy with parameters
        
        Default parameters match the original TradingView implementation
        """
        default_params = {
            'fast_length': 12,          # Short EMA period
            'slow_length': 26,          # Long EMA period  
            'signal_length': 9,         # Signal line EMA period
            'signal_threshold': 0.6,    # Minimum signal strength to trade
            'use_histogram': True,      # Use MACD histogram for signal strength
            'use_zero_line': True,      # Consider zero line crossings
            'histogram_threshold': 0.0001,  # Minimum histogram value for strength
            'show_regular_macd': False, # Option to calculate regular MACD for comparison
        }
        
        if parameters:
            default_params.update(parameters)
        
        super().__init__(
            name="Strategy_039_MacdLeader",
            parameters=default_params
        )
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate MACD Leader indicator and supporting metrics
        
        The MACD Leader calculation follows the LazyBear implementation:
        1. Calculate standard EMAs for fast and slow periods
        2. Apply zero-lag technique by adding price difference components
        3. Calculate MACD Leader as difference of modified EMAs
        4. Generate signal line using EMA of MACD Leader
        """
        # Ensure we have required columns
        if 'close' not in data.columns and 'price' in data.columns:
            data['close'] = data['price']
        
        # Extract parameters
        fast_len = self.parameters['fast_length']
        slow_len = self.parameters['slow_length']
        sig_len = self.parameters['signal_length']
        
        # Calculate standard EMAs
        data['sema'] = pine_ema(data['close'], fast_len)
        data['lema'] = pine_ema(data['close'], slow_len)
        
        # Apply zero-lag technique (LazyBear's innovation)
        # i1 = sema + ma(src - sema, shortLength)
        # i2 = lema + ma(src - lema, longLength)
        price_sema_diff = data['close'] - data['sema']
        price_lema_diff = data['close'] - data['lema']
        
        data['i1'] = data['sema'] + pine_ema(price_sema_diff, fast_len)
        data['i2'] = data['lema'] + pine_ema(price_lema_diff, slow_len)
        
        # Calculate MACD Leader
        data['macd_leader'] = data['i1'] - data['i2']
        
        # Calculate signal line (EMA of MACD Leader)
        data['macd_leader_signal'] = pine_ema(data['macd_leader'], sig_len)
        
        # Calculate MACD Leader histogram
        data['macd_leader_hist'] = data['macd_leader'] - data['macd_leader_signal']
        
        # Optional: Calculate regular MACD for comparison
        if self.parameters['show_regular_macd']:
            data['regular_macd'], data['regular_signal'], data['regular_hist'] = talib.MACD(
                data['close'].values,
                fastperiod=fast_len,
                slowperiod=slow_len,
                signalperiod=sig_len
            )
        
        # Calculate momentum indicators for additional context
        data['macd_leader_momentum'] = data['macd_leader'].diff()
        data['signal_momentum'] = data['macd_leader_signal'].diff()
        
        # Price-based context indicators
        data['price_sma'] = talib.SMA(data['close'].values, timeperiod=20)
        data['price_vs_sma'] = (data['close'] - data['price_sma']) / data['price_sma']
        
        # Store key indicators for debugging
        self.indicators = data[['macd_leader', 'macd_leader_signal', 'macd_leader_hist']].copy()
        
        return data
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate buy/sell signals based on MACD Leader crossovers and momentum
        
        Signal Logic:
        - Buy: MACD Leader crosses above signal line + supporting conditions
        - Sell: MACD Leader crosses below signal line + supporting conditions
        - Strength: Based on histogram magnitude and momentum alignment
        """
        # Initialize signal columns
        data['buy_signal'] = False
        data['sell_signal'] = False
        data['signal_strength'] = 0.0
        
        # Extract parameters
        threshold = self.parameters['signal_threshold']
        use_histogram = self.parameters['use_histogram']
        use_zero_line = self.parameters['use_zero_line']
        hist_threshold = self.parameters['histogram_threshold']
        
        # Primary signal conditions - MACD Leader crossovers
        macd_cross_up = crossover(data['macd_leader'], data['macd_leader_signal'])
        macd_cross_down = crossunder(data['macd_leader'], data['macd_leader_signal'])
        
        # Additional confirmation conditions
        buy_conditions = [macd_cross_up]
        sell_conditions = [macd_cross_down]
        
        # Zero line consideration
        if use_zero_line:
            # Prefer signals in direction of zero line position
            macd_above_zero = data['macd_leader'] > 0
            macd_below_zero = data['macd_leader'] < 0
            
            # Buy signals stronger when MACD Leader is above zero
            buy_zero_align = macd_above_zero | (data['macd_leader'] > data['macd_leader'].shift(1))
            buy_conditions.append(buy_zero_align)
            
            # Sell signals stronger when MACD Leader is below zero  
            sell_zero_align = macd_below_zero | (data['macd_leader'] < data['macd_leader'].shift(1))
            sell_conditions.append(sell_zero_align)
        
        # Momentum confirmation
        momentum_up = data['macd_leader_momentum'] > 0
        momentum_down = data['macd_leader_momentum'] < 0
        buy_conditions.append(momentum_up)
        sell_conditions.append(momentum_down)
        
        # Combine conditions
        data['buy_signal'] = pd.concat(buy_conditions, axis=1).all(axis=1)
        data['sell_signal'] = pd.concat(sell_conditions, axis=1).all(axis=1)
        
        # Calculate signal strength
        strength_factors = []
        
        # Factor 1: Histogram magnitude (momentum strength)
        if use_histogram:
            hist_magnitude = np.abs(data['macd_leader_hist'])
            hist_max = hist_magnitude.rolling(window=50, min_periods=10).max()
            hist_strength = (hist_magnitude / (hist_max + 1e-8)).clip(0, 1)
            strength_factors.append(hist_strength)
        
        # Factor 2: Distance from zero line
        if use_zero_line:
            macd_magnitude = np.abs(data['macd_leader'])
            macd_max = macd_magnitude.rolling(window=50, min_periods=10).max()
            zero_strength = (macd_magnitude / (macd_max + 1e-8)).clip(0, 1)
            strength_factors.append(zero_strength)
        
        # Factor 3: Momentum alignment
        momentum_strength = pd.Series(0.0, index=data.index)
        
        # Strong momentum alignment
        strong_momentum_up = (data['macd_leader_momentum'] > 0) & (data['signal_momentum'] > 0)
        strong_momentum_down = (data['macd_leader_momentum'] < 0) & (data['signal_momentum'] < 0)
        momentum_strength[strong_momentum_up | strong_momentum_down] = 1.0
        
        # Weak momentum alignment
        weak_momentum = (data['macd_leader_momentum'] * data['signal_momentum']) > 0
        momentum_strength[weak_momentum & ~(strong_momentum_up | strong_momentum_down)] = 0.5
        
        strength_factors.append(momentum_strength)
        
        # Factor 4: Distance between MACD Leader and signal
        if len(strength_factors) > 0:
            signal_separation = np.abs(data['macd_leader'] - data['macd_leader_signal'])
            sep_max = signal_separation.rolling(window=50, min_periods=10).max()
            separation_strength = (signal_separation / (sep_max + 1e-8)).clip(0, 1)
            strength_factors.append(separation_strength)
        
        # Combine strength factors
        if strength_factors:
            data['signal_strength'] = calculate_signal_strength(
                strength_factors,
                weights=[0.3, 0.25, 0.25, 0.2] if len(strength_factors) == 4 
                        else [1.0/len(strength_factors)] * len(strength_factors)
            )
        else:
            # Fallback strength calculation
            data['signal_strength'] = 0.5
        
        # Apply minimum threshold
        weak_signals = data['signal_strength'] < threshold
        data.loc[weak_signals, 'buy_signal'] = False
        data.loc[weak_signals, 'sell_signal'] = False
        
        # Prevent look-ahead bias by shifting signals
        data['buy_signal'] = data['buy_signal'].shift(1).fillna(False).astype(bool)
        data['sell_signal'] = data['sell_signal'].shift(1).fillna(False).astype(bool)
        data['signal_strength'] = data['signal_strength'].shift(1).fillna(0.0).astype(float)
        
        # Apply position constraints
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
        """
        required = ['buy_signal', 'sell_signal', 'signal_strength']
        if not all(col in data.columns for col in required):
            return False
        
        # Check for NaN values in recent data (allow some at beginning)
        recent_data = data.tail(max(50, len(data) // 2))
        if recent_data[required].isna().any().any():
            return False
        
        # Check for simultaneous buy and sell signals
        if (data['buy_signal'] & data['sell_signal']).any():
            return False
        
        # Check signal strength range
        valid_strength = (data['signal_strength'] >= 0) & (data['signal_strength'] <= 1)
        if not valid_strength.all():
            return False
        
        return True


if __name__ == "__main__":
    # Test the strategy with sample data
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    
    # Create sample data with realistic price movements
    dates = pd.date_range(start='2024-01-01', end='2024-02-01', freq='5min')
    np.random.seed(42)  # For reproducible results
    
    # Generate trending price data
    trend = np.cumsum(np.random.randn(len(dates)) * 0.001)
    noise = np.random.randn(len(dates)) * 0.01
    base_price = 100
    
    sample_data = pd.DataFrame({
        'timestamp': dates,
        'open': base_price + trend + noise,
        'high': base_price + trend + noise + np.abs(np.random.randn(len(dates)) * 0.02),
        'low': base_price + trend + noise - np.abs(np.random.randn(len(dates)) * 0.02),
        'close': base_price + trend + noise + np.random.randn(len(dates)) * 0.005,
        'volume': np.random.randint(1000, 10000, len(dates))
    })
    
    # Ensure OHLC consistency
    sample_data['high'] = sample_data[['open', 'high', 'low', 'close']].max(axis=1)
    sample_data['low'] = sample_data[['open', 'high', 'low', 'close']].min(axis=1)
    sample_data['price'] = sample_data['close']
    
    print("Testing Strategy 039: MACD Leader...")
    print(f"Sample data shape: {sample_data.shape}")
    print(f"Price range: {sample_data['close'].min():.4f} - {sample_data['close'].max():.4f}")
    
    # Test strategy
    strategy = Strategy015MACDLeader()
    print(f"Strategy parameters: {strategy.parameters}")
    
    # Calculate indicators
    try:
        data_with_indicators = strategy.calculate_indicators(sample_data.copy())
        print(f"✅ Indicators calculated successfully")
        print(f"MACD Leader range: {data_with_indicators['macd_leader'].min():.6f} - {data_with_indicators['macd_leader'].max():.6f}")
        print(f"Non-NaN MACD Leader values: {data_with_indicators['macd_leader'].count()}/{len(data_with_indicators)}")
    except Exception as e:
        print(f"❌ Error calculating indicators: {e}")
        exit(1)
    
    # Generate signals  
    try:
        data_with_signals = strategy.generate_signals(data_with_indicators)
        print(f"✅ Signals generated successfully")
    except Exception as e:
        print(f"❌ Error generating signals: {e}")
        exit(1)
    
    # Validate signals
    if strategy.validate_signals(data_with_signals):
        print("✅ Signal validation passed")
        
        # Report results
        total_signals = data_with_signals['buy_signal'].sum() + data_with_signals['sell_signal'].sum()
        buy_signals = data_with_signals['buy_signal'].sum()
        sell_signals = data_with_signals['sell_signal'].sum()
        avg_strength = data_with_signals['signal_strength'].mean()
        max_strength = data_with_signals['signal_strength'].max()
        
        print(f"📊 Results:")
        print(f"   Buy signals: {buy_signals}")
        print(f"   Sell signals: {sell_signals}")  
        print(f"   Total signals: {total_signals}")
        print(f"   Average signal strength: {avg_strength:.4f}")
        print(f"   Maximum signal strength: {max_strength:.4f}")
        
        # Show some sample indicator values
        print(f"\n📈 Sample MACD Leader values (last 10):")
        for i in range(max(0, len(data_with_signals)-10), len(data_with_signals)):
            row = data_with_signals.iloc[i]
            print(f"   {i}: MACD={row['macd_leader']:.6f}, Signal={row['macd_leader_signal']:.6f}, "
                  f"Hist={row['macd_leader_hist']:.6f}, Strength={row['signal_strength']:.3f}")
        
        print(f"\n🎯 Strategy 039: MACD Leader implementation completed successfully!")
        
    else:
        print("❌ Signal validation failed")
        
        # Debug information
        print("Checking for issues...")
        required = ['buy_signal', 'sell_signal', 'signal_strength']
        for col in required:
            if col not in data_with_signals.columns:
                print(f"Missing column: {col}")
            else:
                nan_count = data_with_signals[col].isna().sum()
                if nan_count > 0:
                    print(f"NaN values in {col}: {nan_count}")
        
        simultaneous = (data_with_signals['buy_signal'] & data_with_signals['sell_signal']).sum()
        if simultaneous > 0:
            print(f"Simultaneous buy/sell signals: {simultaneous}")