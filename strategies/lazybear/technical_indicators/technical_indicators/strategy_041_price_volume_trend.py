#!/usr/bin/env python3
"""
Strategy 041: Price Volume Trend (PVT)

A volume-momentum indicator that combines price and volume to show the strength 
of price movements. Unlike OBV which adds or subtracts full volume based on 
direction, PVT weights the volume by the percentage price change.

Strategy Number: 018
LazyBear Name: Price Volume Trend
Type: volume/momentum
TradingView URL: https://www.tradingview.com/v/2mI0kMJV/ (404 - using standard implementation)

Description:
Implements the standard Price Volume Trend formula as LazyBear-specific variant
was not found. PVT accumulates volume weighted by price change percentage,
providing insight into the conviction behind price movements.

Formula: PVT = [((Close - PrevClose) / PrevClose) × Volume] + PrevPVT
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
from utils.vectorized_helpers import (
    crossover, crossunder, apply_position_constraints,
    calculate_signal_strength
)


class Strategy041PriceVolumeTrend(BaseStrategy):
    """
    Strategy 041: Price Volume Trend (PVT)
    
    Combines price and volume to measure the strength of price movements.
    Uses standard PVT calculation with EMA signal line for trading signals.
    """
    
    def __init__(self, parameters: Dict = None):
        """
        Initialize strategy with parameters
        
        Args:
            parameters: Dictionary of strategy parameters
        """
        # Define default parameters
        default_params = {
            'signal_period': 9,         # EMA period for PVT signal line
            'signal_threshold': 0.6,    # Minimum signal strength to trade
            'volume_filter': True,      # Require volume confirmation
            'volume_threshold': 1.2,    # Volume must be X times average
        }
        
        # Merge with provided parameters
        if parameters:
            default_params.update(parameters)
        
        # Initialize parent class
        super().__init__(
            name="Strategy_041_PriceVolumeTrend",
            parameters=default_params
        )
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Price Volume Trend indicator and signal line
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with PVT indicators
        """
        # Ensure we have required columns
        if 'close' not in data.columns and 'price' in data.columns:
            data['close'] = data['price']
        
        if 'volume' not in data.columns:
            raise ValueError("Volume data is required for PVT calculation")
        
        # Extract parameters
        signal_period = self.parameters['signal_period']
        
        # Calculate price change percentage
        data['price_change_pct'] = data['close'].pct_change()
        
        # Calculate volume-weighted price change
        data['pvt_step'] = data['price_change_pct'] * data['volume']
        
        # Handle NaN values (first row will be NaN due to pct_change)
        data['pvt_step'] = data['pvt_step'].fillna(0)
        
        # Calculate cumulative PVT
        data['pvt'] = data['pvt_step'].cumsum()
        
        # Calculate PVT signal line using EMA
        data['pvt_signal'] = talib.EMA(
            data['pvt'].astype(np.float64).to_numpy(),
            timeperiod=signal_period
        )
        
        # Calculate volume moving average for volume filter
        if self.parameters['volume_filter']:
            data['volume_ma'] = talib.MA(
                data['volume'].astype(np.float64).to_numpy(),
                timeperiod=20
            )
            data['volume_ratio'] = data['volume'] / data['volume_ma']
        
        # Store indicators for debugging
        indicator_cols = ['pvt', 'pvt_signal', 'price_change_pct', 'pvt_step']
        if self.parameters['volume_filter']:
            indicator_cols.extend(['volume_ma', 'volume_ratio'])
            
        self.indicators = data[indicator_cols].copy()
        
        return data
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate buy/sell signals based on PVT crossovers
        
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
        threshold = self.parameters['signal_threshold']
        
        # Primary signals: PVT vs Signal line crossovers
        pvt_cross_up = crossover(data['pvt'], data['pvt_signal'])
        pvt_cross_down = crossunder(data['pvt'], data['pvt_signal'])
        
        # Apply volume filter if enabled
        volume_confirmed = pd.Series([True] * len(data), index=data.index)
        if self.parameters['volume_filter']:
            volume_surge = data['volume_ratio'] > self.parameters['volume_threshold']
            volume_confirmed = volume_surge
        
        # Combine conditions for buy/sell signals
        buy_conditions = pvt_cross_up & volume_confirmed
        sell_conditions = pvt_cross_down & volume_confirmed
        
        # Calculate signal strength based on multiple factors
        strength_factors = []
        
        # Factor 1: Magnitude of PVT vs Signal line divergence
        pvt_divergence = np.abs(data['pvt'] - data['pvt_signal'])
        pvt_strength = (pvt_divergence / data['pvt'].abs()).clip(0, 1)
        strength_factors.append(pvt_strength)
        
        # Factor 2: Price change magnitude
        price_change_strength = np.abs(data['price_change_pct']).clip(0, 0.1) / 0.1
        strength_factors.append(price_change_strength)
        
        # Factor 3: Volume confirmation strength
        if self.parameters['volume_filter']:
            vol_strength = ((data['volume_ratio'] - 1) / 2).clip(0, 1)
            strength_factors.append(vol_strength)
        
        # Combine strength factors
        if strength_factors:
            data['signal_strength'] = calculate_signal_strength(strength_factors)
        
        # Apply signals with strength consideration
        strong_buy = buy_conditions & (data['signal_strength'] >= threshold)
        strong_sell = sell_conditions & (data['signal_strength'] >= threshold)
        
        # Set signal strength with direction (positive for buy, negative for sell)
        data.loc[strong_buy, 'buy_signal'] = True
        data.loc[strong_buy, 'signal_strength'] = data.loc[strong_buy, 'signal_strength']
        
        data.loc[strong_sell, 'sell_signal'] = True
        data.loc[strong_sell, 'signal_strength'] = -data.loc[strong_sell, 'signal_strength']
        
        # Clear signal strength where no signals
        no_signals = ~(data['buy_signal'] | data['sell_signal'])
        data.loc[no_signals, 'signal_strength'] = 0.0
        
        # Apply position constraints
        data['buy_signal'], data['sell_signal'] = apply_position_constraints(
            data['buy_signal'], data['sell_signal'], allow_short=False
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
        
        # Check for NaN values
        if data[required].isna().any().any():
            return False
        
        # Check for simultaneous buy and sell signals
        simultaneous = (data['buy_signal'] & data['sell_signal']).any()
        if simultaneous:
            return False
        
        # Check signal strength is in valid range
        if (data['signal_strength'] < -1).any() or (data['signal_strength'] > 1).any():
            return False
        
        return True


if __name__ == "__main__":
    # Test the strategy with sample data
    
    # Create realistic price and volume data
    np.random.seed(42)  # For reproducible results
    dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='1h')
    n = len(dates)
    
    # Generate trending price data with realistic volume
    base_price = 100
    trend = np.linspace(0, 10, n)
    noise = np.random.randn(n).cumsum() * 0.3
    prices = base_price + trend + noise
    
    # Generate volume correlated with price changes
    price_changes = np.diff(prices, prepend=prices[0])
    base_volume = 10000
    volume_multiplier = 1 + np.abs(price_changes) / np.mean(np.abs(price_changes))
    volumes = base_volume * volume_multiplier * (0.8 + 0.4 * np.random.random(n))
    
    sample_data = pd.DataFrame({
        'timestamp': dates,
        'close': prices,
        'price': prices,  # For compatibility
        'volume': volumes.astype(int),
        'open': np.roll(prices, 1),  # Simple open approximation
        'high': prices * 1.01,       # Simple high approximation  
        'low': prices * 0.99         # Simple low approximation
    })
    
    # Fix first open value
    sample_data.iloc[0, sample_data.columns.get_loc('open')] = base_price
    
    # Test strategy
    print("Testing Strategy 041: Price Volume Trend")
    print("=" * 50)
    
    try:
        strategy = Strategy041PriceVolumeTrend()
        print(f"✅ Strategy initialized: {strategy.name}")
        print(f"Parameters: {strategy.parameters}")
        
        # Calculate indicators
        data_with_indicators = strategy.calculate_indicators(sample_data.copy())
        print(f"✅ Indicators calculated. Shape: {data_with_indicators.shape}")
        
        # Show PVT statistics
        pvt_range = data_with_indicators['pvt'].max() - data_with_indicators['pvt'].min()
        print(f"PVT range: {pvt_range:.2f}")
        
        # Generate signals
        data_with_signals = strategy.generate_signals(data_with_indicators)
        print(f"✅ Signals generated. Shape: {data_with_signals.shape}")
        
        # Validate signals
        if strategy.validate_signals(data_with_signals):
            print("✅ Signal validation passed")
            
            buy_signals = data_with_signals['buy_signal'].sum()
            sell_signals = data_with_signals['sell_signal'].sum()
            avg_strength = data_with_signals['signal_strength'].abs().mean()
            
            print(f"Buy signals: {buy_signals}")
            print(f"Sell signals: {sell_signals}")
            print(f"Average signal strength: {avg_strength:.3f}")
            
            # Show sample signals
            if buy_signals > 0 or sell_signals > 0:
                signal_rows = data_with_signals[
                    data_with_signals['buy_signal'] | data_with_signals['sell_signal']
                ]
                print(f"\nSample signals generated:")
                cols = ['timestamp', 'close', 'volume', 'pvt', 'pvt_signal', 'buy_signal', 'sell_signal', 'signal_strength']
                print(signal_rows[cols].head())
            
            print("\n✅ Strategy 018 implementation completed successfully")
            
        else:
            print("❌ Signal validation failed")
            
    except Exception as e:
        print(f"❌ Strategy test failed: {e}")
        import traceback
        traceback.print_exc()