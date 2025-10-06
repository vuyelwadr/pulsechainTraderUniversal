#!/usr/bin/env python3
"""
On Balance Volume Oscillator Strategy

Strategy Number: 019
LazyBear Name: On Balance Volume Oscillator
Type: volume/oscillator
TradingView URL: https://www.tradingview.com/v/lxlg3B68/

Description:
The On Balance Volume (OBV) Oscillator transforms the traditional OBV indicator
into an oscillator format by calculating the difference between OBV and its
exponential moving average. This makes it easier to identify divergences and
momentum shifts in volume flow.

OBV measures buying and selling pressure by adding volume on up days and
subtracting volume on down days. The oscillator version highlights when
volume momentum deviates from its trend, providing clearer signals for
trend strength and potential reversals.
"""

import pandas as pd
import numpy as np
import talib
from typing import Dict, Optional
import sys
import os

# Add parent directories to path for imports
repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, repo_root)
from strategies.base_strategy import BaseStrategy
from utils.vectorized_helpers import (
    crossover, crossunder, highest, lowest, 
    barssince, track_position_state, apply_position_constraints,
    calculate_signal_strength, pine_ema, pine_rma
)


class Strategy042OnBalanceVolumeOscillator(BaseStrategy):
    """
    On Balance Volume Oscillator Strategy
    
    Strategy Number: 019
    LazyBear Name: On Balance Volume Oscillator
    Type: volume/oscillator
    TradingView URL: https://www.tradingview.com/v/lxlg3B68/
    
    Description:
    An oscillator that transforms OBV into a momentum indicator by comparing
    it to its exponential moving average. Provides clearer signals for volume-based
    trend analysis and divergence detection.
    """
    
    def __init__(self, parameters: Dict = None):
        """
        Initialize On Balance Volume Oscillator Strategy
        
        Default parameters based on LazyBear's implementation
        """
        # Define default parameters
        default_params = {
            'obv_ma_period': 20,          # EMA period for OBV smoothing
            'signal_threshold': 0.6,      # Minimum signal strength to trade
            'use_zero_cross': True,       # Use zero-line crossovers for signals
            'use_volume_filter': False,   # Already volume-based, no additional filter needed
            'use_trend_filter': True,     # Price trend confirmation
            'trend_ma_period': 50,        # MA period for trend filter
        }
        
        # Merge with provided parameters
        if parameters:
            default_params.update(parameters)
        
        # Initialize parent class
        super().__init__(
            name="Strategy_042_OnBalanceVolumeOscillator",
            parameters=default_params
        )
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate On Balance Volume Oscillator and related indicators
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with additional indicator columns
        """
        # Ensure we have required columns
        if 'close' not in data.columns and 'price' in data.columns:
            data['close'] = data['price']
        if 'open' not in data.columns:
            data['open'] = data['close'].shift(1).fillna(data['close'])
        if 'high' not in data.columns:
            data['high'] = data['close']
        if 'low' not in data.columns:
            data['low'] = data['close']
        
        # Handle volume data - if missing, use default values
        if 'volume' not in data.columns:
            data['volume'] = 1.0  # Default volume for price-only data
        
        # Fill any NaN values in volume
        data['volume'] = data['volume'].fillna(1.0)
        
        # Extract parameters
        obv_ma_period = self.parameters['obv_ma_period']
        trend_ma_period = self.parameters['trend_ma_period']
        
        # Calculate On Balance Volume using TA-Lib
        # OBV adds volume when close > close.shift(1), subtracts when close < close.shift(1)
        # Ensure data types are correct for TA-Lib (requires float64)
        close_values = data['close'].astype(np.float64).values
        volume_values = data['volume'].astype(np.float64).values
        
        data['obv'] = talib.OBV(close_values, volume_values)
        
        # Calculate exponential moving average of OBV
        obv_values = data['obv'].astype(np.float64).values
        data['obv_ma'] = talib.EMA(obv_values, timeperiod=obv_ma_period)
        
        # Calculate OBV Oscillator (difference between OBV and its EMA)
        data['obv_oscillator'] = data['obv'] - data['obv_ma']
        
        # Calculate trend filter moving average
        if self.parameters['use_trend_filter']:
            data['trend_ma'] = talib.EMA(close_values, timeperiod=trend_ma_period)
        
        # Calculate additional indicators for signal strength
        # Momentum of the oscillator
        data['obv_osc_momentum'] = data['obv_oscillator'] - data['obv_oscillator'].shift(5)
        
        # Normalize oscillator for signal strength calculation
        # Use rolling standard deviation for normalization
        obv_osc_std = data['obv_oscillator'].rolling(window=50, min_periods=10).std()
        data['obv_osc_normalized'] = data['obv_oscillator'] / (obv_osc_std + 1e-8)  # Avoid division by zero
        
        # Store indicators for debugging
        self.indicators = data[['obv', 'obv_ma', 'obv_oscillator', 'obv_osc_normalized']].copy()
        
        return data
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate buy/sell signals based on OBV Oscillator logic
        
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
        use_zero_cross = self.parameters['use_zero_cross']
        use_trend_filter = self.parameters['use_trend_filter']
        
        # Define primary signal conditions using zero-line crossovers
        if use_zero_cross:
            # Buy when OBV oscillator crosses above zero (volume momentum turning bullish)
            obv_cross_up = crossover(data['obv_oscillator'], pd.Series([0] * len(data), index=data.index))
            
            # Sell when OBV oscillator crosses below zero (volume momentum turning bearish)  
            obv_cross_down = crossunder(data['obv_oscillator'], pd.Series([0] * len(data), index=data.index))
        else:
            # Alternative: use momentum changes
            obv_cross_up = (data['obv_osc_momentum'] > 0) & (data['obv_osc_momentum'].shift(1) <= 0)
            obv_cross_down = (data['obv_osc_momentum'] < 0) & (data['obv_osc_momentum'].shift(1) >= 0)
        
        # Apply trend filter if enabled
        if use_trend_filter and 'trend_ma' in data.columns:
            # Only buy signals when price is above trend MA
            trend_up = data['close'] > data['trend_ma']
            # Only sell signals when price is below trend MA  
            trend_down = data['close'] < data['trend_ma']
            
            buy_condition = obv_cross_up & trend_up
            sell_condition = obv_cross_down & trend_down
        else:
            buy_condition = obv_cross_up
            sell_condition = obv_cross_down
        
        # Apply position state management to prevent look-ahead bias
        # Use vectorized approach similar to other strategies
        raw_signal = np.select(
            [buy_condition, sell_condition],
            [1, -1],
            default=0
        )
        
        # Convert to position tracking
        position = pd.Series(raw_signal).replace(0, np.nan).ffill().fillna(0)
        is_flat = (position.shift(1) == 0)
        
        # Generate signals with position constraints
        data['buy_signal'] = buy_condition & (is_flat | (position.shift(1) < 0))
        data['sell_signal'] = sell_condition & (position.shift(1) > 0)
        
        # Calculate signal strength based on multiple factors
        strength_factors = []
        
        # Factor 1: Normalized oscillator magnitude
        if 'obv_osc_normalized' in data.columns:
            osc_strength = np.abs(data['obv_osc_normalized']).clip(0, 1)
            osc_strength = osc_strength.fillna(0.0)
            strength_factors.append(osc_strength)
        
        # Factor 2: Momentum strength
        if 'obv_osc_momentum' in data.columns:
            momentum_strength = np.abs(data['obv_osc_momentum'])
            # Normalize by rolling max to get 0-1 scale
            momentum_max = momentum_strength.rolling(window=50, min_periods=10).max()
            momentum_strength = (momentum_strength / (momentum_max + 1e-8)).clip(0, 1)
            momentum_strength = momentum_strength.fillna(0.0)
            strength_factors.append(momentum_strength)
        
        # Factor 3: Distance from zero for oscillator
        zero_distance = np.abs(data['obv_oscillator'])
        zero_distance_max = zero_distance.rolling(window=50, min_periods=10).max()
        zero_distance_normalized = (zero_distance / (zero_distance_max + 1e-8)).clip(0, 1)
        zero_distance_normalized = zero_distance_normalized.fillna(0.0)
        strength_factors.append(zero_distance_normalized)
        
        # Combine strength factors
        if strength_factors:
            data['signal_strength'] = calculate_signal_strength(strength_factors)
            # Fill any remaining NaN values with 0
            data['signal_strength'] = data['signal_strength'].fillna(0.0)
        
        # Apply signal direction to strength
        data.loc[data['buy_signal'], 'signal_strength'] = data.loc[data['buy_signal'], 'signal_strength']
        data.loc[data['sell_signal'], 'signal_strength'] = -data.loc[data['sell_signal'], 'signal_strength']
        
        # Apply minimum threshold
        weak_signals = np.abs(data['signal_strength']) < threshold
        data.loc[weak_signals, 'buy_signal'] = False
        data.loc[weak_signals, 'sell_signal'] = False
        data.loc[weak_signals, 'signal_strength'] = 0.0
        
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
        
        # Check for NaN values in signal columns
        if data[required].isna().any().any():
            return False
        
        # Check for simultaneous buy and sell signals
        simultaneous = (data['buy_signal'] & data['sell_signal']).any()
        if simultaneous:
            return False
        
        # Check signal strength is in valid range
        if (data['signal_strength'] < -1).any() or (data['signal_strength'] > 1).any():
            return False
        
        # Check that we have the core indicator
        if 'obv_oscillator' not in data.columns:
            return False
        
        return True


if __name__ == "__main__":
    # Test the strategy with sample data
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Create sample data with volume
    dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='5min')
    np.random.seed(42)  # For reproducible testing
    
    sample_data = pd.DataFrame({
        'timestamp': dates,
        'open': np.random.randn(len(dates)).cumsum() + 100,
        'high': np.random.randn(len(dates)).cumsum() + 101,
        'low': np.random.randn(len(dates)).cumsum() + 99,
        'close': np.random.randn(len(dates)).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, len(dates))
    })
    sample_data['price'] = sample_data['close']
    
    # Test strategy
    try:
        strategy = Strategy042OnBalanceVolumeOscillator()
        print(f"Testing {strategy.name}...")
        
        # Calculate indicators
        data_with_indicators = strategy.calculate_indicators(sample_data.copy())
        print("✅ Indicators calculated successfully")
        print(f"OBV range: {data_with_indicators['obv'].min():.2f} to {data_with_indicators['obv'].max():.2f}")
        print(f"OBV Oscillator range: {data_with_indicators['obv_oscillator'].min():.2f} to {data_with_indicators['obv_oscillator'].max():.2f}")
        
        # Generate signals
        data_with_signals = strategy.generate_signals(data_with_indicators)
        print("✅ Signals generated successfully")
        
        # Validate
        if strategy.validate_signals(data_with_signals):
            print("✅ Signal validation passed")
            print(f"Buy signals: {data_with_signals['buy_signal'].sum()}")
            print(f"Sell signals: {data_with_signals['sell_signal'].sum()}")
            print(f"Average signal strength: {data_with_signals['signal_strength'].abs().mean():.3f}")
            
            # Show some sample signals
            signal_data = data_with_signals[data_with_signals['buy_signal'] | data_with_signals['sell_signal']]
            if len(signal_data) > 0:
                print("\nSample signals:")
                for idx, row in signal_data.head(3).iterrows():
                    signal_type = "BUY" if row['buy_signal'] else "SELL"
                    print(f"{signal_type}: OBV Osc={row['obv_oscillator']:.2f}, Strength={row['signal_strength']:.3f}")
        else:
            print("❌ Signal validation failed")
            
    except Exception as e:
        print(f"❌ Strategy test failed: {str(e)}")
        import traceback
        traceback.print_exc()