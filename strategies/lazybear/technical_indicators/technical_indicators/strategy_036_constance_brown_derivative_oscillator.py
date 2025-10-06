#!/usr/bin/env python3
"""
Constance Brown Derivative Oscillator Strategy

Strategy Number: 012
LazyBear Name: Constance Brown Derivative Oscillator
Type: oscillator
TradingView URL: https://www.tradingview.com/v/pUZIbYnu/

Description:
The Derivative Oscillator is a technical indicator that applies a MACD histogram
to a double smoothed RSI to create a more advanced version of the RSI indicator.
It was developed by Constance Brown of Aerodynamic Investments.

Formula:
1. Calculate RSI (14-period by default)
2. First smoothing - Apply EMA to RSI
3. Second smoothing - Apply EMA again  
4. Calculate Signal Line (SMA of double smoothed RSI)
5. Derivative Oscillator = Double Smoothed RSI - Signal Line

Signals: Positive readings are bullish, negative readings are bearish.
Buy when oscillator crosses above zero, sell when it crosses below zero.
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


class Strategy036ConstanceBrownDerivativeOscillator(BaseStrategy):
    """
    Constance Brown Derivative Oscillator Strategy
    
    Strategy Number: 012
    LazyBear Name: Constance Brown Derivative Oscillator
    Type: oscillator
    TradingView URL: https://www.tradingview.com/v/pUZIbYnu/
    
    Description:
    The Derivative Oscillator applies MACD histogram logic to a double-smoothed RSI.
    It provides more refined momentum signals than standard RSI.
    """
    
    def __init__(self, parameters: Dict = None):
        """
        Initialize strategy with parameters
        
        Default parameters match the original TradingView implementation
        """
        # Define default parameters
        default_params = {
            # RSI calculation
            'rsi_period': 14,
            
            # Double smoothing periods for RSI
            'ema1_period': 5,  # First EMA smoothing
            'ema2_period': 3,  # Second EMA smoothing
            
            # Signal line period
            'signal_period': 9,  # SMA period for signal line
            
            # Signal thresholds
            'signal_threshold': 0.6,  # Minimum signal strength to trade
            'zero_line_filter': True,  # Only trade zero line crosses
            
            # Additional filters
            'use_volume_filter': False,
            'use_trend_filter': False,
        }
        
        # Merge with provided parameters
        if parameters:
            default_params.update(parameters)
        
        # Initialize parent class
        super().__init__(
            name="Strategy_036_ConstanceBrownDerivativeOscillator",
            parameters=default_params
        )
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all indicators needed for the strategy
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with additional indicator columns
        """
        # Ensure we have required columns
        if 'close' not in data.columns and 'price' in data.columns:
            data['close'] = data['price']
        
        # Extract parameters for readability
        rsi_period = self.parameters['rsi_period']
        ema1_period = self.parameters['ema1_period']
        ema2_period = self.parameters['ema2_period']
        signal_period = self.parameters['signal_period']
        
        # Step 1: Calculate RSI (14-period)
        data['rsi'] = talib.RSI(data['close'].to_numpy(), timeperiod=rsi_period)
        
        # Step 2: First smoothing - Apply EMA to RSI
        data['rsi_ema1'] = talib.EMA(data['rsi'].to_numpy(), timeperiod=ema1_period)
        
        # Step 3: Second smoothing - Apply EMA again
        data['rsi_ema2'] = talib.EMA(data['rsi_ema1'].to_numpy(), timeperiod=ema2_period)
        
        # Step 4: Calculate Signal Line (SMA of double smoothed RSI)
        data['signal_line'] = talib.SMA(data['rsi_ema2'].to_numpy(), timeperiod=signal_period)
        
        # Step 5: Calculate Derivative Oscillator (double smoothed RSI - signal line)
        data['derivative_osc'] = data['rsi_ema2'] - data['signal_line']
        
        # Additional indicators for filtering and strength calculation
        if self.parameters['use_trend_filter']:
            data['ema_20'] = talib.EMA(data['close'].to_numpy(), timeperiod=20)
        
        if self.parameters['use_volume_filter'] and 'volume' in data.columns:
            data['volume_ma'] = talib.MA(data['volume'].to_numpy(), timeperiod=20)
            data['volume_ratio'] = data['volume'] / data['volume_ma']
        
        # Calculate oscillator momentum for strength
        data['osc_momentum'] = data['derivative_osc'] - data['derivative_osc'].shift(1)
        
        # Store indicators for debugging
        self.indicators = data[['rsi', 'rsi_ema1', 'rsi_ema2', 'signal_line', 'derivative_osc']].copy()
        
        return data
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate buy/sell signals based on strategy logic
        
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
        zero_line_filter = self.parameters['zero_line_filter']
        
        # Helper function for crossover detection
        def crossover(series1, series2):
            """Detect when series1 crosses above series2"""
            return (series1 > series2) & (series1.shift(1) <= series2.shift(1))
        
        def crossunder(series1, series2):
            """Detect when series1 crosses below series2"""
            return (series1 < series2) & (series1.shift(1) >= series2.shift(1))
        
        # Define primary buy/sell conditions based on derivative oscillator
        if zero_line_filter:
            # Primary condition: Zero line crossovers
            zero_line = pd.Series(0.0, index=data.index)
            buy_condition = crossover(data['derivative_osc'], zero_line)
            sell_condition = crossunder(data['derivative_osc'], zero_line)
        else:
            # Alternative: Signal line crossovers
            buy_condition = crossover(data['rsi_ema2'], data['signal_line'])
            sell_condition = crossunder(data['rsi_ema2'], data['signal_line'])
        
        # Apply trend filter if enabled
        if self.parameters['use_trend_filter'] and 'ema_20' in data.columns:
            trend_up = data['close'] > data['ema_20']
            trend_down = data['close'] < data['ema_20']
            
            buy_condition = buy_condition & trend_up
            sell_condition = sell_condition & trend_down
        
        # Apply volume filter if enabled
        if self.parameters['use_volume_filter'] and 'volume_ratio' in data.columns:
            volume_surge = data['volume_ratio'] > 1.2
            
            buy_condition = buy_condition & volume_surge
            sell_condition = sell_condition & volume_surge
        
        # Calculate signal strength based on oscillator characteristics
        # Factor 1: Distance from zero line (normalized)
        osc_abs = np.abs(data['derivative_osc'].fillna(0))
        osc_strength = np.clip(osc_abs / 10, 0, 1)  # Normalize to 0-1
        
        # Factor 2: Oscillator momentum strength
        momentum_strength = np.abs(data['osc_momentum'].fillna(0))
        momentum_strength = np.clip(momentum_strength / 5, 0, 1)
        
        # Factor 3: RSI extremes support
        rsi_strength = pd.Series(0.0, index=data.index)
        rsi_strength[data['rsi'] < 30] = 0.8  # Oversold
        rsi_strength[data['rsi'] > 70] = 0.8  # Overbought
        rsi_strength[data['rsi'] < 20] = 1.0  # Very oversold
        rsi_strength[data['rsi'] > 80] = 1.0  # Very overbought
        
        # Factor 4: Signal line divergence
        divergence = np.abs(data['rsi_ema2'].fillna(0) - data['signal_line'].fillna(0))
        divergence_strength = np.clip(divergence / 20, 0, 1)
        
        # Combine strength factors (weighted average)
        weights = [0.3, 0.3, 0.2, 0.2]
        combined_strength = (
            weights[0] * osc_strength +
            weights[1] * momentum_strength +
            weights[2] * rsi_strength +
            weights[3] * divergence_strength
        )
        data['signal_strength'] = combined_strength
        
        # Apply position constraints - simple state tracking
        position = 0  # 0 = flat, 1 = long
        buy_signals = []
        sell_signals = []
        
        for i in range(len(data)):
            buy_sig = False
            sell_sig = False
            
            # Check conditions and signal strength
            if (buy_condition.iloc[i] and 
                data['signal_strength'].iloc[i] >= threshold and 
                position == 0):  # Only buy when flat
                buy_sig = True
                position = 1
            elif (sell_condition.iloc[i] and 
                  data['signal_strength'].iloc[i] >= threshold and 
                  position == 1):  # Only sell when long
                sell_sig = True
                position = 0
            
            buy_signals.append(buy_sig)
            sell_signals.append(sell_sig)
        
        # Apply with look-ahead bias prevention
        buy_series = pd.Series(buy_signals, dtype=bool, index=data.index).shift(1)
        sell_series = pd.Series(sell_signals, dtype=bool, index=data.index).shift(1)
        data['buy_signal'] = buy_series.fillna(False).astype(bool)
        data['sell_signal'] = sell_series.fillna(False).astype(bool)
        
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
        
        # Check for NaN values in critical columns
        if data[required].isna().any().any():
            return False
        
        # Check for simultaneous buy and sell signals
        simultaneous = (data['buy_signal'] & data['sell_signal']).any()
        if simultaneous:
            return False
        
        # Check signal strength is in valid range
        if (data['signal_strength'] < 0).any() or (data['signal_strength'] > 1).any():
            return False
        
        # Check that derivative oscillator was calculated
        if 'derivative_osc' not in data.columns:
            return False
        
        # Check for reasonable oscillator values (should be bounded)
        if data['derivative_osc'].abs().max() > 100:  # Sanity check
            return False
        
        return True


if __name__ == "__main__":
    # Test the strategy with sample data
    import sys
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    sys.path.insert(0, repo_root)
    
    # Create sample data
    dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='5min')
    np.random.seed(42)  # For reproducible results
    
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
    strategy = Strategy012ConstanceBrownDerivativeOscillator()
    data_with_indicators = strategy.calculate_indicators(sample_data.copy())
    data_with_signals = strategy.generate_signals(data_with_indicators)
    
    # Validate
    if strategy.validate_signals(data_with_signals):
        print("✅ Strategy 012 validation passed")
        print(f"Buy signals: {data_with_signals['buy_signal'].sum()}")
        print(f"Sell signals: {data_with_signals['sell_signal'].sum()}")
        print(f"Average signal strength: {data_with_signals['signal_strength'].mean():.3f}")
        
        # Show sample of derivative oscillator values
        valid_osc = data_with_signals['derivative_osc'].dropna()
        if len(valid_osc) > 0:
            print(f"Derivative Oscillator range: {valid_osc.min():.3f} to {valid_osc.max():.3f}")
    else:
        print("❌ Strategy 012 validation failed")