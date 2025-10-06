#!/usr/bin/env python3
"""
EMA Wave Indicator Strategy Implementation

Strategy Number: 161
LazyBear Name: EMA Wave Indicator
Type: momentum/trend analysis
TradingView URL: https://www.tradingview.com/v/5sofTZ6c/

Description:
EMA Wave Indicator uses multiple Exponential Moving Averages (EMAs) on HLC3 price
to create waves by subtracting EMA from source and applying SMA smoothing.
Creates three waves (A, B, C) at different lengths to visualize relative momentum
across multiple timeframes and detect wave alignment patterns.
"""

import pandas as pd
import numpy as np
import talib
from typing import Dict, Optional
import sys
import os

# Add parent directories to path for imports
repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.insert(0, repo_root)
from strategies.base_strategy import BaseStrategy

# Inline helper functions to avoid import dependency
def crossover(series1: pd.Series, series2) -> pd.Series:
    """Detect when series1 crosses above series2."""
    if isinstance(series2, (int, float)):
        series2 = pd.Series([series2] * len(series1), index=series1.index)
    return (series1.shift(1) <= series2.shift(1)) & (series1 > series2)

def crossunder(series1: pd.Series, series2) -> pd.Series:
    """Detect when series1 crosses below series2."""
    if isinstance(series2, (int, float)):
        series2 = pd.Series([series2] * len(series1), index=series1.index)
    return (series1.shift(1) >= series2.shift(1)) & (series1 < series2)

def apply_position_constraints(buy_signals: pd.Series, sell_signals: pd.Series, allow_short: bool = False) -> tuple:
    """Apply position constraints to prevent invalid signals."""
    position = 0
    filtered_buy = buy_signals.copy()
    filtered_sell = sell_signals.copy()
    
    for i in range(1, len(buy_signals)):
        if buy_signals.iloc[i] and position == 0:
            position = 1
        elif sell_signals.iloc[i] and position == 1:
            position = 0
        else:
            filtered_buy.iloc[i] = False
            filtered_sell.iloc[i] = False
    
    return filtered_buy, filtered_sell

def calculate_signal_strength(conditions: list, weights: list = None) -> pd.Series:
    """Calculate composite signal strength from multiple conditions."""
    if weights is None:
        weights = [1.0 / len(conditions)] * len(conditions)
    
    strength = pd.Series([0.0] * len(conditions[0]), index=conditions[0].index)
    for condition, weight in zip(conditions, weights):
        if condition.dtype == bool:
            condition = condition.astype(float)
        strength += condition * weight
    
    return strength.clip(0, 1)


class Strategy161EmaWaveIndicator(BaseStrategy):
    """
    EMA Wave Indicator Strategy Implementation
    
    Uses multiple EMA waves to detect momentum alignment across different timeframes.
    Calculates waves by subtracting EMA from HLC3 price and applying SMA smoothing.
    """
    
    def __init__(self, parameters: Dict = None):
        """
        Initialize EMA Wave Indicator strategy with parameters
        """
        # Define default parameters based on TradingView script
        default_params = {
            # Wave periods
            'wave_a_length': 5,    # Fast wave period
            'wave_b_length': 25,   # Medium wave period
            'wave_c_length': 50,   # Slow wave period
            
            # SMA smoothing period for waves
            'wave_sma_length': 4,
            
            # Signal generation parameters
            'cutoff_threshold': 10,      # Spike/exhaustion detection threshold
            'alignment_threshold': 2,    # Minimum waves aligned for signal
            'signal_threshold': 0.6,     # Minimum signal strength to trade
            'use_spike_filter': True,    # Filter out spike conditions
            'use_momentum_filter': True, # Require momentum confirmation
        }
        
        # Merge with provided parameters
        if parameters:
            default_params.update(parameters)
        
        # Initialize parent class
        super().__init__(
            name="Strategy_161_EmaWaveIndicator",
            parameters=default_params
        )
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate EMA Wave indicators
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with EMA Wave indicators
        """
        # Ensure we have required columns
        if 'close' not in data.columns and 'price' in data.columns:
            data['close'] = data['price']
        
        # Fill missing OHLC columns with close price if needed
        for col in ['open', 'high', 'low']:
            if col not in data.columns:
                data[col] = data['close']
        
        # Calculate HLC3 (typical price)
        data['hlc3'] = (data['high'] + data['low'] + data['close']) / 3
        
        # Extract parameters
        wave_a_len = self.parameters['wave_a_length']
        wave_b_len = self.parameters['wave_b_length']
        wave_c_len = self.parameters['wave_c_length']
        sma_len = self.parameters['wave_sma_length']
        cutoff = self.parameters['cutoff_threshold']
        
        # Calculate EMAs for each wave
        data['ema_a'] = talib.EMA(data['hlc3'].to_numpy().astype(np.float64), timeperiod=wave_a_len)
        data['ema_b'] = talib.EMA(data['hlc3'].to_numpy().astype(np.float64), timeperiod=wave_b_len)
        data['ema_c'] = talib.EMA(data['hlc3'].to_numpy().astype(np.float64), timeperiod=wave_c_len)
        
        # Calculate wave differences (source - EMA)
        data['wave_a_diff'] = data['hlc3'] - data['ema_a']
        data['wave_b_diff'] = data['hlc3'] - data['ema_b']
        data['wave_c_diff'] = data['hlc3'] - data['ema_c']
        
        # Apply SMA smoothing to wave differences
        data['wave_a'] = talib.SMA(data['wave_a_diff'].to_numpy().astype(np.float64), timeperiod=sma_len)
        data['wave_b'] = talib.SMA(data['wave_b_diff'].to_numpy().astype(np.float64), timeperiod=sma_len)
        data['wave_c'] = talib.SMA(data['wave_c_diff'].to_numpy().astype(np.float64), timeperiod=sma_len)
        
        # Calculate wave directions (positive/negative)
        data['wave_a_direction'] = np.where(data['wave_a'] > 0, 1, -1)
        data['wave_b_direction'] = np.where(data['wave_b'] > 0, 1, -1)
        data['wave_c_direction'] = np.where(data['wave_c'] > 0, 1, -1)
        
        # Count waves pointing up/down
        direction_matrix = data[['wave_a_direction', 'wave_b_direction', 'wave_c_direction']]
        data['waves_up'] = (direction_matrix == 1).sum(axis=1)
        data['waves_down'] = (direction_matrix == -1).sum(axis=1)
        
        # Calculate wave alignment strength
        total_waves = 3
        data['alignment_strength'] = np.maximum(
            data['waves_up'] / total_waves,
            data['waves_down'] / total_waves
        )
        
        # Calculate average wave magnitude for momentum
        data['wave_avg_magnitude'] = (np.abs(data['wave_a']) + 
                                    np.abs(data['wave_b']) + 
                                    np.abs(data['wave_c'])) / 3
        
        # Calculate wave momentum (rate of change)
        data['wave_a_momentum'] = data['wave_a'].diff()
        data['wave_b_momentum'] = data['wave_b'].diff()
        data['wave_c_momentum'] = data['wave_c'].diff()
        
        # Calculate composite wave momentum
        data['wave_momentum'] = (data['wave_a_momentum'] + 
                               data['wave_b_momentum'] + 
                               data['wave_c_momentum']) / 3
        
        # Spike/exhaustion detection
        # Calculate wave ratios relative to average magnitude
        data['wave_spike_ratio'] = data['wave_avg_magnitude'] / data['wave_avg_magnitude'].rolling(20).mean()
        data['is_spike'] = data['wave_spike_ratio'] > (cutoff / 100.0 + 1.0)
        
        # Additional momentum indicators for confirmation
        data['rsi'] = talib.RSI(data['close'].to_numpy().astype(np.float64), timeperiod=14)
        
        # Volume analysis if available
        if 'volume' in data.columns:
            data['volume_ma'] = talib.MA(data['volume'].to_numpy().astype(np.float64), timeperiod=20)
            data['volume_ratio'] = data['volume'] / data['volume_ma']
        
        # Store key indicators for debugging
        key_indicators = ['wave_a', 'wave_b', 'wave_c', 'alignment_strength', 
                         'wave_momentum', 'wave_avg_magnitude', 'is_spike']
        self.indicators = data[key_indicators].copy()
        
        return data
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate buy/sell signals based on EMA Wave logic
        
        Args:
            data: DataFrame with EMA Wave indicators
            
        Returns:
            DataFrame with buy_signal, sell_signal, and signal_strength columns
        """
        # Initialize signal columns
        data['buy_signal'] = False
        data['sell_signal'] = False
        data['signal_strength'] = 0.0
        
        # Extract parameters
        alignment_thresh = self.parameters['alignment_threshold']
        threshold = self.parameters['signal_threshold']
        use_spike = self.parameters['use_spike_filter']
        use_momentum = self.parameters['use_momentum_filter']
        
        # Primary buy conditions
        buy_conditions = []
        
        # Condition 1: Strong wave alignment (bullish)
        strong_alignment_up = data['waves_up'] >= alignment_thresh
        buy_conditions.append(strong_alignment_up)
        
        # Condition 2: Waves trending positive
        waves_positive = (data['wave_a'] > 0) & (data['wave_b'] > 0)
        buy_conditions.append(waves_positive)
        
        # Condition 3: Wave momentum is positive
        if use_momentum:
            momentum_positive = data['wave_momentum'] > 0
            buy_conditions.append(momentum_positive)
        
        # Condition 4: Wave A crossover above zero (fast signal)
        wave_a_cross_up = crossover(data['wave_a'], 0)
        buy_conditions.append(wave_a_cross_up)
        
        # Condition 5: RSI not overbought
        rsi_not_overbought = data['rsi'] < 70
        buy_conditions.append(rsi_not_overbought)
        
        # Primary sell conditions
        sell_conditions = []
        
        # Condition 1: Strong wave alignment (bearish)
        strong_alignment_down = data['waves_down'] >= alignment_thresh
        sell_conditions.append(strong_alignment_down)
        
        # Condition 2: Waves trending negative
        waves_negative = (data['wave_a'] < 0) & (data['wave_b'] < 0)
        sell_conditions.append(waves_negative)
        
        # Condition 3: Wave momentum is negative
        if use_momentum:
            momentum_negative = data['wave_momentum'] < 0
            sell_conditions.append(momentum_negative)
        
        # Condition 4: Wave A crossunder below zero (fast signal)
        wave_a_cross_down = crossunder(data['wave_a'], 0)
        sell_conditions.append(wave_a_cross_down)
        
        # Condition 5: RSI not oversold
        rsi_not_oversold = data['rsi'] > 30
        sell_conditions.append(rsi_not_oversold)
        
        # Apply spike filter if enabled
        if use_spike:
            no_spike = ~data['is_spike']
            buy_conditions.append(no_spike)
            sell_conditions.append(no_spike)
        
        # Volume filter if available
        if 'volume_ratio' in data.columns:
            volume_confirmation = data['volume_ratio'] > 1.0
            buy_conditions.append(volume_confirmation)
            sell_conditions.append(volume_confirmation)
        
        # Combine conditions using shift(1) to prevent look-ahead bias
        shifted_buy_conditions = [cond.shift(1).fillna(False).astype(bool) for cond in buy_conditions]
        shifted_sell_conditions = [cond.shift(1).fillna(False).astype(bool) for cond in sell_conditions]
        
        # Generate preliminary signals
        data['buy_signal'] = pd.concat(shifted_buy_conditions, axis=1).all(axis=1)
        data['sell_signal'] = pd.concat(shifted_sell_conditions, axis=1).all(axis=1)
        
        # Calculate signal strength based on multiple factors
        strength_factors = []
        
        # Factor 1: Wave alignment strength
        alignment_strength_shifted = data['alignment_strength'].shift(1).fillna(0)
        strength_factors.append(alignment_strength_shifted)
        
        # Factor 2: Wave momentum strength
        momentum_abs = np.abs(data['wave_momentum'].shift(1)).fillna(0)
        momentum_normalized = (momentum_abs / (momentum_abs.rolling(20).std() + 1e-8)).clip(0, 1)
        strength_factors.append(momentum_normalized)
        
        # Factor 3: Wave magnitude (relative strength)
        magnitude_strength = data['wave_avg_magnitude'].shift(1).fillna(0)
        magnitude_normalized = (magnitude_strength / (magnitude_strength.rolling(20).mean() + 1e-8)).clip(0, 1)
        strength_factors.append(magnitude_normalized)
        
        # Factor 4: RSI momentum
        rsi_shifted = data['rsi'].shift(1).fillna(50)
        rsi_strength = pd.Series(0.0, index=data.index)
        rsi_strength[rsi_shifted < 30] = (30 - rsi_shifted[rsi_shifted < 30]) / 30  # Oversold strength
        rsi_strength[rsi_shifted > 70] = (rsi_shifted[rsi_shifted > 70] - 70) / 30  # Overbought strength
        rsi_strength = rsi_strength.clip(0, 1)
        strength_factors.append(rsi_strength)
        
        # Combine strength factors
        data['signal_strength'] = calculate_signal_strength(
            strength_factors,
            weights=[0.4, 0.3, 0.2, 0.1]  # Prioritize alignment and momentum
        )
        
        # Fill any NaN values in signal strength
        data['signal_strength'] = data['signal_strength'].fillna(0.0)
        
        # Apply minimum threshold
        weak_signals = data['signal_strength'] < threshold
        data.loc[weak_signals, 'buy_signal'] = False
        data.loc[weak_signals, 'sell_signal'] = False
        
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
        Validate that EMA Wave signals are properly formed
        
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
        if (data['signal_strength'] < 0).any() or (data['signal_strength'] > 1).any():
            return False
        
        # Check that we have wave indicators
        wave_indicators = ['wave_a', 'wave_b', 'wave_c', 'alignment_strength']
        if not all(col in data.columns for col in wave_indicators):
            return False
        
        return True


if __name__ == "__main__":
    # Test the EMA Wave Indicator strategy with sample data
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Create sample data with realistic price movement
    np.random.seed(42)  # For reproducible results
    dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='5min')
    n_periods = len(dates)
    
    # Generate realistic OHLCV data
    base_price = 100
    price_changes = np.random.normal(0, 0.002, n_periods).cumsum()
    close_prices = base_price * (1 + price_changes)
    
    # Generate OHLC from close prices
    highs = close_prices * (1 + np.abs(np.random.normal(0, 0.001, n_periods)))
    lows = close_prices * (1 - np.abs(np.random.normal(0, 0.001, n_periods)))
    opens = np.roll(close_prices, 1)
    opens[0] = close_prices[0]
    
    sample_data = pd.DataFrame({
        'timestamp': dates,
        'open': opens,
        'high': highs,
        'low': lows,
        'close': close_prices,
        'volume': np.random.randint(1000, 10000, n_periods)
    })
    sample_data['price'] = sample_data['close']
    
    # Test strategy
    print("Testing EMA Wave Indicator Strategy...")
    strategy = Strategy161EmaWaveIndicator()
    
    try:
        # Calculate indicators
        data_with_indicators = strategy.calculate_indicators(sample_data.copy())
        print(f"✅ Indicators calculated. Shape: {data_with_indicators.shape}")
        
        # Generate signals
        data_with_signals = strategy.generate_signals(data_with_indicators)
        print(f"✅ Signals generated. Shape: {data_with_signals.shape}")
        
        # Validate signals
        if strategy.validate_signals(data_with_signals):
            print("✅ EMA Wave validation passed")
            print(f"Buy signals: {data_with_signals['buy_signal'].sum()}")
            print(f"Sell signals: {data_with_signals['sell_signal'].sum()}")
            print(f"Average signal strength: {data_with_signals['signal_strength'].mean():.3f}")
            print(f"Max alignment strength: {data_with_signals['alignment_strength'].max():.3f}")
            
            # Show some key statistics
            print("\n📊 Key Statistics:")
            print(f"Wave A range: [{data_with_signals['wave_a'].min():.4f}, {data_with_signals['wave_a'].max():.4f}]")
            print(f"Wave B range: [{data_with_signals['wave_b'].min():.4f}, {data_with_signals['wave_b'].max():.4f}]")
            print(f"Wave C range: [{data_with_signals['wave_c'].min():.4f}, {data_with_signals['wave_c'].max():.4f}]")
            print(f"Max waves pointing up: {data_with_signals['waves_up'].max()}")
            print(f"Max waves pointing down: {data_with_signals['waves_down'].max()}")
            print(f"Average alignment strength: {data_with_signals['alignment_strength'].mean():.3f}")
            
        else:
            print("❌ EMA Wave validation failed")
            
    except Exception as e:
        print(f"❌ Error testing EMA Wave strategy: {e}")
        import traceback
        traceback.print_exc()