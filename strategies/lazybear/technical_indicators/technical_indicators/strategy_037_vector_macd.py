#!/usr/bin/env python3
"""
Vector MACD Strategy Implementation

Strategy Number: 013
LazyBear Name: Vector MACD
Type: momentum/trend hybrid
TradingView URL: https://www.tradingview.com/script/UL8mHBNf-Vector-MACD/

Description:
Vector MACD uses multiple Hull Moving Averages as vectors compared to an ALMA origin line.
It calculates six vector moving averages using Fibonacci-based periods and generates signals
when all vectors align in the same direction, indicating strong momentum.
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
    # Simple implementation - only buy when not already long, only sell when long
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


def hull_ma(series: pd.Series, period: int) -> pd.Series:
    """
    Calculate Hull Moving Average (HMA)
    
    Hull MA formula:
    HMA = WMA(2 * WMA(price, period/2) - WMA(price, period), sqrt(period))
    
    Args:
        series: Input price series
        period: Moving average period
        
    Returns:
        Hull Moving Average series
    """
    # Calculate WMA for half period
    wma_half = talib.WMA(series.to_numpy().astype(np.float64), timeperiod=int(period/2))
    
    # Calculate WMA for full period
    wma_full = talib.WMA(series.to_numpy().astype(np.float64), timeperiod=period)
    
    # Calculate raw hull value
    raw_hull = 2 * wma_half - wma_full
    
    # Final smoothing with sqrt(period)
    sqrt_period = int(np.sqrt(period))
    if sqrt_period < 2:
        sqrt_period = 2
        
    hull_ma_values = talib.WMA(raw_hull.astype(np.float64), timeperiod=sqrt_period)
    
    return pd.Series(hull_ma_values, index=series.index)


def alma(series: pd.Series, period: int = 21, offset: float = 0.85, sigma: float = 6.0) -> pd.Series:
    """
    Calculate Arnaud Legoux Moving Average (ALMA)
    
    Args:
        series: Input price series
        period: ALMA period
        offset: Phase/offset parameter (0-1, default 0.85)
        sigma: Smoothing parameter (default 6.0)
        
    Returns:
        ALMA series
    """
    # ALMA calculation
    m = offset * (period - 1)  # Midpoint
    s = period / sigma  # Smoothing factor
    
    # Calculate weights
    weights = np.exp(-0.5 * ((np.arange(period) - m) / s) ** 2)
    weights = weights / weights.sum()
    
    # Apply weights using convolution
    alma_values = np.convolve(series.to_numpy(), weights[::-1], mode='same')
    
    # Handle edge effects
    for i in range(period):
        if i < len(alma_values):
            # Use available data for edge cases
            available_weights = weights[period-i-1:]
            if len(available_weights) > 0:
                available_weights = available_weights / available_weights.sum()
                if i < len(series):
                    alma_values[i] = np.sum(series.iloc[:i+1].to_numpy()[::-1] * available_weights[:i+1])
    
    return pd.Series(alma_values, index=series.index)


class Strategy037VectorMacd(BaseStrategy):
    """
    Vector MACD Strategy Implementation
    
    Uses multiple Hull Moving Average vectors compared to an ALMA origin line
    to detect momentum and trend alignment.
    """
    
    def __init__(self, parameters: Dict = None):
        """
        Initialize Vector MACD strategy with parameters
        """
        # Define default parameters based on Fibonacci sequence
        default_params = {
            # Vector periods (Fibonacci-based)
            'vector_periods': [8, 13, 21, 34, 55, 89],
            
            # ALMA parameters for origin line
            'alma_period': 21,
            'alma_offset': 0.85,
            'alma_sigma': 6.0,
            
            # Signal generation parameters
            'alignment_threshold': 4,  # Minimum vectors pointing same direction
            'signal_threshold': 0.6,   # Minimum signal strength to trade
            'use_divergence_filter': True,
            'use_volume_filter': False,
        }
        
        # Merge with provided parameters
        if parameters:
            default_params.update(parameters)
        
        # Initialize parent class
        super().__init__(
            name="Strategy_037_VectorMacd",
            parameters=default_params
        )
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Vector MACD indicators
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with Vector MACD indicators
        """
        # Ensure we have required columns
        if 'close' not in data.columns and 'price' in data.columns:
            data['close'] = data['price']
        
        # Calculate high-low average (hl2) for ALMA origin
        data['hl2'] = (data['high'] + data['low']) / 2
        
        # Extract parameters
        vector_periods = self.parameters['vector_periods']
        alma_period = self.parameters['alma_period']
        alma_offset = self.parameters['alma_offset']
        alma_sigma = self.parameters['alma_sigma']
        
        # Calculate ALMA origin line
        data['alma_origin'] = alma(
            data['hl2'], 
            period=alma_period,
            offset=alma_offset, 
            sigma=alma_sigma
        )
        
        # Calculate Hull MA vectors
        vector_columns = []
        for i, period in enumerate(vector_periods):
            col_name = f'hma_vector_{i+1}'
            data[col_name] = hull_ma(data['close'], period)
            vector_columns.append(col_name)
        
        # Calculate vector differences from origin
        vector_diff_columns = []
        for i, col_name in enumerate(vector_columns):
            diff_col = f'vector_diff_{i+1}'
            data[diff_col] = data[col_name] - data['alma_origin']
            vector_diff_columns.append(diff_col)
        
        # Calculate vector directions (positive/negative)
        vector_direction_columns = []
        for diff_col in vector_diff_columns:
            dir_col = diff_col.replace('diff', 'direction')
            data[dir_col] = np.where(data[diff_col] > 0, 1, -1)
            vector_direction_columns.append(dir_col)
        
        # Count vectors pointing up/down
        direction_matrix = data[vector_direction_columns]
        data['vectors_up'] = (direction_matrix == 1).sum(axis=1)
        data['vectors_down'] = (direction_matrix == -1).sum(axis=1)
        
        # Calculate vector alignment strength
        total_vectors = len(vector_periods)
        data['alignment_strength'] = np.maximum(
            data['vectors_up'] / total_vectors,
            data['vectors_down'] / total_vectors
        )
        
        # Calculate average vector difference for momentum
        data['avg_vector_diff'] = data[vector_diff_columns].mean(axis=1)
        
        # Calculate vector MACD line (average of all vector differences)
        data['vector_macd'] = data['avg_vector_diff']
        
        # Calculate signal line (EMA of vector MACD)
        data['vector_signal'] = talib.EMA(data['vector_macd'].to_numpy().astype(np.float64), timeperiod=9)
        
        # Calculate histogram
        data['vector_histogram'] = data['vector_macd'] - data['vector_signal']
        
        # Additional momentum indicators for confirmation
        data['rsi'] = talib.RSI(data['close'].to_numpy().astype(np.float64), timeperiod=14)
        
        # Volume analysis if available
        if 'volume' in data.columns:
            data['volume_ma'] = talib.MA(data['volume'].to_numpy().astype(np.float64), timeperiod=20)
            data['volume_ratio'] = data['volume'] / data['volume_ma']
        
        # Store key indicators for debugging
        key_indicators = ['vector_macd', 'vector_signal', 'vector_histogram', 
                         'alignment_strength', 'vectors_up', 'vectors_down']
        self.indicators = data[key_indicators].copy()
        
        return data
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate buy/sell signals based on Vector MACD logic
        
        Args:
            data: DataFrame with Vector MACD indicators
            
        Returns:
            DataFrame with buy_signal, sell_signal, and signal_strength columns
        """
        # Initialize signal columns
        data['buy_signal'] = False
        data['sell_signal'] = False
        data['signal_strength'] = 0.0
        
        # Extract parameters
        alignment_threshold = self.parameters['alignment_threshold']
        threshold = self.parameters['signal_threshold']
        use_divergence = self.parameters['use_divergence_filter']
        use_volume = self.parameters['use_volume_filter']
        
        # Primary buy conditions
        buy_conditions = []
        
        # Condition 1: Vector MACD crossover above signal line
        macd_cross_up = crossover(data['vector_macd'], data['vector_signal'])
        buy_conditions.append(macd_cross_up)
        
        # Condition 2: Strong vector alignment (bullish)
        strong_alignment_up = data['vectors_up'] >= alignment_threshold
        buy_conditions.append(strong_alignment_up)
        
        # Condition 3: Vector histogram trending positive
        hist_positive = data['vector_histogram'] > 0
        buy_conditions.append(hist_positive)
        
        # Condition 4: RSI not overbought (momentum filter)
        rsi_not_overbought = data['rsi'] < 70
        buy_conditions.append(rsi_not_overbought)
        
        # Primary sell conditions
        sell_conditions = []
        
        # Condition 1: Vector MACD crossover below signal line
        macd_cross_down = crossunder(data['vector_macd'], data['vector_signal'])
        sell_conditions.append(macd_cross_down)
        
        # Condition 2: Strong vector alignment (bearish)
        strong_alignment_down = data['vectors_down'] >= alignment_threshold
        sell_conditions.append(strong_alignment_down)
        
        # Condition 3: Vector histogram trending negative
        hist_negative = data['vector_histogram'] < 0
        sell_conditions.append(hist_negative)
        
        # Condition 4: RSI not oversold (momentum filter)
        rsi_not_oversold = data['rsi'] > 30
        sell_conditions.append(rsi_not_oversold)
        
        # Volume filter if enabled
        if use_volume and 'volume_ratio' in data.columns:
            volume_confirmation = data['volume_ratio'] > 1.2
            buy_conditions.append(volume_confirmation)
            sell_conditions.append(volume_confirmation)
        
        # Combine conditions using shift(1) to prevent look-ahead bias
        shifted_buy_conditions = [cond.shift(1).fillna(False) for cond in buy_conditions]
        shifted_sell_conditions = [cond.shift(1).fillna(False) for cond in sell_conditions]
        
        # Generate preliminary signals
        data['buy_signal'] = pd.concat(shifted_buy_conditions, axis=1).all(axis=1)
        data['sell_signal'] = pd.concat(shifted_sell_conditions, axis=1).all(axis=1)
        
        # Calculate signal strength based on multiple factors
        strength_factors = []
        
        # Factor 1: Vector alignment strength
        alignment_strength_shifted = data['alignment_strength'].shift(1).fillna(0)
        strength_factors.append(alignment_strength_shifted)
        
        # Factor 2: Vector MACD momentum
        macd_momentum = np.abs(data['vector_macd'].shift(1)) / data['close'].shift(1) * 100
        macd_momentum_normalized = macd_momentum.clip(0, 1)
        strength_factors.append(macd_momentum_normalized.fillna(0))
        
        # Factor 3: Histogram strength
        hist_strength = np.abs(data['vector_histogram'].shift(1)) / data['close'].shift(1) * 100
        hist_strength_normalized = hist_strength.clip(0, 1)
        strength_factors.append(hist_strength_normalized.fillna(0))
        
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
            weights=[0.4, 0.3, 0.2, 0.1]  # Prioritize alignment and MACD
        )
        
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
        Validate that Vector MACD signals are properly formed
        
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
        
        # Check that we have vector indicators
        vector_indicators = ['vector_macd', 'vector_signal', 'alignment_strength']
        if not all(col in data.columns for col in vector_indicators):
            return False
        
        return True


if __name__ == "__main__":
    # Test the Vector MACD strategy with sample data
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
    print("Testing Vector MACD Strategy...")
    strategy = Strategy013VectorMACD()
    
    try:
        # Calculate indicators
        data_with_indicators = strategy.calculate_indicators(sample_data.copy())
        print(f"✅ Indicators calculated. Shape: {data_with_indicators.shape}")
        
        # Generate signals
        data_with_signals = strategy.generate_signals(data_with_indicators)
        print(f"✅ Signals generated. Shape: {data_with_signals.shape}")
        
        # Validate signals
        if strategy.validate_signals(data_with_signals):
            print("✅ Vector MACD validation passed")
            print(f"Buy signals: {data_with_signals['buy_signal'].sum()}")
            print(f"Sell signals: {data_with_signals['sell_signal'].sum()}")
            print(f"Average signal strength: {data_with_signals['signal_strength'].mean():.3f}")
            print(f"Max alignment strength: {data_with_signals['alignment_strength'].max():.3f}")
            
            # Show some key statistics
            print("\n📊 Key Statistics:")
            print(f"Vector MACD range: [{data_with_signals['vector_macd'].min():.4f}, {data_with_signals['vector_macd'].max():.4f}]")
            print(f"Max vectors pointing up: {data_with_signals['vectors_up'].max()}")
            print(f"Max vectors pointing down: {data_with_signals['vectors_down'].max()}")
            print(f"Average alignment strength: {data_with_signals['alignment_strength'].mean():.3f}")
            
        else:
            print("❌ Vector MACD validation failed")
            
    except Exception as e:
        print(f"❌ Error testing Vector MACD strategy: {e}")
        import traceback
        traceback.print_exc()