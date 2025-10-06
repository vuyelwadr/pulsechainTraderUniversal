#!/usr/bin/env python3
"""
Variable Moving Average (VMA) Strategy Implementation

Strategy Number: 156
LazyBear Name: Variable Moving Average
Type: Adaptive trend-following indicator
TradingView URL: https://www.tradingview.com/v/6Ix0E5Yr/

Description:
Variable Moving Average (VMA) by Tushar S. Chande automatically adjusts its smoothing
constant based on market volatility. It calculates directional movement, smooths it,
and uses the volatility index to create a dynamic moving average that adapts to
changing market conditions. More responsive than traditional MAs during volatile periods,
and smoother during consolidating periods.

Key Features:
- Volatility-adjusted smoothing constant
- Adaptive response to market conditions
- Trend direction indication
- Dynamic coloring based on trend state

Mathematical Formula:
1. Calculate positive/negative directional movement
2. Smooth directional movements
3. Calculate volatility index (vI)
4. VMA = (1 - k*vI)*VMA[1] + k*vI*source
   where k = 1/length
"""

import pandas as pd
import numpy as np
import talib

# Suppress pandas future warnings for this strategy
pd.set_option('future.no_silent_downcasting', True)
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


def calculate_vma(price: pd.Series, length: int = 6) -> pd.Series:
    """
    Calculate Variable Moving Average (VMA)
    
    VMA adjusts its smoothing constant based on volatility, making it more
    responsive during volatile periods and smoother during consolidation.
    
    Args:
        price: Input price series
        length: Period for VMA calculation (default 6)
        
    Returns:
        Variable Moving Average series
    """
    # Initialize arrays
    vma = pd.Series(np.nan, index=price.index)
    pos_dm = pd.Series(0.0, index=price.index)
    neg_dm = pd.Series(0.0, index=price.index)
    pos_dm_smoothed = pd.Series(0.0, index=price.index)
    neg_dm_smoothed = pd.Series(0.0, index=price.index)
    
    # Calculate k factor
    k = 1.0 / length
    
    # Calculate directional movements
    for i in range(1, len(price)):
        price_diff = price.iloc[i] - price.iloc[i-1]
        pos_dm.iloc[i] = max(price_diff, 0)
        neg_dm.iloc[i] = max(-price_diff, 0)
    
    # Smooth the directional movements using EMA-like calculation
    alpha = 2.0 / (length + 1)  # EMA smoothing factor
    
    # Initialize first values
    pos_dm_smoothed.iloc[length-1] = pos_dm.iloc[:length].mean()
    neg_dm_smoothed.iloc[length-1] = neg_dm.iloc[:length].mean()
    
    # Calculate smoothed directional movements
    for i in range(length, len(price)):
        pos_dm_smoothed.iloc[i] = alpha * pos_dm.iloc[i] + (1 - alpha) * pos_dm_smoothed.iloc[i-1]
        neg_dm_smoothed.iloc[i] = alpha * neg_dm.iloc[i] + (1 - alpha) * neg_dm_smoothed.iloc[i-1]
    
    # Calculate volatility index (vI)
    volatility_index = pd.Series(0.0, index=price.index)
    
    for i in range(length, len(price)):
        total_dm = pos_dm_smoothed.iloc[i] + neg_dm_smoothed.iloc[i]
        if total_dm > 0:
            volatility_index.iloc[i] = abs(pos_dm_smoothed.iloc[i] - neg_dm_smoothed.iloc[i]) / total_dm
        else:
            volatility_index.iloc[i] = 0.0
    
    # Calculate VMA using the formula: VMA = (1 - k*vI)*VMA[1] + k*vI*source
    vma.iloc[length-1] = price.iloc[:length].mean()  # Initialize with SMA
    
    for i in range(length, len(price)):
        vi = volatility_index.iloc[i]
        smoothing_factor = k * vi
        vma.iloc[i] = (1 - smoothing_factor) * vma.iloc[i-1] + smoothing_factor * price.iloc[i]
    
    return vma


class Strategy156VariableMovingAverage(BaseStrategy):
    """
    Variable Moving Average Strategy Implementation
    
    Uses VMA to identify trend direction and generate trading signals.
    The VMA adapts to volatility, providing more responsive signals during
    volatile periods and smoother signals during consolidation.
    """
    
    def __init__(self, parameters: Dict = None):
        """
        Initialize Variable Moving Average strategy with parameters
        """
        # Define default parameters
        default_params = {
            # VMA parameters
            'vma_length': 6,           # VMA period
            'vma_source': 'close',     # Price source (close, hl2, hlc3, ohlc4)
            
            # Signal generation parameters
            'use_price_cross': True,    # Use price crossing VMA for signals
            'use_slope_filter': True,   # Use VMA slope for trend confirmation
            'slope_threshold': 0.0001,  # Minimum slope to consider trending
            
            # Additional filters
            'use_volatility_filter': True,  # Filter signals based on volatility
            'min_volatility': 0.001,       # Minimum volatility for signals
            'max_volatility': 0.05,        # Maximum volatility for signals
            
            # Signal strength parameters
            'signal_threshold': 0.6,    # Minimum signal strength to trade
            'use_momentum_filter': True, # Use momentum confirmation
            'momentum_period': 3,       # Period for momentum calculation
        }
        
        # Merge with provided parameters
        if parameters:
            default_params.update(parameters)
        
        # Initialize parent class
        super().__init__(
            name="Strategy_156_VariableMovingAverage",
            parameters=default_params
        )
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Variable Moving Average indicators
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with VMA indicators
        """
        # Ensure we have required columns
        if 'close' not in data.columns and 'price' in data.columns:
            data['close'] = data['price']
        
        # Extract parameters
        vma_length = self.parameters['vma_length']
        vma_source = self.parameters['vma_source']
        slope_threshold = self.parameters['slope_threshold']
        momentum_period = self.parameters['momentum_period']
        
        # Calculate price source
        if vma_source == 'close':
            source = data['close']
        elif vma_source == 'hl2':
            source = (data['high'] + data['low']) / 2
        elif vma_source == 'hlc3':
            source = (data['high'] + data['low'] + data['close']) / 3
        elif vma_source == 'ohlc4':
            source = (data['open'] + data['high'] + data['low'] + data['close']) / 4
        else:
            source = data['close']
        
        # Calculate Variable Moving Average
        data['vma'] = calculate_vma(source, length=vma_length)
        
        # Calculate VMA slope (rate of change)
        data['vma_slope'] = data['vma'].diff() / data['close']
        
        # Calculate VMA trend direction
        data['vma_trend'] = np.where(data['vma_slope'] > slope_threshold, 1,
                                   np.where(data['vma_slope'] < -slope_threshold, -1, 0))
        
        # Calculate distance between price and VMA (normalized)
        data['price_vma_distance'] = (source - data['vma']) / source
        
        # Calculate VMA momentum (rate of change over period)
        data['vma_momentum'] = data['vma'].pct_change(periods=momentum_period)
        
        # Calculate price volatility for filtering
        data['volatility'] = data['close'].rolling(window=20).std() / data['close'].rolling(window=20).mean()
        
        # Calculate ATR for additional volatility measure
        if all(col in data.columns for col in ['high', 'low']):
            data['atr'] = talib.ATR(
                data['high'].to_numpy().astype(np.float64),
                data['low'].to_numpy().astype(np.float64),
                data['close'].to_numpy().astype(np.float64),
                timeperiod=14
            )
            data['atr_normalized'] = data['atr'] / data['close']
        
        # Calculate RSI for additional confirmation
        data['rsi'] = talib.RSI(data['close'].to_numpy().astype(np.float64), timeperiod=14)
        
        # Calculate trend strength based on VMA characteristics
        data['trend_strength'] = np.abs(data['vma_slope']) * 100
        
        # Store key indicators for debugging
        key_indicators = ['vma', 'vma_slope', 'vma_trend', 'price_vma_distance', 
                         'vma_momentum', 'volatility', 'trend_strength']
        self.indicators = data[key_indicators].copy()
        
        return data
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate buy/sell signals based on Variable Moving Average logic
        
        Args:
            data: DataFrame with VMA indicators
            
        Returns:
            DataFrame with buy_signal, sell_signal, and signal_strength columns
        """
        # Initialize signal columns
        data['buy_signal'] = False
        data['sell_signal'] = False
        data['signal_strength'] = 0.0
        
        # Extract parameters
        use_price_cross = self.parameters['use_price_cross']
        use_slope_filter = self.parameters['use_slope_filter']
        use_volatility_filter = self.parameters['use_volatility_filter']
        use_momentum_filter = self.parameters['use_momentum_filter']
        min_volatility = self.parameters['min_volatility']
        max_volatility = self.parameters['max_volatility']
        threshold = self.parameters['signal_threshold']
        
        # Determine price source
        vma_source = self.parameters['vma_source']
        if vma_source == 'close':
            source = data['close']
        elif vma_source == 'hl2':
            source = (data['high'] + data['low']) / 2
        elif vma_source == 'hlc3':
            source = (data['high'] + data['low'] + data['close']) / 3
        elif vma_source == 'ohlc4':
            source = (data['open'] + data['high'] + data['low'] + data['close']) / 4
        else:
            source = data['close']
        
        # Primary buy conditions
        buy_conditions = []
        
        if use_price_cross:
            # Condition 1: Price crosses above VMA
            price_cross_up = crossover(source, data['vma'])
            buy_conditions.append(price_cross_up)
        else:
            # Alternative: Price above VMA
            price_above_vma = source > data['vma']
            buy_conditions.append(price_above_vma)
        
        if use_slope_filter:
            # Condition 2: VMA slope is positive (uptrend)
            vma_uptrend = data['vma_trend'] >= 0
            buy_conditions.append(vma_uptrend)
        
        # Condition 3: RSI not overbought
        rsi_not_overbought = data['rsi'] < 70
        buy_conditions.append(rsi_not_overbought)
        
        # Primary sell conditions
        sell_conditions = []
        
        if use_price_cross:
            # Condition 1: Price crosses below VMA
            price_cross_down = crossunder(source, data['vma'])
            sell_conditions.append(price_cross_down)
        else:
            # Alternative: Price below VMA
            price_below_vma = source < data['vma']
            sell_conditions.append(price_below_vma)
        
        if use_slope_filter:
            # Condition 2: VMA slope is negative (downtrend)
            vma_downtrend = data['vma_trend'] <= 0
            sell_conditions.append(vma_downtrend)
        
        # Condition 3: RSI not oversold
        rsi_not_oversold = data['rsi'] > 30
        sell_conditions.append(rsi_not_oversold)
        
        # Volatility filter if enabled
        if use_volatility_filter and 'volatility' in data.columns:
            volatility_ok = (data['volatility'] >= min_volatility) & (data['volatility'] <= max_volatility)
            buy_conditions.append(volatility_ok)
            sell_conditions.append(volatility_ok)
        
        # Momentum filter if enabled
        if use_momentum_filter:
            positive_momentum = data['vma_momentum'] > 0
            negative_momentum = data['vma_momentum'] < 0
            buy_conditions.append(positive_momentum)
            sell_conditions.append(negative_momentum)
        
        # Combine conditions using shift(1) to prevent look-ahead bias
        shifted_buy_conditions = [cond.shift(1).fillna(False).astype(bool) for cond in buy_conditions]
        shifted_sell_conditions = [cond.shift(1).fillna(False).astype(bool) for cond in sell_conditions]
        
        # Generate preliminary signals
        data['buy_signal'] = pd.concat(shifted_buy_conditions, axis=1).all(axis=1)
        data['sell_signal'] = pd.concat(shifted_sell_conditions, axis=1).all(axis=1)
        
        # Calculate signal strength based on multiple factors
        strength_factors = []
        
        # Factor 1: Trend strength (VMA slope magnitude)
        trend_strength_shifted = data['trend_strength'].shift(1).fillna(0)
        trend_strength_normalized = (trend_strength_shifted / trend_strength_shifted.quantile(0.95)).clip(0, 1)
        strength_factors.append(trend_strength_normalized)
        
        # Factor 2: Price-VMA distance (momentum)
        price_distance_shifted = np.abs(data['price_vma_distance'].shift(1)).fillna(0)
        distance_normalized = (price_distance_shifted / price_distance_shifted.quantile(0.95)).clip(0, 1)
        strength_factors.append(distance_normalized)
        
        # Factor 3: RSI momentum
        rsi_shifted = data['rsi'].shift(1).fillna(50)
        rsi_strength = pd.Series(0.0, index=data.index)
        rsi_strength[rsi_shifted < 30] = (30 - rsi_shifted[rsi_shifted < 30]) / 30  # Oversold strength
        rsi_strength[rsi_shifted > 70] = (rsi_shifted[rsi_shifted > 70] - 70) / 30  # Overbought strength
        rsi_strength = rsi_strength.clip(0, 1)
        strength_factors.append(rsi_strength)
        
        # Factor 4: VMA momentum strength
        vma_momentum_shifted = np.abs(data['vma_momentum'].shift(1)).fillna(0)
        momentum_normalized = (vma_momentum_shifted / vma_momentum_shifted.quantile(0.95)).clip(0, 1)
        strength_factors.append(momentum_normalized)
        
        # Combine strength factors
        data['signal_strength'] = calculate_signal_strength(
            strength_factors,
            weights=[0.4, 0.3, 0.2, 0.1]  # Prioritize trend strength and distance
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
        Validate that VMA signals are properly formed
        
        Returns:
            True if signals are valid, False otherwise
        """
        # Check for required columns
        required = ['buy_signal', 'sell_signal', 'signal_strength']
        if not all(col in data.columns for col in required):
            return False
        
        # Check for NaN values in recent data (allow some initial NaN due to indicators)
        recent_data = data.iloc[-50:] if len(data) > 50 else data
        if recent_data[required].isna().any().any():
            return False
        
        # Check for simultaneous buy and sell signals
        simultaneous = (data['buy_signal'] & data['sell_signal']).any()
        if simultaneous:
            return False
        
        # Check signal strength is in valid range
        valid_strength = recent_data['signal_strength']
        if (valid_strength < 0).any() or (valid_strength > 1).any():
            return False
        
        # Check that we have VMA indicator
        vma_indicators = ['vma', 'vma_slope', 'vma_trend']
        if not all(col in data.columns for col in vma_indicators):
            return False
        
        return True


if __name__ == "__main__":
    # Test the Variable Moving Average strategy with sample data
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Create sample data with realistic price movement
    np.random.seed(42)  # For reproducible results
    dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='5min')
    n_periods = len(dates)
    
    # Generate realistic OHLCV data with trending behavior
    base_price = 100
    # Add trend component and noise
    trend = np.linspace(0, 0.1, n_periods)  # 10% upward trend over period
    noise = np.random.normal(0, 0.002, n_periods)
    price_changes = trend + noise.cumsum()
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
    print("Testing Variable Moving Average Strategy...")
    strategy = Strategy156VariableMovingAverage()
    
    try:
        # Calculate indicators
        data_with_indicators = strategy.calculate_indicators(sample_data.copy())
        print(f"✅ Indicators calculated. Shape: {data_with_indicators.shape}")
        
        # Generate signals
        data_with_signals = strategy.generate_signals(data_with_indicators)
        print(f"✅ Signals generated. Shape: {data_with_signals.shape}")
        
        # Validate signals
        if strategy.validate_signals(data_with_signals):
            print("✅ Variable Moving Average validation passed")
            print(f"Buy signals: {data_with_signals['buy_signal'].sum()}")
            print(f"Sell signals: {data_with_signals['sell_signal'].sum()}")
            print(f"Average signal strength: {data_with_signals['signal_strength'].mean():.3f}")
            
            # Show some key statistics
            print("\n📊 Key Statistics:")
            vma_data = data_with_signals['vma'].dropna()
            if len(vma_data) > 0:
                print(f"VMA range: [{vma_data.min():.4f}, {vma_data.max():.4f}]")
                print(f"Average VMA slope: {data_with_signals['vma_slope'].mean():.6f}")
                print(f"Trend strength range: [{data_with_signals['trend_strength'].min():.6f}, {data_with_signals['trend_strength'].max():.6f}]")
                
                # Show trend distribution
                uptrend = (data_with_signals['vma_trend'] == 1).sum()
                sideways = (data_with_signals['vma_trend'] == 0).sum()
                downtrend = (data_with_signals['vma_trend'] == -1).sum()
                print(f"Trend distribution: Uptrend: {uptrend}, Sideways: {sideways}, Downtrend: {downtrend}")
            
        else:
            print("❌ Variable Moving Average validation failed")
            
    except Exception as e:
        print(f"❌ Error testing Variable Moving Average strategy: {e}")
        import traceback
        traceback.print_exc()