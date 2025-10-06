#!/usr/bin/env python3
"""
Variable Moving Average Bands Strategy Implementation

Strategy Number: 157
LazyBear Name: VMA Bands
Type: trend following with volatility bands
TradingView URL: https://www.tradingview.com/v/HT99VkZb/

Description:
Variable Moving Average Bands uses a Variable Moving Average (VMA) as the center line
with ATR-based bands above and below. VMA automatically adjusts its smoothing constant
based on market volatility, making it more responsive during volatile periods.

The bands are constructed as:
- Upper Band: VMA + (Bands Multiplier * ATR)
- Lower Band: VMA - (Bands Multiplier * ATR)

Signals are generated based on:
- Buy: Price crosses above lower band with VMA trending up
- Sell: Price crosses below upper band with VMA trending down
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


def calculate_vma(src: pd.Series, length: int) -> pd.Series:
    """
    Calculate Variable Moving Average (VMA) based on LazyBear's Pine Script implementation.
    
    VMA automatically adjusts its smoothing constant based on market volatility.
    
    Args:
        src: Source price series (typically close)
        length: VMA period length
        
    Returns:
        VMA series
    """
    n = len(src)
    if n < length:
        return pd.Series([np.nan] * n, index=src.index)
    
    # Initialize arrays
    vma = np.full(n, np.nan)
    pdmS = np.full(n, np.nan)
    mdmS = np.full(n, np.nan)
    pdiS = np.full(n, np.nan)
    mdiS = np.full(n, np.nan)
    iS = np.full(n, np.nan)
    
    # Smoothing constant
    k = 1.0 / length
    
    # Convert to numpy array for faster computation
    src_values = src.to_numpy()
    
    for i in range(1, n):
        # Calculate directional movements
        pdm = max((src_values[i] - src_values[i-1]), 0)
        mdm = max((src_values[i-1] - src_values[i]), 0)
        
        # Smooth directional movements
        pdmS[i] = ((1 - k) * (pdmS[i-1] if not np.isnan(pdmS[i-1]) else 0) + k * pdm)
        mdmS[i] = ((1 - k) * (mdmS[i-1] if not np.isnan(mdmS[i-1]) else 0) + k * mdm)
        
        # Calculate directional indicators
        s = pdmS[i] + mdmS[i]
        if s > 0:
            pdi = pdmS[i] / s
            mdi = mdmS[i] / s
        else:
            pdi = 0.5
            mdi = 0.5
            
        # Smooth directional indicators
        pdiS[i] = ((1 - k) * (pdiS[i-1] if not np.isnan(pdiS[i-1]) else 0.5) + k * pdi)
        mdiS[i] = ((1 - k) * (mdiS[i-1] if not np.isnan(mdiS[i-1]) else 0.5) + k * mdi)
        
        # Calculate volatility index
        d = abs(pdiS[i] - mdiS[i])
        s1 = pdiS[i] + mdiS[i]
        if s1 > 0:
            iS[i] = ((1 - k) * (iS[i-1] if not np.isnan(iS[i-1]) else 0) + k * d / s1)
        else:
            iS[i] = iS[i-1] if not np.isnan(iS[i-1]) else 0
        
        # Calculate VMA when we have enough data
        if i >= length:
            # Get highest and lowest iS values over the period
            start_idx = max(0, i - length + 1)
            iS_period = iS[start_idx:i+1]
            valid_iS = iS_period[~np.isnan(iS_period)]
            
            if len(valid_iS) > 0:
                hhv = np.max(valid_iS)
                llv = np.min(valid_iS)
                d1 = hhv - llv
                
                if d1 > 0:
                    vI = (iS[i] - llv) / d1
                else:
                    vI = 0.5
                
                # Calculate VMA
                vma[i] = ((1 - k * vI) * (vma[i-1] if not np.isnan(vma[i-1]) else src_values[i]) + 
                         k * vI * src_values[i])
            else:
                vma[i] = src_values[i]
    
    return pd.Series(vma, index=src.index)


class Strategy157VmaBands(BaseStrategy):
    """
    Variable Moving Average Bands Strategy Implementation
    
    Uses Variable Moving Average as center line with ATR-based bands
    for trend following and volatility-adjusted support/resistance.
    """
    
    def __init__(self, parameters: Dict = None):
        """
        Initialize VMA Bands strategy with parameters
        """
        # Define default parameters based on LazyBear's implementation
        default_params = {
            # VMA parameters
            'vma_length': 6,           # VMA period (LazyBear default)
            'bands_multiplier': 1.5,   # ATR multiplier for bands
            'atr_period': 14,          # ATR period for bands
            
            # Signal generation parameters
            'signal_threshold': 0.6,   # Minimum signal strength to trade
            'use_trend_filter': True,  # Filter signals by VMA trend direction
            'use_volume_filter': False, # Use volume confirmation
            'volume_threshold': 1.2,   # Volume ratio threshold
            
            # Risk management
            'max_lookback': 100,       # Maximum bars to look back for calculations
        }
        
        # Merge with provided parameters
        if parameters:
            default_params.update(parameters)
        
        # Initialize parent class
        super().__init__(
            name="Strategy_157_VmaBands",
            parameters=default_params
        )
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate VMA Bands indicators
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with VMA Bands indicators
        """
        # Ensure we have required columns
        if 'close' not in data.columns and 'price' in data.columns:
            data['close'] = data['price']
        
        # Extract parameters
        vma_length = self.parameters['vma_length']
        bands_multiplier = self.parameters['bands_multiplier']
        atr_period = self.parameters['atr_period']
        max_lookback = self.parameters['max_lookback']
        
        # Limit data to prevent excessive computation
        if len(data) > max_lookback:
            data = data.tail(max_lookback).copy()
        
        # Calculate Variable Moving Average
        data['vma'] = calculate_vma(data['close'], vma_length)
        
        # Calculate Average True Range for bands
        if all(col in data.columns for col in ['high', 'low', 'close']):
            data['atr'] = talib.ATR(
                data['high'].to_numpy().astype(np.float64),
                data['low'].to_numpy().astype(np.float64),
                data['close'].to_numpy().astype(np.float64),
                timeperiod=atr_period
            )
        else:
            # Fallback ATR calculation using close prices only
            price_changes = data['close'].diff().abs()
            data['atr'] = price_changes.rolling(window=atr_period).mean()
        
        # Calculate bands
        band_width = bands_multiplier * data['atr']
        data['vma_upper'] = data['vma'] + band_width
        data['vma_lower'] = data['vma'] - band_width
        
        # Calculate VMA trend direction
        data['vma_trend'] = np.where(data['vma'] > data['vma'].shift(1), 1, 
                            np.where(data['vma'] < data['vma'].shift(1), -1, 0))
        
        # Calculate price position relative to bands
        data['price_vs_upper'] = data['close'] - data['vma_upper']
        data['price_vs_lower'] = data['close'] - data['vma_lower']
        data['price_vs_vma'] = data['close'] - data['vma']
        
        # Additional momentum indicators for confirmation
        data['rsi'] = talib.RSI(data['close'].to_numpy().astype(np.float64), timeperiod=14)
        
        # Band width as volatility measure
        data['band_width'] = (data['vma_upper'] - data['vma_lower']) / data['vma']
        
        # Volume analysis if available
        if 'volume' in data.columns:
            data['volume_ma'] = talib.MA(data['volume'].to_numpy().astype(np.float64), timeperiod=20)
            data['volume_ratio'] = data['volume'] / data['volume_ma']
        
        # Store key indicators for debugging
        key_indicators = ['vma', 'vma_upper', 'vma_lower', 'vma_trend', 
                         'band_width', 'price_vs_vma', 'rsi']
        self.indicators = data[key_indicators].copy()
        
        return data
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate buy/sell signals based on VMA Bands logic
        
        Args:
            data: DataFrame with VMA Bands indicators
            
        Returns:
            DataFrame with buy_signal, sell_signal, and signal_strength columns
        """
        # Initialize signal columns
        data['buy_signal'] = False
        data['sell_signal'] = False
        data['signal_strength'] = 0.0
        
        # Extract parameters
        threshold = self.parameters['signal_threshold']
        use_trend_filter = self.parameters['use_trend_filter']
        use_volume_filter = self.parameters['use_volume_filter']
        volume_threshold = self.parameters['volume_threshold']
        
        # Primary buy conditions
        buy_conditions = []
        
        # Condition 1: Price crosses above lower band (bounce from support)
        price_above_lower = crossover(data['close'], data['vma_lower'])
        buy_conditions.append(price_above_lower)
        
        # Condition 2: VMA trending upward (trend filter)
        if use_trend_filter:
            vma_uptrend = data['vma_trend'] >= 0
            buy_conditions.append(vma_uptrend)
        
        # Condition 3: Price above VMA (momentum confirmation)
        price_above_vma = data['close'] > data['vma']
        buy_conditions.append(price_above_vma)
        
        # Condition 4: RSI not overbought
        rsi_not_overbought = data['rsi'] < 70
        buy_conditions.append(rsi_not_overbought)
        
        # Primary sell conditions
        sell_conditions = []
        
        # Condition 1: Price crosses below upper band (rejection at resistance)
        price_below_upper = crossunder(data['close'], data['vma_upper'])
        sell_conditions.append(price_below_upper)
        
        # Condition 2: VMA trending downward (trend filter)
        if use_trend_filter:
            vma_downtrend = data['vma_trend'] <= 0
            sell_conditions.append(vma_downtrend)
        
        # Condition 3: Price below VMA (momentum confirmation)
        price_below_vma = data['close'] < data['vma']
        sell_conditions.append(price_below_vma)
        
        # Condition 4: RSI not oversold
        rsi_not_oversold = data['rsi'] > 30
        sell_conditions.append(rsi_not_oversold)
        
        # Volume filter if enabled
        if use_volume_filter and 'volume_ratio' in data.columns:
            volume_confirmation = data['volume_ratio'] > volume_threshold
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
        
        # Factor 1: Distance from bands (closer to opposite band = stronger signal)
        # For buy signals: closer to lower band = stronger
        # For sell signals: closer to upper band = stronger
        band_range = data['vma_upper'] - data['vma_lower']
        price_position = (data['close'] - data['vma_lower']) / band_range
        
        buy_strength = (1 - price_position).clip(0, 1)  # Stronger when price near lower band
        sell_strength = price_position.clip(0, 1)       # Stronger when price near upper band
        
        position_strength = pd.Series(0.0, index=data.index)
        position_strength[data['buy_signal']] = buy_strength[data['buy_signal']]
        position_strength[data['sell_signal']] = sell_strength[data['sell_signal']]
        strength_factors.append(position_strength.shift(1).fillna(0))
        
        # Factor 2: VMA trend strength
        vma_momentum = np.abs(data['vma'].diff()) / data['vma'] * 100
        vma_strength = vma_momentum.clip(0, 1)
        strength_factors.append(vma_strength.shift(1).fillna(0))
        
        # Factor 3: Band width (higher volatility = potentially stronger signals)
        band_width_normalized = data['band_width'].clip(0, 0.1) * 10  # Normalize to 0-1 range
        strength_factors.append(band_width_normalized.shift(1).fillna(0))
        
        # Factor 4: RSI extremity (more extreme = stronger signal)
        rsi_shifted = data['rsi'].shift(1).fillna(50)
        rsi_strength = pd.Series(0.0, index=data.index)
        rsi_strength[rsi_shifted < 30] = (30 - rsi_shifted[rsi_shifted < 30]) / 30
        rsi_strength[rsi_shifted > 70] = (rsi_shifted[rsi_shifted > 70] - 70) / 30
        rsi_strength = rsi_strength.clip(0, 1)
        strength_factors.append(rsi_strength)
        
        # Combine strength factors
        data['signal_strength'] = calculate_signal_strength(
            strength_factors,
            weights=[0.4, 0.3, 0.2, 0.1]  # Prioritize position and trend
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
        Validate that VMA Bands signals are properly formed
        
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
        
        # Check that we have VMA indicators
        vma_indicators = ['vma', 'vma_upper', 'vma_lower']
        if not all(col in data.columns for col in vma_indicators):
            return False
        
        return True


if __name__ == "__main__":
    # Test the VMA Bands strategy with sample data
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
    print("Testing VMA Bands Strategy...")
    strategy = Strategy157VmaBands()
    
    try:
        # Calculate indicators
        data_with_indicators = strategy.calculate_indicators(sample_data.copy())
        print(f"✅ Indicators calculated. Shape: {data_with_indicators.shape}")
        
        # Generate signals
        data_with_signals = strategy.generate_signals(data_with_indicators)
        print(f"✅ Signals generated. Shape: {data_with_signals.shape}")
        
        # Validate signals
        if strategy.validate_signals(data_with_signals):
            print("✅ VMA Bands validation passed")
            print(f"Buy signals: {data_with_signals['buy_signal'].sum()}")
            print(f"Sell signals: {data_with_signals['sell_signal'].sum()}")
            print(f"Average signal strength: {data_with_signals['signal_strength'].mean():.3f}")
            
            # Show some key statistics
            print("\n📊 Key Statistics:")
            print(f"VMA range: [{data_with_signals['vma'].min():.4f}, {data_with_signals['vma'].max():.4f}]")
            print(f"Upper band range: [{data_with_signals['vma_upper'].min():.4f}, {data_with_signals['vma_upper'].max():.4f}]")
            print(f"Lower band range: [{data_with_signals['vma_lower'].min():.4f}, {data_with_signals['vma_lower'].max():.4f}]")
            print(f"Average band width: {data_with_signals['band_width'].mean():.4f}")
            print(f"VMA trend distribution: Up={sum(data_with_signals['vma_trend'] == 1)}, Down={sum(data_with_signals['vma_trend'] == -1)}, Flat={sum(data_with_signals['vma_trend'] == 0)}")
            
        else:
            print("❌ VMA Bands validation failed")
            
    except Exception as e:
        print(f"❌ Error testing VMA Bands strategy: {e}")
        import traceback
        traceback.print_exc()