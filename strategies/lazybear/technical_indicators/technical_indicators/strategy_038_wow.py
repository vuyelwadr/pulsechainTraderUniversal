#!/usr/bin/env python3
"""
WOW Oscillator Strategy

Strategy Number: 014
LazyBear Name: WOW Oscillator
Type: oscillator
TradingView URL: https://www.tradingview.com/v/hOXxJ2ym/

Description:
The WOW Oscillator is a momentum-based oscillator that combines price action
with volatility analysis. It uses exponential moving averages applied to
typical price and true range to create normalized oscillator values.

The oscillator oscillates around zero, with positive values indicating bullish
momentum and negative values indicating bearish momentum. Buy signals occur
when the oscillator crosses above the oversold level, and sell signals occur
when it crosses below the overbought level.
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


class Strategy038Wow(BaseStrategy):
    """
    WOW Oscillator Strategy
    
    Strategy Number: 014
    LazyBear Name: WOW Oscillator  
    Type: oscillator
    TradingView URL: https://www.tradingview.com/v/hOXxJ2ym/
    
    Description:
    A momentum oscillator that combines price and volatility analysis to generate
    normalized oscillator values. Provides overbought/oversold signals with
    additional momentum confirmation.
    """
    
    def __init__(self, parameters: Dict = None):
        """
        Initialize WOW Oscillator Strategy
        
        Default parameters based on typical oscillator configurations
        """
        # Define default parameters
        default_params = {
            'channel_length': 10,        # Period for channel calculation
            'average_length': 21,        # Period for smoothing
            'over_bought_1': 60,         # First overbought level
            'over_bought_2': 53,         # Second overbought level  
            'over_sold_1': -60,          # First oversold level
            'over_sold_2': -53,          # Second oversold level
            'signal_threshold': 0.6,     # Minimum signal strength to trade
            'use_volume_filter': True,   # Volume confirmation
            'use_trend_filter': True,    # Trend filter
        }
        
        # Merge with provided parameters
        if parameters:
            default_params.update(parameters)
        
        # Initialize parent class
        super().__init__(
            name="Strategy_038_Wow",
            parameters=default_params
        )
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate WOW Oscillator and related indicators
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with additional indicator columns
        """
        # Ensure we have required columns
        if 'close' not in data.columns and 'price' in data.columns:
            data['close'] = data['price']
        
        # Extract parameters
        channel_length = self.parameters['channel_length']
        average_length = self.parameters['average_length']
        
        # Calculate typical price (HLC/3)
        data['hlc3'] = (data['high'] + data['low'] + data['close']) / 3
        
        # Calculate True Range
        data['tr'] = talib.TRANGE(
            data['high'].astype(float).values, 
            data['low'].astype(float).values, 
            data['close'].astype(float).values
        )
        
        # Apply EMA to typical price and true range
        data['esa'] = talib.EMA(data['hlc3'].astype(float).values, timeperiod=channel_length)
        data['d'] = talib.EMA(np.abs(data['hlc3'] - data['esa']).astype(float).values, timeperiod=channel_length)
        
        # Calculate channel bands
        data['ci'] = (data['hlc3'] - data['esa']) / (0.015 * data['d'])
        
        # Apply secondary smoothing to create the main oscillator
        data['wt1'] = talib.EMA(data['ci'].astype(float).values, timeperiod=average_length)
        
        # Calculate signal line (additional smoothing)
        data['wt2'] = talib.SMA(data['wt1'].astype(float).values, timeperiod=4)
        
        # Calculate WOW oscillator as difference
        data['wow'] = data['wt1'] - data['wt2']
        
        # Calculate additional momentum indicators
        data['rsi'] = talib.RSI(data['close'].astype(float).values, timeperiod=14)
        data['momentum'] = data['close'] - data['close'].shift(10)
        
        # Volume indicators
        if 'volume' in data.columns:
            data['volume_ma'] = talib.MA(data['volume'].astype(float).values, timeperiod=20)
            data['volume_ratio'] = data['volume'] / data['volume_ma']
        
        # Trend filter using EMA
        data['ema_fast'] = talib.EMA(data['close'].astype(float).values, timeperiod=12)
        data['ema_slow'] = talib.EMA(data['close'].astype(float).values, timeperiod=26)
        
        # Store indicators for debugging
        self.indicators = data[['wt1', 'wt2', 'wow', 'rsi', 'momentum']].copy()
        
        return data
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate buy/sell signals based on WOW Oscillator logic
        
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
        ob1 = self.parameters['over_bought_1']
        ob2 = self.parameters['over_bought_2'] 
        os1 = self.parameters['over_sold_1']
        os2 = self.parameters['over_sold_2']
        threshold = self.parameters['signal_threshold']
        
        # Define buy conditions
        buy_conditions = []
        
        # Condition 1: WOW oscillator crosses above oversold level
        wow_oversold_cross = crossover(data['wow'], pd.Series([os2] * len(data), index=data.index))
        buy_conditions.append(wow_oversold_cross)
        
        # Condition 2: WOW oscillator is in oversold territory
        wow_oversold = data['wow'] < os1
        buy_conditions.append(wow_oversold.shift(1))  # Previous bar was oversold
        
        # Condition 3: WT1 crosses above WT2
        wt_cross_up = crossover(data['wt1'], data['wt2'])
        buy_conditions.append(wt_cross_up)
        
        # Condition 4: Trend filter
        if self.parameters['use_trend_filter']:
            trend_up = data['close'] > data['ema_slow']
            buy_conditions.append(trend_up)
        
        # Condition 5: Volume confirmation
        if self.parameters['use_volume_filter'] and 'volume_ratio' in data.columns:
            volume_surge = data['volume_ratio'] > 1.2
            buy_conditions.append(volume_surge)
        
        # Combine buy conditions - need at least core conditions
        if len(buy_conditions) >= 3:
            buy_score = pd.concat(buy_conditions, axis=1).sum(axis=1)
            data['buy_signal'] = buy_score >= 2  # At least 2 conditions must be true
        else:
            data['buy_signal'] = pd.concat(buy_conditions, axis=1).all(axis=1)
        
        # Define sell conditions
        sell_conditions = []
        
        # Condition 1: WOW oscillator crosses below overbought level
        wow_overbought_cross = crossunder(data['wow'], pd.Series([ob2] * len(data), index=data.index))
        sell_conditions.append(wow_overbought_cross)
        
        # Condition 2: WOW oscillator is in overbought territory
        wow_overbought = data['wow'] > ob1
        sell_conditions.append(wow_overbought.shift(1))  # Previous bar was overbought
        
        # Condition 3: WT1 crosses below WT2
        wt_cross_down = crossunder(data['wt1'], data['wt2'])
        sell_conditions.append(wt_cross_down)
        
        # Condition 4: Trend filter
        if self.parameters['use_trend_filter']:
            trend_down = data['close'] < data['ema_slow']
            sell_conditions.append(trend_down)
        
        # Combine sell conditions - need at least core conditions
        if len(sell_conditions) >= 3:
            sell_score = pd.concat(sell_conditions, axis=1).sum(axis=1)
            data['sell_signal'] = sell_score >= 2  # At least 2 conditions must be true
        else:
            data['sell_signal'] = pd.concat(sell_conditions, axis=1).all(axis=1)
        
        # Calculate signal strength
        strength_factors = []
        
        # Factor 1: WOW oscillator strength
        wow_strength = np.abs(data['wow']) / 100.0  # Normalize to 0-1
        wow_strength = wow_strength.clip(0, 1)
        strength_factors.append(wow_strength)
        
        # Factor 2: WT1/WT2 divergence
        wt_divergence = np.abs(data['wt1'] - data['wt2']) / 50.0
        wt_divergence = wt_divergence.clip(0, 1)
        strength_factors.append(wt_divergence)
        
        # Factor 3: RSI extremes
        rsi_strength = pd.Series(0.0, index=data.index)
        rsi_strength[data['rsi'] < 20] = 1.0
        rsi_strength[(data['rsi'] >= 20) & (data['rsi'] < 30)] = 0.8
        rsi_strength[(data['rsi'] > 70) & (data['rsi'] <= 80)] = 0.8
        rsi_strength[data['rsi'] > 80] = 1.0
        strength_factors.append(rsi_strength)
        
        # Factor 4: Volume confirmation
        if 'volume_ratio' in data.columns:
            volume_strength = (data['volume_ratio'] - 1.0).clip(0, 1)
            strength_factors.append(volume_strength)
        
        # Combine strength factors
        if strength_factors:
            data['signal_strength'] = calculate_signal_strength(strength_factors)
        
        # Apply minimum threshold
        weak_signals = data['signal_strength'] < threshold
        data.loc[weak_signals, 'buy_signal'] = False
        data.loc[weak_signals, 'sell_signal'] = False
        
        # Apply look-ahead bias prevention
        data['buy_signal'] = data['buy_signal'].shift(1).fillna(False).infer_objects(copy=False)
        data['sell_signal'] = data['sell_signal'].shift(1).fillna(False).infer_objects(copy=False)
        data['signal_strength'] = data['signal_strength'].shift(1).fillna(0.0)
        
        # Apply position constraints - ensure boolean values
        data['buy_signal'] = data['buy_signal'].astype(bool)
        data['sell_signal'] = data['sell_signal'].astype(bool)
        
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
        
        # Check for NaN values in recent data
        recent_data = data.tail(100)  # Check last 100 rows
        if recent_data[required].isna().any().any():
            return False
        
        # Check for simultaneous buy and sell signals
        simultaneous = (data['buy_signal'] & data['sell_signal']).any()
        if simultaneous:
            return False
        
        # Check signal strength is in valid range
        valid_strength = data['signal_strength'].dropna()
        if len(valid_strength) > 0:
            if (valid_strength < 0).any() or (valid_strength > 1).any():
                return False
        
        return True


if __name__ == "__main__":
    # Test the strategy with sample data
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    
    # Create sample data
    np.random.seed(42)  # For reproducible testing
    dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='5min')
    n_periods = len(dates)
    
    # Generate realistic price data
    price_base = 100
    price_walk = np.random.randn(n_periods).cumsum() * 0.5
    
    sample_data = pd.DataFrame({
        'timestamp': dates,
        'open': price_base + price_walk + np.random.randn(n_periods) * 0.1,
        'high': price_base + price_walk + np.random.randn(n_periods) * 0.1 + 0.5,
        'low': price_base + price_walk + np.random.randn(n_periods) * 0.1 - 0.5,
        'close': price_base + price_walk,
        'volume': np.random.randint(1000, 10000, n_periods)
    })
    
    # Ensure high >= low >= close relationships
    sample_data['high'] = np.maximum(sample_data['high'], sample_data[['open', 'close']].max(axis=1))
    sample_data['low'] = np.minimum(sample_data['low'], sample_data[['open', 'close']].min(axis=1))
    sample_data['price'] = sample_data['close']
    
    # Test strategy
    try:
        strategy = Strategy038Wow()
        print("🧪 Testing WOW Oscillator Strategy...")
        
        data_with_indicators = strategy.calculate_indicators(sample_data.copy())
        print("✅ Indicators calculated successfully")
        
        data_with_signals = strategy.generate_signals(data_with_indicators)
        print("✅ Signals generated successfully")
        
        # Validate
        if strategy.validate_signals(data_with_signals):
            print("✅ WOW Oscillator validation passed")
            
            # Print statistics
            total_signals = len(data_with_signals)
            buy_signals = data_with_signals['buy_signal'].sum()
            sell_signals = data_with_signals['sell_signal'].sum() 
            avg_strength = data_with_signals['signal_strength'].mean()
            
            print(f"📊 Strategy Statistics:")
            print(f"   Total periods: {total_signals}")
            print(f"   Buy signals: {buy_signals}")
            print(f"   Sell signals: {sell_signals}")
            print(f"   Signal rate: {((buy_signals + sell_signals) / total_signals * 100):.2f}%")
            print(f"   Avg signal strength: {avg_strength:.3f}")
            
            # Show sample of indicators
            print(f"\n📈 Sample Indicator Values (last 5 periods):")
            cols_to_show = ['wt1', 'wt2', 'wow', 'buy_signal', 'sell_signal', 'signal_strength']
            print(data_with_signals[cols_to_show].tail().round(3))
            
        else:
            print("❌ WOW Oscillator validation failed")
            
    except Exception as e:
        print(f"❌ Error testing strategy: {str(e)}")
        import traceback
        traceback.print_exc()