"""
MACD (Moving Average Convergence Divergence) Strategy for PulseChain Trading Bot

This strategy uses MACD crossovers and histogram for trend identification and momentum confirmation.
Generates signals based on MACD line crossing signal line and zero line crossovers.
"""
import pandas as pd
import numpy as np
from typing import Dict
import logging

from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

class MACDStrategy(BaseStrategy):
    """
    MACD Strategy
    
    Generates signals based on MACD crossovers and momentum confirmation.
    Uses MACD line, signal line, and histogram for comprehensive analysis.
    
    Parameters:
        fast_period (int): Fast EMA period (default: 12)
        slow_period (int): Slow EMA period (default: 26)
        signal_period (int): Signal line EMA period (default: 9)
        min_strength (float): Minimum signal strength to trigger trade (default: 0.6)
        histogram_threshold (float): Minimum histogram value for signal confirmation (default: 0.01)
        timeframe_minutes (int): Timeframe for analysis (default: 5)
    """
    
    def __init__(self, parameters: Dict = None):
        default_params = {
            'fast_period': 12,
            'slow_period': 26,
            'signal_period': 9,
            'min_strength': 0.6,
            'histogram_threshold': 0.01,
            'timeframe_minutes': 5
        }
        
        if parameters:
            default_params.update(parameters)
            
        super().__init__("MACDStrategy", default_params)
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate MACD and related indicators"""
        if not self.validate_data(data):
            return data
        
        df = data.copy()
        fast_period = self.parameters['fast_period']
        slow_period = self.parameters['slow_period']
        signal_period = self.parameters['signal_period']
        
        # Ensure we have enough data
        min_required = max(slow_period, fast_period) + signal_period + 5
        if len(df) < min_required:
            logger.warning(f"Not enough data for MACD calculation ({min_required} required)")
            return df
        
        # Calculate EMAs
        df['ema_fast'] = df['price'].ewm(span=fast_period, adjust=False).mean()
        df['ema_slow'] = df['price'].ewm(span=slow_period, adjust=False).mean()
        
        # Calculate MACD line
        df['macd'] = df['ema_fast'] - df['ema_slow']
        
        # Calculate signal line
        df['macd_signal'] = df['macd'].ewm(span=signal_period, adjust=False).mean()
        
        # Calculate histogram
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Calculate momentum indicators
        df['macd_momentum'] = df['macd'].diff()
        df['signal_momentum'] = df['macd_signal'].diff()
        df['histogram_momentum'] = df['macd_histogram'].diff()
        
        # Calculate crossover indicators
        df['macd_above_signal'] = df['macd'] > df['macd_signal']
        df['macd_above_zero'] = df['macd'] > 0
        df['signal_above_zero'] = df['macd_signal'] > 0
        
        # Detect crossovers
        df['bullish_crossover'] = (
            (df['macd'] > df['macd_signal']) & 
            (df['macd'].shift(1) <= df['macd_signal'].shift(1))
        )
        
        df['bearish_crossover'] = (
            (df['macd'] < df['macd_signal']) & 
            (df['macd'].shift(1) >= df['macd_signal'].shift(1))
        )
        
        # Zero line crossovers
        df['zero_line_bullish'] = (
            (df['macd'] > 0) & 
            (df['macd'].shift(1) <= 0)
        )
        
        df['zero_line_bearish'] = (
            (df['macd'] < 0) & 
            (df['macd'].shift(1) >= 0)
        )
        
        # Calculate trend strength
        df['trend_strength'] = abs(df['macd']) / (abs(df['macd']) + abs(df['macd_signal']))
        df['trend_strength'] = df['trend_strength'].fillna(0)
        
        return df
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate buy/sell signals based on MACD analysis"""
        df = data.copy()
        
        if 'macd' not in df.columns:
            df = self.calculate_indicators(df)
        
        # Initialize signal columns
        df['buy_signal'] = False
        df['sell_signal'] = False
        df['signal_strength'] = 0.0
        
        histogram_threshold = self.parameters['histogram_threshold']
        min_strength = self.parameters['min_strength']
        
        # Generate buy signals
        buy_conditions = (
            df['bullish_crossover'] &  # MACD crosses above signal
            (df['macd'] > 0) &  # MACD above zero line (uptrend confirmation)
            (df['macd_histogram'] > histogram_threshold)  # Positive histogram expansion
        )
        
        # Generate sell signals  
        sell_conditions = (
            df['bearish_crossover'] &  # MACD crosses below signal
            (df['macd'] < 0) &  # MACD below zero line (downtrend confirmation)
            (df['macd_histogram'] < -histogram_threshold)  # Negative histogram expansion
        )
        
        # Alternative buy signals (zero line crossover)
        zero_buy_conditions = (
            df['zero_line_bullish'] &  # MACD crosses above zero
            (df['macd_signal'] > df['macd_signal'].shift(1))  # Signal line rising
        )
        
        # Alternative sell signals (zero line crossover)
        zero_sell_conditions = (
            df['zero_line_bearish'] &  # MACD crosses below zero
            (df['macd_signal'] < df['macd_signal'].shift(1))  # Signal line falling
        )
        
        # Combine conditions
        df['buy_signal'] = buy_conditions | zero_buy_conditions
        df['sell_signal'] = sell_conditions | zero_sell_conditions
        
        # Calculate signal strength
        # Stronger signals when MACD and signal are far from zero and moving in same direction
        macd_distance = abs(df['macd']) / (df['price'] * 0.01)  # Normalize by price
        signal_distance = abs(df['macd_signal']) / (df['price'] * 0.01)
        histogram_strength = abs(df['macd_histogram']) / (df['price'] * 0.01)
        
        # Momentum component
        momentum_strength = np.where(
            df['buy_signal'],
            np.maximum(df['macd_momentum'], 0) + np.maximum(df['histogram_momentum'], 0),
            np.where(
                df['sell_signal'],
                abs(np.minimum(df['macd_momentum'], 0)) + abs(np.minimum(df['histogram_momentum'], 0)),
                0
            )
        )
        
        # Combine strength components
        base_strength = (macd_distance + signal_distance + histogram_strength) / 3
        df['signal_strength'] = np.minimum(
            (base_strength * 0.7 + momentum_strength * 0.3),
            1.0
        )
        
        # Apply minimum strength filter
        df.loc[df['signal_strength'] < min_strength, 'buy_signal'] = False
        df.loc[df['signal_strength'] < min_strength, 'sell_signal'] = False
        df.loc[(~df['buy_signal']) & (~df['sell_signal']), 'signal_strength'] = 0.0
        
        return df