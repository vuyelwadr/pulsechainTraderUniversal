"""
ATR-Based Channel Breakout Strategy for PulseChain Trading Bot

This strategy uses ATR (Average True Range) for dynamic channel creation similar to EKT.
Provides volatility-adjusted trend following with dynamic stops.
"""
import pandas as pd
import numpy as np
from typing import Dict
import logging

from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

class ATRChannelStrategy(BaseStrategy):
    """
    ATR-Based Channel Breakout Strategy
    
    Uses ATR for dynamic channel creation around a moving average.
    Trades breakouts from volatility-adjusted bands.
    
    Parameters:
        atr_period (int): Period for ATR calculation (default: 14)
        atr_multiplier (float): ATR multiplier for channel width (default: 2.0)
        ma_period (int): Moving average period for center line (default: 21)
        ma_type (str): Moving average type - 'sma' or 'ema' (default: 'ema')
        min_strength (float): Minimum signal strength (default: 0.6)
        timeframe_minutes (int): Timeframe for analysis (default: 5)
    """
    
    def __init__(self, parameters: Dict = None):
        default_params = {
            'atr_period': 14,
            'atr_multiplier': 2.0,
            'ma_period': 21,
            'ma_type': 'ema',
            'min_strength': 0.6,
            'timeframe_minutes': 5
        }
        
        if parameters:
            default_params.update(parameters)
            
        super().__init__("ATRChannelStrategy", default_params)
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate ATR channel and related indicators"""
        if not self.validate_data(data):
            return data
        
        df = data.copy()
        atr_period = self.parameters['atr_period']
        atr_multiplier = self.parameters['atr_multiplier']
        ma_period = self.parameters['ma_period']
        ma_type = self.parameters['ma_type']
        
        # Ensure we have enough data
        min_required = max(atr_period, ma_period) + 10
        if len(df) < min_required:
            logger.warning(f"Not enough data for ATR Channel calculation ({min_required} required)")
            return df
        
        # Calculate True Range
        df['prev_close'] = df['price'].shift(1)
        df['high_low'] = df['high'] - df['low']
        df['high_close'] = abs(df['high'] - df['prev_close'])
        df['low_close'] = abs(df['low'] - df['prev_close'])
        
        df['true_range'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
        
        # Calculate ATR
        df['atr'] = df['true_range'].ewm(span=atr_period, adjust=False).mean()
        
        # Calculate moving average center line
        if ma_type == 'ema':
            df['ma_center'] = df['price'].ewm(span=ma_period, adjust=False).mean()
        else:
            df['ma_center'] = df['price'].rolling(window=ma_period).mean()
        
        # Calculate channel bands
        df['upper_channel'] = df['ma_center'] + (df['atr'] * atr_multiplier)
        df['lower_channel'] = df['ma_center'] - (df['atr'] * atr_multiplier)
        
        # Calculate price position within channel
        channel_width = df['upper_channel'] - df['lower_channel']
        df['channel_position'] = (df['price'] - df['lower_channel']) / channel_width
        df['channel_position'] = df['channel_position'].fillna(0.5)
        
        # Breakout detection
        df['upper_breakout'] = (
            (df['price'] > df['upper_channel']) & 
            (df['price'].shift(1) <= df['upper_channel'].shift(1))
        )
        
        df['lower_breakout'] = (
            (df['price'] < df['lower_channel']) & 
            (df['price'].shift(1) >= df['lower_channel'].shift(1))
        )
        
        # Channel squeeze detection (low volatility)
        df['atr_sma'] = df['atr'].rolling(window=20).mean()
        df['channel_squeeze'] = df['atr'] < (df['atr_sma'] * 0.8)
        
        # Trend detection
        df['trend_up'] = df['ma_center'] > df['ma_center'].shift(5)
        df['trend_down'] = df['ma_center'] < df['ma_center'].shift(5)
        
        # Momentum indicators
        df['price_momentum'] = df['price'].pct_change(3)
        df['ma_momentum'] = df['ma_center'].pct_change(3)
        
        # Volatility expansion
        df['volatility_expansion'] = df['atr'] > df['atr'].shift(1)
        
        # Distance from center (normalized)
        df['distance_from_center'] = abs(df['price'] - df['ma_center']) / df['atr']
        
        # Support/Resistance levels
        df['at_upper_channel'] = abs(df['price'] - df['upper_channel']) / df['price'] < 0.01
        df['at_lower_channel'] = abs(df['price'] - df['lower_channel']) / df['price'] < 0.01
        
        return df
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate buy/sell signals based on ATR channel analysis"""
        df = data.copy()
        
        if 'upper_channel' not in df.columns:
            df = self.calculate_indicators(df)
        
        # Guard: if indicators couldn't be computed (insufficient data),
        # return empty signals to avoid KeyError on missing columns.
        required_cols = [
            'upper_breakout', 'lower_breakout', 'trend_up', 'trend_down',
            'volatility_expansion', 'price_momentum', 'at_lower_channel',
            'channel_position', 'distance_from_center', 'at_upper_channel',
            'atr', 'atr_sma', 'upper_channel', 'lower_channel', 'ma_momentum',
            'channel_squeeze'
        ]
        missing = [c for c in required_cols if c not in df.columns]
        
        # Initialize signal columns
        df['buy_signal'] = False
        df['sell_signal'] = False
        df['signal_strength'] = 0.0
        
        if missing:
            # Not enough data; no signals
            return df
        
        min_strength = self.parameters['min_strength']
        
        # Primary Breakout Signals
        # Buy on upward breakout
        breakout_buy_conditions = (
            df['upper_breakout'] &  # Price breaks above upper channel
            df['trend_up'] &  # Overall trend is up
            df['volatility_expansion'] &  # Volatility is expanding
            (df['price_momentum'] > 0)  # Price momentum positive
        )
        
        # Sell on downward breakout
        breakout_sell_conditions = (
            df['lower_breakout'] &  # Price breaks below lower channel
            df['trend_down'] &  # Overall trend is down
            df['volatility_expansion'] &  # Volatility is expanding
            (df['price_momentum'] < 0)  # Price momentum negative
        )
        
        # Mean Reversion Signals (when channel is wide)
        # Buy when price touches lower channel in uptrend
        reversion_buy_conditions = (
            df['at_lower_channel'] &  # Price at lower channel
            df['trend_up'] &  # But overall trend is up
            (df['channel_position'] < 0.2) &  # In lower 20% of channel
            (df['distance_from_center'] > 1.5)  # Far from center
        )
        
        # Sell when price touches upper channel in downtrend
        reversion_sell_conditions = (
            df['at_upper_channel'] &  # Price at upper channel
            df['trend_down'] &  # But overall trend is down
            (df['channel_position'] > 0.8) &  # In upper 20% of channel
            (df['distance_from_center'] > 1.5)  # Far from center
        )
        
        # Squeeze Breakout Signals (after low volatility)
        squeeze_buy_conditions = (
            df['upper_breakout'] &  # Breaks upper channel
            df['channel_squeeze'].shift(1) &  # Previous period was squeeze
            df['volatility_expansion'] &  # Now volatility expanding
            (df['ma_momentum'] > 0)  # MA momentum positive
        )
        
        squeeze_sell_conditions = (
            df['lower_breakout'] &  # Breaks lower channel
            df['channel_squeeze'].shift(1) &  # Previous period was squeeze
            df['volatility_expansion'] &  # Now volatility expanding
            (df['ma_momentum'] < 0)  # MA momentum negative
        )
        
        # Combine all conditions
        df['buy_signal'] = (
            breakout_buy_conditions | 
            reversion_buy_conditions | 
            squeeze_buy_conditions
        )
        
        df['sell_signal'] = (
            breakout_sell_conditions | 
            reversion_sell_conditions | 
            squeeze_sell_conditions
        )
        
        # Calculate signal strength
        # Base strength from breakout magnitude
        breakout_strength = np.where(
            df['buy_signal'],
            np.minimum((df['price'] - df['upper_channel']) / df['atr'], 2.0),
            np.where(
                df['sell_signal'],
                np.minimum((df['lower_channel'] - df['price']) / df['atr'], 2.0),
                0
            )
        ) / 2.0  # Normalize to 0-1
        
        # Momentum strength
        momentum_strength = np.minimum(abs(df['price_momentum']) * 20, 1.0)
        
        # Trend alignment strength
        trend_strength = np.where(
            (df['buy_signal'] & df['trend_up']) | (df['sell_signal'] & df['trend_down']),
            0.5,
            0.0
        )
        
        # Volatility strength (higher volatility = better for breakouts)
        volatility_strength = np.minimum(df['atr'] / df['atr_sma'], 2.0) / 2.0
        
        # Distance strength (further from center = stronger signal)
        distance_strength = np.minimum(df['distance_from_center'] / 3.0, 1.0)
        
        # Combine strength components
        df['signal_strength'] = np.minimum(
            (breakout_strength * 0.3 + 
             momentum_strength * 0.25 + 
             trend_strength * 0.2 + 
             volatility_strength * 0.15 + 
             distance_strength * 0.1),
            1.0
        )
        
        # Apply minimum strength filter
        df.loc[df['signal_strength'] < min_strength, 'buy_signal'] = False
        df.loc[df['signal_strength'] < min_strength, 'sell_signal'] = False
        df.loc[(~df['buy_signal']) & (~df['sell_signal']), 'signal_strength'] = 0.0
        
        return df
