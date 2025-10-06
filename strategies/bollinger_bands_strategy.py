"""
Bollinger Bands Strategy for PulseChain Trading Bot

This strategy uses Bollinger Bands for volatility breakout and mean reversion trading.
Combines both breakout signals (price moving outside bands) and mean reversion signals.
"""
import pandas as pd
import numpy as np
from typing import Dict
import logging

from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

class BollingerBandsStrategy(BaseStrategy):
    """
    Bollinger Bands Strategy
    
    Generates signals based on Bollinger Bands volatility and mean reversion patterns.
    Uses band width, price position relative to bands, and squeeze patterns.
    
    Parameters:
        period (int): Period for moving average and standard deviation (default: 20)
        std_dev (float): Standard deviation multiplier (default: 2.0)
        min_strength (float): Minimum signal strength to trigger trade (default: 0.6)
        squeeze_threshold (float): Band width threshold for squeeze detection (default: 0.02)
        breakout_confirmation (int): Periods to confirm breakout (default: 2)
        timeframe_minutes (int): Timeframe for analysis (default: 5)
    """
    
    def __init__(self, parameters: Dict = None):
        default_params = {
            'period': 20,
            'std_dev': 2.0,
            'min_strength': 0.6,
            'squeeze_threshold': 0.02,
            'breakout_confirmation': 2,
            'timeframe_minutes': 5
        }
        
        if parameters:
            default_params.update(parameters)
            
        super().__init__("BollingerBandsStrategy", default_params)
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Bollinger Bands and related indicators"""
        if not self.validate_data(data):
            return data
        
        df = data.copy()
        period = self.parameters['period']
        std_dev = self.parameters['std_dev']
        
        # Ensure we have enough data
        min_required = period + 10
        if len(df) < min_required:
            logger.warning(f"Not enough data for Bollinger Bands calculation ({min_required} required)")
            return df
        
        # Calculate middle band (SMA)
        df['bb_middle'] = df['price'].rolling(window=period).mean()
        
        # Calculate standard deviation
        df['bb_std'] = df['price'].rolling(window=period).std()
        
        # Calculate upper and lower bands
        df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * std_dev)
        df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * std_dev)
        
        # Calculate band width (volatility measure)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_width'] = df['bb_width'].fillna(0)
        
        # Calculate %B (price position within bands)
        df['bb_percent'] = (df['price'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        df['bb_percent'] = df['bb_percent'].fillna(0.5)
        
        # Price position relative to bands
        df['above_upper'] = df['price'] > df['bb_upper']
        df['below_lower'] = df['price'] < df['bb_lower']
        df['within_bands'] = (~df['above_upper']) & (~df['below_lower'])
        
        # Calculate band squeeze (low volatility)
        df['bb_squeeze'] = df['bb_width'] < self.parameters['squeeze_threshold']
        
        # Calculate band expansion (high volatility)
        df['bb_expanding'] = df['bb_width'] > df['bb_width'].shift(1)
        df['bb_contracting'] = df['bb_width'] < df['bb_width'].shift(1)
        
        # Calculate momentum indicators
        df['price_momentum'] = df['price'].pct_change()
        df['bb_middle_momentum'] = df['bb_middle'].pct_change()
        
        # Distance from bands (normalized)
        df['distance_upper'] = (df['bb_upper'] - df['price']) / df['price']
        df['distance_lower'] = (df['price'] - df['bb_lower']) / df['price']
        df['distance_middle'] = abs(df['price'] - df['bb_middle']) / df['price']
        
        # Breakout detection
        df['upper_breakout'] = (
            (df['price'] > df['bb_upper']) & 
            (df['price'].shift(1) <= df['bb_upper'].shift(1))
        )
        
        df['lower_breakout'] = (
            (df['price'] < df['bb_lower']) & 
            (df['price'].shift(1) >= df['bb_lower'].shift(1))
        )
        
        # Mean reversion signals
        df['reversion_from_upper'] = (
            (df['price'] < df['bb_upper']) & 
            (df['price'].shift(1) >= df['bb_upper'].shift(1))
        )
        
        df['reversion_from_lower'] = (
            (df['price'] > df['bb_lower']) & 
            (df['price'].shift(1) <= df['bb_lower'].shift(1))
        )
        
        # Consecutive periods outside bands
        df['periods_above_upper'] = 0
        df['periods_below_lower'] = 0
        
        for i in range(1, len(df)):
            if df.iloc[i]['above_upper']:
                df.iloc[i, df.columns.get_loc('periods_above_upper')] = (
                    df.iloc[i-1]['periods_above_upper'] + 1 if df.iloc[i-1]['above_upper'] else 1
                )
            
            if df.iloc[i]['below_lower']:
                df.iloc[i, df.columns.get_loc('periods_below_lower')] = (
                    df.iloc[i-1]['periods_below_lower'] + 1 if df.iloc[i-1]['below_lower'] else 1
                )
        
        # Volume surge detection (if volume data available)
        if 'volume' in df.columns:
            df['volume_sma'] = df['volume'].rolling(window=period).mean()
            df['volume_surge'] = df['volume'] > (df['volume_sma'] * 1.5)
        else:
            df['volume_surge'] = False
        
        return df
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate buy/sell signals based on Bollinger Bands analysis"""
        df = data.copy()
        
        if 'bb_upper' not in df.columns:
            df = self.calculate_indicators(df)
        
        # Initialize signal columns
        df['buy_signal'] = False
        df['sell_signal'] = False
        df['signal_strength'] = 0.0
        
        min_strength = self.parameters['min_strength']
        confirmation_periods = self.parameters['breakout_confirmation']
        
        # Breakout Strategy Signals
        # Buy on upward breakout with confirmation
        breakout_buy_conditions = (
            df['upper_breakout'] & 
            df['bb_expanding'] &  # Bands are expanding (volatility increasing)
            (df['price_momentum'] > 0) &  # Price momentum positive
            df['volume_surge']  # Volume confirmation
        )
        
        # Sell on downward breakout with confirmation
        breakout_sell_conditions = (
            df['lower_breakout'] & 
            df['bb_expanding'] &  # Bands are expanding
            (df['price_momentum'] < 0) &  # Price momentum negative
            df['volume_surge']  # Volume confirmation
        )
        
        # Mean Reversion Strategy Signals
        # Buy when price bounces from lower band
        reversion_buy_conditions = (
            df['reversion_from_lower'] & 
            (df['bb_percent'] < 0.2) &  # Price was in lower 20% of bands
            (df['periods_below_lower'] >= 2) &  # Was below for at least 2 periods
            (df['price_momentum'] > 0)  # Now showing upward momentum
        )
        
        # Sell when price falls from upper band  
        reversion_sell_conditions = (
            df['reversion_from_upper'] & 
            (df['bb_percent'] > 0.8) &  # Price was in upper 20% of bands
            (df['periods_above_upper'] >= 2) &  # Was above for at least 2 periods
            (df['price_momentum'] < 0)  # Now showing downward momentum
        )
        
        # Squeeze Breakout Signals (after low volatility)
        squeeze_buy_conditions = (
            df['upper_breakout'] & 
            df['bb_squeeze'].shift(1) &  # Previous period was squeeze
            df['bb_expanding'] &  # Bands now expanding
            (df['price'] > df['bb_middle'])  # Price above middle band
        )
        
        squeeze_sell_conditions = (
            df['lower_breakout'] & 
            df['bb_squeeze'].shift(1) &  # Previous period was squeeze
            df['bb_expanding'] &  # Bands now expanding
            (df['price'] < df['bb_middle'])  # Price below middle band
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
        # Base strength from band position and volatility
        volatility_strength = np.minimum(df['bb_width'] * 50, 1.0)  # Normalize volatility
        
        # Position strength (how far outside bands or from middle)
        position_strength = np.where(
            df['buy_signal'],
            np.maximum(
                1 - df['bb_percent'],  # Strength from lower band position
                df['distance_lower'] * 10  # Distance from lower band
            ),
            np.where(
                df['sell_signal'],
                np.maximum(
                    df['bb_percent'],  # Strength from upper band position  
                    df['distance_upper'] * 10  # Distance from upper band
                ),
                0
            )
        )
        
        # Momentum strength
        momentum_strength = np.minimum(abs(df['price_momentum']) * 100, 1.0)
        
        # Volume strength (if available)
        volume_strength = np.where(df['volume_surge'], 0.3, 0.0) if 'volume_surge' in df.columns else 0.0
        
        # Combine strength components
        df['signal_strength'] = np.minimum(
            (volatility_strength * 0.3 + 
             position_strength * 0.4 + 
             momentum_strength * 0.2 + 
             volume_strength * 0.1),
            1.0
        )
        
        # Apply minimum strength filter
        df.loc[df['signal_strength'] < min_strength, 'buy_signal'] = False
        df.loc[df['signal_strength'] < min_strength, 'sell_signal'] = False
        df.loc[(~df['buy_signal']) & (~df['sell_signal']), 'signal_strength'] = 0.0
        
        return df