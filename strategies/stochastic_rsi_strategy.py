"""
Stochastic RSI Strategy for PulseChain Trading Bot

This strategy combines Stochastic oscillator with RSI for enhanced signal accuracy.
Provides momentum analysis with overbought/oversold detection.
"""
import pandas as pd
import numpy as np
from typing import Dict
import logging

from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

class StochasticRSIStrategy(BaseStrategy):
    """
    Stochastic RSI Strategy
    
    Combines RSI with Stochastic oscillator for momentum and reversal signals.
    Uses %K and %D lines for signal generation with overbought/oversold levels.
    
    Parameters:
        rsi_period (int): Period for RSI calculation (default: 14)
        stoch_period (int): Period for Stochastic calculation (default: 14)
        k_smooth (int): %K smoothing periods (default: 3)
        d_smooth (int): %D smoothing periods (default: 3)
        oversold_level (float): Oversold threshold (default: 0.2)
        overbought_level (float): Overbought threshold (default: 0.8)
        min_strength (float): Minimum signal strength (default: 0.6)
        timeframe_minutes (int): Timeframe for analysis (default: 5)
    """
    
    def __init__(self, parameters: Dict = None):
        default_params = {
            'rsi_period': 14,
            'stoch_period': 14,
            'k_smooth': 3,
            'd_smooth': 3,
            'oversold_level': 0.2,
            'overbought_level': 0.8,
            'min_strength': 0.6,
            'timeframe_minutes': 5
        }
        
        if parameters:
            default_params.update(parameters)
            
        super().__init__("StochasticRSIStrategy", default_params)
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Stochastic RSI and related indicators"""
        if not self.validate_data(data):
            return data
        
        df = data.copy()
        rsi_period = self.parameters['rsi_period']
        stoch_period = self.parameters['stoch_period']
        k_smooth = self.parameters['k_smooth']
        d_smooth = self.parameters['d_smooth']
        
        # Ensure we have enough data
        min_required = max(rsi_period, stoch_period) + max(k_smooth, d_smooth) + 10
        if len(df) < min_required:
            logger.warning(f"Not enough data for Stochastic RSI calculation ({min_required} required)")
            return df
        
        # First calculate RSI
        df['price_change'] = df['price'].diff()
        df['gain'] = df['price_change'].where(df['price_change'] > 0, 0)
        df['loss'] = (-df['price_change']).where(df['price_change'] < 0, 0)
        
        # Calculate average gain and loss
        df['avg_gain'] = df['gain'].ewm(span=rsi_period, adjust=False).mean()
        df['avg_loss'] = df['loss'].ewm(span=rsi_period, adjust=False).mean()
        
        # Calculate RSI
        df['rs'] = df['avg_gain'] / df['avg_loss']
        df['rsi'] = 100 - (100 / (1 + df['rs']))
        df['rsi'] = df['rsi'].fillna(50)
        
        # Calculate Stochastic of RSI
        df['rsi_lowest'] = df['rsi'].rolling(window=stoch_period).min()
        df['rsi_highest'] = df['rsi'].rolling(window=stoch_period).max()
        
        # Calculate raw %K
        df['stoch_rsi_raw'] = (
            (df['rsi'] - df['rsi_lowest']) / 
            (df['rsi_highest'] - df['rsi_lowest'])
        ).fillna(0.5)
        
        # Smooth %K
        df['stoch_rsi_k'] = df['stoch_rsi_raw'].rolling(window=k_smooth).mean()
        
        # Calculate %D (moving average of %K)
        df['stoch_rsi_d'] = df['stoch_rsi_k'].rolling(window=d_smooth).mean()
        
        # Calculate momentum indicators
        df['k_momentum'] = df['stoch_rsi_k'].diff()
        df['d_momentum'] = df['stoch_rsi_d'].diff()
        df['kd_spread'] = df['stoch_rsi_k'] - df['stoch_rsi_d']
        
        # Crossover detection
        df['k_above_d'] = df['stoch_rsi_k'] > df['stoch_rsi_d']
        df['bullish_crossover'] = (
            (df['stoch_rsi_k'] > df['stoch_rsi_d']) & 
            (df['stoch_rsi_k'].shift(1) <= df['stoch_rsi_d'].shift(1))
        )
        df['bearish_crossover'] = (
            (df['stoch_rsi_k'] < df['stoch_rsi_d']) & 
            (df['stoch_rsi_k'].shift(1) >= df['stoch_rsi_d'].shift(1))
        )
        
        # Overbought/Oversold conditions
        oversold = self.parameters['oversold_level']
        overbought = self.parameters['overbought_level']
        
        df['oversold'] = (df['stoch_rsi_k'] < oversold) & (df['stoch_rsi_d'] < oversold)
        df['overbought'] = (df['stoch_rsi_k'] > overbought) & (df['stoch_rsi_d'] > overbought)
        
        # Divergence detection (simplified)
        price_direction = np.sign(df['price'].diff(5))
        stoch_direction = np.sign(df['stoch_rsi_k'].diff(5))
        df['bullish_divergence'] = (price_direction < 0) & (stoch_direction > 0) & df['oversold']
        df['bearish_divergence'] = (price_direction > 0) & (stoch_direction < 0) & df['overbought']
        
        # Calculate trend strength
        df['trend_strength'] = abs(df['stoch_rsi_k'] - 0.5) * 2  # 0 to 1 scale
        
        # RSI trend confirmation
        df['rsi_rising'] = df['rsi'] > df['rsi'].shift(3)
        df['rsi_falling'] = df['rsi'] < df['rsi'].shift(3)
        
        # Multi-timeframe confirmation (if enough data)
        if len(df) > 50:
            df['rsi_long_trend'] = df['rsi'].rolling(window=20).mean()
            df['long_trend_up'] = df['rsi'] > df['rsi_long_trend']
            df['long_trend_down'] = df['rsi'] < df['rsi_long_trend']
        else:
            df['long_trend_up'] = True
            df['long_trend_down'] = True
        
        return df
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate buy/sell signals based on Stochastic RSI analysis"""
        df = data.copy()
        
        if 'stoch_rsi_k' not in df.columns:
            df = self.calculate_indicators(df)
        
        # Initialize signal columns
        df['buy_signal'] = False
        df['sell_signal'] = False
        df['signal_strength'] = 0.0
        
        min_strength = self.parameters['min_strength']
        oversold = self.parameters['oversold_level']
        overbought = self.parameters['overbought_level']
        
        # Primary Buy Signals
        primary_buy_conditions = (
            df['bullish_crossover'] &  # %K crosses above %D
            (df['stoch_rsi_k'] < 0.5) &  # In lower half (coming from oversold)
            df['k_momentum'] > 0  # %K momentum positive
        )
        
        # Primary Sell Signals
        primary_sell_conditions = (
            df['bearish_crossover'] &  # %K crosses below %D
            (df['stoch_rsi_k'] > 0.5) &  # In upper half (coming from overbought)
            df['k_momentum'] < 0  # %K momentum negative
        )
        
        # Oversold Reversal Signals
        oversold_buy_conditions = (
            df['oversold'] & 
            (df['stoch_rsi_k'] > df['stoch_rsi_k'].shift(1)) &  # %K turning up
            (df['stoch_rsi_k'] > df['stoch_rsi_d']) &  # %K above %D
            df['rsi_rising']  # RSI also rising
        )
        
        # Overbought Reversal Signals
        overbought_sell_conditions = (
            df['overbought'] & 
            (df['stoch_rsi_k'] < df['stoch_rsi_k'].shift(1)) &  # %K turning down
            (df['stoch_rsi_k'] < df['stoch_rsi_d']) &  # %K below %D
            df['rsi_falling']  # RSI also falling
        )
        
        # Divergence Signals (higher quality)
        divergence_buy_conditions = (
            df['bullish_divergence'] & 
            df['bullish_crossover']  # Crossover confirmation
        )
        
        divergence_sell_conditions = (
            df['bearish_divergence'] & 
            df['bearish_crossover']  # Crossover confirmation
        )
        
        # Trend Continuation Signals
        trend_buy_conditions = (
            df['bullish_crossover'] & 
            df['long_trend_up'] &  # Long-term trend up
            (df['stoch_rsi_k'] > oversold) &  # Not in extreme oversold
            (df['stoch_rsi_k'] < 0.7)  # Not too overbought
        )
        
        trend_sell_conditions = (
            df['bearish_crossover'] & 
            df['long_trend_down'] &  # Long-term trend down
            (df['stoch_rsi_k'] < overbought) &  # Not in extreme overbought
            (df['stoch_rsi_k'] > 0.3)  # Not too oversold
        )
        
        # Combine all conditions
        df['buy_signal'] = (
            primary_buy_conditions | 
            oversold_buy_conditions | 
            divergence_buy_conditions |
            trend_buy_conditions
        )
        
        df['sell_signal'] = (
            primary_sell_conditions | 
            overbought_sell_conditions | 
            divergence_sell_conditions |
            trend_sell_conditions
        )
        
        # Calculate signal strength
        # Base strength from position and momentum
        position_strength = np.where(
            df['buy_signal'],
            (oversold + 0.1 - df['stoch_rsi_k']).clip(0, 1),  # Stronger when more oversold
            np.where(
                df['sell_signal'],
                (df['stoch_rsi_k'] - (overbought - 0.1)).clip(0, 1),  # Stronger when more overbought
                0
            )
        )
        
        # Momentum strength
        momentum_strength = np.minimum(
            abs(df['k_momentum']) * 10 + abs(df['d_momentum']) * 5,
            1.0
        )
        
        # Crossover strength (stronger when %K and %D are far apart)
        crossover_strength = np.minimum(abs(df['kd_spread']) * 2, 1.0)
        
        # Divergence bonus
        divergence_bonus = np.where(
            df['bullish_divergence'] | df['bearish_divergence'],
            0.3,
            0.0
        )
        
        # Trend alignment bonus
        trend_bonus = np.where(
            (df['buy_signal'] & df['long_trend_up']) |
            (df['sell_signal'] & df['long_trend_down']),
            0.2,
            0.0
        )
        
        # Combine strength components
        df['signal_strength'] = np.minimum(
            (position_strength * 0.35 + 
             momentum_strength * 0.25 + 
             crossover_strength * 0.15 + 
             divergence_bonus + 
             trend_bonus),
            1.0
        )
        
        # Apply minimum strength filter
        df.loc[df['signal_strength'] < min_strength, 'buy_signal'] = False
        df.loc[df['signal_strength'] < min_strength, 'sell_signal'] = False
        df.loc[(~df['buy_signal']) & (~df['sell_signal']), 'signal_strength'] = 0.0
        
        return df