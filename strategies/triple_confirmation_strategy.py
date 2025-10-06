"""
RSI + MACD + Bollinger Bands Hybrid Strategy for PulseChain Trading Bot

This strategy combines three powerful indicators for triple confirmation.
Uses momentum (RSI), trend (MACD), and volatility (Bollinger Bands) for high-probability signals.
"""
import pandas as pd
import numpy as np
from typing import Dict
import logging

from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

class TripleConfirmationStrategy(BaseStrategy):
    """
    RSI + MACD + Bollinger Bands Hybrid Strategy
    
    Triple confirmation system using momentum, trend, and volatility indicators.
    All three indicators must align for signal generation.
    
    Parameters:
        rsi_period (int): RSI calculation period (default: 14)
        rsi_oversold (float): RSI oversold level (default: 30)
        rsi_overbought (float): RSI overbought level (default: 70)
        macd_fast (int): MACD fast EMA period (default: 12)
        macd_slow (int): MACD slow EMA period (default: 26)
        macd_signal (int): MACD signal line period (default: 9)
        bb_period (int): Bollinger Bands period (default: 20)
        bb_std_dev (float): Bollinger Bands standard deviation (default: 2.0)
        min_strength (float): Minimum signal strength (default: 0.8)
        confirmation_periods (int): Periods to wait for confirmation (default: 2)
        timeframe_minutes (int): Timeframe for analysis (default: 5)
    """
    
    def __init__(self, parameters: Dict = None):
        default_params = {
            'rsi_period': 14,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'bb_period': 20,
            'bb_std_dev': 2.0,
            'min_strength': 0.8,
            'confirmation_periods': 2,
            'timeframe_minutes': 5
        }
        
        if parameters:
            default_params.update(parameters)
            
        super().__init__("TripleConfirmationStrategy", default_params)
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate RSI, MACD, and Bollinger Bands indicators"""
        if not self.validate_data(data):
            return data
        
        df = data.copy()
        
        # Get parameters
        rsi_period = self.parameters['rsi_period']
        rsi_oversold = self.parameters['rsi_oversold']
        rsi_overbought = self.parameters['rsi_overbought']
        macd_fast = self.parameters['macd_fast']
        macd_slow = self.parameters['macd_slow']
        macd_signal = self.parameters['macd_signal']
        bb_period = self.parameters['bb_period']
        bb_std_dev = self.parameters['bb_std_dev']
        
        # Ensure we have enough data
        min_required = max(rsi_period, macd_slow, bb_period) + 20
        if len(df) < min_required:
            logger.warning(f"Not enough data for Triple Confirmation calculation ({min_required} required)")
            return df
        
        # === RSI CALCULATION ===
        delta = df['price'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=rsi_period).mean()
        avg_loss = loss.rolling(window=rsi_period).mean()
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))
        df['rsi'] = df['rsi'].fillna(50)
        
        # RSI conditions
        df['rsi_oversold'] = df['rsi'] < rsi_oversold
        df['rsi_overbought'] = df['rsi'] > rsi_overbought
        df['rsi_bullish'] = df['rsi'] > 50
        df['rsi_bearish'] = df['rsi'] < 50
        df['rsi_rising'] = df['rsi'] > df['rsi'].shift(1)
        df['rsi_falling'] = df['rsi'] < df['rsi'].shift(1)
        
        # RSI divergence detection (simplified)
        price_direction = np.sign(df['price'].diff(5))
        rsi_direction = np.sign(df['rsi'].diff(5))
        df['rsi_bullish_divergence'] = (price_direction < 0) & (rsi_direction > 0) & df['rsi_oversold']
        df['rsi_bearish_divergence'] = (price_direction > 0) & (rsi_direction < 0) & df['rsi_overbought']
        
        # === MACD CALCULATION ===
        df['ema_fast'] = df['price'].ewm(span=macd_fast).mean()
        df['ema_slow'] = df['price'].ewm(span=macd_slow).mean()
        df['macd'] = df['ema_fast'] - df['ema_slow']
        df['macd_signal'] = df['macd'].ewm(span=macd_signal).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # MACD conditions
        df['macd_bullish'] = df['macd'] > df['macd_signal']
        df['macd_bearish'] = df['macd'] < df['macd_signal']
        df['macd_above_zero'] = df['macd'] > 0
        df['macd_below_zero'] = df['macd'] < 0
        
        # MACD crossovers
        df['macd_bullish_cross'] = (
            (df['macd'] > df['macd_signal']) & 
            (df['macd'].shift(1) <= df['macd_signal'].shift(1))
        )
        df['macd_bearish_cross'] = (
            (df['macd'] < df['macd_signal']) & 
            (df['macd'].shift(1) >= df['macd_signal'].shift(1))
        )
        
        # MACD histogram analysis
        df['macd_histogram_rising'] = df['macd_histogram'] > df['macd_histogram'].shift(1)
        df['macd_histogram_falling'] = df['macd_histogram'] < df['macd_histogram'].shift(1)
        
        # === BOLLINGER BANDS CALCULATION ===
        df['bb_middle'] = df['price'].rolling(window=bb_period).mean()
        df['bb_std'] = df['price'].rolling(window=bb_period).std()
        df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * bb_std_dev)
        df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * bb_std_dev)
        
        # Bollinger Bands conditions
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['price'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        df['bb_position'] = df['bb_position'].fillna(0.5)
        
        df['price_above_bb_upper'] = df['price'] > df['bb_upper']
        df['price_below_bb_lower'] = df['price'] < df['bb_lower']
        df['price_near_bb_upper'] = abs(df['price'] - df['bb_upper']) / df['price'] < 0.02
        df['price_near_bb_lower'] = abs(df['price'] - df['bb_lower']) / df['price'] < 0.02
        
        # Bollinger Band squeeze and expansion
        df['bb_squeeze'] = df['bb_width'] < df['bb_width'].rolling(window=20).mean() * 0.8
        df['bb_expansion'] = df['bb_width'] > df['bb_width'].shift(1)
        
        # Bollinger Band reversals
        df['bb_reversal_from_lower'] = (
            (df['price'] > df['bb_lower']) & 
            (df['price'].shift(1) <= df['bb_lower'].shift(1))
        )
        df['bb_reversal_from_upper'] = (
            (df['price'] < df['bb_upper']) & 
            (df['price'].shift(1) >= df['bb_upper'].shift(1))
        )
        
        # === TRIPLE CONFIRMATION ANALYSIS ===
        
        # Bullish confluence conditions
        df['rsi_bullish_condition'] = (
            df['rsi_oversold'] | 
            (df['rsi_bullish'] & df['rsi_rising']) |
            df['rsi_bullish_divergence']
        )
        
        df['macd_bullish_condition'] = (
            df['macd_bullish_cross'] | 
            (df['macd_bullish'] & df['macd_histogram_rising']) |
            (df['macd_above_zero'] & df['macd_bullish'])
        )
        
        df['bb_bullish_condition'] = (
            df['bb_reversal_from_lower'] |
            (df['price_near_bb_lower'] & (df['bb_position'] < 0.3)) |
            (df['bb_expansion'] & (df['price'] > df['bb_middle']))
        )
        
        # Bearish confluence conditions
        df['rsi_bearish_condition'] = (
            df['rsi_overbought'] |
            (df['rsi_bearish'] & df['rsi_falling']) |
            df['rsi_bearish_divergence']
        )
        
        df['macd_bearish_condition'] = (
            df['macd_bearish_cross'] |
            (df['macd_bearish'] & df['macd_histogram_falling']) |
            (df['macd_below_zero'] & df['macd_bearish'])
        )
        
        df['bb_bearish_condition'] = (
            df['bb_reversal_from_upper'] |
            (df['price_near_bb_upper'] & (df['bb_position'] > 0.7)) |
            (df['bb_expansion'] & (df['price'] < df['bb_middle']))
        )
        
        # Triple confirmation scores
        df['bullish_confirmation_score'] = (
            df['rsi_bullish_condition'].astype(int) +
            df['macd_bullish_condition'].astype(int) +
            df['bb_bullish_condition'].astype(int)
        )
        
        df['bearish_confirmation_score'] = (
            df['rsi_bearish_condition'].astype(int) +
            df['macd_bearish_condition'].astype(int) +
            df['bb_bearish_condition'].astype(int)
        )
        
        # Strong confluence (all three indicators agree)
        df['strong_bullish_confluence'] = df['bullish_confirmation_score'] >= 3
        df['strong_bearish_confluence'] = df['bearish_confirmation_score'] >= 3
        
        # Partial confluence (at least 2 out of 3 indicators)
        df['partial_bullish_confluence'] = df['bullish_confirmation_score'] >= 2
        df['partial_bearish_confluence'] = df['bearish_confirmation_score'] >= 2
        
        # === ADDITIONAL CONFIRMATION FACTORS ===
        
        # Volume confirmation (if available)
        if 'volume' in df.columns:
            df['volume_ma'] = df['volume'].rolling(window=20).mean()
            df['volume_surge'] = df['volume'] > (df['volume_ma'] * 1.5)
        else:
            df['volume_surge'] = False
        
        # Momentum confirmation
        df['price_momentum'] = df['price'].pct_change(3)
        df['momentum_bullish'] = df['price_momentum'] > 0
        df['momentum_bearish'] = df['price_momentum'] < 0
        
        # Trend context
        df['sma_50'] = df['price'].rolling(window=50).mean()
        df['uptrend_context'] = df['price'] > df['sma_50']
        df['downtrend_context'] = df['price'] < df['sma_50']
        
        return df
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate buy/sell signals based on triple confirmation"""
        df = data.copy()
        
        if 'strong_bullish_confluence' not in df.columns:
            df = self.calculate_indicators(df)
        
        # Initialize signal columns
        df['buy_signal'] = False
        df['sell_signal'] = False
        df['signal_strength'] = 0.0
        
        min_strength = self.parameters['min_strength']
        confirmation_periods = self.parameters['confirmation_periods']
        
        # === PRIMARY SIGNALS (All 3 indicators must agree) ===
        
        # Strong buy signals - Perfect confluence
        strong_buy_signals = (
            df['strong_bullish_confluence'] &  # All 3 indicators bullish
            df['uptrend_context'] &  # Overall uptrend
            (df['rsi'] < 60)  # Not too overbought
        )
        
        # Strong sell signals - Perfect confluence
        strong_sell_signals = (
            df['strong_bearish_confluence'] &  # All 3 indicators bearish
            df['downtrend_context'] &  # Overall downtrend
            (df['rsi'] > 40)  # Not too oversold
        )
        
        # === SECONDARY SIGNALS (2 out of 3 indicators + confirmation) ===
        
        # Partial buy with strong momentum
        partial_buy_signals = (
            df['partial_bullish_confluence'] &  # 2 out of 3 bullish
            df['momentum_bullish'] &  # Price momentum bullish
            df['volume_surge'] &  # Volume confirmation
            (df['bullish_confirmation_score'] >= 2)
        )
        
        # Partial sell with strong momentum
        partial_sell_signals = (
            df['partial_bearish_confluence'] &  # 2 out of 3 bearish
            df['momentum_bearish'] &  # Price momentum bearish
            df['volume_surge'] &  # Volume confirmation
            (df['bearish_confirmation_score'] >= 2)
        )
        
        # === SPECIAL OPPORTUNITY SIGNALS ===
        
        # Divergence-based signals (high quality)
        divergence_buy_signals = (
            df['rsi_bullish_divergence'] &  # RSI divergence
            df['macd_bullish_cross'] &  # MACD confirmation
            df['bb_reversal_from_lower'] &  # BB reversal
            df['volume_surge']  # Volume confirmation
        )
        
        divergence_sell_signals = (
            df['rsi_bearish_divergence'] &  # RSI divergence
            df['macd_bearish_cross'] &  # MACD confirmation
            df['bb_reversal_from_upper'] &  # BB reversal
            df['volume_surge']  # Volume confirmation
        )
        
        # Squeeze breakout signals
        squeeze_buy_signals = (
            df['bb_squeeze'].shift(1) &  # Previous period was squeeze
            df['bb_expansion'] &  # Now expanding
            df['macd_bullish_cross'] &  # MACD bullish crossover
            df['rsi_rising'] &  # RSI rising
            (df['price'] > df['bb_middle'])  # Price above middle band
        )
        
        squeeze_sell_signals = (
            df['bb_squeeze'].shift(1) &  # Previous period was squeeze
            df['bb_expansion'] &  # Now expanding
            df['macd_bearish_cross'] &  # MACD bearish crossover
            df['rsi_falling'] &  # RSI falling
            (df['price'] < df['bb_middle'])  # Price below middle band
        )
        
        # === COMBINE ALL SIGNALS ===
        
        df['buy_signal'] = (
            strong_buy_signals |
            partial_buy_signals |
            divergence_buy_signals |
            squeeze_buy_signals
        )
        
        df['sell_signal'] = (
            strong_sell_signals |
            partial_sell_signals |
            divergence_sell_signals |
            squeeze_sell_signals
        )
        
        # === SIGNAL STRENGTH CALCULATION ===
        
        # Base strength from confirmation score
        confirmation_strength = np.where(
            df['buy_signal'],
            df['bullish_confirmation_score'] / 3.0,
            np.where(
                df['sell_signal'],
                df['bearish_confirmation_score'] / 3.0,
                0.0
            )
        )
        
        # Perfect confluence bonus
        perfect_confluence_bonus = np.where(
            (df['strong_bullish_confluence'] & df['buy_signal']) |
            (df['strong_bearish_confluence'] & df['sell_signal']),
            0.3,
            0.0
        )
        
        # Divergence bonus (high-quality signals)
        divergence_bonus = np.where(
            (df['rsi_bullish_divergence'] & df['buy_signal']) |
            (df['rsi_bearish_divergence'] & df['sell_signal']),
            0.2,
            0.0
        )
        
        # Volume confirmation bonus
        volume_bonus = np.where(df['volume_surge'], 0.1, 0.0)
        
        # Momentum alignment bonus
        momentum_bonus = np.where(
            (df['momentum_bullish'] & df['buy_signal']) |
            (df['momentum_bearish'] & df['sell_signal']),
            0.1,
            0.0
        )
        
        # Trend alignment bonus
        trend_bonus = np.where(
            (df['uptrend_context'] & df['buy_signal']) |
            (df['downtrend_context'] & df['sell_signal']),
            0.1,
            0.0
        )
        
        # Squeeze breakout bonus
        squeeze_bonus = np.where(
            squeeze_buy_signals | squeeze_sell_signals,
            0.15,
            0.0
        )
        
        # Combine all strength components
        df['signal_strength'] = np.minimum(
            (confirmation_strength * 0.4 +
             perfect_confluence_bonus +
             divergence_bonus +
             volume_bonus +
             momentum_bonus +
             trend_bonus +
             squeeze_bonus),
            1.0
        )
        
        # Apply minimum strength filter
        df.loc[df['signal_strength'] < min_strength, 'buy_signal'] = False
        df.loc[df['signal_strength'] < min_strength, 'sell_signal'] = False
        df.loc[(~df['buy_signal']) & (~df['sell_signal']), 'signal_strength'] = 0.0
        
        return df