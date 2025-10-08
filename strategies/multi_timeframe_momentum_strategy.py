"""
Multi-Timeframe Momentum Strategy for PulseChain Trading Bot

This strategy uses multiple timeframe alignment for high-probability entries.
Combines higher timeframe bias with lower timeframe precision timing.
"""
import pandas as pd
import numpy as np
from typing import Dict
import logging

from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

class MultiTimeframeMomentumStrategy(BaseStrategy):
    """
    Multi-Timeframe Momentum Strategy
    
    Uses multiple timeframe alignment for high-probability entries.
    Analyzes trend on higher timeframe and times entries on lower timeframe.
    
    Parameters:
        higher_tf_multiplier (int): Higher timeframe multiplier (default: 12)
        trend_ma_period (int): Moving average for trend identification (default: 50)
        momentum_rsi_period (int): RSI period for momentum (default: 14)
        momentum_macd_fast (int): MACD fast EMA (default: 12)
        momentum_macd_slow (int): MACD slow EMA (default: 26)
        momentum_macd_signal (int): MACD signal line (default: 9)
        min_strength (float): Minimum signal strength (default: 0.7)
        trend_strength_threshold (float): Minimum trend strength (default: 0.02)
        timeframe_minutes (int): Base timeframe for analysis (default: 5)
    """
    
    def __init__(self, parameters: Dict = None):
        default_params = {
            'higher_tf_multiplier': 12,  # 12x base timeframe (5min -> 1h)
            'trend_ma_period': 50,
            'momentum_rsi_period': 14,
            'momentum_macd_fast': 12,
            'momentum_macd_slow': 26,
            'momentum_macd_signal': 9,
            'min_strength': 0.7,
            'trend_strength_threshold': 0.02,
            'timeframe_minutes': 5
        }
        
        if parameters:
            default_params.update(parameters)
            
        super().__init__("MultiTimeframeMomentumStrategy", default_params)
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate multi-timeframe momentum indicators"""
        if not self.validate_data(data):
            return data
        
        df = data.copy()
        
        # Get parameters
        htf_mult = self.parameters['higher_tf_multiplier']
        trend_ma = self.parameters['trend_ma_period']
        rsi_period = self.parameters['momentum_rsi_period']
        macd_fast = self.parameters['momentum_macd_fast']
        macd_slow = self.parameters['momentum_macd_slow']
        macd_signal = self.parameters['momentum_macd_signal']
        
        # Ensure we have enough data
        min_required = max(trend_ma * htf_mult, macd_slow * htf_mult) + 20
        if len(df) < min_required:
            logger.warning(f"Not enough data for Multi-Timeframe calculation ({min_required} required)")
            return df
        
        # === HIGHER TIMEFRAME ANALYSIS ===
        # Create higher timeframe data by resampling
        htf_data = self._resample_to_higher_timeframe(df, htf_mult)
        
        # Higher timeframe trend indicators
        htf_data['htf_ma_trend'] = htf_data['price'].rolling(window=trend_ma//htf_mult).mean()
        htf_data['htf_ma_fast'] = htf_data['price'].rolling(window=20//htf_mult).mean()
        htf_data['htf_ma_slow'] = htf_data['price'].rolling(window=50//htf_mult).mean()
        
        # Higher timeframe trend direction
        htf_data['htf_uptrend'] = htf_data['htf_ma_fast'] > htf_data['htf_ma_slow']
        htf_data['htf_downtrend'] = htf_data['htf_ma_fast'] < htf_data['htf_ma_slow']
        htf_data['htf_trend_strength'] = abs(htf_data['htf_ma_fast'] - htf_data['htf_ma_slow']) / htf_data['price']
        
        # Higher timeframe momentum
        htf_data['htf_rsi'] = self._calculate_rsi(htf_data['price'], rsi_period//htf_mult)
        htf_data['htf_momentum'] = htf_data['price'].pct_change(5//htf_mult)
        
        # Expand higher timeframe data back to original timeframe
        df = self._expand_htf_to_base_timeframe(df, htf_data, htf_mult)
        
        # === BASE TIMEFRAME ANALYSIS ===
        
        # Moving averages for current timeframe
        df['ma_20'] = df['price'].rolling(window=20).mean()
        df['ma_50'] = df['price'].rolling(window=50).mean()
        df['ema_20'] = df['price'].ewm(span=20).mean()
        
        # RSI for momentum
        df['rsi'] = self._calculate_rsi(df['price'], rsi_period)
        
        # MACD for momentum confirmation
        df['ema_fast'] = df['price'].ewm(span=macd_fast).mean()
        df['ema_slow'] = df['price'].ewm(span=macd_slow).mean()
        df['macd'] = df['ema_fast'] - df['ema_slow']
        df['macd_signal'] = df['macd'].ewm(span=macd_signal).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # MACD crossovers
        df['macd_bullish_cross'] = (
            (df['macd'] > df['macd_signal']) & 
            (df['macd'].shift(1) <= df['macd_signal'].shift(1))
        )
        df['macd_bearish_cross'] = (
            (df['macd'] < df['macd_signal']) & 
            (df['macd'].shift(1) >= df['macd_signal'].shift(1))
        )
        
        # Base timeframe momentum
        df['price_momentum'] = df['price'].pct_change(3)
        df['momentum_acceleration'] = df['price_momentum'].diff()
        
        # Pullback detection
        df['pullback_in_uptrend'] = (
            df['htf_uptrend'] &  # Higher timeframe uptrend
            (df['price'] < df['price'].rolling(window=10).max()) &  # Price below recent high
            (df['price'] > df['ma_20'])  # But still above short-term MA
        )
        
        df['pullback_in_downtrend'] = (
            df['htf_downtrend'] &  # Higher timeframe downtrend  
            (df['price'] > df['price'].rolling(window=10).min()) &  # Price above recent low
            (df['price'] < df['ma_20'])  # But still below short-term MA
        )
        
        # Support/Resistance at moving averages
        df['at_ma_20'] = abs(df['price'] - df['ma_20']) / df['price'] < 0.01
        df['at_ema_20'] = abs(df['price'] - df['ema_20']) / df['price'] < 0.01
        
        # Volume analysis (if available)
        if 'volume' in df.columns:
            df['volume_ma'] = df['volume'].rolling(window=20).mean()
            df['volume_surge'] = df['volume'] > (df['volume_ma'] * 1.5)
        else:
            df['volume_surge'] = False
        
        # === TIMEFRAME ALIGNMENT ANALYSIS ===
        
        # Trend alignment score
        df['trend_alignment'] = 0.0
        
        # Higher timeframe trend contribution
        df['trend_alignment'] += np.where(df['htf_uptrend'], 0.4, 0.0)
        df['trend_alignment'] += np.where(df['htf_downtrend'], -0.4, 0.0)
        
        # Base timeframe trend contribution
        df['trend_alignment'] += np.where(df['ma_20'] > df['ma_50'], 0.3, -0.3)
        
        # Momentum alignment
        df['momentum_alignment'] = 0.0
        
        # RSI alignment (both timeframes)
        df['momentum_alignment'] += np.where(
            (df['rsi'] > 50) & (df['htf_rsi'] > 50), 0.25,
            np.where((df['rsi'] < 50) & (df['htf_rsi'] < 50), -0.25, 0.0)
        )
        
        # MACD alignment
        df['momentum_alignment'] += np.where(df['macd'] > df['macd_signal'], 0.25, -0.25)
        
        # Price momentum alignment
        df['momentum_alignment'] += np.where(
            (df['price_momentum'] > 0) & (df['htf_momentum'] > 0), 0.25,
            np.where((df['price_momentum'] < 0) & (df['htf_momentum'] < 0), -0.25, 0.0)
        )
        
        # Overall signal strength from alignment
        df['timeframe_alignment'] = df['trend_alignment'] + df['momentum_alignment']
        df['strong_bullish_alignment'] = df['timeframe_alignment'] > 0.7
        df['strong_bearish_alignment'] = df['timeframe_alignment'] < -0.7
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
    
    def _resample_to_higher_timeframe(self, df: pd.DataFrame, multiplier: int) -> pd.DataFrame:
        """Resample data to higher timeframe"""
        if len(df) < multiplier:
            return df.copy()
        
        # Group data by higher timeframe periods
        htf_data = []
        for i in range(0, len(df), multiplier):
            end_idx = min(i + multiplier, len(df))
            period_data = df.iloc[i:end_idx]
            
            if len(period_data) > 0:
                htf_row = {
                    'price': period_data['price'].iloc[-1],  # Close price
                    'high': period_data['high'].max() if 'high' in period_data.columns else period_data['price'].max(),
                    'low': period_data['low'].min() if 'low' in period_data.columns else period_data['price'].min(),
                    'open': period_data['price'].iloc[0],  # Open price
                    'timestamp': period_data.index[-1] if hasattr(period_data.index, 'name') else i
                }
                htf_data.append(htf_row)
        
        return pd.DataFrame(htf_data)
    
    def _expand_htf_to_base_timeframe(self, base_df: pd.DataFrame, 
                                     htf_df: pd.DataFrame, multiplier: int) -> pd.DataFrame:
        """Expand higher timeframe data back to base timeframe"""
        df = base_df.copy()
        
        # Initialize higher timeframe columns
        htf_columns = [col for col in htf_df.columns if col.startswith('htf_')]
        for col in htf_columns:
            df[col] = np.nan
        
        # Fill higher timeframe data
        for i, htf_row in htf_df.iterrows():
            start_idx = i * multiplier
            end_idx = min(start_idx + multiplier, len(df))
            
            for col in htf_columns:
                if col in htf_row and not pd.isna(htf_row[col]):
                    value = htf_row[col]
                    if isinstance(value, bool):
                        value = float(value)
                    df.iloc[start_idx:end_idx, df.columns.get_loc(col)] = value

        # Forward fill any remaining NaN values
        df[htf_columns] = df[htf_columns].ffill().infer_objects(copy=False)
        
        return df
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate buy/sell signals based on multi-timeframe momentum"""
        df = data.copy()
        
        if 'timeframe_alignment' not in df.columns:
            df = self.calculate_indicators(df)
        
        # Initialize signal columns
        df['buy_signal'] = False
        df['sell_signal'] = False
        df['signal_strength'] = 0.0
        
        min_strength = self.parameters['min_strength']
        trend_threshold = self.parameters['trend_strength_threshold']
        
        # === BUY SIGNALS ===
        
        # Primary buy: Strong bullish alignment + pullback entry
        primary_buy = (
            df['strong_bullish_alignment'] &  # Strong multi-timeframe bullish alignment
            df['pullback_in_uptrend'] &  # Price pulling back in uptrend
            (df['at_ma_20'] | df['at_ema_20']) &  # Price at moving average support
            df['macd_bullish_cross'] &  # MACD momentum confirmation
            (df['htf_trend_strength'] > trend_threshold)  # Strong higher timeframe trend
        )
        
        # Secondary buy: Momentum alignment + oversold bounce
        momentum_buy = (
            df['htf_uptrend'] &  # Higher timeframe uptrend
            (df['momentum_alignment'] > 0.3) &  # Good momentum alignment
            (df['rsi'] < 40) &  # Oversold on base timeframe
            (df['rsi'] > df['rsi'].shift(1)) &  # RSI turning up
            (df['macd_histogram'] > df['macd_histogram'].shift(1))  # MACD histogram improving
        )
        
        # Volume-confirmed buy
        volume_buy = (
            df['htf_uptrend'] &  # Higher timeframe uptrend
            df['macd_bullish_cross'] &  # MACD crossover
            df['volume_surge'] &  # Volume confirmation
            (df['timeframe_alignment'] > 0.5)  # Good overall alignment
        )
        
        # === SELL SIGNALS ===
        
        # Primary sell: Strong bearish alignment + pullback entry
        primary_sell = (
            df['strong_bearish_alignment'] &  # Strong multi-timeframe bearish alignment
            df['pullback_in_downtrend'] &  # Price bouncing in downtrend
            (df['at_ma_20'] | df['at_ema_20']) &  # Price at moving average resistance
            df['macd_bearish_cross'] &  # MACD momentum confirmation
            (df['htf_trend_strength'] > trend_threshold)  # Strong higher timeframe trend
        )
        
        # Secondary sell: Momentum alignment + overbought rejection
        momentum_sell = (
            df['htf_downtrend'] &  # Higher timeframe downtrend
            (df['momentum_alignment'] < -0.3) &  # Good bearish momentum alignment
            (df['rsi'] > 60) &  # Overbought on base timeframe
            (df['rsi'] < df['rsi'].shift(1)) &  # RSI turning down
            (df['macd_histogram'] < df['macd_histogram'].shift(1))  # MACD histogram deteriorating
        )
        
        # Volume-confirmed sell
        volume_sell = (
            df['htf_downtrend'] &  # Higher timeframe downtrend
            df['macd_bearish_cross'] &  # MACD crossover
            df['volume_surge'] &  # Volume confirmation
            (df['timeframe_alignment'] < -0.5)  # Good overall bearish alignment
        )
        
        # Combine conditions
        df['buy_signal'] = primary_buy | momentum_buy | volume_buy
        df['sell_signal'] = primary_sell | momentum_sell | volume_sell
        
        # === SIGNAL STRENGTH CALCULATION ===
        
        # Base strength from timeframe alignment
        alignment_strength = np.minimum(abs(df['timeframe_alignment']), 1.0)
        
        # Trend strength component
        trend_strength_component = np.minimum(df['htf_trend_strength'] * 20, 1.0)
        
        # Momentum confirmation strength
        momentum_strength = np.minimum(abs(df['momentum_alignment']) * 2, 1.0)
        
        # MACD strength
        macd_strength = np.where(
            df['macd_bullish_cross'] | df['macd_bearish_cross'],
            np.minimum(abs(df['macd_histogram']) * 100, 1.0),
            0.0
        )
        
        # Volume confirmation strength
        volume_strength = np.where(df['volume_surge'], 0.2, 0.0)
        
        # Pullback timing strength (better entries on pullbacks)
        pullback_strength = np.where(
            df['pullback_in_uptrend'] | df['pullback_in_downtrend'],
            0.3,
            0.0
        )
        
        # Combine strength components
        df['signal_strength'] = np.where(
            df['buy_signal'] | df['sell_signal'],
            np.minimum(
                (alignment_strength * 0.3 +
                 trend_strength_component * 0.25 +
                 momentum_strength * 0.2 +
                 macd_strength * 0.15 +
                 pullback_strength * 0.05 +
                 volume_strength * 0.05),
                1.0
            ),
            0.0
        )
        
        # Apply minimum strength filter
        df.loc[df['signal_strength'] < min_strength, 'buy_signal'] = False
        df.loc[df['signal_strength'] < min_strength, 'sell_signal'] = False
        df.loc[(~df['buy_signal']) & (~df['sell_signal']), 'signal_strength'] = 0.0
        
        return df
