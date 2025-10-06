"""
Volume + Price Action + Moving Average Hybrid Strategy for PulseChain Trading Bot

This strategy combines volume analysis with price action patterns and trend confirmation.
Uses volume spikes, candlestick patterns, and moving average alignment for signals.
"""
import pandas as pd
import numpy as np
from typing import Dict
import logging

from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

class VolumePriceActionStrategy(BaseStrategy):
    """
    Volume + Price Action + Moving Average Hybrid Strategy
    
    Combines volume analysis with price structure and trend confirmation.
    Looks for volume confirmation with price action patterns and MA alignment.
    
    Parameters:
        volume_ma_period (int): Volume moving average period (default: 20)
        volume_spike_threshold (float): Volume spike multiplier (default: 2.0)
        ma_fast_period (int): Fast moving average period (default: 20)
        ma_slow_period (int): Slow moving average period (default: 50)
        ma_trend_period (int): Trend filter MA period (default: 200)
        price_action_periods (int): Periods for price action analysis (default: 3)
        min_strength (float): Minimum signal strength (default: 0.6)
        trend_alignment_weight (float): Weight for trend alignment (default: 0.3)
        timeframe_minutes (int): Timeframe for analysis (default: 5)
    """
    
    def __init__(self, parameters: Dict = None):
        default_params = {
            'volume_ma_period': 20,
            'volume_spike_threshold': 2.0,
            'ma_fast_period': 20,
            'ma_slow_period': 50,
            'ma_trend_period': 200,
            'price_action_periods': 3,
            'min_strength': 0.6,
            'trend_alignment_weight': 0.3,
            'timeframe_minutes': 5
        }
        
        if parameters:
            default_params.update(parameters)
            
        super().__init__("VolumePriceActionStrategy", default_params)
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume, price action, and moving average indicators"""
        if not self.validate_data(data):
            return data
        
        df = data.copy()
        
        # Get parameters
        vol_ma_period = self.parameters['volume_ma_period']
        vol_spike_threshold = self.parameters['volume_spike_threshold']
        ma_fast = self.parameters['ma_fast_period']
        ma_slow = self.parameters['ma_slow_period']
        ma_trend = self.parameters['ma_trend_period']
        pa_periods = self.parameters['price_action_periods']
        
        # Ensure we have enough data
        min_required = max(vol_ma_period, ma_trend) + 20
        if len(df) < min_required:
            logger.warning(f"Not enough data for Volume Price Action calculation ({min_required} required)")
            return df
        
        # === VOLUME ANALYSIS ===
        
        # Handle missing volume data
        if 'volume' not in df.columns:
            # Create synthetic volume based on price volatility
            df['volume'] = abs(df['price'].pct_change()) * 1000000
            logger.info("Volume data not available, using synthetic volume based on price volatility")
        
        # Volume indicators
        df['volume_ma'] = df['volume'].rolling(window=vol_ma_period).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        df['volume_spike'] = df['volume_ratio'] > vol_spike_threshold
        df['volume_dry_up'] = df['volume_ratio'] < 0.5
        
        # Volume trend
        df['volume_increasing'] = df['volume'] > df['volume'].shift(1)
        df['volume_decreasing'] = df['volume'] < df['volume'].shift(1)
        
        # Accumulation/Distribution approximation
        df['close_position'] = (df['price'] - df['low']) / (df['high'] - df['low'])
        df['close_position'] = df['close_position'].fillna(0.5)
        df['volume_pressure'] = (df['close_position'] - 0.5) * df['volume']
        df['accumulation_distribution'] = df['volume_pressure'].cumsum()
        df['ad_trending_up'] = df['accumulation_distribution'] > df['accumulation_distribution'].shift(5)
        
        # === MOVING AVERAGE ANALYSIS ===
        
        # Moving averages
        df['ma_fast'] = df['price'].rolling(window=ma_fast).mean()
        df['ma_slow'] = df['price'].rolling(window=ma_slow).mean()
        df['ma_trend'] = df['price'].rolling(window=ma_trend).mean()
        
        # EMA for faster response
        df['ema_fast'] = df['price'].ewm(span=ma_fast).mean()
        df['ema_slow'] = df['price'].ewm(span=ma_slow).mean()
        
        # Moving average conditions
        df['ma_bullish_alignment'] = (df['ma_fast'] > df['ma_slow']) & (df['ma_slow'] > df['ma_trend'])
        df['ma_bearish_alignment'] = (df['ma_fast'] < df['ma_slow']) & (df['ma_slow'] < df['ma_trend'])
        df['ma_fast_cross_up'] = (df['ma_fast'] > df['ma_slow']) & (df['ma_fast'].shift(1) <= df['ma_slow'].shift(1))
        df['ma_fast_cross_down'] = (df['ma_fast'] < df['ma_slow']) & (df['ma_fast'].shift(1) >= df['ma_slow'].shift(1))
        
        # Price position relative to MAs
        df['price_above_all_mas'] = (df['price'] > df['ma_fast']) & (df['price'] > df['ma_slow']) & (df['price'] > df['ma_trend'])
        df['price_below_all_mas'] = (df['price'] < df['ma_fast']) & (df['price'] < df['ma_slow']) & (df['price'] < df['ma_trend'])
        
        # === PRICE ACTION ANALYSIS ===
        
        # Basic OHLC calculations
        if 'open' not in df.columns:
            df['open'] = df['price'].shift(1)
        if 'high' not in df.columns:
            df['high'] = df[['price', 'open']].max(axis=1)
        if 'low' not in df.columns:
            df['low'] = df[['price', 'open']].min(axis=1)
        
        # Candlestick body and shadows
        df['body_size'] = abs(df['price'] - df['open'])
        df['upper_shadow'] = df['high'] - df[['price', 'open']].max(axis=1)
        df['lower_shadow'] = df[['price', 'open']].min(axis=1) - df['low']
        df['total_range'] = df['high'] - df['low']
        
        # Avoid division by zero
        df['total_range'] = df['total_range'].replace(0, 0.001)
        
        # Body and shadow ratios
        df['body_ratio'] = df['body_size'] / df['total_range']
        df['upper_shadow_ratio'] = df['upper_shadow'] / df['total_range']
        df['lower_shadow_ratio'] = df['lower_shadow'] / df['total_range']
        
        # Candlestick patterns
        df['bullish_candle'] = df['price'] > df['open']
        df['bearish_candle'] = df['price'] < df['open']
        df['doji'] = df['body_ratio'] < 0.1
        
        # Hammer and hanging man (long lower shadow, small body)
        df['hammer_pattern'] = (
            (df['lower_shadow_ratio'] > 0.6) & 
            (df['body_ratio'] < 0.3) & 
            (df['upper_shadow_ratio'] < 0.1)
        )
        
        # Shooting star (long upper shadow, small body)
        df['shooting_star_pattern'] = (
            (df['upper_shadow_ratio'] > 0.6) & 
            (df['body_ratio'] < 0.3) & 
            (df['lower_shadow_ratio'] < 0.1)
        )
        
        # Engulfing patterns
        df['bullish_engulfing'] = (
            df['bullish_candle'] & 
            df['bearish_candle'].shift(1) & 
            (df['body_size'] > df['body_size'].shift(1) * 1.2) &
            (df['price'] > df['open'].shift(1)) &
            (df['open'] < df['price'].shift(1))
        )
        
        df['bearish_engulfing'] = (
            df['bearish_candle'] & 
            df['bullish_candle'].shift(1) & 
            (df['body_size'] > df['body_size'].shift(1) * 1.2) &
            (df['price'] < df['open'].shift(1)) &
            (df['open'] > df['price'].shift(1))
        )
        
        # Price action momentum
        df['strong_bullish_candle'] = df['bullish_candle'] & (df['body_ratio'] > 0.7)
        df['strong_bearish_candle'] = df['bearish_candle'] & (df['body_ratio'] > 0.7)
        
        # Support and resistance levels (simplified)
        df['local_high'] = df['high'].rolling(window=pa_periods*2+1, center=True).max() == df['high']
        df['local_low'] = df['low'].rolling(window=pa_periods*2+1, center=True).min() == df['low']
        
        # Breakout detection
        df['breakout_above'] = (
            (df['price'] > df['high'].rolling(window=pa_periods).max().shift(1)) &
            df['volume_spike']
        )
        df['breakdown_below'] = (
            (df['price'] < df['low'].rolling(window=pa_periods).min().shift(1)) &
            df['volume_spike']
        )
        
        # === COMBINED ANALYSIS ===
        
        # Volume + Price Action combinations
        df['bullish_volume_pa'] = (
            (df['hammer_pattern'] | df['bullish_engulfing'] | df['strong_bullish_candle']) &
            df['volume_spike'] &
            df['ad_trending_up']
        )
        
        df['bearish_volume_pa'] = (
            (df['shooting_star_pattern'] | df['bearish_engulfing'] | df['strong_bearish_candle']) &
            df['volume_spike'] &
            (~df['ad_trending_up'])
        )
        
        # Momentum indicators
        df['price_momentum'] = df['price'].pct_change(pa_periods)
        df['momentum_bullish'] = df['price_momentum'] > 0.01
        df['momentum_bearish'] = df['price_momentum'] < -0.01
        
        # Volatility
        df['volatility'] = df['price'].rolling(window=10).std() / df['price']
        df['low_volatility'] = df['volatility'] < df['volatility'].rolling(window=20).mean() * 0.8
        df['high_volatility'] = df['volatility'] > df['volatility'].rolling(window=20).mean() * 1.5
        
        return df
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate buy/sell signals based on volume, price action, and MA analysis"""
        df = data.copy()
        
        if 'bullish_volume_pa' not in df.columns:
            df = self.calculate_indicators(df)
        
        # Initialize signal columns
        df['buy_signal'] = False
        df['sell_signal'] = False
        df['signal_strength'] = 0.0
        
        min_strength = self.parameters['min_strength']
        trend_weight = self.parameters['trend_alignment_weight']
        
        # === BUY SIGNALS ===
        
        # Primary buy: Volume + Price Action + Trend alignment
        primary_buy = (
            df['bullish_volume_pa'] &  # Bullish volume + price action
            df['ma_bullish_alignment'] &  # MA trend bullish
            df['price_above_all_mas']  # Price above all MAs
        )
        
        # Breakout buy with volume
        breakout_buy = (
            df['breakout_above'] &  # Price breakout with volume
            df['ma_fast_cross_up'] &  # MA crossover confirmation
            df['momentum_bullish'] &  # Momentum confirmation
            (df['price'] > df['ma_trend'])  # Above long-term trend
        )
        
        # Pullback buy in uptrend
        pullback_buy = (
            df['ma_bullish_alignment'] &  # Overall uptrend
            df['hammer_pattern'] &  # Hammer pattern (reversal)
            df['volume_spike'] &  # Volume confirmation
            (df['price'] > df['ma_fast'] * 0.98) &  # Near fast MA support
            df['ad_trending_up']  # Accumulation
        )
        
        # Volume surge with bullish pattern
        volume_surge_buy = (
            df['volume_spike'] &  # Volume surge
            (df['bullish_engulfing'] | df['strong_bullish_candle']) &  # Strong bullish pattern
            (df['price'] > df['ma_slow']) &  # Above medium-term trend
            df['momentum_bullish']  # Momentum confirmation
        )
        
        # === SELL SIGNALS ===
        
        # Primary sell: Volume + Price Action + Trend alignment
        primary_sell = (
            df['bearish_volume_pa'] &  # Bearish volume + price action
            df['ma_bearish_alignment'] &  # MA trend bearish
            df['price_below_all_mas']  # Price below all MAs
        )
        
        # Breakdown sell with volume
        breakdown_sell = (
            df['breakdown_below'] &  # Price breakdown with volume
            df['ma_fast_cross_down'] &  # MA crossover confirmation
            df['momentum_bearish'] &  # Momentum confirmation
            (df['price'] < df['ma_trend'])  # Below long-term trend
        )
        
        # Pullback sell in downtrend
        pullback_sell = (
            df['ma_bearish_alignment'] &  # Overall downtrend
            df['shooting_star_pattern'] &  # Shooting star (reversal)
            df['volume_spike'] &  # Volume confirmation
            (df['price'] < df['ma_fast'] * 1.02) &  # Near fast MA resistance
            (~df['ad_trending_up'])  # Distribution
        )
        
        # Volume surge with bearish pattern
        volume_surge_sell = (
            df['volume_spike'] &  # Volume surge
            (df['bearish_engulfing'] | df['strong_bearish_candle']) &  # Strong bearish pattern
            (df['price'] < df['ma_slow']) &  # Below medium-term trend
            df['momentum_bearish']  # Momentum confirmation
        )
        
        # === COMBINE SIGNALS ===
        
        df['buy_signal'] = (
            primary_buy | breakout_buy | pullback_buy | volume_surge_buy
        )
        
        df['sell_signal'] = (
            primary_sell | breakdown_sell | pullback_sell | volume_surge_sell
        )
        
        # === SIGNAL STRENGTH CALCULATION ===
        
        # Base strength components
        volume_strength = np.minimum(df['volume_ratio'] / 3.0, 1.0)  # Volume spike strength
        
        # Price action strength
        pa_strength = 0.0
        pa_strength += np.where(df['hammer_pattern'] | df['shooting_star_pattern'], 0.3, 0.0)
        pa_strength += np.where(df['bullish_engulfing'] | df['bearish_engulfing'], 0.4, 0.0)
        pa_strength += np.where(df['strong_bullish_candle'] | df['strong_bearish_candle'], 0.3, 0.0)
        pa_strength += np.where(df['breakout_above'] | df['breakdown_below'], 0.3, 0.0)
        
        # Trend alignment strength
        trend_strength = np.where(
            (df['ma_bullish_alignment'] & df['buy_signal']) |
            (df['ma_bearish_alignment'] & df['sell_signal']),
            1.0,
            np.where(
                df['price_above_all_mas'] | df['price_below_all_mas'],
                0.7,
                0.3
            )
        )
        
        # Momentum strength
        momentum_strength = np.minimum(abs(df['price_momentum']) * 20, 1.0)
        
        # Accumulation/Distribution strength
        ad_strength = np.where(
            (df['ad_trending_up'] & df['buy_signal']) |
            ((~df['ad_trending_up']) & df['sell_signal']),
            0.3,
            0.0
        )
        
        # Combine strength components
        df['signal_strength'] = np.where(
            df['buy_signal'] | df['sell_signal'],
            np.minimum(
                (volume_strength * 0.3 +
                 pa_strength * 0.25 +
                 trend_strength * trend_weight +
                 momentum_strength * 0.1 +
                 ad_strength * 0.05),
                1.0
            ),
            0.0
        )
        
        # Apply minimum strength filter
        df.loc[df['signal_strength'] < min_strength, 'buy_signal'] = False
        df.loc[df['signal_strength'] < min_strength, 'sell_signal'] = False
        df.loc[(~df['buy_signal']) & (~df['sell_signal']), 'signal_strength'] = 0.0
        
        return df