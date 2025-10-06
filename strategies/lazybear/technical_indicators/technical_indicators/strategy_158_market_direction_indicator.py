"""
LazyBear Strategy #158: Market Direction Indicator (MDI)
Based on: https://www.tradingview.com/v/wGAhnKHS/

The Market Direction Indicator (MDI) is a trend-following indicator that compares 
moving averages of different lengths to determine market direction and momentum.

MDI Formula:
1. Calculate CP2 = (len1*sum(src, len2-1) - len2*sum(src, len1-1)) / (len2-len1)
2. MDI = 100 * (nz(CP2[1]) - CP2) / ((src + src[1]) / 2)

The indicator oscillates around zero:
- Positive values indicate upward trend
- Negative values indicate downward trend
- Values within cutoff range are considered neutral
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Tuple
from strategies.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

class MarketDirectionIndicatorStrategy(BaseStrategy):
    """
    LazyBear Strategy #158: Market Direction Indicator
    
    A trend-following strategy that uses Market Direction Indicator (MDI)
    to determine trend direction and generate trading signals.
    """
    
    def __init__(self, parameters: Dict = None):
        default_params = {
            'short_length': 13,      # Short MA length (len1)
            'long_length': 55,       # Long MA length (len2)
            'cutoff': 2.0,          # Neutral zone cutoff
            'signal_threshold': 0.7, # Minimum signal strength to trade
            'timeframe_minutes': 15  # 15-minute timeframe
        }
        
        if parameters:
            default_params.update(parameters)
            
        super().__init__("LazyBear_MDI_Strategy_158", default_params)
    
    def _calculate_cp2(self, src: pd.Series, len1: int, len2: int) -> pd.Series:
        """
        Calculate CP2 component of MDI
        
        Formula: (len1*sum(src, len2-1) - len2*sum(src, len1-1)) / (len2-len1)
        
        Args:
            src: Price series (typically close prices)
            len1: Short length
            len2: Long length
            
        Returns:
            CP2 series
        """
        if len2 <= len1:
            logger.error(f"Long length ({len2}) must be greater than short length ({len1})")
            return pd.Series(np.nan, index=src.index)
        
        # Calculate rolling sums
        sum_len2_minus_1 = src.rolling(window=len2-1, min_periods=len2-1).sum()
        sum_len1_minus_1 = src.rolling(window=len1-1, min_periods=len1-1).sum()
        
        # Calculate CP2
        numerator = len1 * sum_len2_minus_1 - len2 * sum_len1_minus_1
        denominator = len2 - len1
        
        cp2 = numerator / denominator
        return cp2
    
    def _calculate_mdi(self, src: pd.Series, len1: int, len2: int) -> pd.Series:
        """
        Calculate Market Direction Indicator
        
        Formula: 100 * (nz(CP2[1]) - CP2) / ((src + src[1]) / 2)
        
        Args:
            src: Price series
            len1: Short length
            len2: Long length
            
        Returns:
            MDI series
        """
        # Calculate CP2
        cp2 = self._calculate_cp2(src, len1, len2)
        
        # Get previous CP2 values (shift by 1)
        cp2_prev = cp2.shift(1)
        
        # Calculate average of current and previous price
        price_avg = (src + src.shift(1)) / 2
        
        # Avoid division by zero
        price_avg = price_avg.replace(0, np.nan)
        
        # Calculate MDI
        mdi = 100 * (cp2_prev.fillna(0) - cp2) / price_avg
        
        return mdi
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Market Direction Indicator and related metrics
        
        Args:
            data: Price data with OHLCV columns
            
        Returns:
            DataFrame with MDI indicators added
        """
        if data.empty:
            logger.warning("Empty data provided to calculate_indicators")
            return data.copy()
        
        df = data.copy()
        
        # Ensure we have the price column
        if 'close' in df.columns:
            price_column = 'close'
        elif 'price' in df.columns:
            price_column = 'price'
        else:
            logger.error("No 'close' or 'price' column found in data")
            return df
        
        # Get parameters
        short_length = self.parameters['short_length']
        long_length = self.parameters['long_length']
        cutoff = self.parameters['cutoff']
        
        # Calculate MDI
        df['mdi'] = self._calculate_mdi(df[price_column], short_length, long_length)
        
        # Calculate trend direction based on MDI
        df['mdi_trend'] = np.where(
            df['mdi'] > cutoff, 1,      # Bullish
            np.where(df['mdi'] < -cutoff, -1, 0)  # Bearish or Neutral
        )
        
        # Calculate signal strength based on MDI absolute value
        df['mdi_strength'] = np.abs(df['mdi']) / 100.0  # Normalize to 0-1 range
        df['mdi_strength'] = np.clip(df['mdi_strength'], 0, 1)
        
        # Smooth the strength to avoid noise
        df['mdi_strength_smooth'] = df['mdi_strength'].rolling(window=3).mean()
        
        # Additional momentum indicators
        df['mdi_momentum'] = df['mdi'] - df['mdi'].shift(1)
        df['mdi_acceleration'] = df['mdi_momentum'] - df['mdi_momentum'].shift(1)
        
        # Trend persistence (how long current trend has been active)
        trend_changes = (df['mdi_trend'] != df['mdi_trend'].shift(1)).astype(int)
        df['trend_persistence'] = trend_changes.cumsum() - trend_changes.cumsum().where(trend_changes == 1).ffill().fillna(0)
        
        return df
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate buy/sell signals based on MDI
        
        Signal Rules:
        - BUY: MDI crosses above cutoff with positive momentum
        - SELL: MDI crosses below -cutoff with negative momentum
        - Signal strength based on MDI absolute value and momentum
        
        Args:
            data: DataFrame with MDI indicators
            
        Returns:
            DataFrame with buy_signal, sell_signal, and signal_strength columns
        """
        if data.empty or 'mdi' not in data.columns:
            logger.warning("MDI indicators not found in data")
            df = data.copy()
            df['buy_signal'] = False
            df['sell_signal'] = False
            df['signal_strength'] = 0.0
            return df
        
        df = data.copy()
        cutoff = self.parameters['cutoff']
        signal_threshold = self.parameters['signal_threshold']
        
        # Initialize signal columns
        df['buy_signal'] = False
        df['sell_signal'] = False
        df['signal_strength'] = 0.0
        
        # Ensure we have enough data for signals
        min_periods = max(self.parameters['short_length'], self.parameters['long_length']) + 5
        if len(df) < min_periods:
            logger.warning(f"Insufficient data for MDI signals. Need at least {min_periods} periods, got {len(df)}")
            return df
        
        # Get trend and momentum information
        mdi_trend = df['mdi_trend'].fillna(0)
        mdi_trend_prev = mdi_trend.shift(1).fillna(0)
        mdi_momentum = df['mdi_momentum'].fillna(0)
        mdi_strength = df['mdi_strength_smooth'].fillna(0)
        
        # Generate BUY signals
        # Buy when MDI crosses above cutoff (trend changes from 0/-1 to 1) with positive momentum
        buy_condition = (
            (mdi_trend == 1) &                    # Currently bullish
            (mdi_trend_prev <= 0) &               # Previously neutral or bearish
            (mdi_momentum > 0) &                  # Positive momentum
            (mdi_strength >= signal_threshold)    # Strong enough signal
        )
        
        # Generate SELL signals
        # Sell when MDI crosses below -cutoff (trend changes from 0/1 to -1) with negative momentum
        sell_condition = (
            (mdi_trend == -1) &                   # Currently bearish
            (mdi_trend_prev >= 0) &               # Previously neutral or bullish
            (mdi_momentum < 0) &                  # Negative momentum
            (mdi_strength >= signal_threshold)    # Strong enough signal
        )
        
        # Apply signal conditions
        df.loc[buy_condition, 'buy_signal'] = True
        df.loc[sell_condition, 'sell_signal'] = True
        
        # Calculate signal strength for active signals
        # Combine MDI strength with momentum strength
        momentum_strength = np.abs(df['mdi_momentum']) / 10.0  # Scale momentum
        momentum_strength = np.clip(momentum_strength, 0, 0.5)  # Cap at 0.5
        
        combined_strength = mdi_strength + momentum_strength
        combined_strength = np.clip(combined_strength, 0, 1)
        
        # Set signal strength only for active signals
        df.loc[buy_condition | sell_condition, 'signal_strength'] = combined_strength[buy_condition | sell_condition]
        
        # Add signal metadata for analysis
        df['mdi_signal_type'] = np.where(
            buy_condition, 'BUY',
            np.where(sell_condition, 'SELL', 'HOLD')
        )
        
        # Log signal statistics
        buy_count = buy_condition.sum()
        sell_count = sell_condition.sum()
        avg_strength = df.loc[df['signal_strength'] > 0, 'signal_strength'].mean()
        
        if buy_count > 0 or sell_count > 0:
            logger.info(f"MDI Strategy signals - BUY: {buy_count}, SELL: {sell_count}, Avg Strength: {avg_strength:.3f}")
        
        return df
    
    def get_strategy_info(self) -> Dict:
        """Get detailed strategy information"""
        return {
            'name': self.name,
            'description': 'LazyBear Market Direction Indicator Strategy #158',
            'type': 'Trend Following',
            'timeframe': f"{self.parameters['timeframe_minutes']} minutes",
            'parameters': {
                'short_length': {
                    'value': self.parameters['short_length'],
                    'description': 'Short moving average period'
                },
                'long_length': {
                    'value': self.parameters['long_length'],
                    'description': 'Long moving average period'
                },
                'cutoff': {
                    'value': self.parameters['cutoff'],
                    'description': 'Neutral zone cutoff threshold'
                },
                'signal_threshold': {
                    'value': self.parameters['signal_threshold'],
                    'description': 'Minimum signal strength to execute trades'
                }
            },
            'indicators': [
                'Market Direction Indicator (MDI)',
                'MDI Trend Direction',
                'MDI Signal Strength',
                'MDI Momentum',
                'Trend Persistence'
            ],
            'signals': {
                'buy': 'MDI crosses above cutoff with positive momentum',
                'sell': 'MDI crosses below -cutoff with negative momentum'
            }
        }
    
    def validate_parameters(self) -> bool:
        """Validate strategy parameters"""
        try:
            short_length = self.parameters['short_length']
            long_length = self.parameters['long_length']
            cutoff = self.parameters['cutoff']
            signal_threshold = self.parameters['signal_threshold']
            
            # Validation checks
            if short_length <= 0 or long_length <= 0:
                logger.error("Length parameters must be positive")
                return False
            
            if long_length <= short_length:
                logger.error("Long length must be greater than short length")
                return False
            
            if cutoff < 0:
                logger.error("Cutoff must be non-negative")
                return False
            
            if not (0 <= signal_threshold <= 1):
                logger.error("Signal threshold must be between 0 and 1")
                return False
            
            return True
            
        except KeyError as e:
            logger.error(f"Missing required parameter: {e}")
            return False
        except Exception as e:
            logger.error(f"Parameter validation error: {e}")
            return False