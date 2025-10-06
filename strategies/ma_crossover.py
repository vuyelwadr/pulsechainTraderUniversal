"""
Moving Average Crossover Strategy for PulseChain Trading Bot

This strategy generates buy signals when the short-term moving average crosses above 
the long-term moving average, and sell signals when it crosses below.
"""
import pandas as pd
import numpy as np
from typing import Dict
import logging

from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

class MovingAverageCrossover(BaseStrategy):
    """
    Moving Average Crossover Strategy
    
    Generates signals based on the crossover of short and long period moving averages.
    
    Parameters:
        short_period (int): Period for short-term moving average (default: 10)
        long_period (int): Period for long-term moving average (default: 30)
        ma_type (str): Type of moving average - 'sma' or 'ema' (default: 'ema')
        min_strength (float): Minimum signal strength to trigger trade (default: 0.6)
    """
    
    def __init__(self, parameters: Dict = None):
        default_params = {
            'short_period': 10,
            'long_period': 30,
            'ma_type': 'ema',  # 'sma' for Simple MA, 'ema' for Exponential MA
            'min_strength': 0.6
        }
        
        if parameters:
            default_params.update(parameters)
            
        super().__init__("MovingAverageCrossover", default_params)
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate moving averages and related indicators"""
        if not self.validate_data(data):
            return data
        
        df = data.copy()
        short_period = self.parameters['short_period']
        long_period = self.parameters['long_period']
        ma_type = self.parameters['ma_type']
        
        # Calculate moving averages even on shorter clips; SMA uses min_periods to avoid NaNs
        if ma_type == 'ema':
            df['ma_short'] = df['price'].ewm(span=max(1, short_period), adjust=False).mean()
            df['ma_long'] = df['price'].ewm(span=max(1, long_period), adjust=False).mean()
        else:  # sma
            df['ma_short'] = df['price'].rolling(window=max(1, short_period), min_periods=1).mean()
            df['ma_long'] = df['price'].rolling(window=max(1, long_period), min_periods=1).mean()

        # Calculate additional indicators
        df['ma_diff'] = df['ma_short'] - df['ma_long']
        df['ma_diff_pct'] = (df['ma_diff'] / df['ma_long']) * 100
        
        # Calculate momentum (rate of change of MA difference)
        df['ma_momentum'] = df['ma_diff'].diff()
        
        # Calculate signal strength based on multiple factors
        df['price_vs_ma_short'] = (df['price'] - df['ma_short']) / df['ma_short']
        df['price_vs_ma_long'] = (df['price'] - df['ma_long']) / df['ma_long']
        
        return df
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate buy/sell signals based on MA crossover"""
        df = data.copy()
        
        for col in ('ma_short', 'ma_long'):
            if col not in df.columns:
                logger.error("Moving averages not calculated")
                return df

        # Drop leading rows where long MA is still NaN to keep signals clean
        df = df.dropna(subset=['ma_long']).reset_index(drop=True)
        
        # Initialize signal columns
        df['buy_signal'] = False
        df['sell_signal'] = False
        df['signal_strength'] = 0.0
        df['signal_type'] = 'hold'
        
        # Find crossover points
        df['ma_crossover'] = np.where(
            (df['ma_short'] > df['ma_long']) & (df['ma_short'].shift(1) <= df['ma_long'].shift(1)),
            1,  # Bullish crossover
            np.where(
                (df['ma_short'] < df['ma_long']) & (df['ma_short'].shift(1) >= df['ma_long'].shift(1)),
                -1,  # Bearish crossover
                0   # No crossover
            )
        )
        
        min_strength = self.parameters['min_strength']
        
        for i in range(1, len(df)):
            crossover = df.iloc[i]['ma_crossover']
            
            if crossover == 1:  # Bullish crossover - BUY signal
                # Calculate signal strength based on multiple factors
                strength = self._calculate_buy_strength(df, i)
                
                if strength >= min_strength:
                    df.iloc[i, df.columns.get_loc('buy_signal')] = True
                    df.iloc[i, df.columns.get_loc('signal_strength')] = strength
                    df.iloc[i, df.columns.get_loc('signal_type')] = 'buy'
            
            elif crossover == -1:  # Bearish crossover - SELL signal
                # Calculate signal strength
                strength = self._calculate_sell_strength(df, i)
                
                if strength >= min_strength:
                    df.iloc[i, df.columns.get_loc('sell_signal')] = True
                    df.iloc[i, df.columns.get_loc('signal_strength')] = strength
                    df.iloc[i, df.columns.get_loc('signal_type')] = 'sell'
        
        return df
    
    def _calculate_buy_strength(self, df: pd.DataFrame, index: int) -> float:
        """Calculate the strength of a buy signal"""
        try:
            current = df.iloc[index]
            
            # Base strength from crossover
            strength = 0.5
            
            # Factor 1: How much above the long MA is the price? (max +0.3)
            price_above_long = current['price_vs_ma_long']
            if price_above_long > 0:
                strength += min(0.3, price_above_long * 10)
            
            # Factor 2: Momentum of MA difference (max +0.2)
            if 'ma_momentum' in df.columns and not pd.isna(current['ma_momentum']):
                if current['ma_momentum'] > 0:
                    momentum_factor = min(0.2, abs(current['ma_momentum']) / current['ma_long'] * 100)
                    strength += momentum_factor
            
            # Factor 3: How separated are the MAs? (max +0.2)
            ma_separation = abs(current['ma_diff_pct'])
            if ma_separation > 0.1:  # At least 0.1% separation
                separation_factor = min(0.2, ma_separation / 5.0)  # 5% separation = max factor
                strength += separation_factor
            
            # Factor 4: Recent price trend (max +0.1)
            if index >= 5:  # Need some history
                recent_prices = df.iloc[index-4:index+1]['price']
                if len(recent_prices) >= 2:
                    trend = (recent_prices.iloc[-1] - recent_prices.iloc[0]) / recent_prices.iloc[0]
                    if trend > 0:
                        strength += min(0.1, trend * 5)  # 2% trend = max factor
            
            return min(1.0, strength)  # Cap at 1.0
            
        except Exception as e:
            logger.warning(f"Error calculating buy strength: {e}")
            return 0.6  # Default strength
    
    def _calculate_sell_strength(self, df: pd.DataFrame, index: int) -> float:
        """Calculate the strength of a sell signal"""
        try:
            current = df.iloc[index]
            
            # Base strength from crossover
            strength = 0.5
            
            # Factor 1: How much below the long MA is the price? (max +0.3)
            price_below_long = current['price_vs_ma_long']
            if price_below_long < 0:
                strength += min(0.3, abs(price_below_long) * 10)
            
            # Factor 2: Momentum of MA difference (max +0.2)
            if 'ma_momentum' in df.columns and not pd.isna(current['ma_momentum']):
                if current['ma_momentum'] < 0:
                    momentum_factor = min(0.2, abs(current['ma_momentum']) / current['ma_long'] * 100)
                    strength += momentum_factor
            
            # Factor 3: How separated are the MAs? (max +0.2)
            ma_separation = abs(current['ma_diff_pct'])
            if ma_separation > 0.1:
                separation_factor = min(0.2, ma_separation / 5.0)
                strength += separation_factor
            
            # Factor 4: Recent price trend (max +0.1)
            if index >= 5:
                recent_prices = df.iloc[index-4:index+1]['price']
                if len(recent_prices) >= 2:
                    trend = (recent_prices.iloc[-1] - recent_prices.iloc[0]) / recent_prices.iloc[0]
                    if trend < 0:
                        strength += min(0.1, abs(trend) * 5)
            
            return min(1.0, strength)
            
        except Exception as e:
            logger.warning(f"Error calculating sell strength: {e}")
            return 0.6
    
    def get_current_position_recommendation(self, data: pd.DataFrame) -> Dict:
        """Get current position recommendation with detailed analysis"""
        if data.empty:
            return {'recommendation': 'hold', 'reason': 'No data available'}
        
        # Get data with indicators
        df_with_indicators = self.calculate_indicators(data)
        df_with_signals = self.generate_signals(df_with_indicators)
        
        if df_with_signals.empty:
            return {'recommendation': 'hold', 'reason': 'No signals generated'}
        
        latest = df_with_signals.iloc[-1]
        
        recommendation = {
            'recommendation': latest.get('signal_type', 'hold'),
            'signal_strength': latest.get('signal_strength', 0.0),
            'current_price': latest['price'],
            'ma_short': latest.get('ma_short', 0),
            'ma_long': latest.get('ma_long', 0),
            'ma_diff_pct': latest.get('ma_diff_pct', 0),
            'price_vs_ma_short': latest.get('price_vs_ma_short', 0) * 100,  # Convert to percentage
            'price_vs_ma_long': latest.get('price_vs_ma_long', 0) * 100,
            'parameters': self.parameters
        }
        
        # Add reasoning
        if latest.get('buy_signal', False):
            recommendation['reason'] = f"Bullish MA crossover detected with {latest['signal_strength']:.2f} strength"
        elif latest.get('sell_signal', False):
            recommendation['reason'] = f"Bearish MA crossover detected with {latest['signal_strength']:.2f} strength"
        else:
            if latest.get('ma_short', 0) > latest.get('ma_long', 0):
                recommendation['reason'] = "In bullish trend but no crossover signal"
            else:
                recommendation['reason'] = "In bearish trend but no crossover signal"
        
        return recommendation
