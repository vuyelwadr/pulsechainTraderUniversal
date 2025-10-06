"""
RSI (Relative Strength Index) Strategy for PulseChain Trading Bot

This strategy uses RSI to identify overbought/oversold conditions.
Buys when RSI is oversold, sells when RSI is overbought.
"""
import pandas as pd
import numpy as np
from typing import Dict
import logging

from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

class RSIStrategy(BaseStrategy):
    """
    RSI Strategy
    
    Generates signals based on RSI overbought/oversold levels.
    
    Parameters:
        rsi_period (int): Period for RSI calculation (default: 14)
        oversold_level (float): RSI level considered oversold (default: 30)
        overbought_level (float): RSI level considered overbought (default: 70)
        min_strength (float): Minimum signal strength to trigger trade (default: 0.6)
    """
    
    def __init__(self, parameters: Dict = None):
        default_params = {
            'rsi_period': 14,
            'oversold_level': 30.0,
            'overbought_level': 70.0,
            'min_strength': 0.6
        }
        
        if parameters:
            default_params.update(parameters)
            
        super().__init__("RSIStrategy", default_params)
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate RSI and related indicators"""
        if not self.validate_data(data):
            return data
        
        df = data.copy()
        rsi_period = self.parameters['rsi_period']
        
        # Ensure we have enough data
        if len(df) < rsi_period + 1:
            logger.warning(f"Not enough data for RSI calculation ({rsi_period + 1} required)")
            return df
        
        # Calculate price changes
        df['price_change'] = df['price'].diff()
        
        # Calculate gains and losses
        df['gain'] = df['price_change'].where(df['price_change'] > 0, 0)
        df['loss'] = (-df['price_change']).where(df['price_change'] < 0, 0)
        
        # Calculate average gain and loss using EMA
        df['avg_gain'] = df['gain'].ewm(span=rsi_period, adjust=False).mean()
        df['avg_loss'] = df['loss'].ewm(span=rsi_period, adjust=False).mean()
        
        # Calculate RS and RSI
        df['rs'] = df['avg_gain'] / df['avg_loss']
        df['rsi'] = 100 - (100 / (1 + df['rs']))
        
        # Handle division by zero
        df['rsi'] = df['rsi'].fillna(50)  # Neutral RSI when no data
        
        # Calculate RSI rate of change (momentum)
        df['rsi_momentum'] = df['rsi'].diff()
        
        # Calculate distance from key levels
        oversold = self.parameters['oversold_level']
        overbought = self.parameters['overbought_level']
        
        df['distance_from_oversold'] = df['rsi'] - oversold
        df['distance_from_overbought'] = overbought - df['rsi']
        
        return df
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate buy/sell signals based on RSI"""
        df = data.copy()
        
        if 'rsi' not in df.columns:
            logger.error("RSI not calculated")
            return df
        
        # Initialize signal columns
        df['buy_signal'] = False
        df['sell_signal'] = False
        df['signal_strength'] = 0.0
        df['signal_type'] = 'hold'
        
        oversold_level = self.parameters['oversold_level']
        overbought_level = self.parameters['overbought_level']
        min_strength = self.parameters['min_strength']
        
        for i in range(1, len(df)):
            current_rsi = df.iloc[i]['rsi']
            prev_rsi = df.iloc[i-1]['rsi']
            
            # Buy signal: RSI crosses above oversold level from below
            if current_rsi > oversold_level and prev_rsi <= oversold_level:
                strength = self._calculate_buy_strength(df, i)
                
                if strength >= min_strength:
                    df.iloc[i, df.columns.get_loc('buy_signal')] = True
                    df.iloc[i, df.columns.get_loc('signal_strength')] = strength
                    df.iloc[i, df.columns.get_loc('signal_type')] = 'buy'
            
            # Sell signal: RSI crosses below overbought level from above
            elif current_rsi < overbought_level and prev_rsi >= overbought_level:
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
            rsi = current['rsi']
            oversold_level = self.parameters['oversold_level']
            
            # Base strength from RSI level
            strength = 0.5
            
            # Factor 1: How oversold was it? (max +0.3)
            if rsi < oversold_level:
                oversold_factor = (oversold_level - rsi) / oversold_level  # 0-1
                strength += min(0.3, oversold_factor * 0.3)
            
            # Factor 2: RSI momentum (max +0.2)
            if 'rsi_momentum' in df.columns and not pd.isna(current['rsi_momentum']):
                if current['rsi_momentum'] > 0:  # RSI increasing
                    momentum_factor = min(5, abs(current['rsi_momentum'])) / 5  # 0-1
                    strength += momentum_factor * 0.2
            
            # Factor 3: Price trend confirmation (max +0.2)
            if index >= 3:
                recent_prices = df.iloc[index-2:index+1]['price']
                if len(recent_prices) >= 2:
                    price_trend = (recent_prices.iloc[-1] - recent_prices.iloc[0]) / recent_prices.iloc[0]
                    if price_trend > 0:  # Price increasing
                        strength += min(0.2, price_trend * 10)  # 2% price increase = max factor
            
            return min(1.0, strength)
            
        except Exception as e:
            logger.warning(f"Error calculating buy strength: {e}")
            return 0.6
    
    def _calculate_sell_strength(self, df: pd.DataFrame, index: int) -> float:
        """Calculate the strength of a sell signal"""
        try:
            current = df.iloc[index]
            rsi = current['rsi']
            overbought_level = self.parameters['overbought_level']
            
            # Base strength from RSI level
            strength = 0.5
            
            # Factor 1: How overbought was it? (max +0.3)
            if rsi > overbought_level:
                overbought_factor = (rsi - overbought_level) / (100 - overbought_level)  # 0-1
                strength += min(0.3, overbought_factor * 0.3)
            
            # Factor 2: RSI momentum (max +0.2)
            if 'rsi_momentum' in df.columns and not pd.isna(current['rsi_momentum']):
                if current['rsi_momentum'] < 0:  # RSI decreasing
                    momentum_factor = min(5, abs(current['rsi_momentum'])) / 5  # 0-1
                    strength += momentum_factor * 0.2
            
            # Factor 3: Price trend confirmation (max +0.2)
            if index >= 3:
                recent_prices = df.iloc[index-2:index+1]['price']
                if len(recent_prices) >= 2:
                    price_trend = (recent_prices.iloc[-1] - recent_prices.iloc[0]) / recent_prices.iloc[0]
                    if price_trend < 0:  # Price decreasing
                        strength += min(0.2, abs(price_trend) * 10)
            
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
            'rsi': latest.get('rsi', 50),
            'rsi_level': self._get_rsi_description(latest.get('rsi', 50)),
            'oversold_level': self.parameters['oversold_level'],
            'overbought_level': self.parameters['overbought_level'],
            'parameters': self.parameters
        }
        
        # Add reasoning
        rsi = latest.get('rsi', 50)
        if latest.get('buy_signal', False):
            recommendation['reason'] = f"RSI oversold signal detected (RSI: {rsi:.1f})"
        elif latest.get('sell_signal', False):
            recommendation['reason'] = f"RSI overbought signal detected (RSI: {rsi:.1f})"
        else:
            if rsi < self.parameters['oversold_level']:
                recommendation['reason'] = f"RSI oversold ({rsi:.1f}) but no signal yet"
            elif rsi > self.parameters['overbought_level']:
                recommendation['reason'] = f"RSI overbought ({rsi:.1f}) but no signal yet"
            else:
                recommendation['reason'] = f"RSI neutral ({rsi:.1f})"
        
        return recommendation
    
    def _get_rsi_description(self, rsi: float) -> str:
        """Get descriptive text for RSI level"""
        if rsi <= 20:
            return "Extremely Oversold"
        elif rsi <= 30:
            return "Oversold"
        elif rsi <= 40:
            return "Weak"
        elif rsi <= 60:
            return "Neutral"
        elif rsi <= 70:
            return "Strong"
        elif rsi <= 80:
            return "Overbought"
        else:
            return "Extremely Overbought"