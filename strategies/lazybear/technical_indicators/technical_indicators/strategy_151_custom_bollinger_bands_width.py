"""
LazyBear Strategy #151: Custom Bollinger Bands Width
Source: http://pastebin.com/9wkQXqEQ

This strategy uses the width of Bollinger Bands as a volatility indicator.
When BBW drops below a threshold, it indicates low volatility which often
precedes significant price moves.

Original Pine Script implementation by LazyBear.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
import logging
import sys
import os

# Add parent directories to path to import BaseStrategy
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
from strategies.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

class CustomBollingerBandsWidthStrategy(BaseStrategy):
    """
    Custom Bollinger Bands Width Strategy by LazyBear
    
    Uses Bollinger Bands Width to identify low volatility periods
    that often precede significant price movements.
    
    Trading Logic:
    - BUY: When BBW is below threshold and price is above basis (bullish setup)
    - SELL: When BBW is below threshold and price is below basis (bearish setup)
    - Signal strength based on how far BBW is below threshold
    """
    
    def __init__(self, parameters: Dict = None):
        default_params = {
            'length': 20,           # Bollinger Bands period
            'mult': 2.0,            # Standard deviation multiplier
            'threshold': 0.0249,    # BBW threshold for low volatility
            'timeframe_minutes': 5, # Base timeframe
            'min_signal_strength': 0.3  # Minimum signal strength to trade
        }
        
        # Merge user parameters with defaults
        if parameters:
            default_params.update(parameters)
            
        super().__init__("Custom Bollinger Bands Width [LazyBear]", default_params)
        
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Bollinger Bands Width indicator
        
        Args:
            data: Price data with columns: timestamp, price, volume, high, low, open, close
            
        Returns:
            DataFrame with BB and BBW indicators
        """
        if not self.validate_data(data):
            return data
            
        df = data.copy()
        
        # Ensure we have a close price column
        if 'close' not in df.columns:
            df['close'] = df['price']
            
        source = df['close']
        length = self.parameters['length']
        mult = self.parameters['mult']
        
        try:
            # Calculate Bollinger Bands components
            df['bb_basis'] = source.rolling(window=length).mean()  # SMA basis
            df['bb_stdev'] = source.rolling(window=length).std()   # Standard deviation
            
            # Calculate upper and lower bands
            dev = mult * df['bb_stdev']
            df['bb_upper'] = df['bb_basis'] + dev
            df['bb_lower'] = df['bb_basis'] - dev
            
            # Calculate Bollinger Bands Width (BBW)
            # BBW = (upper - lower) / basis
            df['bbw'] = (df['bb_upper'] - df['bb_lower']) / df['bb_basis']
            
            # Handle division by zero or invalid values
            df['bbw'] = df['bbw'].replace([np.inf, -np.inf], np.nan)
            df['bbw'] = df['bbw'].fillna(0)
            
            # Calculate position relative to basis
            df['price_above_basis'] = df['close'] > df['bb_basis']
            df['price_below_basis'] = df['close'] < df['bb_basis']
            
            # Store threshold for reference
            df['bbw_threshold'] = self.parameters['threshold']
            
            # Identify low volatility periods
            df['low_volatility'] = df['bbw'] <= self.parameters['threshold']
            
            logger.debug(f"Calculated BBW indicators for {len(df)} data points")
            
        except Exception as e:
            logger.error(f"Error calculating BBW indicators: {e}")
            
        return df
        
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate buy/sell signals based on BBW and price position
        
        Trading Logic:
        - BUY: BBW below threshold AND price above basis (bullish squeeze)
        - SELL: BBW below threshold AND price below basis (bearish squeeze)
        
        Args:
            data: Price data with BBW indicators
            
        Returns:
            DataFrame with signal columns
        """
        df = data.copy()
        
        # Initialize signal columns
        df['buy_signal'] = False
        df['sell_signal'] = False
        df['signal_strength'] = 0.0
        
        if len(df) < self.parameters['length']:
            logger.warning(f"Insufficient data for BBW strategy: {len(df)} < {self.parameters['length']}")
            return df
            
        try:
            threshold = self.parameters['threshold']
            min_strength = self.parameters['min_signal_strength']
            
            # Identify valid signal conditions
            low_vol_condition = df['low_volatility'] & (df['bbw'] > 0)  # Ensure valid BBW
            
            # Buy signals: Low volatility + price above basis (bullish setup)
            buy_condition = (
                low_vol_condition &
                df['price_above_basis'] &
                df['bb_basis'].notna() &
                df['bbw'].notna()
            )
            
            # Sell signals: Low volatility + price below basis (bearish setup)  
            sell_condition = (
                low_vol_condition &
                df['price_below_basis'] &
                df['bb_basis'].notna() &
                df['bbw'].notna()
            )
            
            # Calculate signal strength based on how far BBW is below threshold
            # Lower BBW = higher signal strength (more compressed = stronger signal)
            bbw_ratio = df['bbw'] / threshold
            signal_strength = np.where(
                df['bbw'] <= threshold,
                np.clip(1.0 - bbw_ratio, 0.0, 1.0),  # Invert so lower BBW = higher strength
                0.0
            )
            
            # Apply signals with minimum strength filter
            df.loc[buy_condition & (signal_strength >= min_strength), 'buy_signal'] = True
            df.loc[sell_condition & (signal_strength >= min_strength), 'sell_signal'] = True
            df.loc[buy_condition | sell_condition, 'signal_strength'] = signal_strength[buy_condition | sell_condition]
            
            # Log signal statistics
            buy_count = df['buy_signal'].sum()
            sell_count = df['sell_signal'].sum()
            avg_strength = df.loc[df['signal_strength'] > 0, 'signal_strength'].mean()
            
            logger.info(f"BBW Strategy generated {buy_count} buy signals, {sell_count} sell signals")
            if avg_strength > 0:
                logger.info(f"Average signal strength: {avg_strength:.3f}")
                
        except Exception as e:
            logger.error(f"Error generating BBW signals: {e}")
            
        return df
        
    def get_current_analysis(self, data: pd.DataFrame) -> Dict:
        """
        Get current market analysis based on BBW
        
        Returns:
            Dictionary with current BBW analysis
        """
        if data.empty:
            return {'error': 'No data available'}
            
        try:
            # Calculate indicators
            data_with_indicators = self.calculate_indicators(data)
            
            if len(data_with_indicators) == 0:
                return {'error': 'No indicators calculated'}
                
            latest = data_with_indicators.iloc[-1]
            
            analysis = {
                'current_price': float(latest.get('close', 0)),
                'bb_basis': float(latest.get('bb_basis', 0)),
                'bb_upper': float(latest.get('bb_upper', 0)), 
                'bb_lower': float(latest.get('bb_lower', 0)),
                'bbw': float(latest.get('bbw', 0)),
                'bbw_threshold': float(latest.get('bbw_threshold', 0)),
                'low_volatility': bool(latest.get('low_volatility', False)),
                'price_above_basis': bool(latest.get('price_above_basis', False)),
                'volatility_status': 'Low' if latest.get('low_volatility', False) else 'Normal',
                'trend_bias': 'Bullish' if latest.get('price_above_basis', False) else 'Bearish'
            }
            
            # Add interpretation
            if analysis['low_volatility']:
                analysis['interpretation'] = 'Low volatility detected - potential breakout setup'
            else:
                analysis['interpretation'] = 'Normal volatility - no squeeze condition'
                
            return analysis
            
        except Exception as e:
            logger.error(f"Error in current analysis: {e}")
            return {'error': str(e)}
            
    def optimize_parameters(self, data: pd.DataFrame) -> Dict:
        """
        Basic parameter optimization for BBW strategy
        
        Tests different threshold values to find optimal settings
        """
        if len(data) < 100:  # Need sufficient data for optimization
            return {'error': 'Insufficient data for optimization'}
            
        original_params = self.parameters.copy()
        best_params = original_params.copy()
        best_score = 0
        
        # Test different threshold values
        thresholds = [0.015, 0.020, 0.0249, 0.030, 0.035, 0.040]
        
        try:
            for threshold in thresholds:
                # Update parameters
                self.parameters['threshold'] = threshold
                
                # Run backtest
                result = self.backtest_signals(data)
                
                if 'error' not in result:
                    # Simple scoring based on signal count and strength
                    total_signals = result.get('total_buy_signals', 0) + result.get('total_sell_signals', 0)
                    avg_strength = result.get('avg_signal_strength', 0)
                    score = total_signals * avg_strength
                    
                    if score > best_score:
                        best_score = score
                        best_params['threshold'] = threshold
                        
            # Restore original parameters
            self.parameters = original_params
            
            return {
                'best_parameters': best_params,
                'best_score': best_score,
                'optimization_method': 'threshold_sweep'
            }
            
        except Exception as e:
            # Restore original parameters on error
            self.parameters = original_params
            logger.error(f"Error in parameter optimization: {e}")
            return {'error': str(e)}