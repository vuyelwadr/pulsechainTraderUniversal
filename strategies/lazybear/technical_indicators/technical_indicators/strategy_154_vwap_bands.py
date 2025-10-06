"""
LazyBear Strategy #154: VWAP Bands
Source: http://pastebin.com/6VqZ8DQ3

VWAP (Volume Weighted Average Price) with Standard Deviation Bands
- Calculates VWAP as the volume-weighted average price
- Creates bands using standard deviation of prices around VWAP
- Generates signals based on price position relative to bands

Trading Logic:
- Buy signals when price moves above lower bands (oversold bounce)
- Sell signals when price moves above upper bands (overbought)
- Uses multiple band levels for signal confirmation
"""
import pandas as pd
import numpy as np
from typing import Dict
import logging

# Import BaseStrategy from the main strategies directory
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))
from strategies.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

class VWAPBandsStrategy(BaseStrategy):
    """
    VWAP Bands Strategy by LazyBear
    
    Implements VWAP with standard deviation bands for trading signals.
    
    Parameters:
        length (int): Period for standard deviation calculation (default: 34)
        l1_multiplier (float): Multiplier for first band level (default: 1.0)
        l2_multiplier (float): Multiplier for second band level (default: 2.0)
        l3_multiplier (float): Optional third band level (default: 2.5)
        use_l3_bands (bool): Whether to use third band level (default: False)
        min_strength (float): Minimum signal strength to trigger trade (default: 0.6)
        volume_threshold (float): Minimum volume multiplier vs average (default: 1.0)
    """
    
    def __init__(self, parameters: Dict = None):
        default_params = {
            'length': 34,
            'l1_multiplier': 1.0,
            'l2_multiplier': 2.0,
            'l3_multiplier': 2.5,
            'use_l3_bands': False,
            'min_strength': 0.6,
            'volume_threshold': 1.0
        }
        
        if parameters:
            default_params.update(parameters)
            
        super().__init__("VWAPBands", default_params)
    
    def calculate_vwap(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate VWAP (Volume Weighted Average Price)"""
        df = data.copy()
        
        # Ensure we have volume data, if not create synthetic volume based on price movement
        if 'volume' not in df.columns or df['volume'].isna().all():
            # Create synthetic volume based on price volatility
            df['volume'] = np.where(df['price'].diff().abs() > df['price'].rolling(10).std(), 
                                  df['price'] * 1000, df['price'] * 500)
            logger.warning("No volume data found, using synthetic volume based on price volatility")
        
        # Use typical price for VWAP calculation when OHLC data available
        if all(col in df.columns for col in ['high', 'low', 'close']):
            df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        else:
            df['typical_price'] = df['price']
        
        # Calculate VWAP
        df['volume_price'] = df['typical_price'] * df['volume']
        df['cum_volume_price'] = df['volume_price'].expanding().sum()
        df['cum_volume'] = df['volume'].expanding().sum()
        df['vwap'] = df['cum_volume_price'] / df['cum_volume']
        
        return df
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate VWAP and standard deviation bands"""
        if not self.validate_data(data):
            return data
        
        df = data.copy()
        length = self.parameters['length']
        l1_mult = self.parameters['l1_multiplier']
        l2_mult = self.parameters['l2_multiplier']
        l3_mult = self.parameters['l3_multiplier']
        use_l3 = self.parameters['use_l3_bands']
        
        # Ensure we have enough data
        if len(df) < length:
            logger.warning(f"Not enough data for VWAP calculation ({length})")
            return df
        
        # Calculate VWAP
        df = self.calculate_vwap(df)
        
        # Use price for standard deviation calculation (matching Pine Script src=close)
        src = df['price']
        
        # Calculate rolling standard deviation
        df['price_std'] = src.rolling(window=length).std()
        
        # Calculate VWAP bands
        df['vwap_upper_1'] = df['vwap'] + l1_mult * df['price_std']
        df['vwap_lower_1'] = df['vwap'] - l1_mult * df['price_std']
        df['vwap_upper_2'] = df['vwap'] + l2_mult * df['price_std']
        df['vwap_lower_2'] = df['vwap'] - l2_mult * df['price_std']
        
        if use_l3:
            df['vwap_upper_3'] = df['vwap'] + l3_mult * df['price_std']
            df['vwap_lower_3'] = df['vwap'] - l3_mult * df['price_std']
        
        # Calculate additional indicators for signal strength
        df['price_vs_vwap'] = (df['price'] - df['vwap']) / df['vwap']
        df['vwap_trend'] = df['vwap'].diff()
        
        # Calculate band position (where price is relative to bands)
        # 0 = at VWAP, +1 = at upper band 1, -1 = at lower band 1, etc.
        upper_1_distance = df['vwap_upper_1'] - df['vwap']
        lower_1_distance = df['vwap'] - df['vwap_lower_1']
        
        df['band_position'] = np.where(
            df['price'] >= df['vwap'],
            (df['price'] - df['vwap']) / upper_1_distance,
            (df['price'] - df['vwap']) / lower_1_distance
        )
        
        # Calculate volume relative to average
        df['volume_ma'] = df['volume'].rolling(window=length).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        return df
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate buy/sell signals based on VWAP bands"""
        df = data.copy()
        
        required_cols = ['vwap', 'vwap_upper_1', 'vwap_lower_1', 'vwap_upper_2', 'vwap_lower_2']
        if not all(col in df.columns for col in required_cols):
            logger.error("VWAP bands not calculated")
            return df
        
        # Initialize signal columns
        df['buy_signal'] = False
        df['sell_signal'] = False
        df['signal_strength'] = 0.0
        df['signal_type'] = 'hold'
        
        min_strength = self.parameters['min_strength']
        volume_threshold = self.parameters['volume_threshold']
        use_l3 = self.parameters['use_l3_bands']
        
        for i in range(1, len(df)):
            current = df.iloc[i]
            previous = df.iloc[i-1]
            
            price = current['price']
            prev_price = previous['price']
            
            # Check volume condition
            volume_ok = current.get('volume_ratio', 1.0) >= volume_threshold
            
            # BUY SIGNALS - Price bouncing from lower bands
            buy_strength = 0.0
            
            # Strong buy: Price crosses above lower band 2 (oversold bounce)
            if (prev_price <= previous['vwap_lower_2'] and 
                price > current['vwap_lower_2'] and volume_ok):
                buy_strength = 0.9
            
            # Medium buy: Price crosses above lower band 1
            elif (prev_price <= previous['vwap_lower_1'] and 
                  price > current['vwap_lower_1'] and 
                  price < current['vwap_upper_1'] and volume_ok):
                buy_strength = 0.7
            
            # Weak buy: Price approaching VWAP from below with momentum
            elif (price < current['vwap'] and price > current['vwap_lower_1'] and
                  current.get('vwap_trend', 0) > 0 and
                  current.get('band_position', 0) > -0.5):
                buy_strength = 0.5
            
            # SELL SIGNALS - Price hitting upper bands (overbought)
            sell_strength = 0.0
            
            # Strong sell: Price at or above upper band 2
            if price >= current['vwap_upper_2'] and volume_ok:
                sell_strength = 0.9
            
            # Medium sell: Price crosses above upper band 1
            elif (prev_price <= previous['vwap_upper_1'] and 
                  price > current['vwap_upper_1'] and volume_ok):
                sell_strength = 0.7
            
            # Weak sell: Price above VWAP with negative momentum
            elif (price > current['vwap'] and 
                  current.get('vwap_trend', 0) < 0 and
                  current.get('band_position', 0) > 0.5):
                sell_strength = 0.5
            
            # Third band signals (if enabled)
            if use_l3 and 'vwap_upper_3' in df.columns:
                # Extreme oversold buy signal
                if (prev_price <= previous['vwap_lower_3'] and 
                    price > current['vwap_lower_3'] and volume_ok):
                    buy_strength = max(buy_strength, 0.95)
                
                # Extreme overbought sell signal
                if price >= current['vwap_upper_3'] and volume_ok:
                    sell_strength = max(sell_strength, 0.95)
            
            # Apply strength enhancements
            buy_strength = self._enhance_signal_strength(df, i, buy_strength, 'buy')
            sell_strength = self._enhance_signal_strength(df, i, sell_strength, 'sell')
            
            # Set signals
            if buy_strength >= min_strength:
                df.iloc[i, df.columns.get_loc('buy_signal')] = True
                df.iloc[i, df.columns.get_loc('signal_strength')] = buy_strength
                df.iloc[i, df.columns.get_loc('signal_type')] = 'buy'
            elif sell_strength >= min_strength:
                df.iloc[i, df.columns.get_loc('sell_signal')] = True
                df.iloc[i, df.columns.get_loc('signal_strength')] = sell_strength
                df.iloc[i, df.columns.get_loc('signal_type')] = 'sell'
        
        return df
    
    def _enhance_signal_strength(self, df: pd.DataFrame, index: int, base_strength: float, signal_type: str) -> float:
        """Enhance signal strength based on additional factors"""
        if base_strength == 0.0:
            return 0.0
        
        try:
            current = df.iloc[index]
            strength = base_strength
            
            # Factor 1: VWAP trend alignment (max +0.1)
            vwap_trend = current.get('vwap_trend', 0)
            if signal_type == 'buy' and vwap_trend > 0:
                strength += min(0.1, abs(vwap_trend) / current['vwap'] * 100)
            elif signal_type == 'sell' and vwap_trend < 0:
                strength += min(0.1, abs(vwap_trend) / current['vwap'] * 100)
            
            # Factor 2: Volume confirmation (max +0.1)
            volume_ratio = current.get('volume_ratio', 1.0)
            if volume_ratio > 1.5:  # Above average volume
                volume_factor = min(0.1, (volume_ratio - 1.0) / 2.0)
                strength += volume_factor
            
            # Factor 3: Band position extremity (max +0.05)
            band_pos = abs(current.get('band_position', 0))
            if band_pos > 1.0:  # Beyond first band
                extremity_factor = min(0.05, (band_pos - 1.0) / 2.0)
                strength += extremity_factor
            
            # Factor 4: Price momentum alignment (max +0.05)
            if index >= 3:
                recent_prices = df.iloc[index-2:index+1]['price']
                if len(recent_prices) >= 2:
                    momentum = (recent_prices.iloc[-1] - recent_prices.iloc[0]) / recent_prices.iloc[0]
                    if signal_type == 'buy' and momentum > 0:
                        strength += min(0.05, momentum * 10)
                    elif signal_type == 'sell' and momentum < 0:
                        strength += min(0.05, abs(momentum) * 10)
            
            return min(1.0, strength)  # Cap at 1.0
            
        except Exception as e:
            logger.warning(f"Error enhancing signal strength: {e}")
            return base_strength
    
    def get_current_position_recommendation(self, data: pd.DataFrame) -> Dict:
        """Get current position recommendation with detailed VWAP analysis"""
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
            'vwap': latest.get('vwap', 0),
            'vwap_upper_1': latest.get('vwap_upper_1', 0),
            'vwap_lower_1': latest.get('vwap_lower_1', 0),
            'vwap_upper_2': latest.get('vwap_upper_2', 0),
            'vwap_lower_2': latest.get('vwap_lower_2', 0),
            'band_position': latest.get('band_position', 0),
            'volume_ratio': latest.get('volume_ratio', 1.0),
            'parameters': self.parameters
        }
        
        # Add reasoning based on band position
        price = latest['price']
        vwap = latest.get('vwap', 0)
        
        if latest.get('buy_signal', False):
            if price <= latest.get('vwap_lower_2', 0):
                recommendation['reason'] = f"Strong oversold bounce signal (below L2 band) with {latest['signal_strength']:.2f} strength"
            elif price <= latest.get('vwap_lower_1', 0):
                recommendation['reason'] = f"Oversold bounce signal (below L1 band) with {latest['signal_strength']:.2f} strength"
            else:
                recommendation['reason'] = f"VWAP support signal with {latest['signal_strength']:.2f} strength"
        
        elif latest.get('sell_signal', False):
            if price >= latest.get('vwap_upper_2', 0):
                recommendation['reason'] = f"Strong overbought signal (above U2 band) with {latest['signal_strength']:.2f} strength"
            elif price >= latest.get('vwap_upper_1', 0):
                recommendation['reason'] = f"Overbought signal (above U1 band) with {latest['signal_strength']:.2f} strength"
            else:
                recommendation['reason'] = f"VWAP resistance signal with {latest['signal_strength']:.2f} strength"
        
        else:
            if price > vwap:
                recommendation['reason'] = f"Price above VWAP ({(price/vwap-1)*100:.2f}%) - bullish bias"
            elif price < vwap:
                recommendation['reason'] = f"Price below VWAP ({(1-price/vwap)*100:.2f}%) - bearish bias"
            else:
                recommendation['reason'] = "Price near VWAP - neutral"
        
        return recommendation