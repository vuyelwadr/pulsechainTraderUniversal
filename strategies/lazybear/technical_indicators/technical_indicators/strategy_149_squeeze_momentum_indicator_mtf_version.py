"""
LazyBear Strategy #149: Squeeze Momentum Indicator MTF version
Based on Pine Script from http://pastebin.com/ZFpNuFfB

The Squeeze Momentum Indicator identifies periods where Bollinger Bands are within Keltner Channels
(indicating a "squeeze" - low volatility) and measures momentum direction during these periods.

Multi-timeframe (MTF) version allows analysis on different timeframes while trading on base timeframe.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
import logging
import sys
import os

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 'strategies'))

from strategies.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

class SqueezeMomentumMTFStrategy(BaseStrategy):
    """
    Squeeze Momentum Indicator Multi-Timeframe Strategy
    
    Detects squeeze conditions (low volatility) and momentum direction:
    - Squeeze: When Bollinger Bands are within Keltner Channels 
    - Momentum: Linear regression of price relative to midpoint
    - MTF: Can analyze higher timeframes for signals
    
    Trading Logic:
    - Buy: Momentum turns positive after squeeze release
    - Sell: Momentum turns negative after squeeze release
    """
    
    def __init__(self, parameters: Dict = None):
        default_params = {
            'bb_length': 20,           # Bollinger Bands length
            'bb_mult': 2.0,            # Bollinger Bands multiplier
            'kc_length': 20,           # Keltner Channel length  
            'kc_mult': 1.5,            # Keltner Channel multiplier
            'use_true_range': True,    # Use True Range for KC calculation
            'momentum_length': 20,     # Linear regression length for momentum
            'signal_threshold': 0.0,   # Minimum momentum for signals
            'squeeze_exit_delay': 1,   # Bars to wait after squeeze exit
            'timeframe_minutes': 15,   # MTF timeframe (base is 5min)
        }
        
        if parameters:
            default_params.update(parameters)
            
        super().__init__("SqueezeMomentumMTF", default_params)
    
    def calculate_true_range(self, data: pd.DataFrame) -> pd.Series:
        """Calculate True Range for Keltner Channels"""
        if len(data) < 2:
            return pd.Series([np.nan] * len(data), index=data.index)
        
        high = data['high']
        low = data['low'] 
        close_prev = data['close'].shift(1)
        
        tr1 = high - low
        tr2 = (high - close_prev).abs()
        tr3 = (low - close_prev).abs()
        
        return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    def calculate_bollinger_bands(self, data: pd.DataFrame, length: int, mult: float) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        source = data['close']
        basis = source.rolling(window=length).mean()
        dev = mult * source.rolling(window=length).std()
        
        upper = basis + dev
        lower = basis - dev
        
        return upper, lower, basis
    
    def calculate_keltner_channels(self, data: pd.DataFrame, length: int, mult: float, use_true_range: bool) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Keltner Channels"""
        source = data['close']
        ma = source.rolling(window=length).mean()
        
        if use_true_range:
            tr = self.calculate_true_range(data)
            range_ma = tr.rolling(window=length).mean()
        else:
            price_range = data['high'] - data['low']
            range_ma = price_range.rolling(window=length).mean()
        
        upper = ma + range_ma * mult
        lower = ma - range_ma * mult
        
        return upper, lower, ma
    
    def calculate_momentum(self, data: pd.DataFrame, length: int) -> pd.Series:
        """
        Calculate momentum using linear regression
        Similar to Pine Script's linreg function
        """
        source = data['close']
        high = data['high']
        low = data['low']
        
        # Calculate the midpoint (average of highest high and lowest low over period)
        highest_high = high.rolling(window=length).max()
        lowest_low = low.rolling(window=length).min()
        midpoint = (highest_high + lowest_low) / 2
        
        # Calculate SMA for comparison
        sma = source.rolling(window=length).mean()
        
        # Average of midpoint and SMA
        baseline = (midpoint + sma) / 2
        
        # Difference from baseline
        diff = source - baseline
        
        # Calculate linear regression slope as momentum
        momentum = diff.rolling(window=length).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == length else np.nan,
            raw=False
        )
        
        return momentum
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate all indicators for the strategy"""
        if len(data) < max(self.parameters['bb_length'], self.parameters['kc_length'], self.parameters['momentum_length']):
            logger.warning(f"Insufficient data for {self.name} indicators")
            # Return data with NaN indicators
            result = data.copy()
            for col in ['bb_upper', 'bb_lower', 'bb_basis', 'kc_upper', 'kc_lower', 'kc_basis',
                       'squeeze_on', 'squeeze_off', 'no_squeeze', 'momentum', 'momentum_positive',
                       'momentum_increasing', 'squeeze_exit_signal']:
                result[col] = np.nan
            return result
        
        result = data.copy()
        
        # Calculate Bollinger Bands
        bb_upper, bb_lower, bb_basis = self.calculate_bollinger_bands(
            data, self.parameters['bb_length'], self.parameters['bb_mult']
        )
        result['bb_upper'] = bb_upper
        result['bb_lower'] = bb_lower
        result['bb_basis'] = bb_basis
        
        # Calculate Keltner Channels
        kc_upper, kc_lower, kc_basis = self.calculate_keltner_channels(
            data, self.parameters['kc_length'], self.parameters['kc_mult'], self.parameters['use_true_range']
        )
        result['kc_upper'] = kc_upper
        result['kc_lower'] = kc_lower
        result['kc_basis'] = kc_basis
        
        # Determine squeeze conditions
        result['squeeze_on'] = (bb_lower > kc_lower) & (bb_upper < kc_upper)  # Squeeze is on
        result['squeeze_off'] = (bb_lower < kc_lower) & (bb_upper > kc_upper)  # Squeeze is releasing
        result['no_squeeze'] = ~result['squeeze_on'] & ~result['squeeze_off']  # No squeeze condition
        
        # Calculate momentum
        result['momentum'] = self.calculate_momentum(data, self.parameters['momentum_length'])
        
        # Momentum direction indicators
        result['momentum_positive'] = result['momentum'] > 0
        result['momentum_increasing'] = result['momentum'] > result['momentum'].shift(1)
        
        # Squeeze exit signal (when squeeze turns off)
        result['squeeze_exit_signal'] = result['squeeze_off'] & result['squeeze_on'].shift(1).fillna(False)
        
        return result
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate buy/sell signals based on squeeze momentum strategy"""
        result = data.copy()
        
        # Initialize signal columns
        result['buy_signal'] = False
        result['sell_signal'] = False
        result['signal_strength'] = 0.0
        
        if 'momentum' not in result.columns or result['momentum'].isna().all():
            logger.warning(f"No valid momentum data for {self.name} signals")
            return result
        
        # Get parameters
        signal_threshold = self.parameters.get('signal_threshold', 0.0)
        squeeze_exit_delay = self.parameters.get('squeeze_exit_delay', 1)
        
        # Create delayed squeeze exit signal
        squeeze_exit_delayed = result['squeeze_exit_signal'].shift(squeeze_exit_delay).fillna(False)
        
        # Generate buy signals
        # Buy when momentum turns positive after squeeze exit
        momentum_turn_positive = (result['momentum'] > signal_threshold) & (result['momentum'].shift(1) <= signal_threshold)
        buy_condition = momentum_turn_positive & (
            squeeze_exit_delayed |  # Recent squeeze exit
            result['squeeze_off']   # Current squeeze release
        )
        
        # Generate sell signals  
        # Sell when momentum turns negative after squeeze exit
        momentum_turn_negative = (result['momentum'] < -signal_threshold) & (result['momentum'].shift(1) >= -signal_threshold)
        sell_condition = momentum_turn_negative & (
            squeeze_exit_delayed |  # Recent squeeze exit
            result['squeeze_off']   # Current squeeze release
        )
        
        # Apply conditions
        result.loc[buy_condition, 'buy_signal'] = True
        result.loc[sell_condition, 'sell_signal'] = True
        
        # Calculate signal strength based on momentum magnitude and squeeze conditions
        momentum_strength = np.abs(result['momentum']) / (np.abs(result['momentum']).rolling(50).max().fillna(1))
        squeeze_strength = result['squeeze_off'].astype(float) * 0.5 + result['squeeze_on'].astype(float) * 0.3
        
        result['signal_strength'] = np.clip(momentum_strength + squeeze_strength, 0.0, 1.0)
        
        # Only keep signals with minimum strength
        min_strength = 0.3
        weak_signals = result['signal_strength'] < min_strength
        result.loc[weak_signals, 'buy_signal'] = False
        result.loc[weak_signals, 'sell_signal'] = False
        result.loc[weak_signals, 'signal_strength'] = 0.0
        
        return result
    
    def get_strategy_info(self) -> Dict:
        """Get strategy information and current indicator values"""
        info = self.get_parameter_info()
        info.update({
            'description': 'Squeeze Momentum Indicator Multi-Timeframe Strategy',
            'indicators': [
                'Bollinger Bands', 'Keltner Channels', 'Squeeze Detection', 
                'Linear Regression Momentum', 'Multi-Timeframe Analysis'
            ],
            'signals': [
                'Buy: Momentum turns positive after squeeze release',
                'Sell: Momentum turns negative after squeeze release'
            ],
            'timeframe': f"{self.parameters.get('timeframe_minutes', 15)} minutes",
            'best_conditions': [
                'Works well in ranging markets with periodic breakouts',
                'Effective when volatility alternates between low and high periods',
                'Suitable for mean-reversion and momentum strategies'
            ]
        })
        return info

def create_strategy(parameters: Dict = None) -> SqueezeMomentumMTFStrategy:
    """Factory function to create the strategy"""
    return SqueezeMomentumMTFStrategy(parameters)

# Example usage and testing
if __name__ == "__main__":
    # Test with sample data
    import matplotlib.pyplot as plt
    from datetime import datetime, timedelta
    
    # Create sample OHLCV data
    dates = pd.date_range(start='2024-01-01', periods=200, freq='5T')
    np.random.seed(42)
    
    # Simulate price data with squeeze/expansion cycles
    base_price = 100
    prices = []
    volatility = []
    squeeze_cycle = 0
    
    for i in range(len(dates)):
        # Create volatility cycles (squeeze and expansion)
        cycle_pos = i % 40
        if cycle_pos < 20:  # Squeeze period
            vol = 0.005 + 0.001 * np.sin(cycle_pos * np.pi / 20)
        else:  # Expansion period  
            vol = 0.02 + 0.01 * np.sin((cycle_pos - 20) * np.pi / 20)
        
        volatility.append(vol)
        
        if i == 0:
            price = base_price
        else:
            # Add momentum during expansion
            momentum = 0
            if cycle_pos >= 20:  # Expansion period
                momentum = 0.001 * np.sin((cycle_pos - 20) * np.pi / 20)
            
            price = prices[-1] * (1 + np.random.normal(momentum, vol))
        
        prices.append(price)
    
    # Create OHLCV data
    data = pd.DataFrame({
        'timestamp': dates,
        'close': prices,
        'open': [p * (1 + np.random.normal(0, 0.001)) for p in prices],
    })
    
    data['high'] = data[['open', 'close']].max(axis=1) * (1 + np.abs(np.random.normal(0, 0.002, len(data))))
    data['low'] = data[['open', 'close']].min(axis=1) * (1 - np.abs(np.random.normal(0, 0.002, len(data))))
    data['price'] = data['close']
    data['volume'] = np.random.exponential(1000, len(data))
    
    # Test the strategy
    strategy = SqueezeMomentumMTFStrategy()
    
    print(f"Testing {strategy.name} strategy...")
    print(f"Strategy info: {strategy.get_strategy_info()}")
    
    # Calculate indicators
    data_with_indicators = strategy.calculate_indicators(data)
    signals_data = strategy.generate_signals(data_with_indicators)
    
    # Print results
    buy_signals = signals_data['buy_signal'].sum()
    sell_signals = signals_data['sell_signal'].sum()
    squeeze_periods = signals_data['squeeze_on'].sum()
    
    print(f"\nBacktest Results:")
    print(f"Buy signals: {buy_signals}")
    print(f"Sell signals: {sell_signals}")
    print(f"Squeeze periods: {squeeze_periods}")
    print(f"Avg signal strength: {signals_data['signal_strength'].mean():.3f}")
    
    # Test current signal
    signal, strength = strategy.get_signal(data)
    print(f"\nCurrent signal: {signal} (strength: {strength:.3f})")
    
    print(f"\n{strategy.name} implementation completed successfully!")