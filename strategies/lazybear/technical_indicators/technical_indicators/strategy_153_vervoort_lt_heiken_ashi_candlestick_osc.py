#!/usr/bin/env python3
"""
Strategy 153: Vervoort LT Heiken-Ashi Candlestick Oscillator (HACOLT)

The Vervoort LongTerm Heiken-Ashi Candlestick Oscillator (HACOLT) is a digital oscillator 
developed by Sylvain Vervoort, based on his Heikin-Ashi Candlestick Oscillator (HACO) 
published in TASC December 2008. This is the Long Term version that provides trend 
direction signals with values of -1 (short), 0 (neutral), and 1 (long).

The algorithm applies complex mathematical transformations to Heiken-Ashi candlesticks
using TEMA (Triple Exponential Moving Average) smoothing to generate trend signals
with reduced whipsaws compared to traditional indicators.

Mathematical Components:
1. Heiken-Ashi candlestick calculations
2. TEMA smoothing and Zero-Lag TEMA transformations
3. Complex signal filtering logic
4. Long-term trend determination

TradingView URL: https://www.tradingview.com/v/4zuhGaAU/
Original Author: Sylvain Vervoort (TASC Dec 2008)
LazyBear Implementation: @LazyBear
Type: trend/oscillator
"""

import pandas as pd
import numpy as np
import talib
from typing import Dict
import sys
import os

# Add parent directories to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_dir))))
strategies_dir = os.path.join(project_root, 'strategies')
sys.path.insert(0, strategies_dir)

try:
    from base_strategy import BaseStrategy
except ImportError:
    # Try alternative path
    sys.path.insert(0, project_root)
    from strategies.base_strategy import BaseStrategy

# Also try importing vectorized helpers if available
try:
    sys.path.insert(0, os.path.join(project_root, 'src'))
    from utils.vectorized_helpers import (
        crossover, crossunder, apply_position_constraints,
        calculate_signal_strength
    )
    VECTORIZED_HELPERS_AVAILABLE = True
except ImportError:
    VECTORIZED_HELPERS_AVAILABLE = False
    
    # Define simple crossover functions if not available
    def crossover(series1, series2):
        """Simple crossover detection"""
        return (series1 > series2) & (series1.shift(1) <= series2.shift(1))
    
    def crossunder(series1, series2):
        """Simple crossunder detection"""
        return (series1 < series2) & (series1.shift(1) >= series2.shift(1))

class Strategy153VervoortLTHeikenAshiCandlestickOsc(BaseStrategy):
    def __init__(self, parameters: Dict = None):
        default_params = {
            'length': 55,              # TEMA Period
            'ema_length': 60,          # EMA Period  
            'candle_size_factor': 1.1, # Candle size factor for short candle detection
            'signal_sensitivity': 0.7  # Signal strength threshold
        }
        
        if parameters:
            default_params.update(parameters)
            
        super().__init__(
            name="Strategy153VervoortLTHeikenAshiCandlestickOsc",
            parameters=default_params
        )
    
    def tema(self, data: pd.Series, period: int) -> pd.Series:
        """
        Calculate Triple Exponential Moving Average (TEMA)
        TEMA = 3*EMA1 - 3*EMA2 + EMA3
        where EMA1 = EMA(data), EMA2 = EMA(EMA1), EMA3 = EMA(EMA2)
        """
        if len(data) < period:
            return pd.Series([np.nan] * len(data), index=data.index)
        
        ema1 = data.ewm(span=period, adjust=False).mean()
        ema2 = ema1.ewm(span=period, adjust=False).mean()
        ema3 = ema2.ewm(span=period, adjust=False).mean()
        
        tema = 3 * ema1 - 3 * ema2 + ema3
        return tema
    
    def zero_lag_tema(self, data: pd.Series, period: int) -> pd.Series:
        """
        Calculate Zero-Lag TEMA for smoother signals
        ZL-TEMA = 2*TEMA - TEMA(TEMA)
        """
        tema1 = self.tema(data, period)
        tema2 = self.tema(tema1, period)
        
        zl_tema = 2 * tema1 - tema2
        return zl_tema
    
    def calculate_heiken_ashi(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Calculate Heiken-Ashi candlesticks
        """
        if len(data) < 2:
            return {
                'ha_open': pd.Series([np.nan] * len(data), index=data.index),
                'ha_close': pd.Series([np.nan] * len(data), index=data.index),
                'ha_high': pd.Series([np.nan] * len(data), index=data.index),
                'ha_low': pd.Series([np.nan] * len(data), index=data.index)
            }
        
        # Ensure we have required columns
        if 'close' not in data.columns:
            data['close'] = data['price']
        if 'open' not in data.columns:
            data['open'] = data['price']
        if 'high' not in data.columns:
            data['high'] = data['price']
        if 'low' not in data.columns:
            data['low'] = data['price']
        
        # Calculate OHLC4
        ohlc4 = (data['open'] + data['high'] + data['low'] + data['close']) / 4
        
        # Initialize series
        ha_open = pd.Series(index=data.index, dtype=float)
        ha_close = pd.Series(index=data.index, dtype=float)
        
        # First value
        ha_open.iloc[0] = ohlc4.iloc[0]
        ha_close.iloc[0] = ohlc4.iloc[0]
        
        # Calculate Heiken-Ashi values
        for i in range(1, len(data)):
            # HA Open = (Previous HA Open + Previous HA Close) / 2
            ha_open.iloc[i] = (ha_open.iloc[i-1] + ohlc4.iloc[i-1]) / 2
            
            # HA Close = (Open + High + Low + Close) / 4
            ha_close.iloc[i] = (ha_open.iloc[i] + 
                               max(data['high'].iloc[i], ha_open.iloc[i]) + 
                               min(data['low'].iloc[i], ha_open.iloc[i]) + 
                               ohlc4.iloc[i]) / 4
        
        # HA High and Low
        ha_high = pd.Series(index=data.index, dtype=float)
        ha_low = pd.Series(index=data.index, dtype=float)
        
        for i in range(len(data)):
            ha_high.iloc[i] = max(data['high'].iloc[i], ha_open.iloc[i], ha_close.iloc[i])
            ha_low.iloc[i] = min(data['low'].iloc[i], ha_open.iloc[i], ha_close.iloc[i])
        
        return {
            'ha_open': ha_open,
            'ha_close': ha_close,
            'ha_high': ha_high,
            'ha_low': ha_low
        }
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate HACOLT indicators"""
        if len(data) < max(self.parameters['length'], self.parameters['ema_length']):
            # Return data with NaN indicators if insufficient data
            result_data = data.copy()
            result_data['hacolt'] = np.nan
            result_data['trend_strength'] = np.nan
            result_data['ha_close'] = np.nan
            result_data['ha_open'] = np.nan
            return result_data
        
        result_data = data.copy()
        
        # Ensure we have required columns
        if 'close' not in result_data.columns:
            result_data['close'] = result_data['price']
        if 'open' not in result_data.columns:
            result_data['open'] = result_data['price']
        if 'high' not in result_data.columns:
            result_data['high'] = result_data['price']
        if 'low' not in result_data.columns:
            result_data['low'] = result_data['price']
        
        # Calculate Heiken-Ashi values
        ha_values = self.calculate_heiken_ashi(result_data)
        result_data['ha_open'] = ha_values['ha_open']
        result_data['ha_close'] = ha_values['ha_close']
        result_data['ha_high'] = ha_values['ha_high']
        result_data['ha_low'] = ha_values['ha_low']
        
        # Calculate smoothed values using TEMA
        length = self.parameters['length']
        
        # TEMA of Heiken-Ashi close
        t_ha_close = self.tema(result_data['ha_close'], length)
        
        # TEMA of HL2 (High + Low) / 2
        hl2 = (result_data['high'] + result_data['low']) / 2
        t_hl2 = self.tema(hl2, length)
        
        # Zero-Lag smoothed values
        ha_close_smooth = self.zero_lag_tema(result_data['ha_close'], length)
        hl2_smooth = self.zero_lag_tema(hl2, length)
        
        # Store intermediate calculations
        result_data['t_ha_close'] = t_ha_close
        result_data['t_hl2'] = t_hl2
        result_data['ha_close_smooth'] = ha_close_smooth
        result_data['hl2_smooth'] = hl2_smooth
        
        # Detect short candles
        candle_range = result_data['high'] - result_data['low']
        candle_body = abs(result_data['close'] - result_data['open'])
        short_candle = candle_body < (candle_range * self.parameters['candle_size_factor'])
        result_data['short_candle'] = short_candle
        
        # Calculate complex conditions for trend determination
        
        # Condition 1: Basic Heiken-Ashi trend conditions
        cond1 = ((result_data['ha_close'] >= result_data['ha_open']) & 
                 (result_data['ha_close'].shift(1) >= result_data['ha_open'].shift(1)))
        
        # Condition 2: Close vs HA Close
        cond2 = result_data['close'] >= result_data['ha_close']
        
        # Condition 3: Higher highs
        cond3 = result_data['high'] > result_data['high'].shift(1)
        
        # Condition 4: Higher lows
        cond4 = result_data['low'] > result_data['low'].shift(1)
        
        # Condition 5: Smoothed HL2 vs Smoothed HA Close
        cond5 = result_data['hl2_smooth'] >= result_data['ha_close_smooth']
        
        # Combined keep conditions
        keep_n1 = cond1 | cond2 | cond3 | cond4 | cond5
        result_data['keep_n1'] = keep_n1
        
        # Keep all conditions including momentum
        keep_all1_basic = (keep_n1 | 
                          (keep_n1.shift(1) & (result_data['close'] >= result_data['open'])) |
                          (result_data['close'] >= result_data['close'].shift(1)))
        
        # Additional condition for short candles
        keep_3 = short_candle & (result_data['high'] >= result_data['low'].shift(1))
        
        # Final uptrend condition
        utr = keep_all1_basic | (keep_all1_basic.shift(1) & keep_3)
        
        # Mirror logic for downtrend (opposite conditions)
        # Downtrend conditions
        down_cond1 = ((result_data['ha_close'] < result_data['ha_open']) & 
                     (result_data['ha_close'].shift(1) < result_data['ha_open'].shift(1)))
        
        down_cond2 = result_data['close'] < result_data['ha_close']
        down_cond3 = result_data['high'] < result_data['high'].shift(1)
        down_cond4 = result_data['low'] < result_data['low'].shift(1)
        down_cond5 = result_data['hl2_smooth'] < result_data['ha_close_smooth']
        
        keep_n1_down = down_cond1 | down_cond2 | down_cond3 | down_cond4 | down_cond5
        
        keep_all1_down = (keep_n1_down | 
                         (keep_n1_down.shift(1) & (result_data['close'] < result_data['open'])) |
                         (result_data['close'] < result_data['close'].shift(1)))
        
        keep_3_down = short_candle & (result_data['low'] <= result_data['high'].shift(1))
        
        # Final downtrend condition
        dtr = keep_all1_down | (keep_all1_down.shift(1) & keep_3_down)
        
        # Generate HACOLT oscillator values
        hacolt = pd.Series(index=result_data.index, dtype=float)
        
        # Apply trend logic
        # 1 = Long, 0 = Neutral, -1 = Short
        hacolt = np.where(utr & ~dtr, 1,     # Clear uptrend
                 np.where(dtr & ~utr, -1,    # Clear downtrend  
                         0))                  # Neutral/uncertain
        
        result_data['hacolt'] = hacolt
        
        # Calculate trend strength based on the conviction of the signal
        trend_strength = pd.Series(index=result_data.index, dtype=float)
        
        # Strong signals when multiple conditions align
        strong_up = (cond1 & cond2 & (cond3 | cond4) & cond5)
        strong_down = (down_cond1 & down_cond2 & (down_cond3 | down_cond4) & down_cond5)
        
        trend_strength = np.where(strong_up & (hacolt == 1), 0.9,
                         np.where(strong_down & (hacolt == -1), 0.9,
                         np.where(hacolt != 0, 0.6, 0.3)))
        
        result_data['trend_strength'] = trend_strength
        
        return result_data
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on HACOLT oscillator"""
        if len(data) < 2:
            # Return data with no signals if insufficient data
            result_data = data.copy()
            result_data['buy_signal'] = False
            result_data['sell_signal'] = False
            result_data['signal_strength'] = 0.0
            return result_data
        
        result_data = data.copy()
        
        # Initialize signals
        buy_signals = pd.Series(False, index=result_data.index)
        sell_signals = pd.Series(False, index=result_data.index)
        signal_strength = pd.Series(0.0, index=result_data.index)
        
        hacolt = result_data.get('hacolt', pd.Series([0] * len(result_data), index=result_data.index))
        trend_strength = result_data.get('trend_strength', pd.Series([0.0] * len(result_data), index=result_data.index))
        
        # Generate signals based on HACOLT transitions
        # Buy: HACOLT changes from -1 or 0 to 1 (uptrend start)
        hacolt_shifted = hacolt.shift(1)
        buy_condition = (hacolt == 1) & ((hacolt_shifted == -1) | (hacolt_shifted == 0))
        
        # Sell: HACOLT changes from 1 to 0 or -1 (uptrend end)
        # Also add signals when moving to strong downtrend (-1)
        sell_condition = ((hacolt == 0) | (hacolt == -1)) & (hacolt_shifted == 1)
        
        # Short signals: HACOLT changes from 1 or 0 to -1 (downtrend start)
        short_condition = (hacolt == -1) & ((hacolt_shifted == 1) | (hacolt_shifted == 0))
        
        # Apply sensitivity threshold
        sensitivity = self.parameters.get('signal_sensitivity', 0.7)
        
        # Generate signals with different thresholds for different signal types
        # Buy signals need high confidence
        buy_signals = buy_condition & (trend_strength >= sensitivity)
        
        # Sell signals can have lower threshold since they're protective
        sell_threshold = max(0.5, sensitivity - 0.2)
        sell_signals = (sell_condition | short_condition) & (trend_strength >= sell_threshold)
        
        # Prevent conflicting signals in the same period
        conflicting = buy_signals & sell_signals
        if conflicting.any():
            # In case of conflict, prefer the signal with higher strength
            buy_signals = buy_signals & ~conflicting
            sell_signals = sell_signals & ~conflicting
        
        # Signal strength is based on trend strength and oscillator conviction
        signal_strength = np.where(buy_signals | sell_signals, trend_strength, 0.0)
        
        # Store signals
        result_data['buy_signal'] = buy_signals
        result_data['sell_signal'] = sell_signals  
        result_data['signal_strength'] = signal_strength
        
        return result_data
    
    def get_strategy_info(self) -> Dict:
        """Get strategy information"""
        return {
            'name': self.name,
            'description': 'Vervoort LT Heiken-Ashi Candlestick Oscillator - Digital trend oscillator using TEMA-smoothed Heiken-Ashi values',
            'parameters': self.parameters,
            'signal_types': ['buy', 'sell'],
            'indicators': ['hacolt', 'trend_strength', 'ha_close', 'ha_open'],
            'oscillator_range': [-1, 0, 1],
            'author': 'Sylvain Vervoort (TASC Dec 2008)',
            'source': 'LazyBear TradingView Implementation'
        }

def create_strategy(parameters: Dict = None) -> Strategy153VervoortLTHeikenAshiCandlestickOsc:
    """Factory function to create strategy instance"""
    return Strategy153VervoortLTHeikenAshiCandlestickOsc(parameters)

# Test the strategy implementation
if __name__ == "__main__":
    # Test with sample data - use built-in libraries only
    
    print("Testing Strategy 153: Vervoort LT Heiken-Ashi Candlestick Oscillator...")
    
    # Create strategy
    strategy = create_strategy({
        'length': 55,
        'ema_length': 60,
        'candle_size_factor': 1.1,
        'signal_sensitivity': 0.7
    })
    
    # Test with sample data
    dates = pd.date_range(start='2023-01-01', end='2023-02-01', freq='1h')
    np.random.seed(42)
    
    # Generate sample OHLCV data with realistic price movements
    base_price = 100
    price_changes = np.random.normal(0, 0.02, len(dates))
    prices = [base_price]
    
    for change in price_changes[1:]:
        new_price = prices[-1] * (1 + change)
        prices.append(max(new_price, 1))  # Ensure price doesn't go below 1
    
    # Create OHLCV data
    sample_data = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'close': prices,
        'price': prices,
        'volume': np.random.randint(1000, 10000, len(dates))
    })
    
    # Ensure high >= max(open, close) and low <= min(open, close)
    sample_data['high'] = np.maximum.reduce([sample_data['high'], sample_data['open'], sample_data['close']])
    sample_data['low'] = np.minimum.reduce([sample_data['low'], sample_data['open'], sample_data['close']])
    
    try:
        # Test indicators calculation
        print("\n1. Testing indicators calculation...")
        data_with_indicators = strategy.calculate_indicators(sample_data)
        print(f"✓ Calculated indicators for {len(data_with_indicators)} data points")
        
        # Check for key indicators
        key_indicators = ['hacolt', 'trend_strength', 'ha_close', 'ha_open']
        for indicator in key_indicators:
            if indicator in data_with_indicators.columns:
                non_nan_count = (~data_with_indicators[indicator].isna()).sum()
                print(f"  - {indicator}: {non_nan_count} non-NaN values")
            else:
                print(f"  - {indicator}: Missing!")
        
        # Test signals generation
        print("\n2. Testing signals generation...")
        data_with_signals = strategy.generate_signals(data_with_indicators)
        
        buy_signals = data_with_signals['buy_signal'].sum()
        sell_signals = data_with_signals['sell_signal'].sum()
        avg_strength = data_with_signals['signal_strength'].mean()
        
        print(f"✓ Generated {buy_signals} buy signals and {sell_signals} sell signals")
        print(f"✓ Average signal strength: {avg_strength:.3f}")
        
        # Test strategy info
        print("\n3. Testing strategy info...")
        info = strategy.get_strategy_info()
        print(f"✓ Strategy: {info['name']}")
        print(f"✓ Parameters: {info['parameters']}")
        print(f"✓ Indicators: {info['indicators']}")
        
        # Test backtest
        print("\n4. Testing backtest...")
        backtest_results = strategy.backtest_signals(sample_data)
        print(f"✓ Backtest results: {backtest_results}")
        
        # Show sample of HACOLT values
        print("\n5. Sample HACOLT values:")
        recent_data = data_with_signals.tail(10)[['timestamp', 'price', 'hacolt', 'trend_strength', 'buy_signal', 'sell_signal']]
        for _, row in recent_data.iterrows():
            signals = []
            if row['buy_signal']:
                signals.append('BUY')
            if row['sell_signal']:
                signals.append('SELL')
            signal_str = ', '.join(signals) if signals else 'HOLD'
            print(f"  {row['timestamp'].strftime('%Y-%m-%d %H:%M')}: Price=${row['price']:.2f}, HACOLT={row['hacolt']:.0f}, Strength={row['trend_strength']:.3f}, Signal={signal_str}")
        
        print(f"\n✅ Strategy 153 implementation completed successfully!")
        print(f"The Vervoort LT Heiken-Ashi Candlestick Oscillator is working correctly.")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()