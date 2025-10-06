#!/usr/bin/env python3
"""
Strategy 163: SAR Alarm Script

LazyBear Strategy Number: 163 (Original)
Strategy ID: 163 (Canonical)
LazyBear Name: SAR Alarm
Type: momentum/trend/alarm
PasteBin URL: http://pastebin.com/ak2M5Dte

Description:
Implements LazyBear's SAR Alarm indicator which uses Parabolic SAR (Stop and Reverse) 
to generate alarm signals when the SAR crosses over price levels. The original Pine Script
plots a zero line and histogram showing SAR crossover points with color-coding.

This strategy translates the SAR alarm concept into actionable buy/sell signals:
- Buy signals when SAR crosses below price (SAR support)  
- Sell signals when SAR crosses above price (SAR resistance)
- Uses configurable SAR parameters: start, increment, maximum
"""

import pandas as pd
import numpy as np
import sys
import os

# Add parent directories to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_dir))))
strategies_dir = os.path.join(repo_root, 'strategies')
utils_dir = os.path.join(repo_root, 'src', 'utils')
sys.path.insert(0, repo_root)
sys.path.insert(0, strategies_dir)
sys.path.insert(0, utils_dir)
from base_strategy import BaseStrategy
from utils.vectorized_helpers import crossover, crossunder


class Strategy163SarAlarmScript(BaseStrategy):
    """
    Strategy 163: SAR Alarm Script
    
    Uses Parabolic SAR to generate alarm signals when SAR values cross price levels.
    Generates buy signals when SAR moves below price (support) and sell signals 
    when SAR moves above price (resistance).
    """

    def __init__(self, parameters: dict = None):
        """
        Initialize strategy with SAR Alarm parameters
        
        Args:
            parameters: Dictionary with strategy parameters
        """
        # Default parameters matching LazyBear implementation
        default_params = {
            'start': 0.02,          # SAR start value (acceleration factor start)
            'increment': 0.02,      # SAR increment (acceleration factor increment) 
            'maximum': 0.2,         # SAR maximum (acceleration factor maximum)
            'signal_threshold': 0.6, # Minimum signal strength to trade
            'trend_confirmation': True,  # Require trend confirmation
            'min_sar_distance': 0.001,   # Minimum SAR distance from price (0.1%)
        }
        
        # Merge with provided parameters
        if parameters:
            default_params.update(parameters)
        
        # Initialize parent class
        super().__init__(
            name="Strategy_163_SarAlarmScript",
            parameters=default_params
        )

    def calculate_parabolic_sar(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate Parabolic SAR using the parameters.
        Vectorized implementation of the SAR algorithm.
        
        Args:
            data: DataFrame with OHLC data
            
        Returns:
            Series with SAR values
        """
        high = data['high']
        low = data['low']
        close = data['close']
        
        start = self.parameters['start']
        increment = self.parameters['increment']  
        maximum = self.parameters['maximum']
        
        # Initialize arrays
        n = len(data)
        sar = np.zeros(n)
        trend = np.ones(n)  # 1 for uptrend, -1 for downtrend
        af = np.zeros(n)    # acceleration factor
        ep = np.zeros(n)    # extreme point
        
        if n < 2:
            return pd.Series(sar, index=data.index)
        
        # Initialize first values
        sar[0] = low.iloc[0]
        trend[0] = 1
        af[0] = start
        ep[0] = high.iloc[0]
        
        # Calculate SAR for each period
        for i in range(1, n):
            # Previous values
            prev_sar = sar[i-1]
            prev_trend = trend[i-1]
            prev_af = af[i-1]
            prev_ep = ep[i-1]
            
            # Current values
            curr_high = high.iloc[i]
            curr_low = low.iloc[i]
            
            if prev_trend == 1:  # Uptrend
                # Calculate SAR
                sar[i] = prev_sar + prev_af * (prev_ep - prev_sar)
                
                # SAR cannot be above previous low or current low
                sar[i] = min(sar[i], low.iloc[i-1])
                if i > 1:
                    sar[i] = min(sar[i], low.iloc[i-2])
                
                # Check for trend reversal
                if curr_low <= sar[i]:
                    # Trend reversal to downtrend
                    trend[i] = -1
                    sar[i] = prev_ep  # SAR becomes previous extreme point
                    af[i] = start     # Reset acceleration factor
                    ep[i] = curr_low  # New extreme point is current low
                else:
                    # Continue uptrend
                    trend[i] = 1
                    
                    # Update extreme point and acceleration factor
                    if curr_high > prev_ep:
                        ep[i] = curr_high
                        af[i] = min(prev_af + increment, maximum)
                    else:
                        ep[i] = prev_ep
                        af[i] = prev_af
                        
            else:  # Downtrend (prev_trend == -1)
                # Calculate SAR
                sar[i] = prev_sar + prev_af * (prev_ep - prev_sar)
                
                # SAR cannot be below previous high or current high  
                sar[i] = max(sar[i], high.iloc[i-1])
                if i > 1:
                    sar[i] = max(sar[i], high.iloc[i-2])
                
                # Check for trend reversal
                if curr_high >= sar[i]:
                    # Trend reversal to uptrend
                    trend[i] = 1
                    sar[i] = prev_ep  # SAR becomes previous extreme point
                    af[i] = start     # Reset acceleration factor
                    ep[i] = curr_high # New extreme point is current high
                else:
                    # Continue downtrend
                    trend[i] = -1
                    
                    # Update extreme point and acceleration factor
                    if curr_low < prev_ep:
                        ep[i] = curr_low
                        af[i] = min(prev_af + increment, maximum)
                    else:
                        ep[i] = prev_ep
                        af[i] = prev_af
        
        return pd.Series(sar, index=data.index)

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate SAR and related indicators
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with SAR indicator columns
        """
        # Ensure we have required columns
        if 'close' not in data.columns and 'price' in data.columns:
            data['close'] = data['price']
        
        # Ensure we have OHLC data
        if 'open' not in data.columns:
            data['open'] = data['close'].shift(1).fillna(data['close'])
        if 'high' not in data.columns:
            data['high'] = data['close']
        if 'low' not in data.columns:
            data['low'] = data['close']
        
        # Calculate Parabolic SAR
        data['sar'] = self.calculate_parabolic_sar(data)
        
        # Calculate SAR difference from low (as in original Pine Script: d=s-low)
        data['sar_diff'] = data['sar'] - data['low']
        
        # Detect SAR crossovers (cross(d,0) in Pine Script)
        # This happens when sar_diff crosses zero line
        zero_line = pd.Series(0, index=data.index)
        data['sar_cross_above'] = crossover(data['sar_diff'], zero_line)
        data['sar_cross_below'] = crossunder(data['sar_diff'], zero_line)
        
        # SAR position relative to price
        data['sar_below_price'] = data['sar'] < data['close']
        data['sar_above_price'] = data['sar'] > data['close']
        
        # SAR trend direction
        data['sar_trend_up'] = data['sar'] < data['close']
        data['sar_trend_down'] = data['sar'] > data['close']
        
        # Distance between SAR and price (for signal strength)
        data['sar_price_distance'] = np.abs(data['sar'] - data['close']) / data['close']
        
        # Store indicators for debugging
        indicator_columns = ['sar', 'sar_diff', 'sar_cross_above', 'sar_cross_below', 
                            'sar_below_price', 'sar_above_price', 'sar_trend_up', 
                            'sar_trend_down', 'sar_price_distance']
        
        self.indicators = data[indicator_columns].copy()
        
        return data

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate buy/sell signals based on SAR alarm logic
        
        CRITICAL: Apply look-ahead bias prevention using .shift(1) for stateful logic.
        
        Args:
            data: DataFrame with price and SAR indicator data
            
        Returns:
            DataFrame with buy_signal, sell_signal, and signal_strength columns
        """
        # Initialize signal columns
        data['buy_signal'] = False
        data['sell_signal'] = False
        data['signal_strength'] = 0.0
        
        # Extract parameters
        threshold = self.parameters['signal_threshold']
        trend_confirmation = self.parameters['trend_confirmation']
        min_distance = self.parameters['min_sar_distance']
        
        # Signal strength components
        buy_strength = pd.Series(0.0, index=data.index)
        sell_strength = pd.Series(0.0, index=data.index)
        
        # Primary SAR alarm signals (based on original Pine Script logic)
        # Buy signal: SAR crosses below price (support), or SAR is below and trending up
        sar_support_signal = (
            data['sar_cross_below'] |  # SAR just crossed below (d crosses 0 from positive to negative)
            (data['sar_below_price'] & data['sar_below_price'].shift(1) == False)  # SAR moves below price
        )
        
        # Sell signal: SAR crosses above price (resistance), or SAR is above and trending down  
        sar_resistance_signal = (
            data['sar_cross_above'] |  # SAR just crossed above (d crosses 0 from negative to positive)
            (data['sar_above_price'] & data['sar_above_price'].shift(1) == False)  # SAR moves above price
        )
        
        # Calculate base signal strength from SAR distance
        # Closer SAR = stronger signal (more responsive to price action)
        distance_strength = 1.0 - np.clip(data['sar_price_distance'] / 0.05, 0, 1)  # Max distance 5% for full strength
        
        # Apply minimum distance filter (avoid whipsaws when SAR too close)
        valid_distance = data['sar_price_distance'] >= min_distance
        
        # Buy signal conditions
        buy_conditions = sar_support_signal & valid_distance
        if trend_confirmation:
            # Require price above recent SAR for trend confirmation
            price_momentum = data['close'] > data['close'].shift(2)  # 2-bar momentum
            buy_conditions = buy_conditions & price_momentum
        
        buy_strength = buy_conditions.astype(float) * distance_strength
        
        # Sell signal conditions
        sell_conditions = sar_resistance_signal & valid_distance
        if trend_confirmation:
            # Require price below recent SAR for trend confirmation
            price_momentum = data['close'] < data['close'].shift(2)  # 2-bar momentum
            sell_conditions = sell_conditions & price_momentum
        
        sell_strength = sell_conditions.astype(float) * distance_strength
        
        # Generate final signals with strength threshold
        strong_buy = buy_strength >= threshold
        strong_sell = sell_strength >= threshold
        
        data.loc[strong_buy, 'buy_signal'] = True
        data.loc[strong_buy, 'signal_strength'] = buy_strength[strong_buy]
        
        data.loc[strong_sell, 'sell_signal'] = True
        data.loc[strong_sell, 'signal_strength'] = sell_strength[strong_sell]
        
        # --- State Management for Look-ahead Bias Prevention ---
        # Create unified signal for position tracking
        raw_signal = np.select(
            [data['buy_signal'], data['sell_signal']],
            [1, -1],  # +1 for buy, -1 for sell
            default=0
        )
        
        # Calculate position state: ffill non-zero signals to track current position
        position = pd.Series(raw_signal).replace(0, np.nan).ffill().fillna(0)
        
        # Apply state-based filtering using previous bar to avoid look-ahead bias
        is_flat = (position.shift(1) == 0)
        is_long = (position.shift(1) == 1) 
        is_short = (position.shift(1) == -1)
        
        # Refine signals based on position state
        # Only allow buy if flat or short, only allow sell if long or flat
        data.loc[~(data['buy_signal'] & (is_flat | is_short)), 'buy_signal'] = False
        data.loc[~(data['sell_signal'] & (is_long | is_flat)), 'sell_signal'] = False
        
        # Reset signal strength where signals were filtered out
        data.loc[~(data['buy_signal'] | data['sell_signal']), 'signal_strength'] = 0.0
        
        # Store signals for debugging
        self.signals = data[['buy_signal', 'sell_signal', 'signal_strength']].copy()
        
        return data


# Test function for smoke testing
if __name__ == "__main__":
    # Create sample data with realistic OHLCV structure
    import datetime
    
    print("🔍 Testing Strategy 163: SAR Alarm Script")
    
    # Generate sample data with trending periods (good for SAR testing)
    np.random.seed(42)  # For reproducible results
    dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='5min')
    n_points = len(dates)
    
    # Create price movements with clear trends (optimal for SAR)
    base_price = 100
    trend = np.linspace(0, 15, n_points)  # Strong upward trend
    noise = np.random.randn(n_points).cumsum() * 0.2
    volatility = 3 * np.sin(np.linspace(0, 6*np.pi, n_points))  # Volatility cycles
    
    price_series = base_price + trend + noise + volatility
    
    sample_data = pd.DataFrame({
        'timestamp': dates,
        'open': price_series + np.random.randn(n_points) * 0.1,
        'high': price_series + np.abs(np.random.randn(n_points)) * 0.4 + 0.3,
        'low': price_series - np.abs(np.random.randn(n_points)) * 0.4 - 0.3,
        'close': price_series,
        'volume': np.random.randint(1000, 10000, n_points)
    })
    
    # Ensure OHLC consistency
    sample_data['high'] = sample_data[['open', 'close', 'high']].max(axis=1)
    sample_data['low'] = sample_data[['open', 'close', 'low']].min(axis=1)
    sample_data['price'] = sample_data['close']
    
    print(f"Sample data shape: {sample_data.shape}")
    print(f"Price range: {sample_data['close'].min():.2f} - {sample_data['close'].max():.2f}")
    
    try:
        # Test strategy initialization
        strategy = Strategy163SarAlarmScript()
        print("✅ Strategy initialization successful")
        
        # Test indicator calculation
        data_with_indicators = strategy.calculate_indicators(sample_data.copy())
        print(f"✅ Indicator calculation successful")
        
        # Show some SAR calculations
        latest_data = data_with_indicators.iloc[-1]
        print(f"📊 Latest SAR Data:")
        print(f"   Price: {latest_data['close']:.2f}")
        print(f"   SAR: {latest_data['sar']:.2f}")
        print(f"   SAR Diff: {latest_data['sar_diff']:.4f}")
        print(f"   SAR Below Price: {latest_data['sar_below_price']}")
        print(f"   SAR-Price Distance: {latest_data['sar_price_distance']:.4f} ({latest_data['sar_price_distance']*100:.2f}%)")
        
        # Test signal generation
        data_with_signals = strategy.generate_signals(data_with_indicators)
        print("✅ Signal generation successful")
        
        # Print summary statistics
        buy_signals = data_with_signals['buy_signal'].sum()
        sell_signals = data_with_signals['sell_signal'].sum()
        avg_strength = data_with_signals['signal_strength'].mean()
        max_strength = data_with_signals['signal_strength'].max()
        
        print(f"📊 Signal Summary:")
        print(f"   Buy signals: {buy_signals}")
        print(f"   Sell signals: {sell_signals}")
        print(f"   Average signal strength: {avg_strength:.3f}")
        print(f"   Maximum signal strength: {max_strength:.3f}")
        
        # Check SAR crossover frequency
        cross_above = data_with_indicators['sar_cross_above'].sum()
        cross_below = data_with_indicators['sar_cross_below'].sum()
        print(f"   SAR crosses above: {cross_above}")
        print(f"   SAR crosses below: {cross_below}")
        
        if buy_signals > 0 or sell_signals > 0:
            print(f"✅ Strategy generated {buy_signals + sell_signals} total signals - Ready for deployment")
        else:
            print("ℹ️  No signals in test data - May need parameter adjustment or different market conditions")
            
        # Test different parameters
        print("\n🔧 Testing with different SAR parameters...")
        alt_params = {'start': 0.01, 'increment': 0.01, 'maximum': 0.1, 'signal_threshold': 0.4}
        strategy_alt = Strategy163SarAlarmScript(parameters=alt_params)
        data_alt = strategy_alt.calculate_indicators(sample_data.copy())
        data_alt = strategy_alt.generate_signals(data_alt)
        
        alt_buy = data_alt['buy_signal'].sum()
        alt_sell = data_alt['sell_signal'].sum()
        print(f"   With start=0.01, increment=0.01, max=0.1: {alt_buy} buy, {alt_sell} sell signals")
        
        # Test more sensitive parameters
        sensitive_params = {'start': 0.05, 'increment': 0.05, 'maximum': 0.3, 'signal_threshold': 0.3}
        strategy_sens = Strategy163SarAlarmScript(parameters=sensitive_params)
        data_sens = strategy_sens.calculate_indicators(sample_data.copy())
        data_sens = strategy_sens.generate_signals(data_sens)
        
        sens_buy = data_sens['buy_signal'].sum()
        sens_sell = data_sens['sell_signal'].sum()
        print(f"   With start=0.05, increment=0.05, max=0.3: {sens_buy} buy, {sens_sell} sell signals")
            
    except Exception as e:
        print(f"❌ Strategy test failed with error: {e}")
        import traceback
        traceback.print_exc()