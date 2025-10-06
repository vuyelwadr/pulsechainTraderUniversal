#!/usr/bin/env python3
"""
Strategy 040: Laguerre RSI

LazyBear Strategy Number: 016 (Original)
Strategy ID: 040 (Canonical)
LazyBear Name: Laguerre RSI
Type: oscillator/momentum
TradingView URL: https://www.tradingview.com/v/PFsgmnZW/ (referenced)

Description:
Implements the Laguerre RSI indicator, which uses Laguerre filters to create
a smoother RSI with reduced lag. This oscillator applies recursive Laguerre 
filtering (L0-L3 components) and then computes cumulative up/down movements
to generate an RSI-like value in the 0-1 range.

Mathematical Formula:
1. Laguerre Filter Components (recursive):
   - L0 = (1-γ)*Src + γ*L0[1]
   - L1 = -γ*L0 + L0[1] + γ*L1[1]
   - L2 = -γ*L1 + L1[1] + γ*L2[1]
   - L3 = -γ*L2 + L2[1] + γ*L3[1]
   where γ = 1-α (alpha is the damping factor)

2. Cumulative Up/Down Movements:
   - cu = sum of positive differences between consecutive L components
   - cd = sum of negative differences between consecutive L components

3. Laguerre RSI:
   - LaRSI = cu / (cu + cd) when (cu + cd) != 0, else 0
"""

import pandas as pd
import numpy as np
import sys
import os

# Add parent directories to path for imports
repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, repo_root)
from strategies.base_strategy import BaseStrategy


class Strategy040LaguerreRsi(BaseStrategy):
    """
    Strategy 040: Laguerre RSI
    
    Uses Laguerre filters to create a smoother RSI with reduced lag.
    This oscillator is more responsive than traditional RSI while maintaining smoothness.
    """
    
    def __init__(self, parameters=None):
        """
        Initialize strategy with parameters
        
        Parameters:
        - alpha: Laguerre filter damping factor (0.0-1.0, default 0.2)
        - overbought: Overbought threshold (default 0.8)
        - oversold: Oversold threshold (default 0.2) 
        - signal_threshold: Minimum signal strength to trade (default 0.6)
        """
        default_params = {
            'alpha': 0.2,           # Laguerre damping factor
            'overbought': 0.8,      # Overbought level 
            'oversold': 0.2,        # Oversold level
            'signal_threshold': 0.6, # Minimum signal strength to trade
        }
        
        if parameters:
            default_params.update(parameters)
        
        # Initialize parent class
        super().__init__(
            name="Strategy_040_LaguerreRsi",
            parameters=default_params
        )
    
    def _laguerre_filter(self, src: pd.Series, alpha: float) -> tuple:
        """
        Calculate Laguerre filter components L0, L1, L2, L3
        
        Args:
            src: Input price series
            alpha: Damping factor (0-1)
            
        Returns:
            Tuple of (L0, L1, L2, L3) series
        """
        gamma = 1 - alpha
        n = len(src)
        
        # Initialize arrays
        L0 = np.zeros(n)
        L1 = np.zeros(n)
        L2 = np.zeros(n)
        L3 = np.zeros(n)
        
        # Set initial values
        L0[0] = src.iloc[0]
        L1[0] = src.iloc[0]
        L2[0] = src.iloc[0]
        L3[0] = src.iloc[0]
        
        # Calculate recursive components
        for i in range(1, n):
            L0[i] = (1 - gamma) * src.iloc[i] + gamma * L0[i-1]
            L1[i] = -gamma * L0[i] + L0[i-1] + gamma * L1[i-1]
            L2[i] = -gamma * L1[i] + L1[i-1] + gamma * L2[i-1]
            L3[i] = -gamma * L2[i] + L2[i-1] + gamma * L3[i-1]
        
        return pd.Series(L0, index=src.index), pd.Series(L1, index=src.index), \
               pd.Series(L2, index=src.index), pd.Series(L3, index=src.index)
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Laguerre RSI indicator
        """
        # Ensure we have required columns
        if 'close' not in data.columns and 'price' in data.columns:
            data['close'] = data['price']
        
        if 'close' not in data.columns:
            raise ValueError("No price data available ('close' or 'price' column required)")
        
        # Extract parameters
        alpha = self.parameters['alpha']
        
        # Calculate Laguerre filter components
        L0, L1, L2, L3 = self._laguerre_filter(data['close'], alpha)
        
        # Store intermediate components for debugging
        data['L0'] = L0
        data['L1'] = L1
        data['L2'] = L2
        data['L3'] = L3
        
        # Calculate cumulative up movements (cu)
        cu1 = np.where(L0 > L1, L0 - L1, 0)
        cu2 = np.where(L1 > L2, L1 - L2, 0)  
        cu3 = np.where(L2 > L3, L2 - L3, 0)
        cu = cu1 + cu2 + cu3
        
        # Calculate cumulative down movements (cd)
        cd1 = np.where(L0 < L1, L1 - L0, 0)
        cd2 = np.where(L1 < L2, L2 - L1, 0)
        cd3 = np.where(L2 < L3, L3 - L2, 0)
        cd = cd1 + cd2 + cd3
        
        # Calculate Laguerre RSI
        # Avoid division by zero safely
        total_movement = cu + cd
        
        # Use numpy divide with where to handle zeros properly
        with np.errstate(divide='ignore', invalid='ignore'):
            lrsi_values = np.divide(cu, total_movement, out=np.zeros_like(cu), where=(total_movement != 0))
        
        # Set neutral value (0.5) where there's no movement
        data['laguerre_rsi'] = np.where(total_movement == 0, 0.5, lrsi_values)
        
        return data
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate buy/sell signals based on Laguerre RSI
        
        Buy signals: LaRSI crosses above oversold level
        Sell signals: LaRSI crosses below overbought level
        """
        # Extract parameters
        overbought = self.parameters['overbought']
        oversold = self.parameters['oversold']
        threshold = self.parameters['signal_threshold']
        
        # Initialize signal columns
        data['buy_signal'] = False
        data['sell_signal'] = False
        data['signal_strength'] = 0.0
        
        # Generate conditions
        lrsi = data['laguerre_rsi']
        
        # Buy when crossing above oversold level
        buy_condition = (
            (lrsi > oversold) & 
            (lrsi.shift(1) <= oversold)
        )
        
        # Sell when crossing below overbought level
        sell_condition = (
            (lrsi < overbought) & 
            (lrsi.shift(1) >= overbought)
        )
        
        # Create raw signal series for position state management
        raw_signal = np.select(
            [buy_condition, sell_condition],
            [1, -1],  # 1 for buy, -1 for sell
            default=0
        )
        
        # Position state management to prevent look-ahead bias
        position = pd.Series(raw_signal).replace(0, np.nan).ffill().fillna(0)
        is_flat = (position.shift(1) == 0)
        is_long = (position.shift(1) == 1)
        is_short = (position.shift(1) == -1)
        
        # Apply signals based on previous position state
        data.loc[buy_condition & (is_flat | is_short), 'buy_signal'] = True
        data.loc[sell_condition & (is_long | is_flat), 'sell_signal'] = True
        
        # Calculate signal strength based on distance from extreme levels
        # Stronger signals when further from neutral (0.5)
        buy_strength = np.where(
            data['buy_signal'],
            np.clip((oversold + 0.3 - lrsi) / 0.3, 0.3, 1.0),  # Stronger near oversold
            0.0
        )
        
        sell_strength = np.where(
            data['sell_signal'], 
            np.clip((lrsi - overbought + 0.3) / 0.3, 0.3, 1.0),  # Stronger near overbought
            0.0
        )
        
        # Set final signal strength (positive for buy, negative for sell)
        data.loc[data['buy_signal'], 'signal_strength'] = buy_strength[data['buy_signal']]
        data.loc[data['sell_signal'], 'signal_strength'] = -sell_strength[data['sell_signal']]
        
        # Apply minimum threshold filter
        weak_signals = np.abs(data['signal_strength']) < threshold
        data.loc[weak_signals, 'buy_signal'] = False
        data.loc[weak_signals, 'sell_signal'] = False
        data.loc[weak_signals, 'signal_strength'] = 0.0
        
        return data
    
    def validate_signals(self, data: pd.DataFrame) -> bool:
        """
        Validate that signals are properly formed
        """
        # Check for required columns
        required = ['buy_signal', 'sell_signal', 'signal_strength', 'laguerre_rsi']
        if not all(col in data.columns for col in required):
            return False
        
        # Check for NaN values in signal columns
        signal_cols = ['buy_signal', 'sell_signal', 'signal_strength']
        if data[signal_cols].isna().any().any():
            return False
        
        # Check for simultaneous buy and sell signals
        if (data['buy_signal'] & data['sell_signal']).any():
            return False
        
        # Check signal strength is in valid range
        if (data['signal_strength'] < -1).any() or (data['signal_strength'] > 1).any():
            return False
        
        # Check Laguerre RSI is in valid range [0, 1]
        if (data['laguerre_rsi'] < 0).any() or (data['laguerre_rsi'] > 1).any():
            return False
        
        return True


# Test function for smoke testing
if __name__ == "__main__":
    import datetime
    
    print("🔍 Testing Strategy 040: Laguerre RSI")
    
    # Create sample data with realistic price movements
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='5min')
    n_points = len(dates)
    
    # Create trending price data with some noise
    base_price = 100
    trend = np.linspace(0, 20, n_points)  # Upward trend
    noise = np.random.randn(n_points).cumsum() * 0.5
    
    sample_data = pd.DataFrame({
        'timestamp': dates,
        'open': base_price + trend + noise + np.random.randn(n_points) * 0.1,
        'high': base_price + trend + noise + np.random.randn(n_points) * 0.1 + 0.5,
        'low': base_price + trend + noise + np.random.randn(n_points) * 0.1 - 0.5,
        'close': base_price + trend + noise,
        'volume': np.random.randint(1000, 10000, n_points)
    })
    
    # Ensure OHLC consistency
    sample_data['high'] = sample_data[['open', 'close', 'high']].max(axis=1)
    sample_data['low'] = sample_data[['open', 'close', 'low']].min(axis=1)
    sample_data['price'] = sample_data['close']
    
    print(f"Sample data shape: {sample_data.shape}")
    
    try:
        # Test strategy initialization
        strategy = Strategy016LaguerreRSI()
        print("✅ Strategy initialization successful")
        
        # Test indicator calculation
        data_with_indicators = strategy.calculate_indicators(sample_data.copy())
        print(f"✅ Indicator calculation successful")
        
        # Print indicator statistics
        lrsi = data_with_indicators['laguerre_rsi']
        print(f"   Laguerre RSI range: [{lrsi.min():.3f}, {lrsi.max():.3f}]")
        print(f"   Laguerre RSI mean: {lrsi.mean():.3f}")
        
        # Test signal generation
        data_with_signals = strategy.generate_signals(data_with_indicators)
        print("✅ Signal generation successful")
        
        # Validate signals
        if strategy.validate_signals(data_with_signals):
            print("✅ Signal validation passed")
            
            # Print summary statistics
            buy_signals = data_with_signals['buy_signal'].sum()
            sell_signals = data_with_signals['sell_signal'].sum()
            avg_strength = data_with_signals['signal_strength'].mean()
            min_strength = data_with_signals['signal_strength'].min()
            max_strength = data_with_signals['signal_strength'].max()
            
            print(f"📊 Signal Summary:")
            print(f"   Buy signals: {buy_signals}")
            print(f"   Sell signals: {sell_signals}")
            print(f"   Average signal strength: {avg_strength:.3f}")
            print(f"   Min signal strength: {min_strength:.3f}")
            print(f"   Max signal strength: {max_strength:.3f}")
            
            # Check extreme values
            overbought_periods = (data_with_signals['laguerre_rsi'] > 0.8).sum()
            oversold_periods = (data_with_signals['laguerre_rsi'] < 0.2).sum()
            print(f"   Overbought periods (>0.8): {overbought_periods}")
            print(f"   Oversold periods (<0.2): {oversold_periods}")
            
            if buy_signals > 0 or sell_signals > 0:
                print(f"✅ Strategy generated {buy_signals + sell_signals} total signals - Ready for deployment")
            else:
                print("ℹ️  No signals in test data - This can be normal for oscillator strategies")
                
        else:
            print("❌ Signal validation failed")
            
    except Exception as e:
        print(f"❌ Strategy test failed with error: {e}")
        import traceback
        traceback.print_exc()