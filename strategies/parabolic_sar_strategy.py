#!/usr/bin/env python3
"""
Parabolic SAR Strategy - Dynamic stop-loss for trending markets
Designed to ride trends with trailing stops that adapt to price movement
"""
import pandas as pd
import numpy as np
from strategies.base_strategy import BaseStrategy

class ParabolicSARStrategy(BaseStrategy):
    def __init__(self, parameters=None):
        base_params = {
            'initial_af': 0.02,
            'max_af': 0.2,
            'af_increment': 0.02,
            'min_strength': 0.3,
            'position_size_pct': 0.7,
            'use_trend_filter': True,
            'trend_period': 50,
        }
        if parameters:
            base_params.update(parameters)
        super().__init__(name="ParabolicSAR", parameters=base_params)
        self.parameters = base_params
        
        # Parabolic SAR parameters
        self.initial_af = float(self.parameters.get('initial_af', 0.02))
        self.max_af = float(self.parameters.get('max_af', 0.2))
        self.af_increment = float(self.parameters.get('af_increment', 0.02))
        
        # Signal parameters
        self.min_strength = float(self.parameters.get('min_strength', 0.3))
        self.position_size_pct = float(self.parameters.get('position_size_pct', 0.7))
        
        # Trend confirmation
        self.use_trend_filter = bool(self.parameters.get('use_trend_filter', True))
        self.trend_period = int(self.parameters.get('trend_period', 50))
        
    def calculate_parabolic_sar(self, data):
        """Calculate Parabolic SAR indicator"""
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values
        
        # Initialize
        sar = np.zeros(len(data))
        ep = np.zeros(len(data))  # Extreme Point
        af = np.zeros(len(data))  # Acceleration Factor
        trend = np.zeros(len(data))  # 1 for uptrend, -1 for downtrend
        
        # Start with uptrend assumption
        sar[0] = low[0]
        ep[0] = high[0]
        af[0] = self.initial_af
        trend[0] = 1
        
        for i in range(1, len(data)):
            # Calculate SAR for current period
            sar[i] = sar[i-1] + af[i-1] * (ep[i-1] - sar[i-1])
            
            if trend[i-1] == 1:  # Uptrend
                # Check for trend reversal
                if low[i] <= sar[i]:
                    trend[i] = -1
                    sar[i] = ep[i-1]  # Set SAR to previous EP
                    ep[i] = low[i]
                    af[i] = self.initial_af
                else:
                    trend[i] = 1
                    
                    # Update EP and AF if new high
                    if high[i] > ep[i-1]:
                        ep[i] = high[i]
                        af[i] = min(af[i-1] + self.af_increment, self.max_af)
                    else:
                        ep[i] = ep[i-1]
                        af[i] = af[i-1]
                    
                    # SAR cannot be above the prior two lows
                    if i >= 2:
                        sar[i] = min(sar[i], low[i-1], low[i-2])
                        
            else:  # Downtrend
                # Check for trend reversal
                if high[i] >= sar[i]:
                    trend[i] = 1
                    sar[i] = ep[i-1]  # Set SAR to previous EP
                    ep[i] = high[i]
                    af[i] = self.initial_af
                else:
                    trend[i] = -1
                    
                    # Update EP and AF if new low
                    if low[i] < ep[i-1]:
                        ep[i] = low[i]
                        af[i] = min(af[i-1] + self.af_increment, self.max_af)
                    else:
                        ep[i] = ep[i-1]
                        af[i] = af[i-1]
                    
                    # SAR cannot be below the prior two highs
                    if i >= 2:
                        sar[i] = max(sar[i], high[i-1], high[i-2])
        
        return sar, trend, af
    
    def calculate_indicators(self, data):
        """Calculate Parabolic SAR and supporting indicators"""
        # Ensure we have OHLC data
        if 'high' not in data.columns or 'low' not in data.columns:
            data['high'] = data['close'] * 1.001
            data['low'] = data['close'] * 0.999
        
        # Calculate Parabolic SAR
        sar, trend, af = self.calculate_parabolic_sar(data)
        
        data['psar'] = sar
        data['psar_trend'] = trend
        data['psar_af'] = af
        
        # Calculate distance from SAR (for signal strength)
        data['psar_distance'] = abs(data['close'] - data['psar']) / data['close']
        
        # Add trend filter (SMA)
        if self.use_trend_filter:
            data['sma_trend'] = data['close'].rolling(window=self.trend_period).mean()
            data['above_trend'] = data['close'] > data['sma_trend']
        
        # Volatility for position sizing
        data['returns'] = data['close'].pct_change()
        data['volatility'] = data['returns'].rolling(20).std()
        
        # Market regime (trending vs ranging)
        data['atr'] = self.calculate_atr(data, 14)
        data['atr_ratio'] = data['atr'] / data['close']
        data['trending'] = data['atr_ratio'].rolling(20).mean() < data['atr_ratio'].rolling(50).mean()
        
        return data
    
    def calculate_atr(self, data, period):
        """Calculate Average True Range"""
        high = data['high']
        low = data['low']
        close = data['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr
    
    def generate_signals(self, data):
        """Generate buy/sell signals based on Parabolic SAR"""
        data['buy_signal'] = False
        data['sell_signal'] = False
        data['signal_strength'] = 0.0
        
        # Skip if insufficient data
        if len(data) < self.trend_period:
            return data
        
        # Detect SAR crossovers
        prev_trend = data['psar_trend'].shift(1)
        
        # Buy when trend changes from -1 to 1
        buy_condition = (
            (data['psar_trend'] == 1) & 
            (prev_trend == -1)
        )
        
        # Apply trend filter if enabled
        if self.use_trend_filter:
            buy_condition = buy_condition & data['above_trend']
        
        # Sell when trend changes from 1 to -1
        sell_condition = (
            (data['psar_trend'] == -1) & 
            (prev_trend == 1)
        )
        
        # Apply signals
        data.loc[buy_condition, 'buy_signal'] = True
        data.loc[sell_condition, 'sell_signal'] = True
        
        # Calculate signal strength
        # Base strength on distance from SAR and acceleration factor
        distance_strength = data['psar_distance'].clip(upper=0.1) * 10
        af_strength = data['psar_af'] / self.max_af
        
        # Combine factors
        base_strength = (distance_strength * 0.5 + af_strength * 0.3)
        
        # Boost in trending markets
        trend_boost = data['trending'].fillna(False).astype(float) * 0.2
        
        # Set signal strength
        data.loc[buy_condition, 'signal_strength'] = (
            base_strength[buy_condition] + trend_boost[buy_condition] + self.min_strength
        ).clip(upper=1.0)
        
        data.loc[sell_condition, 'signal_strength'] = (
            base_strength[sell_condition] + self.min_strength
        ).clip(upper=1.0)
        
        return data
    
    def get_position_size(self, signal_strength, current_price, portfolio_value):
        """
        Dynamic position sizing based on signal strength and volatility
        """
        # Base position from configuration
        base_position = self.position_size_pct
        
        # Adjust for signal strength
        strength_multiplier = 0.5 + (signal_strength * 0.5)
        
        # Final position size with cap
        position_pct = base_position * strength_multiplier
        position_pct = min(position_pct, 0.9)
        
        return portfolio_value * position_pct

    @classmethod
    def parameter_space(cls):
        return {
            'initial_af': (0.01, 0.05),
            'max_af': (0.1, 0.4),
            'af_increment': (0.01, 0.05),
            'min_strength': (0.1, 0.6),
            'position_size_pct': (0.3, 1.0),
            'trend_period': (20, 80),
        }
