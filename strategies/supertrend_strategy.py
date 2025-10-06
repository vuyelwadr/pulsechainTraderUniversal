#!/usr/bin/env python3
"""
Supertrend Strategy - Designed for trending crypto markets
Uses ATR-based trend following to stay in positions longer
"""
import pandas as pd
import numpy as np
from strategies.base_strategy import BaseStrategy

class SupertrendStrategy(BaseStrategy):
    def __init__(self, parameters=None):
        base_params = {
            'atr_period': 10,
            'multiplier': 3.0,
            'min_strength': 0.3,
            'position_size_pct': 0.7,
        }
        if parameters:
            base_params.update(parameters)
        super().__init__(name="Supertrend", parameters=base_params)
        self.parameters = base_params
        
        # Supertrend parameters
        self.atr_period = int(self.parameters.get('atr_period', 10))
        self.multiplier = float(self.parameters.get('multiplier', 3.0))
        
        # Signal strength threshold
        self.min_strength = float(self.parameters.get('min_strength', 0.3))
        
        # Position sizing
        self.position_size_pct = float(self.parameters.get('position_size_pct', 0.7))
        
    def calculate_atr(self, data, period):
        """Calculate Average True Range"""
        high = data['high']
        low = data['low']
        close = data['close']
        
        # True Range calculation
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr
    
    def calculate_supertrend(self, data):
        """Calculate Supertrend indicator"""
        df = data.copy()
        
        # Calculate ATR
        atr = self.calculate_atr(df, self.atr_period)
        
        # Calculate basic bands
        hl_avg = (df['high'] + df['low']) / 2
        final_upperband = hl_avg + (self.multiplier * atr)
        final_lowerband = hl_avg - (self.multiplier * atr)
        
        # Initialize Supertrend
        supertrend = pd.Series(index=df.index, dtype=float)
        direction = pd.Series(index=df.index, dtype=float)
        
        for i in range(self.atr_period, len(df)):
            # Determine trend direction
            if i == self.atr_period:
                if df['close'].iloc[i] <= final_upperband.iloc[i]:
                    direction.iloc[i] = 1
                else:
                    direction.iloc[i] = -1
            else:
                if df['close'].iloc[i] <= final_upperband.iloc[i]:
                    if direction.iloc[i-1] == 1:
                        if final_lowerband.iloc[i] <= final_lowerband.iloc[i-1]:
                            final_lowerband.iloc[i] = final_lowerband.iloc[i-1]
                    direction.iloc[i] = 1
                elif df['close'].iloc[i] > final_upperband.iloc[i]:
                    if direction.iloc[i-1] == -1:
                        if final_upperband.iloc[i] >= final_upperband.iloc[i-1]:
                            final_upperband.iloc[i] = final_upperband.iloc[i-1]
                    direction.iloc[i] = -1
                else:
                    direction.iloc[i] = direction.iloc[i-1]
                    
                    if direction.iloc[i] == 1:
                        if final_lowerband.iloc[i] <= final_lowerband.iloc[i-1]:
                            final_lowerband.iloc[i] = final_lowerband.iloc[i-1]
                    else:
                        if final_upperband.iloc[i] >= final_upperband.iloc[i-1]:
                            final_upperband.iloc[i] = final_upperband.iloc[i-1]
            
            # Set Supertrend value
            if direction.iloc[i] == 1:
                supertrend.iloc[i] = final_lowerband.iloc[i]
            else:
                supertrend.iloc[i] = final_upperband.iloc[i]
        
        return supertrend, direction
    
    def calculate_indicators(self, data):
        """Calculate Supertrend indicators"""
        # Ensure we have OHLC data
        if 'high' not in data.columns or 'low' not in data.columns:
            data['high'] = data['close'] * 1.001
            data['low'] = data['close'] * 0.999
        
        # Calculate Supertrend
        supertrend, direction = self.calculate_supertrend(data)
        
        data['supertrend'] = supertrend
        data['st_direction'] = direction
        
        # Calculate trend strength (distance from Supertrend)
        data['trend_strength'] = abs(data['close'] - data['supertrend']) / data['close']
        
        # Market regime detection (volatility-based)
        data['returns'] = data['close'].pct_change()
        data['volatility_20'] = data['returns'].rolling(20).std()
        data['volatility_60'] = data['returns'].rolling(60).std()
        data['trending_market'] = data['volatility_20'] < data['volatility_60']
        
        return data
    
    def generate_signals(self, data):
        """Generate buy/sell signals based on Supertrend"""
        data['buy_signal'] = False
        data['sell_signal'] = False
        data['signal_strength'] = 0.0
        
        # Skip if insufficient data
        if len(data) < max(self.atr_period, 60):
            return data
        
        # Detect Supertrend crossovers
        prev_direction = data['st_direction'].shift(1)
        
        # Buy when price crosses above Supertrend (direction changes from -1 to 1)
        buy_condition = (
            (data['st_direction'] == 1) & 
            (prev_direction == -1) &
            (data['trend_strength'] > 0.001)
        )
        
        # Sell when price crosses below Supertrend (direction changes from 1 to -1)
        sell_condition = (
            (data['st_direction'] == -1) & 
            (prev_direction == 1)
        )
        
        # Apply signals
        data.loc[buy_condition, 'buy_signal'] = True
        data.loc[sell_condition, 'sell_signal'] = True
        
        # Calculate signal strength based on trend strength and market regime
        strength_base = data['trend_strength'].fillna(0)
        
        # Boost strength in trending markets
        trending_boost = data['trending_market'].fillna(False).astype(float) * 0.3
        
        # Set signal strength
        data.loc[buy_condition, 'signal_strength'] = (
            strength_base[buy_condition] + trending_boost[buy_condition] + self.min_strength
        )
        
        data.loc[sell_condition, 'signal_strength'] = (
            strength_base[sell_condition] + trending_boost[sell_condition] + self.min_strength
        )
        
        # Ensure minimum strength
        data['signal_strength'] = data['signal_strength'].clip(lower=0, upper=1)
        
        return data
    
    def get_position_size(self, signal_strength, current_price, portfolio_value):
        """
        Dynamic position sizing based on signal strength
        Uses modified Kelly Criterion with safety cap
        """
        # Base position from configuration
        base_position = self.position_size_pct
        
        # Apply Kelly-inspired sizing with 25% cap
        kelly_fraction = min(signal_strength * 0.5, 0.25)
        
        # Combine base and Kelly sizing
        position_pct = base_position * (0.7 + kelly_fraction)
        
        # Cap at 90% of portfolio
        position_pct = min(position_pct, 0.9)
        
        return portfolio_value * position_pct

    @classmethod
    def parameter_space(cls):
        return {
            'atr_period': (5, 40),
            'multiplier': (1.5, 5.0),
            'min_strength': (0.1, 0.6),
            'position_size_pct': (0.2, 1.0),
        }
