"""
Enhanced RSI Strategy based on optimization findings
Implements low-frequency, high-conviction trading approach
"""
import pandas as pd
import numpy as np
from typing import Dict, Any
from .base_strategy import BaseStrategy

class EnhancedRSIStrategy(BaseStrategy):
    """
    Enhanced RSI Strategy with optimizations from early findings:
    - Low frequency trading (target < 100 trades)
    - High signal strength requirements (> 0.7)
    - Conservative RSI levels
    - Trend confirmation
    - Volatility filtering
    """
    
    def __init__(self, name: str = "Enhanced RSI Strategy", parameters: Dict = None):
        base_params = {
            'rsi_period': 18,
            'rsi_oversold': 28,
            'rsi_overbought': 72,
            'ma_period': 50,
            'trend_strength': 0.02,
            'min_signal_strength': 0.75,
            'rsi_extreme_oversold': 20,
            'rsi_extreme_overbought': 80,
            'atr_period': 14,
            'min_volatility': 0.01,
            'max_volatility': 0.10,
            'base_position_size': 0.3,
            'max_position_size': 0.6,
            'trade_cooldown': 10,
        }
        if parameters:
            base_params.update(parameters)
        super().__init__(name=name, parameters=base_params)
        parameters = base_params
        
        # Optimized RSI parameters based on findings
        self.rsi_period = parameters.get('rsi_period', 18)  # Longer period for stability
        self.rsi_oversold = parameters.get('rsi_oversold', 28)  # More conservative
        self.rsi_overbought = parameters.get('rsi_overbought', 72)  # More conservative
        
        # Trend confirmation parameters
        self.ma_period = parameters.get('ma_period', 50)  # Medium-term trend
        self.trend_strength = parameters.get('trend_strength', 0.02)  # 2% trend threshold
        
        # Signal strength requirements (key finding: need high conviction)
        self.min_signal_strength = parameters.get('min_signal_strength', 0.75)  # Higher threshold
        self.rsi_extreme_oversold = parameters.get('rsi_extreme_oversold', 20)  # Very oversold
        self.rsi_extreme_overbought = parameters.get('rsi_extreme_overbought', 80)  # Very overbought
        
        # Volatility filter (ATR-based)
        self.atr_period = parameters.get('atr_period', 14)
        self.min_volatility = parameters.get('min_volatility', 0.01)  # 1% minimum volatility
        self.max_volatility = parameters.get('max_volatility', 0.10)  # 10% maximum volatility
        
        # Position sizing based on signal strength
        self.base_position_size = parameters.get('base_position_size', 0.3)  # Conservative base size
        self.max_position_size = parameters.get('max_position_size', 0.6)  # Maximum position
        
        # Cooldown period to reduce overtrading
        self.trade_cooldown = parameters.get('trade_cooldown', 10)  # Bars between trades
        self.last_trade_bar = -self.trade_cooldown
        
    def calculate_rsi(self, data: pd.DataFrame, period: int = None) -> pd.Series:
        """Calculate RSI with specified period"""
        if period is None:
            period = self.rsi_period
        
        # Use 'price' column if available, otherwise 'close'
        price_col = 'price' if 'price' in data.columns else 'close'
        
        delta = data[price_col].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_atr(self, data: pd.DataFrame, period: int = None) -> pd.Series:
        """Calculate Average True Range for volatility filtering"""
        if period is None:
            period = self.atr_period
        
        # Use 'price' column if available, otherwise 'close'
        price_col = 'price' if 'price' in data.columns else 'close'
        
        high = data['high'] if 'high' in data.columns else data[price_col] * 1.02
        low = data['low'] if 'low' in data.columns else data[price_col] * 0.98
        
        tr1 = high - low
        tr2 = abs(high - data[price_col].shift())
        tr3 = abs(low - data[price_col].shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        # Normalize ATR as percentage of price
        atr_pct = atr / data[price_col]
        return atr_pct
    
    def calculate_trend_strength(self, data: pd.DataFrame) -> pd.Series:
        """Calculate trend strength using moving average slope"""
        price_col = 'price' if 'price' in data.columns else 'close'
        ma = data[price_col].rolling(window=self.ma_period).mean()
        ma_change = ma.pct_change(periods=5)  # 5-bar rate of change
        return ma_change
    
    def calculate_signal_strength(self, rsi: float, trend: float, volatility: float) -> float:
        """
        Calculate composite signal strength (0-1)
        Higher values indicate stronger signals
        """
        strength = 0.0
        
        # RSI component (40% weight)
        if rsi <= self.rsi_extreme_oversold:
            rsi_strength = 1.0
        elif rsi <= self.rsi_oversold:
            rsi_strength = (self.rsi_oversold - rsi) / (self.rsi_oversold - self.rsi_extreme_oversold)
        elif rsi >= self.rsi_extreme_overbought:
            rsi_strength = 1.0
        elif rsi >= self.rsi_overbought:
            rsi_strength = (rsi - self.rsi_overbought) / (self.rsi_extreme_overbought - self.rsi_overbought)
        else:
            rsi_strength = 0.0
        
        # Trend component (40% weight)
        trend_strength = min(abs(trend) / self.trend_strength, 1.0) if abs(trend) > 0.005 else 0.0
        
        # Volatility component (20% weight)
        if self.min_volatility <= volatility <= self.max_volatility:
            vol_strength = 1.0 - abs(volatility - 0.03) / 0.07  # Optimal around 3%
        else:
            vol_strength = 0.0
        
        # Weighted combination
        strength = (rsi_strength * 0.4) + (trend_strength * 0.4) + (vol_strength * 0.2)
        
        return strength
    
    def calculate_position_size(self, signal_strength: float) -> float:
        """Dynamic position sizing based on signal strength"""
        if signal_strength < self.min_signal_strength:
            return 0.0
        
        # Scale position size with signal strength
        position_pct = (signal_strength - self.min_signal_strength) / (1.0 - self.min_signal_strength)
        position_size = self.base_position_size + (position_pct * (self.max_position_size - self.base_position_size))
        
        return min(position_size, self.max_position_size)
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators"""
        # RSI
        data['rsi'] = self.calculate_rsi(data)
        
        # Moving average for trend
        price_col = 'price' if 'price' in data.columns else 'close'
        data['ma'] = data[price_col].rolling(window=self.ma_period).mean()
        data['trend_strength'] = self.calculate_trend_strength(data)
        
        # ATR for volatility
        data['atr_pct'] = self.calculate_atr(data)
        
        # Signal strength
        data['signal_strength'] = data.apply(
            lambda row: self.calculate_signal_strength(
                row['rsi'], 
                row['trend_strength'], 
                row['atr_pct']
            ) if pd.notna(row['rsi']) else 0,
            axis=1
        )
        
        # Position sizing
        data['position_size'] = data['signal_strength'].apply(self.calculate_position_size)
        
        return data
    
    def check_cooldown(self, current_bar: int) -> bool:
        """Check if enough time has passed since last trade"""
        return (current_bar - self.last_trade_bar) >= self.trade_cooldown
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate buy/sell signals with strict criteria"""
        data['buy_signal'] = False
        data['sell_signal'] = False
        
        for i in range(self.ma_period + self.rsi_period, len(data)):
            # Skip if indicators not ready
            if pd.isna(data.iloc[i]['rsi']) or pd.isna(data.iloc[i]['ma']):
                continue
            
            # Check cooldown period
            if not self.check_cooldown(i):
                continue
            
            current = data.iloc[i]
            prev = data.iloc[i-1]
            
            price_col = 'price' if 'price' in data.columns else 'close'
            
            # Buy signal criteria
            buy_conditions = [
                current['rsi'] < self.rsi_oversold,  # RSI oversold
                prev['rsi'] >= self.rsi_oversold,  # RSI crossing up
                current[price_col] > current['ma'],  # Price above MA (uptrend)
                current['trend_strength'] > 0.001,  # Positive trend
                current['signal_strength'] >= self.min_signal_strength,  # Strong signal
                self.min_volatility <= current['atr_pct'] <= self.max_volatility,  # Good volatility
            ]
            
            # Sell signal criteria  
            sell_conditions = [
                current['rsi'] > self.rsi_overbought,  # RSI overbought
                prev['rsi'] <= self.rsi_overbought,  # RSI crossing down
                current[price_col] < current['ma'],  # Price below MA (downtrend)
                current['trend_strength'] < -0.001,  # Negative trend
                current['signal_strength'] >= self.min_signal_strength,  # Strong signal
                self.min_volatility <= current['atr_pct'] <= self.max_volatility,  # Good volatility
            ]
            
            if all(buy_conditions):
                data.at[data.index[i], 'buy_signal'] = True
                data.at[data.index[i], 'signal_strength'] = current['signal_strength']
                self.last_trade_bar = i
                
            elif all(sell_conditions):
                data.at[data.index[i], 'sell_signal'] = True
                data.at[data.index[i], 'signal_strength'] = current['signal_strength']
                self.last_trade_bar = i
        
        return data
    
    def get_strategy_name(self) -> str:
        """Return strategy name"""
        return "Enhanced RSI Strategy"
    
    def get_parameters(self) -> Dict[str, Any]:
        """Return current parameters"""
        return {
            'rsi_period': self.rsi_period,
            'rsi_oversold': self.rsi_oversold,
            'rsi_overbought': self.rsi_overbought,
            'ma_period': self.ma_period,
            'trend_strength': self.trend_strength,
            'min_signal_strength': self.min_signal_strength,
            'atr_period': self.atr_period,
            'min_volatility': self.min_volatility,
            'max_volatility': self.max_volatility,
            'base_position_size': self.base_position_size,
            'trade_cooldown': self.trade_cooldown
        }

    @classmethod
    def parameter_space(cls):
        return {
            'rsi_period': (10, 30),
            'rsi_oversold': (20, 35),
            'rsi_overbought': (65, 80),
            'ma_period': (30, 80),
            'trend_strength': (0.01, 0.05),
            'min_signal_strength': (0.6, 0.9),
            'rsi_extreme_oversold': (15, 25),
            'rsi_extreme_overbought': (75, 90),
            'atr_period': (10, 25),
            'min_volatility': (0.005, 0.02),
            'max_volatility': (0.05, 0.15),
            'base_position_size': (0.1, 0.5),
            'max_position_size': (0.4, 0.9),
            'trade_cooldown': (5, 20),
        }
