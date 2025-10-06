"""
Adaptive Hybrid Strategy - Combines best features from multiple strategies
Dynamically adjusts parameters based on market conditions
"""

from typing import Dict, Optional
import pandas as pd
import numpy as np
import logging
from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

class AdaptiveHybridStrategy(BaseStrategy):
    """
    Advanced hybrid strategy that combines:
    - RSI for momentum detection
    - MACD for trend confirmation  
    - Bollinger Bands for volatility analysis
    - Volume analysis for signal strength
    - ATR for dynamic position sizing
    
    Features adaptive parameter adjustment based on market regime
    """
    
    def __init__(self, parameters: Dict = None):
        default_params = {
            # RSI Parameters
            'rsi_period': 14,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            
            # MACD Parameters
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            
            # Bollinger Bands
            'bb_period': 20,
            'bb_std': 2.0,
            
            # ATR for volatility
            'atr_period': 14,
            'atr_multiplier': 2.0,
            
            # Volume analysis
            'volume_ma_period': 20,
            'volume_spike_threshold': 2.0,
            
            # Market regime detection
            'trend_ma_period': 50,
            'regime_lookback': 100,
            
            # Signal thresholds
            'min_strength': 0.65,
            'confirmation_required': 2,  # Number of indicators that must agree
            
            # Adaptive parameters
            'adaptive_mode': True,
            'volatility_adjustment': True,
            
            # Timeframe
            'timeframe_minutes': 15
        }
        
        if parameters:
            default_params.update(parameters)
        
        super().__init__("AdaptiveHybridStrategy", default_params)
        
        # Track market regime
        self.current_regime = 'neutral'
        self.regime_confidence = 0.0
    
    def detect_market_regime(self, data: pd.DataFrame) -> str:
        """
        Detect current market regime: trending, ranging, or volatile
        """
        if len(data) < self.parameters['regime_lookback']:
            return 'neutral'
        
        recent_data = data.tail(self.parameters['regime_lookback'])
        
        # Calculate trend strength
        trend_ma = recent_data['price'].rolling(window=self.parameters['trend_ma_period']).mean()
        price_above_ma = (recent_data['price'] > trend_ma).sum() / len(recent_data)
        
        # Calculate volatility
        returns = recent_data['price'].pct_change()
        volatility = returns.std()
        avg_volatility = returns.rolling(window=20).std().mean()
        
        # Detect regime
        if volatility > avg_volatility * 1.5:
            regime = 'volatile'
            confidence = min(volatility / (avg_volatility * 2), 1.0)
        elif price_above_ma > 0.65 or price_above_ma < 0.35:
            regime = 'trending'
            confidence = abs(price_above_ma - 0.5) * 2
        else:
            regime = 'ranging'
            confidence = 1.0 - abs(price_above_ma - 0.5) * 2
        
        self.current_regime = regime
        self.regime_confidence = confidence
        
        return regime
    
    def adapt_parameters(self, data: pd.DataFrame):
        """
        Dynamically adjust parameters based on market conditions
        """
        if not self.parameters['adaptive_mode']:
            return
        
        regime = self.detect_market_regime(data)
        
        if regime == 'trending':
            # In trending markets, use longer periods and wider bands
            self.parameters['rsi_period'] = 21
            self.parameters['bb_std'] = 2.5
            self.parameters['min_strength'] = 0.6
            
        elif regime == 'ranging':
            # In ranging markets, use shorter periods and tighter bands
            self.parameters['rsi_period'] = 9
            self.parameters['bb_std'] = 1.5
            self.parameters['min_strength'] = 0.7
            
        elif regime == 'volatile':
            # In volatile markets, be more conservative
            self.parameters['rsi_period'] = 14
            self.parameters['bb_std'] = 3.0
            self.parameters['min_strength'] = 0.75
            self.parameters['confirmation_required'] = 3
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators"""
        if not self.validate_data(data):
            return data
        
        df = data.copy()
        
        # Adapt parameters based on market conditions
        self.adapt_parameters(df)
        
        # RSI
        delta = df['price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.parameters['rsi_period']).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.parameters['rsi_period']).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['price'].ewm(span=self.parameters['macd_fast'], adjust=False).mean()
        exp2 = df['price'].ewm(span=self.parameters['macd_slow'], adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=self.parameters['macd_signal'], adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        df['bb_middle'] = df['price'].rolling(window=self.parameters['bb_period']).mean()
        bb_std = df['price'].rolling(window=self.parameters['bb_period']).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * self.parameters['bb_std'])
        df['bb_lower'] = df['bb_middle'] - (bb_std * self.parameters['bb_std'])
        df['bb_width'] = df['bb_upper'] - df['bb_lower']
        df['bb_position'] = (df['price'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # ATR for volatility
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['atr'] = true_range.rolling(window=self.parameters['atr_period']).mean()
        
        # Volume analysis
        df['volume_ma'] = df['volume'].rolling(window=self.parameters['volume_ma_period']).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # Trend MA
        df['trend_ma'] = df['price'].rolling(window=self.parameters['trend_ma_period']).mean()
        df['price_position'] = (df['price'] - df['trend_ma']) / df['trend_ma']
        
        # Market regime
        df['market_regime'] = self.current_regime
        df['regime_confidence'] = self.regime_confidence
        
        return df
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals from multiple indicators"""
        df = data.copy()
        
        if 'rsi' not in df.columns:
            logger.error("Indicators not calculated")
            df['buy_signal'] = False
            df['sell_signal'] = False
            df['signal_strength'] = 0.0
            return df
        
        # Initialize signals
        df['buy_signal'] = False
        df['sell_signal'] = False
        df['signal_strength'] = 0.0
        
        # Skip early rows with NaN values
        min_periods = max(
            self.parameters['rsi_period'],
            self.parameters['macd_slow'],
            self.parameters['bb_period'],
            self.parameters['trend_ma_period']
        )
        
        if len(df) < min_periods:
            return df
        
        # Calculate individual signals
        df['rsi_buy'] = (df['rsi'] < self.parameters['rsi_oversold']) & (df['rsi'].shift(1) >= self.parameters['rsi_oversold'])
        df['rsi_sell'] = (df['rsi'] > self.parameters['rsi_overbought']) & (df['rsi'].shift(1) <= self.parameters['rsi_overbought'])
        
        df['macd_buy'] = (df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1))
        df['macd_sell'] = (df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1))
        
        df['bb_buy'] = df['price'] <= df['bb_lower']
        df['bb_sell'] = df['price'] >= df['bb_upper']
        
        df['volume_confirmation'] = df['volume_ratio'] > self.parameters['volume_spike_threshold']
        
        # Count confirmations
        df['buy_confirmations'] = (
            df['rsi_buy'].astype(int) +
            df['macd_buy'].astype(int) +
            df['bb_buy'].astype(int)
        )
        
        df['sell_confirmations'] = (
            df['rsi_sell'].astype(int) +
            df['macd_sell'].astype(int) +
            df['bb_sell'].astype(int)
        )
        
        # Generate signals based on confirmations
        min_confirmations = self.parameters['confirmation_required']
        
        # Buy signals
        buy_condition = (
            (df['buy_confirmations'] >= min_confirmations) |
            ((df['buy_confirmations'] >= min_confirmations - 1) & df['volume_confirmation'])
        )
        
        # Sell signals
        sell_condition = (
            (df['sell_confirmations'] >= min_confirmations) |
            ((df['sell_confirmations'] >= min_confirmations - 1) & df['volume_confirmation'])
        )
        
        # Calculate signal strength
        for i in range(len(df)):
            if i < min_periods:
                continue
            
            strength = 0.0
            
            if buy_condition.iloc[i]:
                # Base strength from confirmations
                strength = 0.3 + (df['buy_confirmations'].iloc[i] * 0.2)
                
                # RSI strength
                if df['rsi'].iloc[i] < 20:
                    strength += 0.1
                    
                # MACD momentum
                if df['macd_histogram'].iloc[i] > 0:
                    strength += 0.1
                    
                # Volume confirmation
                if df['volume_confirmation'].iloc[i]:
                    strength += 0.15
                    
                # Market regime bonus
                if self.current_regime == 'ranging' and df['bb_position'].iloc[i] < 0.2:
                    strength += 0.1
                elif self.current_regime == 'trending' and df['price'].iloc[i] > df['trend_ma'].iloc[i]:
                    strength += 0.05
                    
                df.loc[df.index[i], 'buy_signal'] = True
                df.loc[df.index[i], 'signal_strength'] = min(strength, 1.0)
                
            elif sell_condition.iloc[i]:
                # Base strength from confirmations
                strength = 0.3 + (df['sell_confirmations'].iloc[i] * 0.2)
                
                # RSI strength
                if df['rsi'].iloc[i] > 80:
                    strength += 0.1
                    
                # MACD momentum
                if df['macd_histogram'].iloc[i] < 0:
                    strength += 0.1
                    
                # Volume confirmation
                if df['volume_confirmation'].iloc[i]:
                    strength += 0.15
                    
                # Market regime bonus
                if self.current_regime == 'ranging' and df['bb_position'].iloc[i] > 0.8:
                    strength += 0.1
                elif self.current_regime == 'trending' and df['price'].iloc[i] < df['trend_ma'].iloc[i]:
                    strength += 0.05
                    
                df.loc[df.index[i], 'sell_signal'] = True
                df.loc[df.index[i], 'signal_strength'] = min(strength, 1.0)
        
        # Filter by minimum strength
        weak_signals = df['signal_strength'] < self.parameters['min_strength']
        df.loc[weak_signals, 'buy_signal'] = False
        df.loc[weak_signals, 'sell_signal'] = False
        df.loc[weak_signals, 'signal_strength'] = 0.0
        
        return df
    
    def get_dynamic_position_size(self, data: pd.DataFrame) -> float:
        """
        Calculate dynamic position size based on volatility (ATR)
        """
        if 'atr' not in data.columns or data['atr'].empty:
            return 0.5  # Default position size
        
        current_atr = data['atr'].iloc[-1]
        avg_atr = data['atr'].mean()
        
        if pd.isna(current_atr) or pd.isna(avg_atr) or avg_atr == 0:
            return 0.5
        
        # Lower position size in high volatility
        volatility_ratio = current_atr / avg_atr
        
        if volatility_ratio > 1.5:
            return 0.3  # Small position in high volatility
        elif volatility_ratio > 1.2:
            return 0.4
        elif volatility_ratio < 0.8:
            return 0.6  # Larger position in low volatility
        else:
            return 0.5  # Normal position
    
    def get_strategy_info(self) -> Dict:
        """Get detailed strategy information"""
        return {
            'name': self.name,
            'description': 'Adaptive hybrid strategy combining RSI, MACD, Bollinger Bands, and Volume analysis',
            'parameters': self.parameters,
            'current_regime': self.current_regime,
            'regime_confidence': self.regime_confidence,
            'features': [
                'Market regime detection',
                'Adaptive parameter adjustment',
                'Multi-indicator confirmation',
                'Volume-based signal strength',
                'Dynamic position sizing based on ATR'
            ]
        }