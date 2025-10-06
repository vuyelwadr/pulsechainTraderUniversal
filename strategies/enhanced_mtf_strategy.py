"""
Enhanced Multi-Timeframe Momentum Strategy
Target: Beat 28.60% return by being more aggressive while maintaining quality
"""
import pandas as pd
import numpy as np
from .base_strategy import BaseStrategy

class EnhancedMTFStrategy(BaseStrategy):
    """
    Enhanced version of the winning MultiTimeframeMomentumStrategy
    Modifications:
    1. More aggressive signal thresholds
    2. Dynamic position sizing based on signal strength
    3. Additional momentum confirmation layers
    4. Volatility-adjusted parameters
    """
    
    def __init__(self, parameters: dict = None):
        base_params = {
            'ltf_period': 3,
            'htf_period': 15,
            'mtf_period': 30,
            'min_strength': 0.3,
            'momentum_threshold': 0.5,
            'volume_multiplier': 0.8,
            'base_position': 0.6,
            'max_position': 0.9,
            'use_trailing_stop': True,
            'trailing_stop_pct': 5.0,
            'use_acceleration': True,
            'use_divergence': True,
            'adaptive_periods': True,
        }
        if parameters:
            base_params.update(parameters)
        super().__init__("EnhancedMTF", base_params)
        self.parameters = base_params
        
        # Core MTF parameters (optimized)
        self.ltf_period = int(self.parameters.get('ltf_period', 3))
        self.htf_period = int(self.parameters.get('htf_period', 15))
        self.mtf_period = int(self.parameters.get('mtf_period', 30))
        
        # Signal parameters (more aggressive)
        self.min_strength = float(self.parameters.get('min_strength', 0.3))
        self.momentum_threshold = float(self.parameters.get('momentum_threshold', 0.5))
        self.volume_multiplier = float(self.parameters.get('volume_multiplier', 0.8))
        
        # Position sizing
        self.base_position = float(self.parameters.get('base_position', 0.6))
        self.max_position = float(self.parameters.get('max_position', 0.9))
        
        # Risk management
        self.use_trailing_stop = bool(self.parameters.get('use_trailing_stop', True))
        self.trailing_stop_pct = float(self.parameters.get('trailing_stop_pct', 5.0))
        
        # Advanced parameters
        self.use_acceleration = bool(self.parameters.get('use_acceleration', True))
        self.use_divergence = bool(self.parameters.get('use_divergence', True))
        self.adaptive_periods = bool(self.parameters.get('adaptive_periods', True))
        
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate enhanced indicators with multiple confirmations"""
        df = data.copy()
        
        # Adaptive period adjustment based on volatility
        if self.adaptive_periods:
            volatility = df['close'].pct_change().rolling(20).std()
            vol_factor = volatility / volatility.mean()
            vol_factor = vol_factor.fillna(1.0).clip(0.5, 1.5)
            
            # Adjust periods dynamically
            ltf_period = (self.ltf_period * vol_factor).round().astype(int).clip(2, 10)
            htf_period = (self.htf_period * vol_factor).round().astype(int).clip(10, 30)
            mtf_period = (self.mtf_period * vol_factor).round().astype(int).clip(20, 50)
        else:
            ltf_period = pd.Series([self.ltf_period] * len(df))
            htf_period = pd.Series([self.htf_period] * len(df))
            mtf_period = pd.Series([self.mtf_period] * len(df))
        
        # Calculate momentum for multiple timeframes
        for i in range(len(df)):
            if i > 0:
                df.loc[df.index[i], 'momentum_ltf'] = df['close'].iloc[max(0, i-ltf_period.iloc[i]):i+1].pct_change(ltf_period.iloc[i]).iloc[-1] * 100 if i >= ltf_period.iloc[i] else 0
                df.loc[df.index[i], 'momentum_htf'] = df['close'].iloc[max(0, i-htf_period.iloc[i]):i+1].pct_change(htf_period.iloc[i]).iloc[-1] * 100 if i >= htf_period.iloc[i] else 0
                df.loc[df.index[i], 'momentum_mtf'] = df['close'].iloc[max(0, i-mtf_period.iloc[i]):i+1].pct_change(mtf_period.iloc[i]).iloc[-1] * 100 if i >= mtf_period.iloc[i] else 0
        
        # Rate of change acceleration
        if self.use_acceleration:
            df['momentum_accel_ltf'] = df['momentum_ltf'].diff()
            df['momentum_accel_htf'] = df['momentum_htf'].diff()
            df['momentum_accel_mtf'] = df['momentum_mtf'].diff()
        
        # Moving averages with dynamic periods
        df['ma_fast'] = df['close'].rolling(window=self.ltf_period).mean()
        df['ma_slow'] = df['close'].rolling(window=self.htf_period).mean()
        df['ma_baseline'] = df['close'].rolling(window=self.mtf_period).mean()
        
        # Price position relative to MAs
        df['price_vs_fast'] = (df['close'] - df['ma_fast']) / df['ma_fast'] * 100
        df['price_vs_slow'] = (df['close'] - df['ma_slow']) / df['ma_slow'] * 100
        df['price_vs_baseline'] = (df['close'] - df['ma_baseline']) / df['ma_baseline'] * 100
        
        # Volume analysis
        df['volume_ma'] = df['volume'].rolling(window=10).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        df['volume_momentum'] = df['volume'].pct_change(5) * 100
        
        # Volatility metrics
        df['volatility'] = df['close'].pct_change().rolling(window=20).std()
        df['volatility_ma'] = df['volatility'].rolling(window=50).mean()
        df['low_volatility'] = df['volatility'] < df['volatility_ma']
        
        # RSI for additional confirmation
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD for trend confirmation
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands for volatility breakout
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        df['bb_std'] = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
        df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Support/Resistance levels
        df['resistance'] = df['high'].rolling(window=20).max()
        df['support'] = df['low'].rolling(window=20).min()
        df['near_resistance'] = (df['close'] >= df['resistance'] * 0.98)
        df['near_support'] = (df['close'] <= df['support'] * 1.02)
        
        return df
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate enhanced trading signals with multiple confirmations"""
        df = data.copy()
        
        # Initialize signal columns
        df['buy_signal'] = False
        df['sell_signal'] = False
        df['signal_strength'] = 0.0
        df['position_size'] = self.base_position
        
        # Calculate composite momentum score
        df['momentum_score'] = 0.0
        
        # Weight different timeframes
        ltf_weight = 0.3
        htf_weight = 0.4
        mtf_weight = 0.3
        
        # Base momentum scoring
        df['momentum_score'] += np.where(df['momentum_ltf'] > self.momentum_threshold, ltf_weight, 0)
        df['momentum_score'] += np.where(df['momentum_htf'] > self.momentum_threshold, htf_weight, 0)
        df['momentum_score'] += np.where(df['momentum_mtf'] > self.momentum_threshold, mtf_weight, 0)
        
        # Acceleration bonus
        if self.use_acceleration:
            df['momentum_score'] += np.where(df['momentum_accel_ltf'] > 0, 0.1, 0)
            df['momentum_score'] += np.where(df['momentum_accel_htf'] > 0, 0.1, 0)
        
        # Trend alignment bonus
        df['trend_aligned'] = (
            (df['close'] > df['ma_fast']) & 
            (df['ma_fast'] > df['ma_slow']) & 
            (df['ma_slow'] > df['ma_baseline'])
        )
        df['momentum_score'] += np.where(df['trend_aligned'], 0.2, 0)
        
        # Volume confirmation bonus
        df['momentum_score'] += np.where(df['volume_ratio'] > self.volume_multiplier, 0.1, 0)
        
        # RSI confirmation
        df['rsi_bullish'] = (df['rsi'] > 30) & (df['rsi'] < 70) & (df['rsi'] > df['rsi'].shift(1))
        df['rsi_bearish'] = (df['rsi'] < 70) & (df['rsi'] > 30) & (df['rsi'] < df['rsi'].shift(1))
        df['momentum_score'] += np.where(df['rsi_bullish'], 0.1, 0)
        
        # MACD confirmation
        df['macd_bullish'] = (df['macd'] > df['macd_signal']) & (df['macd_histogram'] > 0)
        df['macd_bearish'] = (df['macd'] < df['macd_signal']) & (df['macd_histogram'] < 0)
        df['momentum_score'] += np.where(df['macd_bullish'], 0.1, 0)
        
        # Bollinger Band breakout bonus
        df['bb_breakout'] = df['close'] > df['bb_upper']
        df['momentum_score'] += np.where(df['bb_breakout'], 0.15, 0)
        
        # Buy signals - more aggressive conditions
        buy_conditions = (
            (df['momentum_score'] >= self.min_strength) &
            (df['momentum_ltf'] > 0) &  # Must have positive short-term momentum
            (df['volume_ratio'] > self.volume_multiplier * 0.8)  # Slightly relaxed volume
        )
        
        # Additional buy triggers for aggressive entries
        aggressive_buy = (
            (df['bb_breakout']) &  # Bollinger breakout
            (df['momentum_htf'] > self.momentum_threshold * 2) &  # Strong higher timeframe
            (df['volume_ratio'] > 1.2)  # Good volume
        )
        
        # Momentum divergence buy (oversold bounce)
        divergence_buy = (
            (df['rsi'] < 35) &  # Oversold
            (df['rsi'] > df['rsi'].shift(1)) &  # RSI turning up
            (df['momentum_ltf'] > df['momentum_ltf'].shift(1)) &  # Momentum improving
            (df['close'] > df['support'] * 1.01)  # Above support
        )
        
        # Combine all buy conditions
        df.loc[buy_conditions | aggressive_buy | divergence_buy, 'buy_signal'] = True
        
        # Calculate signal strength for buys
        df.loc[df['buy_signal'], 'signal_strength'] = df.loc[df['buy_signal'], 'momentum_score']
        
        # Dynamic position sizing based on signal strength
        df.loc[df['buy_signal'], 'position_size'] = (
            self.base_position + 
            (df.loc[df['buy_signal'], 'signal_strength'] - self.min_strength) * 
            (self.max_position - self.base_position) / (1.0 - self.min_strength)
        ).clip(self.base_position, self.max_position)
        
        # Sell signals - protect profits aggressively
        # Basic momentum reversal
        sell_conditions = (
            (df['momentum_ltf'] < -self.momentum_threshold) |  # Short-term reversal
            (df['momentum_htf'] < -self.momentum_threshold) |  # Long-term reversal
            ((df['momentum_score'] < self.min_strength * 0.5) & (df['momentum_ltf'] < 0))  # Weak momentum
        )
        
        # Take profit conditions
        take_profit = (
            (df['close'] > df['resistance'] * 0.98) &  # Near resistance
            (df['rsi'] > 70) &  # Overbought
            (df['momentum_ltf'] < df['momentum_ltf'].shift(1))  # Momentum slowing
        )
        
        # Stop loss conditions
        stop_loss = (
            (df['close'] < df['ma_slow']) &  # Below key MA
            (df['momentum_htf'] < -self.momentum_threshold * 2)  # Strong downward momentum
        )
        
        # Combine sell conditions
        df.loc[sell_conditions | take_profit | stop_loss, 'sell_signal'] = True
        df.loc[df['sell_signal'], 'signal_strength'] = 1.0  # Full strength for exits
        
        # Trailing stop implementation (simplified for backtesting)
        if self.use_trailing_stop:
            # Mark potential trailing stop exits
            df['high_since_entry'] = df['close'].cummax()
            df['trailing_stop_price'] = df['high_since_entry'] * (1 - self.trailing_stop_pct / 100)
            trailing_stop_hit = df['close'] < df['trailing_stop_price']
            df.loc[trailing_stop_hit, 'sell_signal'] = True
        
        return df
    
    def get_position_size(self, data: pd.DataFrame, index: int) -> float:
        """Dynamic position sizing based on market conditions"""
        if 'position_size' in data.columns:
            return data.iloc[index]['position_size']
        return self.base_position

    @classmethod
    def parameter_space(cls):
        return {
            'ltf_period': (2, 6),
            'htf_period': (10, 25),
            'mtf_period': (20, 50),
            'min_strength': (0.2, 0.5),
            'momentum_threshold': (0.2, 0.8),
            'volume_multiplier': (0.6, 1.2),
            'base_position': (0.3, 0.7),
            'max_position': (0.6, 1.0),
            'trailing_stop_pct': (2.0, 8.0),
        }
