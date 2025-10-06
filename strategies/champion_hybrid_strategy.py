"""
Champion Hybrid Strategy - Combines best performing strategies
Integrates MultiTimeframeMomentum, MACD, and RSI signals
"""
import pandas as pd
import numpy as np
from .base_strategy import BaseStrategy

class ChampionHybridStrategy(BaseStrategy):
    """
    Hybrid strategy combining top 3 performers from optimization:
    1. MultiTimeframeMomentum (28.6% return)
    2. MACD (up to 8.3% return)
    3. RSI (various positive returns)
    
    Uses weighted voting system based on historical performance
    """
    
    def __init__(self, parameters: dict = None):
        base_params = {
            'mtf_ltf_period': 5,
            'mtf_htf_period': 20,
            'mtf_weight': 0.5,
            'macd_fast': 14,
            'macd_slow': 32,
            'macd_signal': 14,
            'macd_weight': 0.3,
            'rsi_period': 19,
            'rsi_oversold': 23,
            'rsi_overbought': 77,
            'rsi_weight': 0.2,
            'min_consensus': 0.5,
            'confirmation_required': 2,
        }
        if parameters:
            base_params.update(parameters)
        super().__init__("ChampionHybrid", base_params)
        self.parameters = base_params
        
        # MTF Momentum parameters (best performer)
        self.mtf_ltf_period = int(self.parameters.get('mtf_ltf_period', 5))
        self.mtf_htf_period = int(self.parameters.get('mtf_htf_period', 20))
        self.mtf_weight = float(self.parameters.get('mtf_weight', 0.5))
        
        # MACD parameters (second best)
        self.macd_fast = int(self.parameters.get('macd_fast', 14))
        self.macd_slow = int(self.parameters.get('macd_slow', 32))
        self.macd_signal = int(self.parameters.get('macd_signal', 14))
        self.macd_weight = float(self.parameters.get('macd_weight', 0.3))
        
        # RSI parameters (third best)
        self.rsi_period = int(self.parameters.get('rsi_period', 19))
        self.rsi_oversold = float(self.parameters.get('rsi_oversold', 23))
        self.rsi_overbought = float(self.parameters.get('rsi_overbought', 77))
        self.rsi_weight = float(self.parameters.get('rsi_weight', 0.2))
        
        # Hybrid parameters
        self.min_consensus = float(self.parameters.get('min_consensus', 0.5))
        self.confirmation_required = int(self.parameters.get('confirmation_required', 2))
        
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate all three strategy indicators"""
        df = data.copy()
        
        # 1. MultiTimeframe Momentum indicators
        # Calculate momentum for different timeframes
        df['momentum_ltf'] = df['close'].pct_change(self.mtf_ltf_period) * 100
        df['momentum_htf'] = df['close'].pct_change(self.mtf_htf_period) * 100
        
        # Moving averages for trend
        df['ma_short'] = df['close'].rolling(window=self.mtf_ltf_period).mean()
        df['ma_long'] = df['close'].rolling(window=self.mtf_htf_period).mean()
        
        # Volume momentum
        df['volume_ma'] = df['volume'].rolling(window=10).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # 2. MACD indicators
        exp1 = df['close'].ewm(span=self.macd_fast, adjust=False).mean()
        exp2 = df['close'].ewm(span=self.macd_slow, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=self.macd_signal, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # 3. RSI indicators
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Additional confluence indicators
        df['price_position'] = (df['close'] - df['close'].rolling(50).min()) / \
                              (df['close'].rolling(50).max() - df['close'].rolling(50).min())
        
        # Volatility for position sizing
        df['volatility'] = df['close'].pct_change().rolling(window=20).std()
        df['atr'] = self._calculate_atr(df, 14)
        
        return df
    
    def _calculate_atr(self, df, period=14):
        """Calculate Average True Range"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        
        return true_range.rolling(period).mean()
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate hybrid signals from all three strategies"""
        df = data.copy()
        
        # Initialize signal columns
        df['buy_signal'] = False
        df['sell_signal'] = False
        df['signal_strength'] = 0.0
        
        # 1. MultiTimeframe Momentum signals
        df['mtf_buy'] = False
        df['mtf_sell'] = False
        df['mtf_strength'] = 0.0
        
        # MTF Buy conditions
        mtf_buy_conditions = (
            (df['momentum_ltf'] > 0) &  # Short-term momentum positive
            (df['momentum_htf'] > 0) &  # Long-term momentum positive
            (df['close'] > df['ma_short']) &  # Price above short MA
            (df['ma_short'] > df['ma_long']) &  # Uptrend
            (df['volume_ratio'] > 1.0)  # Volume confirmation
        )
        
        # MTF Sell conditions
        mtf_sell_conditions = (
            (df['momentum_ltf'] < 0) &  # Short-term momentum negative
            (df['momentum_htf'] < 0) &  # Long-term momentum negative
            (df['close'] < df['ma_short']) &  # Price below short MA
            (df['ma_short'] < df['ma_long']) &  # Downtrend
            (df['volume_ratio'] > 0.8)  # Some volume activity
        )
        
        df.loc[mtf_buy_conditions, 'mtf_buy'] = True
        df.loc[mtf_sell_conditions, 'mtf_sell'] = True
        
        # MTF signal strength based on momentum magnitude
        df['mtf_strength'] = np.where(
            mtf_buy_conditions,
            np.minimum(1.0, (abs(df['momentum_ltf']) + abs(df['momentum_htf'])) / 10),
            0
        )
        df['mtf_strength'] = np.where(
            mtf_sell_conditions,
            np.minimum(1.0, (abs(df['momentum_ltf']) + abs(df['momentum_htf'])) / 10),
            df['mtf_strength']
        )
        
        # 2. MACD signals
        df['macd_buy'] = False
        df['macd_sell'] = False
        df['macd_strength'] = 0.0
        
        # MACD crossover signals
        macd_buy_conditions = (
            (df['macd'] > df['macd_signal']) &
            (df['macd'].shift(1) <= df['macd_signal'].shift(1)) &  # Crossover
            (df['macd_histogram'] > 0)
        )
        
        macd_sell_conditions = (
            (df['macd'] < df['macd_signal']) &
            (df['macd'].shift(1) >= df['macd_signal'].shift(1)) &  # Crossunder
            (df['macd_histogram'] < 0)
        )
        
        df.loc[macd_buy_conditions, 'macd_buy'] = True
        df.loc[macd_sell_conditions, 'macd_sell'] = True
        
        # MACD signal strength based on histogram magnitude
        df['macd_strength'] = np.where(
            macd_buy_conditions | macd_sell_conditions,
            np.minimum(1.0, abs(df['macd_histogram']) / df['atr']),
            0
        )
        
        # 3. RSI signals
        df['rsi_buy'] = False
        df['rsi_sell'] = False
        df['rsi_strength'] = 0.0
        
        # RSI reversal signals
        rsi_buy_conditions = (
            (df['rsi'] < self.rsi_oversold) &
            (df['rsi'] > df['rsi'].shift(1)) &  # RSI turning up
            (df['close'] > df['close'].shift(1))  # Price confirmation
        )
        
        rsi_sell_conditions = (
            (df['rsi'] > self.rsi_overbought) &
            (df['rsi'] < df['rsi'].shift(1)) &  # RSI turning down
            (df['close'] < df['close'].shift(1))  # Price confirmation
        )
        
        df.loc[rsi_buy_conditions, 'rsi_buy'] = True
        df.loc[rsi_sell_conditions, 'rsi_sell'] = True
        
        # RSI signal strength based on extremity
        df['rsi_strength'] = np.where(
            rsi_buy_conditions,
            (self.rsi_oversold - df['rsi']) / self.rsi_oversold,
            0
        )
        df['rsi_strength'] = np.where(
            rsi_sell_conditions,
            (df['rsi'] - self.rsi_overbought) / (100 - self.rsi_overbought),
            df['rsi_strength']
        )
        
        # HYBRID SIGNAL GENERATION
        # Calculate weighted consensus scores
        df['buy_consensus'] = (
            df['mtf_buy'].astype(int) * self.mtf_weight * df['mtf_strength'] +
            df['macd_buy'].astype(int) * self.macd_weight * df['macd_strength'] +
            df['rsi_buy'].astype(int) * self.rsi_weight * df['rsi_strength']
        )
        
        df['sell_consensus'] = (
            df['mtf_sell'].astype(int) * self.mtf_weight * df['mtf_strength'] +
            df['macd_sell'].astype(int) * self.macd_weight * df['macd_strength'] +
            df['rsi_sell'].astype(int) * self.rsi_weight * df['rsi_strength']
        )
        
        # Count how many strategies agree
        df['buy_count'] = df['mtf_buy'].astype(int) + df['macd_buy'].astype(int) + df['rsi_buy'].astype(int)
        df['sell_count'] = df['mtf_sell'].astype(int) + df['macd_sell'].astype(int) + df['rsi_sell'].astype(int)
        
        # Generate final signals based on consensus and confirmation
        buy_signal_conditions = (
            (df['buy_consensus'] >= self.min_consensus) &
            (df['buy_count'] >= self.confirmation_required) &
            (df['volatility'] < df['volatility'].rolling(50).mean() * 1.5)  # Not too volatile
        )
        
        sell_signal_conditions = (
            (df['sell_consensus'] >= self.min_consensus) &
            (df['sell_count'] >= self.confirmation_required)
        )
        
        df.loc[buy_signal_conditions, 'buy_signal'] = True
        df.loc[sell_signal_conditions, 'sell_signal'] = True
        
        # Final signal strength is the consensus score
        df.loc[buy_signal_conditions, 'signal_strength'] = df.loc[buy_signal_conditions, 'buy_consensus']
        df.loc[sell_signal_conditions, 'signal_strength'] = df.loc[sell_signal_conditions, 'sell_consensus']
        
        # Risk management: Reduce signal strength in high volatility
        high_vol_mask = df['volatility'] > df['volatility'].rolling(50).mean() * 1.2
        df.loc[high_vol_mask, 'signal_strength'] *= 0.7
        
        return df
    
    def get_position_size(self, data: pd.DataFrame, index: int) -> float:
        """
        Dynamic position sizing based on volatility and signal strength
        """
        current_vol = data.iloc[index]['volatility']
        avg_vol = data['volatility'].rolling(50).mean().iloc[index]
        signal_strength = data.iloc[index]['signal_strength']
        
        # Base position size
        base_size = 0.5
        
        # Adjust for volatility (inverse relationship)
        vol_adjustment = avg_vol / max(current_vol, 0.001)
        vol_adjustment = np.clip(vol_adjustment, 0.5, 1.5)
        
        # Adjust for signal strength
        strength_adjustment = 0.5 + (signal_strength * 0.5)
        
        # Final position size
        position_size = base_size * vol_adjustment * strength_adjustment
        
        return np.clip(position_size, 0.1, 1.0)

    @classmethod
    def parameter_space(cls):
        return {
            'mtf_ltf_period': (3, 15),
            'mtf_htf_period': (15, 60),
            'mtf_weight': (0.2, 0.7),
            'macd_fast': (8, 20),
            'macd_slow': (20, 60),
            'macd_signal': (6, 20),
            'macd_weight': (0.1, 0.5),
            'rsi_period': (10, 35),
            'rsi_oversold': (15, 35),
            'rsi_overbought': (65, 85),
            'rsi_weight': (0.1, 0.4),
            'min_consensus': (0.3, 0.8),
            'confirmation_required': (1, 3),
        }
