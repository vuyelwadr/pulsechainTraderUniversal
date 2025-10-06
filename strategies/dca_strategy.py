"""
Dollar Cost Averaging (DCA) Bot Strategy for PulseChain Trading Bot

This strategy systematically accumulates positions at regular intervals or on dips.
Provides risk management and accumulation with various triggering mechanisms.
"""
import pandas as pd
import numpy as np
from typing import Dict
import logging

from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

class DCAStrategy(BaseStrategy):
    """
    Dollar Cost Averaging (DCA) Strategy
    
    Systematically buys at regular intervals or on price dips.
    Provides consistent accumulation with controlled risk.
    
    Parameters:
        dca_type (str): Type of DCA - 'time', 'price', 'hybrid' (default: 'hybrid')
        time_interval (int): Time-based DCA interval in periods (default: 24)
        price_drop_threshold (float): Price drop % to trigger DCA (default: 0.05)
        base_position_size (float): Base position size multiplier (default: 1.0)
        scaling_factor (float): Position scaling on larger dips (default: 1.5)
        max_scaling (float): Maximum position scaling (default: 3.0)
        take_profit_pct (float): Take profit percentage (default: 0.20)
        stop_loss_pct (float): Stop loss percentage (default: 0.30)
        min_strength (float): Minimum signal strength (default: 0.5)
        timeframe_minutes (int): Timeframe for analysis (default: 5)
    """
    
    def __init__(self, parameters: Dict = None):
        default_params = {
            'dca_type': 'hybrid',
            'time_interval': 24,  # Every 24 periods
            'price_drop_threshold': 0.05,  # 5% drop triggers DCA
            'base_position_size': 1.0,
            'scaling_factor': 1.5,
            'max_scaling': 3.0,
            'take_profit_pct': 0.20,  # 20% profit target
            'stop_loss_pct': 0.30,   # 30% stop loss
            'min_strength': 0.5,
            'timeframe_minutes': 5
        }
        
        if parameters:
            default_params.update(parameters)
            
        super().__init__("DCAStrategy", default_params)
        
        # DCA state tracking
        self.last_dca_time = 0
        self.dca_counter = 0
        self.avg_entry_price = 0.0
        self.total_position = 0.0
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate DCA-related indicators"""
        if not self.validate_data(data):
            return data
        
        df = data.copy()
        time_interval = self.parameters['time_interval']
        price_drop_threshold = self.parameters['price_drop_threshold']
        
        # Ensure we have enough data
        min_required = time_interval + 10
        if len(df) < min_required:
            logger.warning(f"Not enough data for DCA calculation ({min_required} required)")
            return df
        
        # Calculate price change metrics
        df['price_change_pct'] = df['price'].pct_change()
        df['price_change_1d'] = df['price'].pct_change(periods=time_interval)
        df['price_change_3d'] = df['price'].pct_change(periods=time_interval*3)
        df['price_change_7d'] = df['price'].pct_change(periods=time_interval*7)
        
        # Calculate moving averages for trend context
        df['sma_short'] = df['price'].rolling(window=time_interval//2).mean()
        df['sma_long'] = df['price'].rolling(window=time_interval*2).mean()
        df['ema_20'] = df['price'].ewm(span=20).mean()
        
        # Trend indicators
        df['uptrend'] = df['sma_short'] > df['sma_long']
        df['downtrend'] = df['sma_short'] < df['sma_long']
        df['sideways'] = abs(df['sma_short'] - df['sma_long']) / df['price'] < 0.02
        
        # Volatility indicators
        df['volatility'] = df['price_change_pct'].rolling(window=time_interval).std()
        df['high_volatility'] = df['volatility'] > df['volatility'].rolling(window=time_interval*2).mean() * 1.5
        
        # Price drop detection for price-based DCA
        df['significant_drop'] = df['price_change_1d'] < -price_drop_threshold
        df['major_drop'] = df['price_change_1d'] < -price_drop_threshold * 2
        df['crash'] = df['price_change_1d'] < -price_drop_threshold * 3
        
        # Support/Resistance levels (simplified)
        df['local_low'] = df['low'].rolling(window=time_interval, center=True).min() == df['low']
        df['local_high'] = df['high'].rolling(window=time_interval, center=True).max() == df['high']
        
        # Time-based DCA triggers
        df['time_dca_trigger'] = False
        for i in range(time_interval, len(df), time_interval):
            if i < len(df):
                df.iloc[i, df.columns.get_loc('time_dca_trigger')] = True
        
        # RSI for oversold conditions (helps with DCA timing)
        delta = df['price'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))
        df['rsi'] = df['rsi'].fillna(50)
        
        df['oversold'] = df['rsi'] < 30
        df['extremely_oversold'] = df['rsi'] < 20
        
        # Market fear indicators (combine multiple factors)
        df['fear_index'] = (
            (df['significant_drop'].astype(int) * 0.3) +
            (df['oversold'].astype(int) * 0.3) +
            (df['high_volatility'].astype(int) * 0.2) +
            (df['downtrend'].astype(int) * 0.2)
        )
        
        # DCA opportunity scoring
        df['dca_opportunity'] = 0.0
        
        # Higher opportunity during dips and oversold conditions
        df['dca_opportunity'] += np.where(df['significant_drop'], 0.4, 0)
        df['dca_opportunity'] += np.where(df['major_drop'], 0.3, 0)
        df['dca_opportunity'] += np.where(df['crash'], 0.2, 0)
        df['dca_opportunity'] += np.where(df['oversold'], 0.3, 0)
        df['dca_opportunity'] += np.where(df['extremely_oversold'], 0.2, 0)
        df['dca_opportunity'] += np.where(df['local_low'], 0.2, 0)
        
        # Reduce opportunity in strong uptrends (avoid buying peaks)
        df['dca_opportunity'] -= np.where(df['price_change_1d'] > 0.1, 0.3, 0)
        df['dca_opportunity'] -= np.where(df['rsi'] > 70, 0.2, 0)
        
        # Ensure opportunity stays in valid range
        df['dca_opportunity'] = df['dca_opportunity'].clip(0, 1)
        
        return df
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate buy/sell signals based on DCA strategy"""
        df = data.copy()
        
        if 'dca_opportunity' not in df.columns:
            df = self.calculate_indicators(df)
        
        # Initialize signal columns
        df['buy_signal'] = False
        df['sell_signal'] = False
        df['signal_strength'] = 0.0
        
        dca_type = self.parameters['dca_type']
        time_interval = self.parameters['time_interval']
        price_drop_threshold = self.parameters['price_drop_threshold']
        base_size = self.parameters['base_position_size']
        scaling_factor = self.parameters['scaling_factor']
        max_scaling = self.parameters['max_scaling']
        take_profit_pct = self.parameters['take_profit_pct']
        stop_loss_pct = self.parameters['stop_loss_pct']
        min_strength = self.parameters['min_strength']
        
        # Time-based DCA signals
        time_dca_signals = (
            df['time_dca_trigger'] &
            (dca_type in ['time', 'hybrid'])
        )
        
        # Price-based DCA signals
        price_dca_signals = (
            df['significant_drop'] &
            (dca_type in ['price', 'hybrid']) &
            (df['dca_opportunity'] > 0.3)
        )
        
        # Enhanced price DCA on major drops
        enhanced_price_dca = (
            (df['major_drop'] | df['crash']) &
            (dca_type in ['price', 'hybrid']) &
            df['oversold']
        )
        
        # Hybrid signals (combine time and price conditions)
        hybrid_dca_signals = (
            (dca_type == 'hybrid') &
            (
                (df['time_dca_trigger'] & (df['dca_opportunity'] > 0.2)) |
                (df['significant_drop'] & df['oversold']) |
                (df['local_low'] & df['extremely_oversold'])
            )
        )
        
        # Take profit signals (sell signals)
        # Simple profit taking logic - in reality this would track actual positions
        take_profit_signals = (
            df['price_change_7d'] > take_profit_pct  # Profit over longer period
        )
        
        # Stop loss signals (in severe market conditions)
        stop_loss_signals = (
            (df['price_change_7d'] < -stop_loss_pct) &
            (df['rsi'] < 25) &  # Extremely oversold
            (df['volatility'] > df['volatility'].rolling(window=50).mean() * 2)  # High volatility
        )
        
        # Combine buy conditions
        df['buy_signal'] = (
            time_dca_signals | 
            price_dca_signals | 
            enhanced_price_dca | 
            hybrid_dca_signals
        )
        
        # Sell conditions (profit taking and stop losses)
        df['sell_signal'] = take_profit_signals | stop_loss_signals
        
        # Calculate signal strength for buy signals
        buy_strength = np.where(df['buy_signal'], 0.5, 0.0)  # Base strength
        
        # Increase strength based on conditions
        buy_strength += np.where(df['significant_drop'], 0.2, 0.0)
        buy_strength += np.where(df['major_drop'], 0.15, 0.0)
        buy_strength += np.where(df['crash'], 0.1, 0.0)
        buy_strength += np.where(df['oversold'], 0.15, 0.0)
        buy_strength += np.where(df['extremely_oversold'], 0.1, 0.0)
        buy_strength += np.where(df['local_low'], 0.1, 0.0)
        
        # Position scaling based on market conditions
        position_scaling = np.where(
            df['buy_signal'],
            np.minimum(
                base_size * (1 + df['fear_index'] * scaling_factor),
                max_scaling
            ),
            1.0
        )
        
        # Calculate signal strength for sell signals
        sell_strength = np.where(df['sell_signal'], 0.6, 0.0)  # Base strength for sells
        
        # Increase sell strength for strong profit conditions
        sell_strength += np.where(df['price_change_7d'] > take_profit_pct * 1.5, 0.2, 0.0)
        sell_strength += np.where(df['rsi'] > 75, 0.15, 0.0)
        
        # Emergency sell strength for stop losses
        sell_strength += np.where(stop_loss_signals, 0.3, 0.0)
        
        # Combine strengths
        df['signal_strength'] = np.maximum(buy_strength, sell_strength)
        
        # Store position scaling information
        df['position_scaling'] = position_scaling
        
        # Apply minimum strength filter
        df.loc[df['signal_strength'] < min_strength, 'buy_signal'] = False
        df.loc[df['signal_strength'] < min_strength, 'sell_signal'] = False
        df.loc[(~df['buy_signal']) & (~df['sell_signal']), 'signal_strength'] = 0.0
        
        return df
    
    def get_position_size(self, current_price: float, signal_strength: float, 
                         position_scaling: float = 1.0) -> float:
        """Calculate position size based on DCA logic"""
        base_size = self.parameters['base_position_size']
        
        # Scale position based on market conditions and signal strength
        scaled_size = base_size * position_scaling * signal_strength
        
        return scaled_size
    
    def update_dca_state(self, price: float, position_size: float):
        """Update DCA tracking state (for position tracking)"""
        if position_size > 0:  # Buy order
            total_cost = self.total_position * self.avg_entry_price + position_size * price
            self.total_position += position_size
            self.avg_entry_price = total_cost / self.total_position if self.total_position > 0 else price
            self.dca_counter += 1
        else:  # Sell order
            sell_amount = abs(position_size)
            if sell_amount >= self.total_position:
                # Full exit
                self.total_position = 0.0
                self.avg_entry_price = 0.0
                self.dca_counter = 0
            else:
                # Partial exit
                self.total_position -= sell_amount
    
    def get_dca_info(self) -> Dict:
        """Get current DCA state information"""
        return {
            'total_position': self.total_position,
            'average_entry_price': self.avg_entry_price,
            'dca_count': self.dca_counter,
            'last_dca_time': self.last_dca_time,
            'parameters': self.parameters
        }