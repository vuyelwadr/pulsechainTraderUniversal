"""
Grid Trading Strategy for PulseChain Trading Bot

This strategy places multiple buy/sell orders at regular intervals to profit from volatility.
Ideal for ranging markets and high-volatility assets.
"""
import pandas as pd
import numpy as np
from typing import Dict, List
import logging

from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

class GridTradingStrategy(BaseStrategy):
    """
    Grid Trading Strategy
    
    Places buy orders below current price and sell orders above current price
    at regular intervals to profit from price oscillations.
    
    Parameters:
        grid_size_percent (float): Percentage distance between grid levels (default: 2.0)
        num_grids (int): Number of grid levels each direction (default: 10)
        price_range_percent (float): Total price range for grid (default: 20.0)
        min_strength (float): Minimum signal strength to trigger trade (default: 0.5)
        volatility_threshold (float): Minimum volatility for grid activation (default: 0.02)
        rebalance_threshold (float): Price movement % to trigger grid rebalancing (default: 10.0)
        timeframe_minutes (int): Timeframe for analysis (default: 5)
    """
    
    def __init__(self, parameters: Dict = None):
        default_params = {
            'grid_size_percent': 2.0,
            'num_grids': 10,
            'price_range_percent': 20.0,
            'min_strength': 0.5,
            'volatility_threshold': 0.02,
            'rebalance_threshold': 10.0,
            'timeframe_minutes': 5
        }
        
        if parameters:
            default_params.update(parameters)
            
        super().__init__("GridTradingStrategy", default_params)
        
        # Grid state tracking
        self.grid_levels = []
        self.grid_center = 0.0
        self.last_rebalance_price = 0.0
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate grid trading indicators and volatility measures"""
        if not self.validate_data(data):
            return data
        
        df = data.copy()
        
        # Ensure we have enough data
        if len(df) < 20:
            logger.warning("Not enough data for grid trading calculation")
            return df
        
        # Calculate volatility indicators
        df['price_change'] = df['price'].pct_change()
        df['volatility'] = df['price_change'].rolling(window=20).std()
        df['volatility'] = df['volatility'].fillna(0)
        
        # Calculate price range over different periods
        df['range_5'] = df['price'].rolling(window=5).max() - df['price'].rolling(window=5).min()
        df['range_20'] = df['price'].rolling(window=20).max() - df['price'].rolling(window=20).min()
        df['range_percent_5'] = df['range_5'] / df['price']
        df['range_percent_20'] = df['range_20'] / df['price']
        
        # Calculate moving averages for trend detection
        df['sma_20'] = df['price'].rolling(window=20).mean()
        df['sma_50'] = df['price'].rolling(window=50).mean()
        
        # Market condition indicators
        df['trending_up'] = df['sma_20'] > df['sma_50']
        df['trending_down'] = df['sma_20'] < df['sma_50']
        df['sideways'] = abs(df['sma_20'] - df['sma_50']) / df['price'] < 0.02
        
        # Calculate current price position within recent range
        df['price_position'] = (
            (df['price'] - df['price'].rolling(window=20).min()) / 
            (df['price'].rolling(window=20).max() - df['price'].rolling(window=20).min())
        ).fillna(0.5)
        
        # Grid setup indicators
        self._setup_grid_levels(df)
        
        # Calculate distance from grid levels
        df['distance_from_grid_levels'] = df['price'].apply(self._distance_from_nearest_grid)
        
        # Grid opportunity scoring
        df['grid_opportunity'] = (
            df['volatility'] * 10 +  # Higher volatility = better for grid
            (1 - abs(df['price_position'] - 0.5)) * 2 +  # Center of range = better
            np.where(df['sideways'], 0.5, 0)  # Sideways market = better
        )
        
        return df
    
    def _setup_grid_levels(self, data: pd.DataFrame):
        """Setup grid levels based on current price and parameters"""
        if data.empty:
            return
        
        current_price = data['price'].iloc[-1]
        grid_size = self.parameters['grid_size_percent'] / 100
        num_grids = self.parameters['num_grids']
        
        # Set grid center (use recent average to avoid noise)
        if len(data) >= 10:
            self.grid_center = data['price'].tail(10).mean()
        else:
            self.grid_center = current_price
        
        # Check if we need to rebalance grid
        rebalance_threshold = self.parameters['rebalance_threshold'] / 100
        price_move_percent = abs(current_price - self.last_rebalance_price) / self.last_rebalance_price if self.last_rebalance_price > 0 else 0
        
        if price_move_percent > rebalance_threshold or not self.grid_levels:
            # Create new grid levels
            self.grid_levels = []
            
            # Buy levels (below current price)
            for i in range(1, num_grids + 1):
                buy_level = self.grid_center * (1 - grid_size * i)
                self.grid_levels.append({
                    'level': buy_level,
                    'type': 'buy',
                    'distance': i
                })
            
            # Sell levels (above current price)
            for i in range(1, num_grids + 1):
                sell_level = self.grid_center * (1 + grid_size * i)
                self.grid_levels.append({
                    'level': sell_level,
                    'type': 'sell',
                    'distance': i
                })
            
            self.last_rebalance_price = current_price
            logger.debug(f"Grid rebalanced: center={self.grid_center:.6f}, levels={len(self.grid_levels)}")
    
    def _distance_from_nearest_grid(self, price: float) -> float:
        """Calculate distance from nearest grid level"""
        if not self.grid_levels:
            return 0.0
        
        distances = [abs(price - level['level']) / price for level in self.grid_levels]
        return min(distances) if distances else 0.0
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate buy/sell signals based on grid trading logic"""
        df = data.copy()
        
        if 'volatility' not in df.columns:
            df = self.calculate_indicators(df)
        
        # Initialize signal columns
        df['buy_signal'] = False
        df['sell_signal'] = False
        df['signal_strength'] = 0.0
        
        min_strength = self.parameters['min_strength']
        volatility_threshold = self.parameters['volatility_threshold']
        
        if not self.grid_levels:
            return df
        
        # Process each data point
        for i in range(len(df)):
            current_price = df.iloc[i]['price']
            volatility = df.iloc[i]['volatility']
            grid_opportunity = df.iloc[i]['grid_opportunity']
            
            # Skip if volatility too low
            if volatility < volatility_threshold:
                continue
            
            # Check for grid level hits
            buy_signal, sell_signal, strength = self._check_grid_signals(
                current_price, volatility, grid_opportunity
            )
            
            df.iloc[i, df.columns.get_loc('buy_signal')] = buy_signal
            df.iloc[i, df.columns.get_loc('sell_signal')] = sell_signal
            df.iloc[i, df.columns.get_loc('signal_strength')] = strength
        
        # Apply minimum strength filter
        df.loc[df['signal_strength'] < min_strength, 'buy_signal'] = False
        df.loc[df['signal_strength'] < min_strength, 'sell_signal'] = False
        df.loc[(~df['buy_signal']) & (~df['sell_signal']), 'signal_strength'] = 0.0
        
        return df
    
    def _check_grid_signals(self, current_price: float, volatility: float, grid_opportunity: float) -> tuple:
        """Check if current price triggers any grid signals"""
        buy_signal = False
        sell_signal = False
        max_strength = 0.0
        
        # Find nearest grid levels
        buy_levels = [level for level in self.grid_levels if level['type'] == 'buy']
        sell_levels = [level for level in self.grid_levels if level['type'] == 'sell']
        
        # Check buy levels (price at or below buy level)
        for level in buy_levels:
            level_price = level['level']
            distance = level['distance']
            
            # Signal if price touches or goes below this level
            price_diff_percent = (level_price - current_price) / level_price
            
            if -0.005 <= price_diff_percent <= 0.01:  # Within 0.5% below to 1% above
                # Calculate signal strength
                strength = self._calculate_grid_strength(
                    distance, volatility, grid_opportunity, 'buy'
                )
                
                if strength > max_strength:
                    buy_signal = True
                    sell_signal = False
                    max_strength = strength
        
        # Check sell levels (price at or above sell level)
        for level in sell_levels:
            level_price = level['level']
            distance = level['distance']
            
            # Signal if price touches or goes above this level
            price_diff_percent = (current_price - level_price) / level_price
            
            if -0.005 <= price_diff_percent <= 0.01:  # Within 0.5% below to 1% above
                # Calculate signal strength
                strength = self._calculate_grid_strength(
                    distance, volatility, grid_opportunity, 'sell'
                )
                
                if strength > max_strength:
                    buy_signal = False
                    sell_signal = True
                    max_strength = strength
        
        return buy_signal, sell_signal, max_strength
    
    def _calculate_grid_strength(self, distance: int, volatility: float, 
                                grid_opportunity: float, signal_type: str) -> float:
        """Calculate signal strength for grid level hit"""
        # Base strength (closer levels = higher strength)
        base_strength = max(0.1, 1.0 - (distance - 1) * 0.1)
        
        # Volatility component (higher volatility = better for grid)
        volatility_component = min(1.0, volatility * 20)
        
        # Opportunity component
        opportunity_component = min(1.0, grid_opportunity / 5.0)
        
        # Market timing component (better signals in middle of range)
        timing_component = 0.8  # Default good timing
        
        # Combine components
        total_strength = (
            base_strength * 0.4 +
            volatility_component * 0.25 +
            opportunity_component * 0.25 +
            timing_component * 0.1
        )
        
        return min(1.0, total_strength)
    
    def get_grid_info(self) -> Dict:
        """Get current grid configuration information"""
        return {
            'grid_center': self.grid_center,
            'last_rebalance_price': self.last_rebalance_price,
            'num_levels': len(self.grid_levels),
            'buy_levels': [l['level'] for l in self.grid_levels if l['type'] == 'buy'],
            'sell_levels': [l['level'] for l in self.grid_levels if l['type'] == 'sell'],
            'parameters': self.parameters
        }