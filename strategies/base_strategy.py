"""
Base Strategy class for PulseChain Trading Bot
All trading strategies should inherit from this base class
"""
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)

class BaseStrategy(ABC):
    """Base class for all trading strategies"""
    
    def __init__(self, name: str, parameters: Dict = None):
        self.name = name
        self.parameters = parameters or {}
        self.signals = pd.DataFrame()
        self.indicators = {}
        
        # Default timeframe is 5 minutes (base resolution)
        self.timeframe_minutes = self.parameters.get('timeframe_minutes', 5)
    
    def resample_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Resample 5-minute base data to specified timeframe
        
        Args:
            data: Base 5-minute price data
            
        Returns:
            Resampled data at the specified timeframe
        """
        if self.timeframe_minutes <= 5 or data.empty:
            return data.copy()
        
        try:
            # Set timestamp as index for resampling
            if 'timestamp' in data.columns:
                df = data.set_index('timestamp')
            else:
                df = data.copy()
            
            # Ensure index is datetime
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            
            # Resample to target timeframe
            timeframe_str = f'{self.timeframe_minutes}min'
            
            resampled = df.resample(timeframe_str).agg({
                'price': 'last',      # Last price in timeframe
                'high': 'max',        # Highest price
                'low': 'min',         # Lowest price  
                'open': 'first',      # First price
                'close': 'last',      # Last price (same as price)
                'volume': 'sum'       # Total volume
            }).dropna()
            
            # Reset index to have timestamp as column
            resampled.reset_index(inplace=True)
            
            # Ensure we have close price equal to price for consistency
            if 'close' not in resampled.columns:
                resampled['close'] = resampled['price']
            
            logger.debug(f"Resampled {len(data)} points to {len(resampled)} points at {self.timeframe_minutes}min")
            return resampled
            
        except Exception as e:
            logger.error(f"Error resampling data to {self.timeframe_minutes}min: {e}")
            return data.copy()
        
    @abstractmethod
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators needed for the strategy
        
        Args:
            data: Price data with columns: timestamp, price, volume, high, low, open, close
            
        Returns:
            DataFrame with additional indicator columns
        """
        pass
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate buy/sell signals based on the strategy
        
        Args:
            data: Price data with indicators
            
        Returns:
            DataFrame with signal columns (buy_signal, sell_signal, signal_strength)
        """
        pass
    
    def get_signal(self, data: pd.DataFrame) -> Tuple[str, float]:
        """
        Get the current trading signal
        
        Args:
            data: Current price data (5-minute base resolution)
            
        Returns:
            Tuple of (signal_type, signal_strength) where:
            signal_type: 'buy', 'sell', or 'hold'
            signal_strength: float between 0 and 1
        """
        if data.empty:
            return 'hold', 0.0
        
        # Resample data to strategy timeframe
        resampled_data = self.resample_data(data)
        
        # Calculate indicators and signals
        data_with_indicators = self.calculate_indicators(resampled_data)
        signals_data = self.generate_signals(data_with_indicators)
        
        if signals_data.empty:
            return 'hold', 0.0
        
        # Get the latest signal
        latest = signals_data.iloc[-1]
        
        if latest.get('buy_signal', False):
            return 'buy', latest.get('signal_strength', 1.0)
        elif latest.get('sell_signal', False):
            return 'sell', latest.get('signal_strength', 1.0)
        else:
            return 'hold', 0.0
    
    def backtest_signals(self, data: pd.DataFrame) -> Dict:
        """
        Backtest the strategy on historical data
        
        Args:
            data: Historical price data (5-minute base resolution)
            
        Returns:
            Dictionary with backtest results
        """
        if data.empty:
            return {'error': 'No data provided for backtesting'}
        
        try:
            # Resample data to strategy timeframe
            resampled_data = self.resample_data(data.copy())
            
            # Calculate indicators and signals
            data_with_indicators = self.calculate_indicators(resampled_data)
            signals_data = self.generate_signals(data_with_indicators)
            
            # Simple backtest metrics
            buy_signals = signals_data['buy_signal'].sum() if 'buy_signal' in signals_data else 0
            sell_signals = signals_data['sell_signal'].sum() if 'sell_signal' in signals_data else 0
            
            # Calculate basic performance metrics
            if 'signal_strength' in signals_data:
                avg_signal_strength = signals_data['signal_strength'].mean()
            else:
                avg_signal_strength = 0.0
            
            results = {
                'strategy_name': self.name,
                'total_buy_signals': int(buy_signals),
                'total_sell_signals': int(sell_signals),
                'avg_signal_strength': float(avg_signal_strength),
                'data_points': len(signals_data),
                'parameters': self.parameters
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error in backtesting {self.name}: {e}")
            return {'error': str(e)}
    
    def get_parameter_info(self) -> Dict:
        """Get information about strategy parameters"""
        return {
            'name': self.name,
            'parameters': self.parameters,
            'description': self.__doc__ or 'No description available'
        }

    # --- Parameter Space (Strategy-owned) ---
    @classmethod
    def parameter_space(cls) -> Dict[str, Tuple[float, float]]:
        """
        Return the tunable parameter bounds for this strategy.

        By default, derive bounds from the class's default parameters by
        instantiating the class with no overrides. Subclasses can override
        this to provide precise ranges.

        Returns a dict: {param_name: (low, high)}. Integer-valued params must
        have integer bounds (e.g., (5, 50)). Boolean flags are mapped to
        (0, 1) and will be interpreted as False/True during optimization.
        """
        try:
            # Registry first: strategy-specific explicit mapping
            try:
                from strategies.param_space_registry import get_param_space_for
                reg_space = get_param_space_for(cls.__name__)
                if reg_space:
                    return reg_space
            except Exception:
                pass
            
            # Instantiate with default parameters to read the base dict
            inst = cls()  # Most strategies support parameters=None
            params: Dict[str, Any] = getattr(inst, 'parameters', {}) or {}
            return cls._derive_param_space_from_defaults(params)
        except Exception:
            # If instantiation fails or strategy has no defaults, return empty
            logging.getLogger(__name__).warning(
                f"parameter_space() fallback for {cls.__name__}: no defaults available"
            )
            return {}

    @staticmethod
    def _derive_param_space_from_defaults(params: Dict[str, Any]) -> Dict[str, Tuple[float, float]]:
        """Heuristically derive reasonable bounds from a parameters dict.

        Rules:
        - period/length/window-like ints: [max(2, 0.5*val), min(1000, 2*val)]
        - generic ints: [max(1, 0.5*val), min(1_000_000, 2*val)]
        - floats in [0,1]: [0.0, 1.0]
        - thresholds/multipliers/factors: widen around default but clamp to [0, 5]
        - other floats: [max(1e-9, 0.5*val), max(1.0, 1.5*val)]
        - booleans: (0, 1) treated as int
        Non-numeric values are ignored.
        """
        space: Dict[str, Tuple[float, float]] = {}
        for k, v in (params or {}).items():
            name = str(k).lower()
            # Exclude non-strategy knobs and structural controls
            if name in ('timeframe', 'timeframe_minutes', 'name'):
                continue
            # Booleans as 0/1 integers
            if isinstance(v, bool):
                space[k] = (0, 1)
                continue
            # Integers
            if isinstance(v, int) and not isinstance(v, bool):
                # Strategy-aware integer rules
                if any(tok in name for tok in ('rsi_period', 'stoch_period', 'stochastic', 'cci_period')):
                    lo, hi = 5, 50
                elif 'atr' in name and 'period' in name:
                    lo, hi = 5, 50
                elif any(tok in name for tok in ('ema_fast', 'fast_period', 'short_period')):
                    lo, hi = 5, 50
                elif any(tok in name for tok in ('ema_slow', 'slow_period', 'long_period')):
                    lo, hi = 20, 200
                elif any(tok in name for tok in ('bb_period', 'bollinger_period', 'keltner_period', 'supertrend_period')):
                    lo, hi = 10, 50
                elif any(tok in name for tok in ('period', 'length', 'window', 'ema', 'sma', 'wma')):
                    lo = max(2, int(round(v * 0.5)))
                    hi = min(1000, int(round(max(v + 2, v * 2))))
                else:
                    lo = max(1, int(round(max(1, v * 0.5))))
                    hi = min(1_000_000, int(round(max(v + 1, v * 2))))
                if lo >= hi:
                    lo, hi = min(lo, hi), max(lo, hi)
                    if lo == hi:
                        hi = lo + 1
                space[k] = (lo, hi)
                continue
            # Floats
            if isinstance(v, (float, np.floating)):
                # Common threshold case
                if 0.0 <= v <= 1.0 or any(tok in name for tok in ('threshold', 'alpha', 'beta', 'gamma')):
                    lo = 0.0
                    hi = 1.0
                    space[k] = (float(lo), float(hi))
                    continue
                # Overbought/oversold levels for oscillators
                if 'overbought' in name:
                    space[k] = (60.0, 90.0)
                    continue
                if 'oversold' in name:
                    space[k] = (10.0, 40.0)
                    continue
                # Multipliers/factors
                if any(tok in name for tok in ('atr_mult', 'atr_multiplier', 'mult', 'multiplier', 'factor', 'k')):
                    # Typical ATR/BB multipliers
                    lo = 0.5 if 'atr' in name or 'bb' in name else max(0.01, v * 0.2)
                    hi = 4.0 if 'atr' in name or 'bb' in name else min(10.0, max(lo + 0.01, v * 5.0))
                    if lo >= hi:
                        lo, hi = min(lo, hi), max(lo, hi) if max(lo, hi) > 0 else (0.01, 0.02)
                    space[k] = (float(lo), float(hi))
                    continue
                # Generic float range
                lo = max(1e-9, v * 0.5)
                hi = max(lo + 1e-6, v * 1.5)
                if lo >= hi:
                    hi = lo + 1.0
                space[k] = (float(lo), float(hi))
                continue
            # Unsupported types are ignored (e.g., strings)
        return space
    
    def update_parameters(self, new_parameters: Dict):
        """Update strategy parameters"""
        self.parameters.update(new_parameters)
        logger.info(f"Updated {self.name} parameters: {new_parameters}")
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate that data has required columns"""
        # Accept either 'price' or 'close' column
        if 'price' not in data.columns and 'close' in data.columns:
            data['price'] = data['close']
        
        required_columns = ['timestamp', 'price']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return False
            
        if data.empty:
            logger.warning("Data is empty")
            return False
            
        return True

class StrategyManager:
    """Manages multiple trading strategies"""
    
    def __init__(self):
        self.strategies = {}
        self.active_strategy = None
        
    def add_strategy(self, strategy: BaseStrategy):
        """Add a strategy to the manager"""
        self.strategies[strategy.name] = strategy
        logger.info(f"Added strategy: {strategy.name}")
        
        if self.active_strategy is None:
            self.active_strategy = strategy.name
            logger.info(f"Set {strategy.name} as active strategy")
    
    def set_active_strategy(self, strategy_name: str):
        """Set the active strategy"""
        if strategy_name in self.strategies:
            self.active_strategy = strategy_name
            logger.info(f"Set {strategy_name} as active strategy")
        else:
            raise ValueError(f"Strategy {strategy_name} not found")
    
    def get_active_strategy(self) -> Optional[BaseStrategy]:
        """Get the currently active strategy"""
        if self.active_strategy and self.active_strategy in self.strategies:
            return self.strategies[self.active_strategy]
        return None
    
    def get_signal(self, data: pd.DataFrame) -> Tuple[str, float]:
        """Get signal from active strategy"""
        strategy = self.get_active_strategy()
        if strategy:
            return strategy.get_signal(data)
        return 'hold', 0.0
    
    def list_strategies(self) -> List[str]:
        """Get list of available strategy names"""
        return list(self.strategies.keys())
    
    def get_strategy_info(self, strategy_name: str = None) -> Dict:
        """Get information about a strategy or all strategies"""
        if strategy_name:
            if strategy_name in self.strategies:
                return self.strategies[strategy_name].get_parameter_info()
            else:
                return {'error': f'Strategy {strategy_name} not found'}
        else:
            return {name: strategy.get_parameter_info() for name, strategy in self.strategies.items()}
    
    def backtest_strategy(self, strategy_name: str, data: pd.DataFrame) -> Dict:
        """Backtest a specific strategy"""
        if strategy_name not in self.strategies:
            return {'error': f'Strategy {strategy_name} not found'}
        
        return self.strategies[strategy_name].backtest_signals(data)
    
    def backtest_all_strategies(self, data: pd.DataFrame) -> Dict:
        """Backtest all strategies and compare results"""
        results = {}
        for name, strategy in self.strategies.items():
            results[name] = strategy.backtest_signals(data)
        return results


class StubStrategy(BaseStrategy):
    """A concrete stub strategy for when a real strategy is not found."""
    
    def __init__(self, name: str = "StubStrategy", parameters: Dict = None):
        super().__init__(name, parameters or {})
        
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return data unchanged with minimal placeholder columns"""
        logger.warning(f"Running StubStrategy.calculate_indicators for {self.name}")
        return data.copy()
        
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate no signals, default to hold"""
        logger.warning(f"Running StubStrategy.generate_signals for {self.name}")
        df = data.copy()
        df['buy_signal'] = False
        df['sell_signal'] = False
        df['signal_strength'] = 0.0
        return df
