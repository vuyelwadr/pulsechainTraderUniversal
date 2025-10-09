"""
Core TradingView Strategies - Top 15 High-Priority Implementations
Based on LazyBear's custom indicators collection
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
import talib
from strategies.base_strategy import BaseStrategy
from .indicator_utils import compute_adx, compute_multi_timeframe_adx, compute_atr
import logging

logger = logging.getLogger(__name__)


class SqueezeMomentumStrategy(BaseStrategy):
    """BB/KC Squeeze Momentum Strategy - Detects volatility compression and expansion"""
    
    def __init__(self, parameters: Dict = None):
        default_params = {
            'bb_length': 20,
            'bb_mult': 2.0,
            'kc_length': 20,
            'kc_mult': 1.5,
            'momentum_length': 12,
            'signal_threshold': 0.6
        }
        if parameters:
            default_params.update(parameters)
        super().__init__("SqueezeMomentum", default_params)
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        
        # Bollinger Bands
        df['bb_basis'] = talib.SMA(df['close'], timeperiod=self.parameters['bb_length'])
        df['bb_dev'] = talib.STDDEV(df['close'], timeperiod=self.parameters['bb_length'])
        df['bb_upper'] = df['bb_basis'] + df['bb_dev'] * self.parameters['bb_mult']
        df['bb_lower'] = df['bb_basis'] - df['bb_dev'] * self.parameters['bb_mult']
        
        # Keltner Channels
        df['kc_basis'] = talib.EMA(df['close'], timeperiod=self.parameters['kc_length'])
        df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=self.parameters['kc_length'])
        df['kc_upper'] = df['kc_basis'] + df['atr'] * self.parameters['kc_mult']
        df['kc_lower'] = df['kc_basis'] - df['atr'] * self.parameters['kc_mult']
        
        # Squeeze Detection (BB inside KC)
        df['squeeze'] = (df['bb_lower'] > df['kc_lower']) & (df['bb_upper'] < df['kc_upper'])
        
        # Momentum Oscillator
        highest = df['high'].rolling(window=self.parameters['momentum_length']).max()
        lowest = df['low'].rolling(window=self.parameters['momentum_length']).min()
        df['momentum'] = (df['close'] - (highest + lowest) / 2) / (highest - lowest + 0.0001)
        
        return df
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        
        # Squeeze Release Signals
        df['squeeze_release'] = df['squeeze'].shift(1) & ~df['squeeze']
        
        # Buy when squeeze releases and momentum is positive
        df['buy_signal'] = df['squeeze_release'] & (df['momentum'] > 0)
        
        # Sell when momentum turns negative after squeeze
        df['sell_signal'] = (df['momentum'] < 0) & (df['momentum'].shift(1) >= 0)
        
        # Signal strength based on momentum magnitude
        df['signal_strength'] = np.abs(df['momentum']).clip(0, 1)
        df.loc[df['signal_strength'] < self.parameters['signal_threshold'], 'buy_signal'] = False
        df.loc[df['signal_strength'] < self.parameters['signal_threshold'], 'sell_signal'] = False
        
        return df

    @classmethod
    def parameter_space(cls) -> Dict[str, Tuple[float, float]]:
        """Define parameter bounds for optimization"""
        return {
            'bb_length': (10, 50),  # Bollinger Bands period
            'bb_mult': (1.5, 3.0),  # BB multiplier
            'kc_length': (10, 50),  # Keltner Channel period
            'kc_mult': (1.0, 2.5),  # KC multiplier
            'momentum_length': (5, 30),  # Momentum period
            'signal_threshold': (0.0, 0.95),  # Signal threshold can't be 1.0
        }


class WaveTrendStrategy(BaseStrategy):
    """WaveTrend Oscillator - Advanced momentum oscillator"""
    
    def __init__(self, parameters: Dict = None):
        default_params = {
            'channel_length': 10,
            'average_length': 21,
            'overbought': 8,
            'oversold': -8,
            'signal_threshold': 0.02
        }
        if parameters:
            default_params.update(parameters)
        super().__init__("WaveTrend", default_params)
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        
        # Calculate Average Price (HLC3)
        df['hlc3'] = (df['high'] + df['low'] + df['close']) / 3
        
        # ESA (EMA of HLC3)
        df['esa'] = talib.EMA(df['hlc3'], timeperiod=self.parameters['channel_length'])
        
        # D (EMA of absolute difference)
        df['d'] = talib.EMA(np.abs(df['hlc3'] - df['esa']), timeperiod=self.parameters['channel_length'])
        
        # CI (Channel Index)
        df['ci'] = (df['hlc3'] - df['esa']) / (0.015 * df['d'] + 0.0001)
        
        # TCI (2-period EMA of CI)
        df['tci'] = talib.EMA(df['ci'], timeperiod=self.parameters['average_length'])
        
        # WT1 and WT2
        df['wt1'] = df['tci']
        df['wt2'] = talib.SMA(df['wt1'], timeperiod=4)
        
        return df
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        
        # Buy signals: WT1 crosses above WT2 in oversold area
        df['wt_cross_up'] = (df['wt1'] > df['wt2']) & (df['wt1'].shift(1) <= df['wt2'].shift(1))
        df['buy_signal'] = df['wt_cross_up'] & (df['wt1'] < self.parameters['oversold'])
        
        # Sell signals: WT1 crosses below WT2 in overbought area
        df['wt_cross_down'] = (df['wt1'] < df['wt2']) & (df['wt1'].shift(1) >= df['wt2'].shift(1))
        df['sell_signal'] = df['wt_cross_down'] & (df['wt1'] > self.parameters['overbought'])
        
        # Signal strength based on divergence magnitude
        df['signal_strength'] = np.abs(df['wt1'] - df['wt2']) / 5.0
        df['signal_strength'] = df['signal_strength'].clip(0, 1)
        df.loc[df['signal_strength'] < self.parameters['signal_threshold'], ['buy_signal', 'sell_signal']] = False
        
        return df

    @classmethod
    def parameter_space(cls) -> Dict[str, Tuple[float, float]]:
        """Define parameter bounds for optimization"""
        return {
            'channel_length': (5, 50),  # EMA period for ESA
            'average_length': (5, 50),  # EMA period for TCI
            'overbought': (6, 18),  # Lower thresholds for low-vol regimes
            'oversold': (-18, -6),  # Lower thresholds for low-vol regimes
            'signal_threshold': (0.01, 0.08),  # Strategy becomes active sooner
        }


class CoralTrendStrategy(BaseStrategy):
    """Coral Trend Filter - Adaptive trend following indicator"""
    
    def __init__(self, parameters: Dict = None):
        default_params = {
            'period': 21,
            'smooth_factor': 0.4,
            'signal_threshold': 0.7
        }
        if parameters:
            default_params.update(parameters)
        super().__init__("CoralTrend", default_params)
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        
        # Triple Exponential Moving Average (TEMA)
        ema1 = talib.EMA(df['close'], timeperiod=self.parameters['period'])
        ema2 = talib.EMA(ema1, timeperiod=self.parameters['period'])
        ema3 = talib.EMA(ema2, timeperiod=self.parameters['period'])
        
        df['coral'] = 3 * ema1 - 3 * ema2 + ema3
        
        # Smoothed coral line
        df['coral_smooth'] = talib.EMA(df['coral'], timeperiod=int(self.parameters['period'] * self.parameters['smooth_factor']))
        
        # Trend direction
        df['trend'] = np.where(df['close'] > df['coral_smooth'], 1, -1)
        
        return df
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        
        # Buy when price crosses above coral and trend turns positive
        df['buy_signal'] = (df['trend'] == 1) & (df['trend'].shift(1) == -1)
        
        # Sell when price crosses below coral and trend turns negative
        df['sell_signal'] = (df['trend'] == -1) & (df['trend'].shift(1) == 1)
        
        # Signal strength based on distance from coral
        df['distance'] = np.abs(df['close'] - df['coral_smooth']) / df['close']
        df['signal_strength'] = df['distance'].clip(0, 0.1) * 10
        
        return df

    @classmethod
    def parameter_space(cls) -> Dict[str, Tuple[float, float]]:
        """Define parameter bounds for optimization"""
        return {
            'period': (10, 50),  # TEMA period
            'smooth_factor': (0.2, 0.8),  # Smoothing factor
            'signal_threshold': (0.0, 0.95),  # Signal threshold can't be 1.0
        }


class SchaffTrendCycleStrategy(BaseStrategy):
    """Schaff Trend Cycle - Combines MACD and Stochastic for trend detection"""
    
    def __init__(self, parameters: Dict = None):
        default_params = {
            'fast_length': 23,
            'slow_length': 50,
            'cycle_length': 10,
            'overbought': 75,
            'oversold': 25,
            'signal_threshold': 0.6
        }
        if parameters:
            default_params.update(parameters)
        super().__init__("SchaffTrendCycle", default_params)
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        
        # MACD Line
        df['macd'] = talib.EMA(df['close'], timeperiod=self.parameters['fast_length']) - \
                     talib.EMA(df['close'], timeperiod=self.parameters['slow_length'])
        
        # First Stochastic
        macd_min = df['macd'].rolling(window=self.parameters['cycle_length']).min()
        macd_max = df['macd'].rolling(window=self.parameters['cycle_length']).max()
        df['stoch1'] = 100 * (df['macd'] - macd_min) / (macd_max - macd_min + 0.0001)
        
        # Smooth first stochastic
        df['stoch1_smooth'] = talib.EMA(df['stoch1'], timeperiod=self.parameters['cycle_length'] // 2)
        
        # Second Stochastic
        stoch1_min = df['stoch1_smooth'].rolling(window=self.parameters['cycle_length']).min()
        stoch1_max = df['stoch1_smooth'].rolling(window=self.parameters['cycle_length']).max()
        df['stc'] = 100 * (df['stoch1_smooth'] - stoch1_min) / (stoch1_max - stoch1_min + 0.0001)
        
        # Smooth STC
        df['stc_smooth'] = talib.EMA(df['stc'], timeperiod=self.parameters['cycle_length'] // 2)
        
        return df
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        
        # Buy when STC crosses above oversold level
        df['buy_signal'] = (df['stc_smooth'] > self.parameters['oversold']) & \
                          (df['stc_smooth'].shift(1) <= self.parameters['oversold'])
        
        # Sell when STC crosses below overbought level
        df['sell_signal'] = (df['stc_smooth'] < self.parameters['overbought']) & \
                           (df['stc_smooth'].shift(1) >= self.parameters['overbought'])
        
        # Signal strength
        df['signal_strength'] = np.abs(df['stc_smooth'] - 50) / 50
        df['signal_strength'] = df['signal_strength'].clip(0, 1)
        
        return df

    @classmethod
    def parameter_space(cls) -> Dict[str, Tuple[float, float]]:
        """Define parameter bounds for optimization"""
        return {
            'fast_length': (5, 30),  # MACD fast period
            'slow_length': (20, 60),  # MACD slow period
            'cycle_length': (5, 20),  # Stochastic cycle length
            'overbought': (70, 90),  # STC overbought level
            'oversold': (10, 30),  # STC oversold level
            'signal_threshold': (0.0, 0.95),  # Signal threshold can't be 1.0
        }


class MESAAdaptiveMAStrategy(BaseStrategy):
    """MESA Adaptive Moving Average - John Ehlers' adaptive indicator"""
    
    def __init__(self, parameters: Dict = None):
        default_params = {
            'fast_limit': 0.5,
            'slow_limit': 0.05,
            'signal_threshold': 0.6
        }
        if parameters:
            default_params.update(parameters)
        super().__init__("MESAAdaptiveMA", default_params)
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()

        df['smooth'] = (4 * df['close'] + 3 * df['close'].shift(1) +
                       2 * df['close'].shift(2) + df['close'].shift(3)) / 10

        df['detrender'] = (0.0962 * df['smooth'] + 0.5769 * df['smooth'].shift(2) -
                          0.5769 * df['smooth'].shift(4) - 0.0962 * df['smooth'].shift(6))

        df['q1'] = (0.0962 * df['detrender'] + 0.5769 * df['detrender'].shift(2) -
                   0.5769 * df['detrender'].shift(4) - 0.0962 * df['detrender'].shift(6))
        df['i1'] = df['detrender'].shift(3)

        df['phase'] = np.arctan2(df['q1'], df['i1'])
        df['delta_phase'] = (df['phase'] - df['phase'].shift(1)).clip(-1.1, 1.1)

        close = df['close'].astype(float).to_numpy()
        delta_phase = df['delta_phase'].to_numpy(dtype=float)
        length = len(df)
        instant_period = np.zeros(length, dtype=float)
        period = np.zeros(length, dtype=float)
        fast_limit = float(self.parameters['fast_limit'])
        slow_limit = float(self.parameters['slow_limit'])
        alpha = np.full(length, fast_limit, dtype=float)
        mama = close.copy()
        fama = close.copy()

        for i in range(6, length):
            dp = delta_phase[i]
            if not np.isnan(dp) and dp != 0.0:
                instant_period[i] = 360.0 / np.abs(dp)
            else:
                instant_period[i] = 0.0
            period[i] = 0.2 * instant_period[i] + 0.8 * period[i - 1]
            period[i] = np.clip(period[i], 6.0, 50.0)
            alpha[i] = 2.0 / (period[i] + 1.0)
            alpha[i] = np.clip(alpha[i], slow_limit, fast_limit)
            mama[i] = alpha[i] * close[i] + (1.0 - alpha[i]) * mama[i - 1]
            fama[i] = 0.5 * alpha[i] * mama[i] + (1.0 - 0.5 * alpha[i]) * fama[i - 1]

        df['instant_period'] = instant_period
        df['period'] = period
        df['alpha'] = alpha
        df['mama'] = mama
        df['fama'] = fama

        return df
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        
        # Buy when MAMA crosses above FAMA
        df['buy_signal'] = (df['mama'] > df['fama']) & (df['mama'].shift(1) <= df['fama'].shift(1))
        
        # Sell when MAMA crosses below FAMA
        df['sell_signal'] = (df['mama'] < df['fama']) & (df['mama'].shift(1) >= df['fama'].shift(1))
        
        # Signal strength based on divergence
        df['signal_strength'] = np.abs(df['mama'] - df['fama']) / df['close']
        df['signal_strength'] = df['signal_strength'].clip(0, 1)
        
        return df

    @classmethod
    def parameter_space(cls) -> Dict[str, Tuple[float, float]]:
        """Define parameter bounds for optimization"""
        return {
            'fast_limit': (0.1, 0.8),  # Alpha fast limit
            'slow_limit': (0.01, 0.2),  # Alpha slow limit
            'signal_threshold': (0.0, 0.95),  # Signal threshold can't be 1.0
        }


class ElderImpulseStrategy(BaseStrategy):
    """Elder Impulse System - Combines trend and momentum"""
    
    def __init__(self, parameters: Dict = None):
        default_params = {
            'ema_period': 13,
            'macd_fast': 12,
            'macd_slow': 26,
            'signal_threshold': 0.6
        }
        if parameters:
            default_params.update(parameters)
        super().__init__("ElderImpulse", default_params)
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        
        # 13-period EMA
        df['ema'] = talib.EMA(df['close'], timeperiod=self.parameters['ema_period'])
        
        # MACD Histogram
        macd, signal, hist = talib.MACD(df['close'], 
                                        fastperiod=self.parameters['macd_fast'],
                                        slowperiod=self.parameters['macd_slow'])
        df['macd_hist'] = hist
        
        # Elder Impulse Colors
        # Green: Both EMA and MACD-H rising
        # Red: Both EMA and MACD-H falling
        # Blue: Mixed signals
        df['ema_rising'] = df['ema'] > df['ema'].shift(1)
        df['macd_rising'] = df['macd_hist'] > df['macd_hist'].shift(1)
        
        df['impulse'] = 0  # Blue/Neutral
        df.loc[df['ema_rising'] & df['macd_rising'], 'impulse'] = 1  # Green
        df.loc[~df['ema_rising'] & ~df['macd_rising'], 'impulse'] = -1  # Red
        
        return df
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        
        # Buy when impulse turns green from red or blue
        df['buy_signal'] = (df['impulse'] == 1) & (df['impulse'].shift(1) != 1)
        
        # Sell when impulse turns red from green or blue
        df['sell_signal'] = (df['impulse'] == -1) & (df['impulse'].shift(1) != -1)
        
        # Signal strength based on momentum
        df['signal_strength'] = np.abs(df['macd_hist']) / df['macd_hist'].rolling(20).std()
        df['signal_strength'] = df['signal_strength'].clip(0, 1)
        
        return df

    @classmethod
    def parameter_space(cls) -> Dict[str, Tuple[float, float]]:
        """Define parameter bounds for optimization"""
        return {
            'ema_period': (5, 30),  # EMA period
            'macd_fast': (5, 20),  # MACD fast period
            'macd_slow': (20, 40),  # MACD slow period
            'signal_threshold': (0.0, 0.95),  # Signal threshold can't be 1.0
        }


class FRAMAStrategy(BaseStrategy):
    """Fractal Adaptive Moving Average - Adapts to market fractality"""
    
    def __init__(self, parameters: Dict = None):
        default_params = {
            'period': 20,
            'slow_period': 200,
            'signal_threshold': 0.6
        }
        if parameters:
            default_params.update(parameters)
        super().__init__("FRAMA", default_params)
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        
        n = self.parameters['period']
        df['frama'] = df['close'].copy()
        
        for i in range(2*n, len(df)):
            # Calculate fractal dimension
            n1 = (df['high'].iloc[i-n+1:i+1].max() - df['low'].iloc[i-n+1:i+1].min()) / n
            n2 = (df['high'].iloc[i-2*n+1:i-n+1].max() - df['low'].iloc[i-2*n+1:i-n+1].min()) / n
            n3 = (df['high'].iloc[i-2*n+1:i+1].max() - df['low'].iloc[i-2*n+1:i+1].min()) / (2*n)
            
            if n1 > 0 and n2 > 0 and n3 > 0:
                d = (np.log(n1 + n2) - np.log(n3)) / np.log(2)
                alpha = np.exp(-4.6 * (d - 1))
                alpha = max(min(alpha, 1), 2/(self.parameters['slow_period']+1))
            else:
                alpha = 2/(self.parameters['slow_period']+1)
            
            df.loc[df.index[i], 'frama'] = alpha * df['close'].iloc[i] + (1 - alpha) * df['frama'].iloc[i-1]
        
        return df
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        
        # Buy when price crosses above FRAMA
        df['buy_signal'] = (df['close'] > df['frama']) & (df['close'].shift(1) <= df['frama'].shift(1))
        
        # Sell when price crosses below FRAMA
        df['sell_signal'] = (df['close'] < df['frama']) & (df['close'].shift(1) >= df['frama'].shift(1))
        
        # Signal strength based on distance from FRAMA
        df['signal_strength'] = np.abs(df['close'] - df['frama']) / df['close']
        df['signal_strength'] = df['signal_strength'].clip(0, 1)
        
        return df

    @classmethod
    def parameter_space(cls) -> Dict[str, Tuple[float, float]]:
        """Define parameter bounds for optimization"""
        return {
            'period': (10, 50),  # FRAMA period
            'slow_period': (100, 300),  # Slow period for alpha calculation
            'signal_threshold': (0.0, 0.95),  # Signal threshold can't be 1.0
        }


class ZeroLagEMAStrategy(BaseStrategy):
    """Zero Lag Exponential Moving Average - Reduces lag in trend following"""
    
    def __init__(self, parameters: Dict = None):
        default_params = {
            'period': 28,
            'gain_limit': 12,
            'signal_threshold': 0.03,
            'trade_amount_pct': 0.15,
            'volatility_floor': 0.0006
        }
        if parameters:
            default_params.update(parameters)
        super().__init__("ZeroLagEMA", default_params)
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        
        # Calculate lag
        lag = int((self.parameters['period'] - 1) / 2)
        
        # Regular EMA
        df['ema1'] = talib.EMA(df['close'], timeperiod=self.parameters['period'])
        
        # EMA of EMA
        df['ema2'] = talib.EMA(df['ema1'], timeperiod=self.parameters['period'])
        
        # Calculate difference
        df['diff'] = df['ema1'] - df['ema2']
        
        # Zero Lag EMA
        df['zlema'] = df['ema1'] + df['diff']
        
        return df
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        
        # Simple volatility gate: require recent absolute return to exceed floor
        df['ret'] = df['close'].pct_change().abs()
        vol_floor = float(self.parameters.get('volatility_floor', 0.0) or 0.0)
        df['vol_ok'] = df['ret'].rolling(window=3, min_periods=1).mean() >= vol_floor
        
        # Buy when price crosses above ZLEMA
        df['buy_signal'] = (
            (df['close'] > df['zlema']) &
            (df['close'].shift(1) <= df['zlema'].shift(1)) &
            df['vol_ok']
        )
        
        # Sell when price crosses below ZLEMA
        df['sell_signal'] = (
            (df['close'] < df['zlema']) &
            (df['close'].shift(1) >= df['zlema'].shift(1)) &
            df['vol_ok']
        )
        
        # Signal strength
        df['signal_strength'] = np.abs(df['close'] - df['zlema']) / df['close']
        df['signal_strength'] = df['signal_strength'].clip(0, 1)
        df.loc[df['signal_strength'] < self.parameters['signal_threshold'], ['buy_signal', 'sell_signal']] = False
        
        df.drop(columns=['ret', 'vol_ok'], inplace=True)
        
        return df

    @classmethod
    def parameter_space(cls) -> Dict[str, Tuple[float, float]]:
        """Define parameter bounds for optimization"""
        return {
            'period': (20, 40),  # EMA period
            'gain_limit': (5, 20),  # Gain limit parameter
            'signal_threshold': (0.015, 0.08),  # Guard against hyper-trading
            'trade_amount_pct': (0.05, 0.2),
            'volatility_floor': (0.0003, 0.0015),
        }


class KaufmannAMAStrategy(BaseStrategy):
    """Kaufmann Adaptive Moving Average - Efficiency ratio based adaptation"""
    
    def __init__(self, parameters: Dict = None):
        default_params = {
            'period': 10,
            'fast_period': 2,
            'slow_period': 30,
            'signal_threshold': 0.6
        }
        if parameters:
            default_params.update(parameters)
        super().__init__("KaufmannAMA", default_params)
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        
        # Calculate Efficiency Ratio
        change = np.abs(df['close'] - df['close'].shift(self.parameters['period']))
        volatility = df['close'].diff().abs().rolling(window=self.parameters['period']).sum()
        df['er'] = change / (volatility + 0.0001)
        
        # Calculate smoothing constants
        fastest_sc = 2 / (self.parameters['fast_period'] + 1)
        slowest_sc = 2 / (self.parameters['slow_period'] + 1)
        df['sc'] = (df['er'] * (fastest_sc - slowest_sc) + slowest_sc) ** 2
        
        # Calculate KAMA
        df['kama'] = df['close'].copy()
        for i in range(self.parameters['period'], len(df)):
            df.loc[df.index[i], 'kama'] = df['kama'].iloc[i-1] + \
                                          df['sc'].iloc[i] * (df['close'].iloc[i] - df['kama'].iloc[i-1])
        
        return df
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        
        # Buy when price crosses above KAMA
        df['buy_signal'] = (df['close'] > df['kama']) & (df['close'].shift(1) <= df['kama'].shift(1))
        
        # Sell when price crosses below KAMA
        df['sell_signal'] = (df['close'] < df['kama']) & (df['close'].shift(1) >= df['kama'].shift(1))
        
        # Signal strength based on efficiency ratio
        df['signal_strength'] = df['er']
        
        return df

    @classmethod
    def parameter_space(cls) -> Dict[str, Tuple[float, float]]:
        """Define parameter bounds for optimization"""
        return {
            'period': (5, 20),  # Efficiency ratio period
            'fast_period': (2, 5),  # Fast smoothing constant base
            'slow_period': (20, 50),  # Slow smoothing constant base
            'signal_threshold': (0.0, 0.95),  # Signal threshold can't be 1.0
        }


class TradersDynamicIndexStrategy(BaseStrategy):
    """Traders Dynamic Index - Combines RSI with volatility bands"""
    
    def __init__(self, parameters: Dict = None):
        default_params = {
            'rsi_period': 13,
            'band_period': 34,
            'fast_period': 2,
            'slow_period': 7,
            'signal_threshold': 0.6
        }
        if parameters:
            default_params.update(parameters)
        super().__init__("TradersDynamicIndex", default_params)
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        
        # Calculate RSI
        df['rsi'] = talib.RSI(df['close'], timeperiod=self.parameters['rsi_period'])
        
        # Smooth RSI
        df['rsi_smooth'] = talib.SMA(df['rsi'], timeperiod=self.parameters['fast_period'])
        
        # Calculate bands on RSI
        df['rsi_ma'] = talib.SMA(df['rsi_smooth'], timeperiod=self.parameters['band_period'])
        df['rsi_std'] = df['rsi_smooth'].rolling(window=self.parameters['band_period']).std()
        df['rsi_upper'] = df['rsi_ma'] + 1.6185 * df['rsi_std']
        df['rsi_lower'] = df['rsi_ma'] - 1.6185 * df['rsi_std']
        
        # Market base line
        df['mbl'] = talib.SMA(df['rsi_smooth'], timeperiod=self.parameters['slow_period'])
        
        return df
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        
        # Buy when RSI crosses above lower band and MBL is rising
        df['buy_signal'] = (df['rsi_smooth'] > df['rsi_lower']) & \
                          (df['rsi_smooth'].shift(1) <= df['rsi_lower'].shift(1)) & \
                          (df['mbl'] > df['mbl'].shift(1))
        
        # Sell when RSI crosses below upper band and MBL is falling
        df['sell_signal'] = (df['rsi_smooth'] < df['rsi_upper']) & \
                           (df['rsi_smooth'].shift(1) >= df['rsi_upper'].shift(1)) & \
                           (df['mbl'] < df['mbl'].shift(1))
        
        # Signal strength
        df['signal_strength'] = np.abs(df['rsi_smooth'] - 50) / 50
        
        return df

    @classmethod
    def parameter_space(cls) -> Dict[str, Tuple[float, float]]:
        """Define parameter bounds for optimization"""
        return {
            'rsi_period': (5, 30),  # RSI period
            'band_period': (20, 50),  # Band calculation period
            'fast_period': (2, 10),  # Fast smoothing period
            'slow_period': (5, 15),  # Slow period for MBL
            'signal_threshold': (0.0, 0.95),  # Signal threshold can't be 1.0
        }


class InsyncIndexStrategy(BaseStrategy):
    """Insync Index - Consensus of multiple indicators"""
    
    def __init__(self, parameters: Dict = None):
        default_params = {
            'period': 14,
            'threshold_buy': -50,
            'threshold_sell': 50,
            'signal_threshold': 0.7
        }
        if parameters:
            default_params.update(parameters)
        super().__init__("InsyncIndex", default_params)
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        period = self.parameters['period']
        
        # Multiple indicators for consensus
        df['rsi'] = talib.RSI(df['close'], timeperiod=period)
        df['stoch'], _ = talib.STOCH(df['high'], df['low'], df['close'], 
                                     fastk_period=period, slowk_period=3, slowd_period=3)
        df['cci'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=period)
        df['mfi'] = talib.MFI(df['high'], df['low'], df['close'], df['volume'], timeperiod=period)
        df['willr'] = talib.WILLR(df['high'], df['low'], df['close'], timeperiod=period)
        
        # Normalize indicators to -100 to 100 scale
        df['rsi_norm'] = (df['rsi'] - 50) * 2
        df['stoch_norm'] = (df['stoch'] - 50) * 2
        df['cci_norm'] = np.clip(df['cci'], -100, 100)
        df['mfi_norm'] = (df['mfi'] - 50) * 2
        df['willr_norm'] = df['willr']
        
        # Calculate Insync Index (average of all normalized indicators)
        df['insync'] = (df['rsi_norm'] + df['stoch_norm'] + df['cci_norm'] + 
                       df['mfi_norm'] + df['willr_norm']) / 5
        
        return df
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        
        # Buy when Insync crosses above oversold threshold
        df['buy_signal'] = (df['insync'] > self.parameters['threshold_buy']) & \
                          (df['insync'].shift(1) <= self.parameters['threshold_buy'])
        
        # Sell when Insync crosses below overbought threshold
        df['sell_signal'] = (df['insync'] < self.parameters['threshold_sell']) & \
                           (df['insync'].shift(1) >= self.parameters['threshold_sell'])
        
        # Signal strength
        df['signal_strength'] = np.abs(df['insync']) / 100
        df['signal_strength'] = df['signal_strength'].clip(0, 1)
        
        return df

    @classmethod
    def parameter_space(cls) -> Dict[str, Tuple[float, float]]:
        """Define parameter bounds for optimization"""
        return {
            'period': (5, 30),  # Indicator period
            'threshold_buy': (-80, -20),  # Buy threshold (negative for oversold)
            'threshold_sell': (20, 80),  # Sell threshold (positive for overbought)
            'signal_threshold': (0.0, 0.95),  # Signal threshold can't be 1.0
        }


class PremierStochasticStrategy(BaseStrategy):
    """Premier Stochastic Oscillator - Enhanced stochastic with normalization"""
    
    def __init__(self, parameters: Dict = None):
        default_params = {
            'period': 8,
            'smooth_period': 25,
            'overbought': 0.9,
            'oversold': -0.9,
            'signal_threshold': 0.6
        }
        if parameters:
            default_params.update(parameters)
        super().__init__("PremierStochastic", default_params)
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        
        # Regular Stochastic
        df['stoch'], _ = talib.STOCH(df['high'], df['low'], df['close'],
                                     fastk_period=self.parameters['period'],
                                     slowk_period=1, slowd_period=1)
        
        # Normalize to -1 to 1
        df['norm_stoch'] = (df['stoch'] - 50) / 50
        
        # Smooth
        df['smooth_stoch'] = talib.EMA(df['norm_stoch'], timeperiod=self.parameters['smooth_period'])
        
        # Apply exponential transformation for Premier Stochastic
        df['pso'] = (np.exp(5 * df['smooth_stoch']) - 1) / (np.exp(5 * df['smooth_stoch']) + 1)
        
        return df
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        
        # Buy when PSO crosses above oversold
        df['buy_signal'] = (df['pso'] > self.parameters['oversold']) & \
                          (df['pso'].shift(1) <= self.parameters['oversold'])
        
        # Sell when PSO crosses below overbought
        df['sell_signal'] = (df['pso'] < self.parameters['overbought']) & \
                           (df['pso'].shift(1) >= self.parameters['overbought'])
        
        # Signal strength
        df['signal_strength'] = np.abs(df['pso'])
        
        return df

    @classmethod
    def parameter_space(cls) -> Dict[str, Tuple[float, float]]:
        """Define parameter bounds for optimization"""
        return {
            'period': (5, 20),  # Stochastic period
            'smooth_period': (10, 50),  # Smoothing period
            'overbought': (0.5, 0.95),  # Overbought level
            'oversold': (-0.95, -0.5),  # Oversold level
            'signal_threshold': (0.0, 0.95),  # Signal threshold can't be 1.0
        }


class MACZStrategy(BaseStrategy):
    """MACD with Z-Score normalization"""
    
    def __init__(self, parameters: Dict = None):
        default_params = {
            'fast_period': 12,
            'slow_period': 26,
            'signal_period': 9,
            'z_period': 20,
            'signal_threshold': 0.6
        }
        if parameters:
            default_params.update(parameters)
        super().__init__("MAC_Z", default_params)
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        
        # Calculate MACD
        df['macd'], df['signal'], df['hist'] = talib.MACD(df['close'],
                                                          fastperiod=self.parameters['fast_period'],
                                                          slowperiod=self.parameters['slow_period'],
                                                          signalperiod=self.parameters['signal_period'])
        
        # Calculate Z-Score of MACD
        df['macd_mean'] = df['macd'].rolling(window=self.parameters['z_period']).mean()
        df['macd_std'] = df['macd'].rolling(window=self.parameters['z_period']).std()
        df['macd_z'] = (df['macd'] - df['macd_mean']) / (df['macd_std'] + 0.0001)
        
        # Z-Score of histogram
        df['hist_mean'] = df['hist'].rolling(window=self.parameters['z_period']).mean()
        df['hist_std'] = df['hist'].rolling(window=self.parameters['z_period']).std()
        df['hist_z'] = (df['hist'] - df['hist_mean']) / (df['hist_std'] + 0.0001)
        
        return df
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        
        # Buy when MACD Z-score crosses above -2 and histogram is positive
        df['buy_signal'] = (df['macd_z'] > -2) & (df['macd_z'].shift(1) <= -2) & (df['hist_z'] > 0)
        
        # Sell when MACD Z-score crosses below 2 and histogram is negative
        df['sell_signal'] = (df['macd_z'] < 2) & (df['macd_z'].shift(1) >= 2) & (df['hist_z'] < 0)
        
        # Signal strength based on Z-score magnitude
        df['signal_strength'] = np.abs(df['macd_z']).clip(0, 3) / 3
        
        return df

    @classmethod
    def parameter_space(cls) -> Dict[str, Tuple[float, float]]:
        """Define parameter bounds for optimization"""
        return {
            'fast_period': (5, 20),  # MACD fast period
            'slow_period': (20, 40),  # MACD slow period
            'signal_period': (5, 15),  # MACD signal period
            'z_period': (10, 50),  # Z-score calculation period
            'signal_threshold': (0.0, 0.95),  # Signal threshold can't be 1.0
        }


class FireflyOscillatorStrategy(BaseStrategy):
    """Firefly Oscillator - Custom momentum oscillator with adaptive smoothing"""
    
    def __init__(self, parameters: Dict = None):
        default_params = {
            'period': 10,
            'smooth_period': 5,
            'threshold': 0.02,
            'signal_threshold': 0.6
        }
        if parameters:
            default_params.update(parameters)
        super().__init__("FireflyOscillator", default_params)
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        
        # Price momentum
        df['momentum'] = df['close'] - df['close'].shift(self.parameters['period'])
        
        # Volume-weighted momentum
        df['vwm'] = df['momentum'] * df['volume']
        df['vwm_sum'] = df['vwm'].rolling(window=self.parameters['period']).sum()
        df['vol_sum'] = df['volume'].rolling(window=self.parameters['period']).sum()
        
        # Firefly oscillator
        df['firefly'] = df['vwm_sum'] / (df['vol_sum'] * df['close'] + 0.0001)
        
        # Smooth the oscillator
        df['firefly_smooth'] = talib.EMA(df['firefly'], timeperiod=self.parameters['smooth_period'])
        
        return df
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        
        # Buy when firefly crosses above positive threshold
        df['buy_signal'] = (df['firefly_smooth'] > self.parameters['threshold']) & \
                          (df['firefly_smooth'].shift(1) <= self.parameters['threshold'])
        
        # Sell when firefly crosses below negative threshold
        df['sell_signal'] = (df['firefly_smooth'] < -self.parameters['threshold']) & \
                           (df['firefly_smooth'].shift(1) >= -self.parameters['threshold'])
        
        # Signal strength
        df['signal_strength'] = np.abs(df['firefly_smooth']).clip(0, 0.1) * 10
        
        return df

    @classmethod
    def parameter_space(cls) -> Dict[str, Tuple[float, float]]:
        """Define parameter bounds for optimization"""
        return {
            'period': (5, 30),  # Momentum period
            'smooth_period': (3, 15),  # Smoothing period
            'threshold': (0.005, 0.05),  # Signal threshold
            'signal_threshold': (0.0, 0.95),  # Signal threshold can't be 1.0
        }


class CompositeMomentumIndexStrategy(BaseStrategy):
    """Composite Momentum Index - Combines multiple momentum indicators"""
    
    def __init__(self, parameters: Dict = None):
        default_params = {
            'rsi_period': 14,
            'roc_period': 10,
            'tsi_long': 25,
            'tsi_short': 13,
            'signal_threshold': 0.7,
            'adx_period': 14,
            'adx_htf_period': 14,
            'adx_htf_minutes': 60,
            'adx_trend_threshold': 25.0,
            'atr_period': 14,
            'atr_floor_pct': 0.002,
            'chandelier_period': 22,
            'chandelier_atr_mult': 3.0,
        }
        if parameters:
            default_params.update(parameters)
        super().__init__("CompositeMomentumIndex", default_params)
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        
        # RSI
        df['rsi'] = talib.RSI(df['close'], timeperiod=self.parameters['rsi_period'])
        
        # Rate of Change
        df['roc'] = talib.ROC(df['close'], timeperiod=self.parameters['roc_period'])
        
        # True Strength Index components
        mom = df['close'] - df['close'].shift(1)
        abs_mom = np.abs(mom)
        
        # Double smoothed momentum
        ema_mom_slow = talib.EMA(mom, timeperiod=self.parameters['tsi_long'])
        ema_mom_fast = talib.EMA(ema_mom_slow, timeperiod=self.parameters['tsi_short'])
        
        # Double smoothed absolute momentum
        ema_abs_slow = talib.EMA(abs_mom, timeperiod=self.parameters['tsi_long'])
        ema_abs_fast = talib.EMA(ema_abs_slow, timeperiod=self.parameters['tsi_short'])
        
        # TSI
        df['tsi'] = 100 * ema_mom_fast / (ema_abs_fast + 0.0001)
        
        # Normalize all indicators to 0-100 scale
        df['rsi_norm'] = df['rsi']
        df['roc_norm'] = 50 + df['roc'].clip(-50, 50)
        df['tsi_norm'] = 50 + df['tsi'].clip(-50, 50) / 2
        
        # Composite Momentum Index (weighted average)
        df['cmi'] = (df['rsi_norm'] * 0.4 + df['roc_norm'] * 0.3 + df['tsi_norm'] * 0.3)

        high = df['high'] if 'high' in df.columns else df['close']
        low = df['low'] if 'low' in df.columns else df['close']
        close = df['close']

        adx_df = compute_adx(high, low, close, int(self.parameters['adx_period']))
        df['adx'] = adx_df['adx']
        df['plus_di'] = adx_df['plus_di']
        df['minus_di'] = adx_df['minus_di']
        df['adx_htf'] = compute_multi_timeframe_adx(
            df[['timestamp', 'open', 'high', 'low', 'close']].copy(),
            period=int(self.parameters['adx_htf_period']),
            timeframe_minutes=int(self.parameters['adx_htf_minutes']),
        )
        adx_threshold = float(self.parameters['adx_trend_threshold'])
        df['is_trending'] = (df['adx'] >= adx_threshold) & (df['adx_htf'] >= adx_threshold)

        atr_period = int(self.parameters['atr_period'])
        atr_series = compute_atr(high, low, close, atr_period)
        df['atr'] = atr_series
        df['atr_pct'] = atr_series / close.replace(0, np.nan)
        atr_floor_pct = float(self.parameters['atr_floor_pct'])
        if atr_floor_pct > 0:
            df['atr_ok'] = df['atr_pct'] >= atr_floor_pct
        else:
            df['atr_ok'] = True

        chandelier_period = int(self.parameters['chandelier_period'])
        chandelier_mult = float(self.parameters['chandelier_atr_mult'])
        rolling_high = df['high'].rolling(chandelier_period, min_periods=1).max()
        df['chandelier_long'] = rolling_high - chandelier_mult * df['atr']

        return df
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        
        trend_mask = df.get('is_trending', True)
        atr_mask = df.get('atr_ok', True)

        # Buy when CMI crosses above 30 (oversold) while in a valid trend/vol regime
        df['buy_signal'] = trend_mask & atr_mask & (df['cmi'] > 30) & (df['cmi'].shift(1) <= 30)
        
        # Sell when CMI crosses below 70 (overbought)
        df['sell_signal'] = (df['cmi'] < 70) & (df['cmi'].shift(1) >= 70)
        if 'chandelier_long' in df.columns:
            df['sell_signal'] |= df['close'] <= df['chandelier_long']
        
        # Signal strength based on extremity
        raw_strength = np.where(
            df['cmi'] < 50,
            (50 - df['cmi']) / 50,
            (df['cmi'] - 50) / 50,
        )
        df['signal_strength'] = np.where(trend_mask & atr_mask, raw_strength, 0.0)
        
        return df

    @classmethod
    def parameter_space(cls) -> Dict[str, Tuple[float, float]]:
        """Define parameter bounds for optimization"""
        return {
            'rsi_period': (5, 30),  # RSI period
            'roc_period': (5, 20),  # Rate of change period
            'tsi_long': (15, 40),  # TSI long period
            'tsi_short': (5, 20),  # TSI short period
            'signal_threshold': (0.0, 0.95),  # Signal threshold can't be 1.0
            'adx_period': (10, 28),
            'adx_htf_period': (10, 28),
            'adx_htf_minutes': (30, 240),
            'adx_trend_threshold': (20.0, 35.0),
            'atr_period': (10, 30),
            'atr_floor_pct': (0.0, 0.01),
            'chandelier_period': (14, 40),
            'chandelier_atr_mult': (2.0, 4.0),
        }


# Strategy factory function
def create_strategy(strategy_name: str, parameters: Dict = None) -> BaseStrategy:
    """Factory function to create strategy instances"""
    
    strategies = {
        'SqueezeMomentum': SqueezeMomentumStrategy,
        'WaveTrend': WaveTrendStrategy,
        'CoralTrend': CoralTrendStrategy,
        'SchaffTrendCycle': SchaffTrendCycleStrategy,
        'MESAAdaptiveMA': MESAAdaptiveMAStrategy,
        'ElderImpulse': ElderImpulseStrategy,
        'FRAMA': FRAMAStrategy,
        'ZeroLagEMA': ZeroLagEMAStrategy,
        'KaufmannAMA': KaufmannAMAStrategy,
        'TradersDynamicIndex': TradersDynamicIndexStrategy,
        'InsyncIndex': InsyncIndexStrategy,
        'PremierStochastic': PremierStochasticStrategy,
        'MAC_Z': MACZStrategy,
        'FireflyOscillator': FireflyOscillatorStrategy,
        'CompositeMomentumIndex': CompositeMomentumIndexStrategy
    }
    
    if strategy_name not in strategies:
        raise ValueError(f"Strategy {strategy_name} not found. Available: {list(strategies.keys())}")
    
    return strategies[strategy_name](parameters)
