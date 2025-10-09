"""
Agent 06 Strategies - LazyBear Collection (Strategies 56-66)
Optimized implementation of specific TradingView strategies for testing
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
import talib
from strategies.base_strategy import BaseStrategy
from .indicator_utils import compute_adx, compute_multi_timeframe_adx, compute_atr
import logging

logger = logging.getLogger(__name__)


class Strategy_56_RSquared(BaseStrategy):
    """R-Squared - Statistical measure of trend strength"""
    
    def __init__(self, parameters: Dict = None):
        default_params = {
            'period': 20,
            'threshold': 0.8,
            'signal_threshold': 0.6
        }
        if parameters:
            default_params.update(parameters)
        super().__init__("R_Squared_56", default_params)
        self.strategy_type = 'volatility'
        self.strategy_id = 56
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        
        # Calculate R-Squared for trend strength
        period = self.parameters['period']
        df['rsquared'] = 0.0
        
        for i in range(period, len(df)):
            # Get price data for period
            prices = df['close'].iloc[i-period:i].values
            x = np.arange(len(prices))
            
            # Calculate R-squared
            if len(prices) > 1:
                correlation = np.corrcoef(x, prices)[0, 1]
                df.loc[df.index[i], 'rsquared'] = correlation ** 2 if not np.isnan(correlation) else 0
        
        # Smooth R-squared
        df['rsquared_smooth'] = talib.EMA(df['rsquared'], timeperiod=5)
        
        return df
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        
        threshold = self.parameters['threshold']
        
        # Buy when R-squared shows strong trend and price is rising
        df['price_rising'] = df['close'] > df['close'].shift(1)
        df['buy_signal'] = (df['rsquared_smooth'] > threshold) & df['price_rising']
        
        # Sell when R-squared falls (trend weakening)
        df['sell_signal'] = (df['rsquared_smooth'] < threshold) & \
                           (df['rsquared_smooth'].shift(1) >= threshold)
        
        # Signal strength based on R-squared value
        df['signal_strength'] = df['rsquared_smooth'].clip(0, 1)
        
        return df


class Strategy_57_GuppyMMA(BaseStrategy):
    """Guppy Multiple Moving Average - Trend analysis with multiple EMAs"""
    
    def __init__(self, parameters: Dict = None):
        default_params = {
            'fast_periods': [3, 5, 8, 10, 12, 15],
            'slow_periods': [30, 35, 40, 45, 50, 60],
            'signal_threshold': 0.6
        }
        if parameters:
            default_params.update(parameters)
        super().__init__("Guppy_MMA_57", default_params)
        self.strategy_type = 'hybrid'
        self.strategy_id = 57
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        
        # Calculate fast EMAs
        fast_emas = []
        for period in self.parameters['fast_periods']:
            col_name = f'fast_ema_{period}'
            df[col_name] = talib.EMA(df['close'], timeperiod=period)
            fast_emas.append(col_name)
        
        # Calculate slow EMAs
        slow_emas = []
        for period in self.parameters['slow_periods']:
            col_name = f'slow_ema_{period}'
            df[col_name] = talib.EMA(df['close'], timeperiod=period)
            slow_emas.append(col_name)
        
        # Calculate ribbon averages
        df['fast_avg'] = df[fast_emas].mean(axis=1)
        df['slow_avg'] = df[slow_emas].mean(axis=1)
        
        # Calculate ribbon separation
        df['ribbon_sep'] = (df['fast_avg'] - df['slow_avg']) / df['slow_avg']
        
        return df
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        
        # Buy when fast ribbon crosses above slow ribbon
        df['buy_signal'] = (df['fast_avg'] > df['slow_avg']) & \
                          (df['fast_avg'].shift(1) <= df['slow_avg'].shift(1))
        
        # Sell when fast ribbon crosses below slow ribbon
        df['sell_signal'] = (df['fast_avg'] < df['slow_avg']) & \
                           (df['fast_avg'].shift(1) >= df['slow_avg'].shift(1))
        
        # Signal strength based on ribbon separation
        df['signal_strength'] = np.abs(df['ribbon_sep']).clip(0, 0.1) * 10
        
        return df


class Strategy_58_GuppyOscillator(BaseStrategy):
    """Guppy Oscillator - Oscillator based on Guppy MMA"""
    
    def __init__(self, parameters: Dict = None):
        default_params = {
            'fast_periods': [3, 5, 8, 10, 12, 15],
            'slow_periods': [30, 35, 40, 45, 50, 60],
            'smooth_period': 5,
            'signal_threshold': 0.6
        }
        if parameters:
            default_params.update(parameters)
        super().__init__("Guppy_Oscillator_58", default_params)
        self.strategy_type = 'momentum'
        self.strategy_id = 58
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        
        # Calculate EMAs like Guppy MMA
        fast_emas = []
        for period in self.parameters['fast_periods']:
            col_name = f'fast_ema_{period}'
            df[col_name] = talib.EMA(df['close'], timeperiod=period)
            fast_emas.append(col_name)
        
        slow_emas = []
        for period in self.parameters['slow_periods']:
            col_name = f'slow_ema_{period}'
            df[col_name] = talib.EMA(df['close'], timeperiod=period)
            slow_emas.append(col_name)
        
        # Calculate ribbon averages
        df['fast_avg'] = df[fast_emas].mean(axis=1)
        df['slow_avg'] = df[slow_emas].mean(axis=1)
        
        # Create oscillator
        df['guppy_osc'] = ((df['fast_avg'] - df['slow_avg']) / df['slow_avg']) * 100
        
        # Smooth oscillator
        df['guppy_osc_smooth'] = talib.EMA(df['guppy_osc'], timeperiod=self.parameters['smooth_period'])
        
        return df
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        
        # Buy when oscillator crosses above zero
        df['buy_signal'] = (df['guppy_osc_smooth'] > 0) & \
                          (df['guppy_osc_smooth'].shift(1) <= 0)
        
        # Sell when oscillator crosses below zero
        df['sell_signal'] = (df['guppy_osc_smooth'] < 0) & \
                           (df['guppy_osc_smooth'].shift(1) >= 0)
        
        # Signal strength based on oscillator magnitude
        df['signal_strength'] = np.abs(df['guppy_osc_smooth']).clip(0, 10) / 10
        
        return df


class Strategy_59_LindaRaschkeOscillator(BaseStrategy):
    """Linda Raschke Oscillator - Advanced momentum oscillator"""
    
    def __init__(self, parameters: Dict = None):
        default_params = {
            'fast_period': 7,
            'slow_period': 21,
            'smooth_period': 5,
            'overbought': 2.0,
            'oversold': -2.0,
            'signal_threshold': 0.6
        }
        if parameters:
            default_params.update(parameters)
        super().__init__("Linda_Raschke_Osc_59", default_params)
        self.strategy_type = 'trend'
        self.strategy_id = 59
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        
        # Calculate price change momentum
        df['price_change'] = df['close'] - df['close'].shift(1)
        
        # Fast and slow EMAs of price changes
        df['fast_momentum'] = talib.EMA(df['price_change'], timeperiod=self.parameters['fast_period'])
        df['slow_momentum'] = talib.EMA(df['price_change'], timeperiod=self.parameters['slow_period'])
        
        # Linda Raschke Oscillator
        df['lr_osc'] = df['fast_momentum'] - df['slow_momentum']
        
        # Normalize by price
        df['lr_osc_norm'] = (df['lr_osc'] / df['close']) * 1000
        
        # Smooth the oscillator
        df['lr_osc_smooth'] = talib.EMA(df['lr_osc_norm'], timeperiod=self.parameters['smooth_period'])
        
        return df
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        
        overbought = self.parameters['overbought']
        oversold = self.parameters['oversold']
        
        # Buy when crossing above oversold level
        df['buy_signal'] = (df['lr_osc_smooth'] > oversold) & \
                          (df['lr_osc_smooth'].shift(1) <= oversold)
        
        # Sell when crossing below overbought level
        df['sell_signal'] = (df['lr_osc_smooth'] < overbought) & \
                           (df['lr_osc_smooth'].shift(1) >= overbought)
        
        # Signal strength based on extremity
        df['signal_strength'] = np.abs(df['lr_osc_smooth']).clip(0, 5) / 5
        
        return df


class Strategy_60_IanOscillator(BaseStrategy):
    """Ian Oscillator - Custom momentum oscillator"""
    
    def __init__(self, parameters: Dict = None):
        default_params = {
            'period': 14,
            'smooth_period': 3,
            'threshold': 0.5,
            'signal_threshold': 0.6
        }
        if parameters:
            default_params.update(parameters)
        super().__init__("Ian_Oscillator_60", default_params)
        self.strategy_type = 'oscillator'
        self.strategy_id = 60
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        
        period = self.parameters['period']
        
        # Calculate True Range
        df['tr'] = talib.TRANGE(df['high'], df['low'], df['close'])
        
        # Calculate directional movement
        df['dm_pos'] = np.where((df['high'] - df['high'].shift(1)) > 
                               (df['low'].shift(1) - df['low']),
                               np.maximum(df['high'] - df['high'].shift(1), 0), 0)
        df['dm_neg'] = np.where((df['low'].shift(1) - df['low']) > 
                               (df['high'] - df['high'].shift(1)),
                               np.maximum(df['low'].shift(1) - df['low'], 0), 0)
        
        # Smooth with EMA
        df['tr_smooth'] = talib.EMA(df['tr'], timeperiod=period)
        df['dm_pos_smooth'] = talib.EMA(df['dm_pos'], timeperiod=period)
        df['dm_neg_smooth'] = talib.EMA(df['dm_neg'], timeperiod=period)
        
        # Ian Oscillator calculation
        df['ian_osc'] = (df['dm_pos_smooth'] - df['dm_neg_smooth']) / \
                       (df['tr_smooth'] + 0.0001)
        
        # Smooth the oscillator
        df['ian_osc_smooth'] = talib.EMA(df['ian_osc'], timeperiod=self.parameters['smooth_period'])
        
        return df
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        
        threshold = self.parameters['threshold']
        
        # Buy when oscillator rises above threshold
        df['buy_signal'] = (df['ian_osc_smooth'] > threshold) & \
                          (df['ian_osc_smooth'].shift(1) <= threshold)
        
        # Sell when oscillator falls below -threshold
        df['sell_signal'] = (df['ian_osc_smooth'] < -threshold) & \
                           (df['ian_osc_smooth'].shift(1) >= -threshold)
        
        # Signal strength based on oscillator value
        df['signal_strength'] = np.abs(df['ian_osc_smooth']).clip(0, 1)
        
        return df


class Strategy_61_ConstanceBrownComposite(BaseStrategy):
    """Constance Brown Composite Index - Multiple indicator composite"""
    
    def __init__(self, parameters: Dict = None):
        default_params = {
            'rsi_period': 14,
            'stoch_period': 14,
            'momentum_period': 10,
            'signal_threshold': 0.6
        }
        if parameters:
            default_params.update(parameters)
        super().__init__("CB_Composite_61", default_params)
        self.strategy_type = 'volume'
        self.strategy_id = 61
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        
        # Calculate individual indicators
        df['rsi'] = talib.RSI(df['close'], timeperiod=self.parameters['rsi_period'])
        df['stoch'], _ = talib.STOCH(df['high'], df['low'], df['close'],
                                     fastk_period=self.parameters['stoch_period'],
                                     slowk_period=3, slowd_period=3)
        df['momentum'] = talib.MOM(df['close'], timeperiod=self.parameters['momentum_period'])
        
        # Normalize momentum
        df['momentum_norm'] = ((df['momentum'] / df['close']) * 1000).clip(-100, 100) + 50
        
        # Calculate composite (average of normalized indicators)
        df['cb_composite'] = (df['rsi'] + df['stoch'] + df['momentum_norm']) / 3
        
        # Smooth composite
        df['cb_smooth'] = talib.EMA(df['cb_composite'], timeperiod=5)
        
        return df
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        
        # Buy when composite moves from oversold to neutral
        df['buy_signal'] = (df['cb_smooth'] > 30) & (df['cb_smooth'].shift(1) <= 30)
        
        # Sell when composite moves from overbought to neutral
        df['sell_signal'] = (df['cb_smooth'] < 70) & (df['cb_smooth'].shift(1) >= 70)
        
        # Signal strength based on extremity
        df['signal_strength'] = np.abs(df['cb_smooth'] - 50) / 50
        
        return df


class Strategy_62_RSIAvgs(BaseStrategy):
    """RSI+Avgs - RSI with multiple moving averages"""
    
    def __init__(self, parameters: Dict = None):
        default_params = {
            'rsi_period': 14,
            'ma_periods': [5, 10, 20],
            'overbought': 70,
            'oversold': 30,
            'signal_threshold': 0.6,
            'adx_period': 14,
            'adx_htf_period': 14,
            'adx_htf_minutes': 60,
            'adx_range_threshold': 22.0,
            'atr_period': 14,
            'atr_stop_mult': 1.5,
            'atr_floor_pct': 0.0015,
            'time_stop_bars': 12,
        }
        if parameters:
            default_params.update(parameters)
        super().__init__("RSI_Avgs_62", default_params)
        self.strategy_type = 'volatility'
        self.strategy_id = 62
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        
        # Calculate RSI
        df['rsi'] = talib.RSI(df['close'], timeperiod=self.parameters['rsi_period'])
        
        # Calculate multiple moving averages of RSI
        ma_cols = []
        for period in self.parameters['ma_periods']:
            col_name = f'rsi_ma_{period}'
            df[col_name] = talib.SMA(df['rsi'], timeperiod=period)
            ma_cols.append(col_name)
        
        # Calculate average of RSI MAs
        df['rsi_ma_avg'] = df[ma_cols].mean(axis=1)
        
        # RSI trend based on MA slope
        df['rsi_trend'] = df['rsi_ma_avg'] - df['rsi_ma_avg'].shift(3)
        
        high = df['high'] if 'high' in df.columns else df['close']
        low = df['low'] if 'low' in df.columns else df['close']
        close = df['close']

        adx_df = compute_adx(high, low, close, int(self.parameters['adx_period']))
        df['adx'] = adx_df['adx']
        df['adx_htf'] = compute_multi_timeframe_adx(
            df[['timestamp', 'open', 'high', 'low', 'close']].copy(),
            period=int(self.parameters['adx_htf_period']),
            timeframe_minutes=int(self.parameters['adx_htf_minutes']),
        )
        adx_threshold = float(self.parameters['adx_range_threshold'])
        df['is_range'] = (df['adx'] <= adx_threshold) & (df['adx_htf'] <= adx_threshold)

        atr_period = int(self.parameters['atr_period'])
        atr_series = compute_atr(high, low, close, atr_period)
        atr_floor = float(self.parameters['atr_floor_pct'])
        if atr_floor > 0:
            atr_series = atr_series.clip(lower=close * atr_floor)
        df['atr'] = atr_series

        return df
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()

        overbought = float(self.parameters['overbought'])
        oversold = float(self.parameters['oversold'])
        atr_mult = float(self.parameters['atr_stop_mult'])
        time_stop_bars = int(self.parameters.get('time_stop_bars', 0))

        df['buy_signal'] = False
        df['sell_signal'] = False
        df['signal_strength'] = 0.0

        rsi = df['rsi'].to_numpy(float)
        rsi_trend = df['rsi_trend'].to_numpy(float)
        close = df['close'].to_numpy(float)
        atr = df.get('atr', pd.Series(np.nan, index=df.index)).to_numpy(float)
        range_mask = df.get('is_range', pd.Series(True, index=df.index)).to_numpy(bool)

        in_position = False
        entry_price = 0.0
        bars_in_trade = 0

        for i in range(len(df)):
            if not in_position:
                if not range_mask[i]:
                    continue
                if (
                    rsi[i] > oversold
                    and rsi[i - 1] <= oversold if i > 0 else False
                    and rsi_trend[i] > 0
                ):
                    df.iat[i, df.columns.get_loc('buy_signal')] = True
                    df.iat[i, df.columns.get_loc('signal_strength')] = min(
                        1.0, abs(rsi[i] - 50) / 50.0
                    )
                    in_position = True
                    entry_price = close[i]
                    bars_in_trade = 0
            else:
                bars_in_trade += 1
                stop_price = entry_price - atr_mult * atr[i] if not np.isnan(atr[i]) else entry_price * 0.97
                take_profit = entry_price + atr_mult * atr[i] * 1.5 if not np.isnan(atr[i]) else entry_price * 1.03

                sell = False
                strength = 0.5
                if rsi[i] < overbought and (i > 0 and rsi[i - 1] >= overbought) and rsi_trend[i] < 0:
                    sell = True
                    strength = 1.0
                elif close[i] <= stop_price:
                    sell = True
                    strength = 0.6
                elif close[i] >= take_profit:
                    sell = True
                    strength = 0.8
                elif time_stop_bars > 0 and bars_in_trade >= time_stop_bars:
                    sell = True
                    strength = 0.5

                if sell or not range_mask[i]:
                    df.iat[i, df.columns.get_loc('sell_signal')] = True
                    df.iat[i, df.columns.get_loc('signal_strength')] = strength
                    in_position = False
                    entry_price = 0.0
                    bars_in_trade = 0

        return df


class Strategy_63_AccumulativeSwingIndex(BaseStrategy):
    """Accumulative Swing Index - Price and volume momentum"""
    
    def __init__(self, parameters: Dict = None):
        default_params = {
            'limit_move': 0.25,
            'smooth_period': 5,
            'signal_threshold': 0.6
        }
        if parameters:
            default_params.update(parameters)
        super().__init__("Accum_Swing_Index_63", default_params)
        self.strategy_type = 'hybrid'
        self.strategy_id = 63
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        
        # Calculate Swing Index components
        df['true_range'] = talib.TRANGE(df['high'], df['low'], df['close'])
        
        # Calculate swing index
        df['k_factor'] = np.maximum(df['true_range'], self.parameters['limit_move'])
        
        # Price relationships
        df['cy'] = df['close'] - df['close'].shift(1)
        df['oy'] = df['open'] - df['close'].shift(1)
        df['hy'] = df['high'] - df['close'].shift(1)
        df['ly'] = df['low'] - df['close'].shift(1)
        
        # R value calculation
        df['r'] = np.where(np.abs(df['hy']) > np.abs(df['ly']),
                          df['hy'], df['ly'])
        
        # Swing Index calculation
        df['si'] = 50 * (df['cy'] + 0.5 * df['oy'] + 0.25 * df['r']) / df['k_factor']
        
        # Accumulative Swing Index
        df['asi'] = df['si'].cumsum()
        
        # Smooth ASI
        df['asi_smooth'] = talib.EMA(df['asi'], timeperiod=self.parameters['smooth_period'])
        
        return df
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        
        # Buy when ASI turns positive
        df['buy_signal'] = (df['asi_smooth'] > 0) & (df['asi_smooth'].shift(1) <= 0)
        
        # Sell when ASI turns negative
        df['sell_signal'] = (df['asi_smooth'] < 0) & (df['asi_smooth'].shift(1) >= 0)
        
        # Signal strength based on ASI magnitude
        asi_range = df['asi_smooth'].rolling(50).std()
        df['signal_strength'] = np.abs(df['asi_smooth']) / (asi_range + 0.0001)
        df['signal_strength'] = df['signal_strength'].clip(0, 1)
        
        return df


class Strategy_64_ASIOscillator(BaseStrategy):
    """ASI Oscillator - Oscillator version of Accumulative Swing Index"""
    
    def __init__(self, parameters: Dict = None):
        default_params = {
            'limit_move': 0.25,
            'oscillator_period': 20,
            'smooth_period': 5,
            'signal_threshold': 0.6
        }
        if parameters:
            default_params.update(parameters)
        super().__init__("ASI_Oscillator_64", default_params)
        self.strategy_type = 'momentum'
        self.strategy_id = 64
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        
        # Calculate Swing Index (same as Strategy_63)
        df['true_range'] = talib.TRANGE(df['high'], df['low'], df['close'])
        df['k_factor'] = np.maximum(df['true_range'], self.parameters['limit_move'])
        
        df['cy'] = df['close'] - df['close'].shift(1)
        df['oy'] = df['open'] - df['close'].shift(1)
        df['hy'] = df['high'] - df['close'].shift(1)
        df['ly'] = df['low'] - df['close'].shift(1)
        df['r'] = np.where(np.abs(df['hy']) > np.abs(df['ly']), df['hy'], df['ly'])
        
        df['si'] = 50 * (df['cy'] + 0.5 * df['oy'] + 0.25 * df['r']) / df['k_factor']
        
        # Create oscillator by normalizing SI over period
        period = self.parameters['oscillator_period']
        df['si_sma'] = talib.SMA(df['si'], timeperiod=period)
        df['si_std'] = df['si'].rolling(window=period).std()
        
        # ASI Oscillator
        df['asi_osc'] = (df['si'] - df['si_sma']) / (df['si_std'] + 0.0001)
        
        # Smooth oscillator
        df['asi_osc_smooth'] = talib.EMA(df['asi_osc'], timeperiod=self.parameters['smooth_period'])
        
        return df
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        
        # Buy when oscillator crosses above -1 (oversold)
        df['buy_signal'] = (df['asi_osc_smooth'] > -1) & \
                          (df['asi_osc_smooth'].shift(1) <= -1)
        
        # Sell when oscillator crosses below 1 (overbought)
        df['sell_signal'] = (df['asi_osc_smooth'] < 1) & \
                           (df['asi_osc_smooth'].shift(1) >= 1)
        
        # Signal strength based on oscillator extremity
        df['signal_strength'] = np.abs(df['asi_osc_smooth']).clip(0, 2) / 2
        
        return df


class Strategy_65_TomDemarkREI(BaseStrategy):
    """Tom Demark Range Expansion Index - Range-based momentum"""
    
    def __init__(self, parameters: Dict = None):
        default_params = {
            'period': 8,
            'smooth_period': 5,
            'threshold': 45,
            'signal_threshold': 0.6
        }
        if parameters:
            default_params.update(parameters)
        super().__init__("TD_REI_65", default_params)
        self.strategy_type = 'trend'
        self.strategy_id = 65
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        
        period = self.parameters['period']
        
        # Calculate price ranges
        df['high_low'] = df['high'] - df['low']
        df['high_close'] = np.abs(df['high'] - df['close'].shift(1))
        df['low_close'] = np.abs(df['low'] - df['close'].shift(1))
        
        # True Range
        df['true_range'] = np.maximum(df['high_low'], 
                          np.maximum(df['high_close'], df['low_close']))
        
        # Range Expansion conditions
        df['rei_up'] = np.where((df['high'] - df['high'].shift(period)) > 0, 
                               df['high'] - df['high'].shift(period), 0)
        df['rei_down'] = np.where((df['low'].shift(period) - df['low']) > 0,
                                 df['low'].shift(period) - df['low'], 0)
        
        # REI calculation
        rei_up_sum = df['rei_up'].rolling(window=period).sum()
        rei_down_sum = df['rei_down'].rolling(window=period).sum()
        
        df['rei'] = 100 * rei_up_sum / (rei_up_sum + rei_down_sum + 0.0001)
        
        # Smooth REI
        df['rei_smooth'] = talib.EMA(df['rei'], timeperiod=self.parameters['smooth_period'])
        
        return df
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        
        threshold = self.parameters['threshold']
        
        # Buy when REI crosses above threshold
        df['buy_signal'] = (df['rei_smooth'] > threshold) & \
                          (df['rei_smooth'].shift(1) <= threshold)
        
        # Sell when REI crosses below (100 - threshold)
        sell_threshold = 100 - threshold
        df['sell_signal'] = (df['rei_smooth'] < sell_threshold) & \
                           (df['rei_smooth'].shift(1) >= sell_threshold)
        
        # Signal strength based on extremity
        df['signal_strength'] = np.abs(df['rei_smooth'] - 50) / 50
        
        return df


# Factory function for Agent 06 strategies
def create_agent06_strategy(strategy_id: int, parameters: Dict = None) -> BaseStrategy:
    """Factory function to create Agent 06 strategy instances"""
    
    strategies = {
        56: Strategy_56_RSquared,
        57: Strategy_57_GuppyMMA,
        58: Strategy_58_GuppyOscillator,
        59: Strategy_59_LindaRaschkeOscillator,
        60: Strategy_60_IanOscillator,
        61: Strategy_61_ConstanceBrownComposite,
        62: Strategy_62_RSIAvgs,
        63: Strategy_63_AccumulativeSwingIndex,
        64: Strategy_64_ASIOscillator,
        65: Strategy_65_TomDemarkREI
    }
    
    if strategy_id not in strategies:
        raise ValueError(f"Strategy {strategy_id} not found in Agent 06 strategies. Available: {list(strategies.keys())}")
    
    return strategies[strategy_id](parameters)
