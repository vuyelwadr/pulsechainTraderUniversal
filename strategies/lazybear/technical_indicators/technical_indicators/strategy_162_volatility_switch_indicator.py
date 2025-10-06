"""
LazyBear Strategy #162: Volatility Switch Indicator

Original concept by Ron McEwan from Stocks & Commodities February 2013
TradingView implementation by LazyBear

The Volatility Switch Indicator estimates current volatility relative to historical data,
indicating whether the market is trending or in mean reversion mode.
Range is normalized to 0 - 1.

Trading Logic:
- Below 0.5: Volatility decreasing, potential trend formation
- Above 0.5: High volatility, potentially choppy market conditions
- Can be used with RSI to identify overbought/oversold levels when volatility is high

Mathematical Formula:
1. Calculate daily change ratio: (Close - Previous Close) / ((Close + Previous Close)/2)
2. Calculate standard deviation of these ratios over lookback period
3. Normalize to 0-1 range using historical statistics

URL: https://www.tradingview.com/v/50YzpVDY/
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from strategies.base_strategy import BaseStrategy
import pandas as pd
import numpy as np
import logging
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

class VolatilitySwitchIndicatorStrategy(BaseStrategy):
    """
    LazyBear Volatility Switch Indicator Strategy
    
    Estimates current volatility relative to historical data to determine
    market regime (trending vs mean reversion).
    """
    
    def __init__(self, parameters: Dict = None):
        default_params = {
            'length': 21,                    # Lookback period for volatility calculation
            'short_length': 14,             # Alternative shorter lookback period
            'normalization_period': 100,     # Period for normalizing to 0-1 range
            'volatility_threshold': 0.5,     # Threshold for regime identification
            'rsi_period': 14,               # RSI period for high volatility signals
            'rsi_overbought': 70,           # RSI overbought level
            'rsi_oversold': 30,             # RSI oversold level
            'signal_strength_factor': 0.8,   # Base signal strength
            'trend_confirm_periods': 3,      # Periods to confirm trend formation
            'regime_memory': 5,             # Memory periods for regime identification
        }
        
        if parameters:
            default_params.update(parameters)
            
        super().__init__("LazyBear_VolatilitySwitch", default_params)
        
    def calculate_volatility_ratio(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate the volatility ratio using Ron McEwan's formula:
        (Close - Previous Close) / ((Close + Previous Close)/2)
        """
        close = data['close']
        prev_close = close.shift(1)
        
        # Calculate the ratio
        numerator = close - prev_close
        denominator = (close + prev_close) / 2
        
        # Avoid division by zero
        denominator = denominator.replace(0, np.nan)
        volatility_ratio = numerator / denominator
        
        return volatility_ratio
    
    def calculate_volatility_switch(self, volatility_ratios: pd.Series, length: int, normalization_period: int) -> pd.Series:
        """
        Calculate the Volatility Switch indicator
        
        Args:
            volatility_ratios: Series of volatility ratios
            length: Lookback period for standard deviation
            normalization_period: Period for normalization
            
        Returns:
            Series of normalized volatility switch values (0-1)
        """
        # Calculate rolling standard deviation of volatility ratios
        volatility_std = volatility_ratios.rolling(window=length, min_periods=length//2).std()
        
        # Calculate rolling statistics for normalization
        rolling_min = volatility_std.rolling(window=normalization_period, min_periods=normalization_period//2).min()
        rolling_max = volatility_std.rolling(window=normalization_period, min_periods=normalization_period//2).max()
        
        # Normalize to 0-1 range
        range_diff = rolling_max - rolling_min
        range_diff = range_diff.replace(0, 1)  # Avoid division by zero
        
        volatility_switch = (volatility_std - rolling_min) / range_diff
        
        # Ensure values are between 0 and 1
        volatility_switch = np.clip(volatility_switch, 0, 1)
        
        return volatility_switch
    
    def calculate_rsi(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate RSI for high volatility periods"""
        close = data['close']
        delta = close.diff()
        
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def identify_regime(self, volatility_switch: pd.Series, threshold: float, memory: int) -> pd.Series:
        """
        Identify market regime with memory to avoid whipsaws
        
        Returns:
            Series with regime: 'trending' or 'choppy'
        """
        # Basic regime identification
        regime = pd.Series(index=volatility_switch.index, dtype='object')
        regime[volatility_switch < threshold] = 'trending'
        regime[volatility_switch >= threshold] = 'choppy'
        
        # Apply memory to reduce regime switching noise
        if memory > 1:
            for i in range(memory, len(regime)):
                recent_regimes = regime.iloc[i-memory:i].values
                if len(set(recent_regimes)) == 1:
                    # All recent regimes are the same, keep current
                    continue
                else:
                    # Mixed regimes, use majority vote
                    trending_count = np.sum(recent_regimes == 'trending')
                    if trending_count > memory / 2:
                        regime.iloc[i] = 'trending'
                    else:
                        regime.iloc[i] = 'choppy'
        
        return regime
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators for the strategy"""
        if not self.validate_data(data):
            return data
        
        df = data.copy()
        
        # Ensure we have a 'close' column
        if 'close' not in df.columns and 'price' in df.columns:
            df['close'] = df['price']
        
        try:
            # Calculate volatility ratios
            volatility_ratios = self.calculate_volatility_ratio(df)
            df['volatility_ratio'] = volatility_ratios
            
            # Calculate primary volatility switch (21 period)
            vol_switch_21 = self.calculate_volatility_switch(
                volatility_ratios, 
                self.parameters['length'], 
                self.parameters['normalization_period']
            )
            df['volatility_switch_21'] = vol_switch_21
            
            # Calculate alternative volatility switch (14 period)
            vol_switch_14 = self.calculate_volatility_switch(
                volatility_ratios,
                self.parameters['short_length'],
                self.parameters['normalization_period']
            )
            df['volatility_switch_14'] = vol_switch_14
            
            # Calculate RSI for high volatility periods
            df['rsi'] = self.calculate_rsi(df, self.parameters['rsi_period'])
            
            # Identify market regime
            df['regime'] = self.identify_regime(
                vol_switch_21,
                self.parameters['volatility_threshold'],
                self.parameters['regime_memory']
            )
            
            # Calculate regime strength (how far from threshold)
            df['regime_strength'] = np.abs(vol_switch_21 - self.parameters['volatility_threshold'])
            
            # Calculate trend confirmation (periods in trending regime)
            trend_periods = []
            trend_count = 0
            for i in range(len(df)):
                if df['regime'].iloc[i] == 'trending':
                    trend_count += 1
                else:
                    trend_count = 0
                trend_periods.append(trend_count)
            df['trend_periods'] = trend_periods
            
            # Calculate volatility momentum (rate of change)
            df['vol_switch_momentum'] = vol_switch_21.diff()
            
            logger.debug(f"Calculated Volatility Switch indicators for {len(df)} data points")
            
        except Exception as e:
            logger.error(f"Error calculating Volatility Switch indicators: {e}")
            
        return df
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on volatility switch analysis"""
        df = data.copy()
        
        # Initialize signal columns
        df['buy_signal'] = False
        df['sell_signal'] = False
        df['signal_strength'] = 0.0
        df['signal_reason'] = 'hold'
        
        if len(df) < max(self.parameters['length'], self.parameters['short_length']):
            return df
        
        try:
            vol_switch = df['volatility_switch_21']
            vol_switch_14 = df['volatility_switch_14']
            regime = df['regime']
            rsi = df['rsi']
            trend_periods = df['trend_periods']
            regime_strength = df['regime_strength']
            vol_momentum = df['vol_switch_momentum']
            
            threshold = self.parameters['volatility_threshold']
            rsi_ob = self.parameters['rsi_overbought']
            rsi_os = self.parameters['rsi_oversold']
            trend_confirm = self.parameters['trend_confirm_periods']
            base_strength = self.parameters['signal_strength_factor']
            
            for i in range(len(df)):
                if pd.isna(vol_switch.iloc[i]) or pd.isna(rsi.iloc[i]):
                    continue
                
                current_vol = vol_switch.iloc[i]
                current_vol_14 = vol_switch_14.iloc[i]
                current_regime = regime.iloc[i]
                current_rsi = rsi.iloc[i]
                current_trend_periods = trend_periods.iloc[i]
                current_strength = regime_strength.iloc[i]
                current_momentum = vol_momentum.iloc[i] if not pd.isna(vol_momentum.iloc[i]) else 0
                
                signal_strength = 0.0
                reason = 'hold'
                
                # Trending Regime Signals (volatility < 0.5)
                if current_regime == 'trending':
                    # Strong trend formation signal
                    if (current_trend_periods >= trend_confirm and 
                        current_vol < threshold - 0.1 and  # Well below threshold
                        current_momentum < -0.01):  # Volatility decreasing
                        
                        # Buy on trend confirmation with decreasing volatility
                        if current_rsi < 50:  # Not overbought
                            df.loc[df.index[i], 'buy_signal'] = True
                            signal_strength = base_strength + current_strength
                            reason = 'trend_formation_buy'
                        
                        # Sell if very overbought even in trending regime
                        elif current_rsi > rsi_ob + 10:
                            df.loc[df.index[i], 'sell_signal'] = True
                            signal_strength = base_strength * 0.6
                            reason = 'trend_overbought_sell'
                    
                    # Early trend detection
                    elif (current_vol < threshold and 
                          current_momentum < 0 and
                          current_vol_14 < current_vol):  # Short period confirming
                        
                        if current_rsi < 60:
                            df.loc[df.index[i], 'buy_signal'] = True
                            signal_strength = base_strength * 0.7
                            reason = 'early_trend_buy'
                
                # Choppy Regime Signals (volatility >= 0.5)
                elif current_regime == 'choppy':
                    # High volatility mean reversion using RSI
                    if current_vol > threshold + 0.1:  # Well above threshold
                        
                        # Buy oversold in high volatility
                        if current_rsi < rsi_os:
                            df.loc[df.index[i], 'buy_signal'] = True
                            signal_strength = base_strength * (1 + current_strength)
                            reason = 'choppy_oversold_buy'
                        
                        # Sell overbought in high volatility
                        elif current_rsi > rsi_ob:
                            df.loc[df.index[i], 'sell_signal'] = True
                            signal_strength = base_strength * (1 + current_strength)
                            reason = 'choppy_overbought_sell'
                
                # Regime transition signals
                if i > 0:
                    prev_regime = regime.iloc[i-1] if i > 0 else None
                    
                    # Transition from choppy to trending
                    if (prev_regime == 'choppy' and current_regime == 'trending' and
                        current_vol < threshold - 0.05):
                        
                        if current_rsi < 65:
                            df.loc[df.index[i], 'buy_signal'] = True
                            signal_strength = base_strength * 0.9
                            reason = 'regime_transition_buy'
                    
                    # Transition from trending to choppy (take profits)
                    elif (prev_regime == 'trending' and current_regime == 'choppy' and
                          current_vol > threshold + 0.05):
                        
                        df.loc[df.index[i], 'sell_signal'] = True
                        signal_strength = base_strength * 0.8
                        reason = 'regime_transition_sell'
                
                # Apply signal strength
                df.loc[df.index[i], 'signal_strength'] = min(signal_strength, 1.0)
                df.loc[df.index[i], 'signal_reason'] = reason
            
            # Log signal summary
            buy_signals = df['buy_signal'].sum()
            sell_signals = df['sell_signal'].sum()
            avg_strength = df[df['signal_strength'] > 0]['signal_strength'].mean() if df['signal_strength'].sum() > 0 else 0
            
            logger.info(f"Generated {buy_signals} buy signals and {sell_signals} sell signals")
            logger.info(f"Average signal strength: {avg_strength:.3f}")
            
        except Exception as e:
            logger.error(f"Error generating Volatility Switch signals: {e}")
        
        return df
    
    def get_strategy_info(self) -> Dict:
        """Get detailed information about the strategy"""
        return {
            'name': self.name,
            'description': self.__doc__,
            'parameters': self.parameters,
            'required_columns': ['timestamp', 'close', 'price'],
            'indicators': [
                'volatility_ratio',
                'volatility_switch_21',
                'volatility_switch_14', 
                'rsi',
                'regime',
                'regime_strength',
                'trend_periods',
                'vol_switch_momentum'
            ],
            'trading_logic': {
                'trending_regime': 'volatility_switch < 0.5 - trend formation signals',
                'choppy_regime': 'volatility_switch >= 0.5 - mean reversion with RSI',
                'regime_transitions': 'signals on regime changes',
                'signal_types': [
                    'trend_formation_buy',
                    'trend_overbought_sell',
                    'early_trend_buy',
                    'choppy_oversold_buy',
                    'choppy_overbought_sell',
                    'regime_transition_buy',
                    'regime_transition_sell'
                ]
            },
            'author': 'LazyBear (TradingView)',
            'original_author': 'Ron McEwan',
            'source': 'Stocks & Commodities February 2013',
            'url': 'https://www.tradingview.com/v/50YzpVDY/'
        }