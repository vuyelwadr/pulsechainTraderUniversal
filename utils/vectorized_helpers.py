"""
Vectorized helper functions for technical analysis and trading strategies.

This module provides vectorized implementations of common technical analysis functions
used in trading strategies, particularly for detecting crossovers and managing signals.
"""

import pandas as pd
import numpy as np


def crossover(series1, series2):
    """
    Returns True when series1 crosses above series2.

    Args:
        series1: First pandas Series
        series2: Second pandas Series or scalar value

    Returns:
        Boolean Series indicating crossover points
    """
    if isinstance(series2, (int, float)):
        series2 = pd.Series([series2] * len(series1), index=series1.index)
    return (series1 > series2) & (series1.shift(1) <= series2.shift(1))


def crossunder(series1, series2):
    """
    Returns True when series1 crosses below series2.

    Args:
        series1: First pandas Series
        series2: Second pandas Series or scalar value

    Returns:
        Boolean Series indicating crossunder points
    """
    if isinstance(series2, (int, float)):
        series2 = pd.Series([series2] * len(series1), index=series1.index)
    return (series1 < series2) & (series1.shift(1) >= series2.shift(1))


def pine_ema(series, length):
    """
    Pine Script EMA (Exponential Moving Average).

    Args:
        series: Input pandas Series
        length: Period length

    Returns:
        Series with EMA values
    """
    return series.ewm(span=length, adjust=False).mean()


def pine_rma(series, length):
    """
    Pine Script RMA (Running Moving Average).
    Used in Pine Script for RSI and other indicators.

    Args:
        series: Input pandas Series
        length: Period length

    Returns:
        Series with RMA values
    """
    return series.ewm(alpha=1/max(1,int(length)), adjust=False, min_periods=1).mean()


def highest(series, period):
    """
    Rolling maximum over a specified period.

    Args:
        series: Input pandas Series
        period: Rolling window period

    Returns:
        Series with rolling maximum values
    """
    return series.rolling(window=period, min_periods=1).max()


def lowest(series, period):
    """
    Rolling minimum over a specified period.

    Args:
        series: Input pandas Series
        period: Rolling window period

    Returns:
        Series with rolling minimum values
    """
    return series.rolling(window=period, min_periods=1).min()


def barssince(condition):
    """
    Count bars since a condition was last True.

    Args:
        condition: Boolean Series

    Returns:
        Series with bar counts since last True condition
    """
    # Find indices where condition is True
    true_indices = condition[condition].index

    if len(true_indices) == 0:
        # No True conditions found, return length of series
        return pd.Series([len(condition)] * len(condition), index=condition.index)

    # For each position, find the most recent True condition
    result = []
    true_idx = 0

    for i, idx in enumerate(condition.index):
        # Move to next true index if we've passed it
        while true_idx < len(true_indices) - 1 and true_indices[true_idx + 1] <= idx:
            true_idx += 1

        if true_indices[true_idx] <= idx:
            result.append(idx - true_indices[true_idx])
        else:
            result.append(len(condition))  # No previous true condition

    return pd.Series(result, index=condition.index)


def track_position_state(buy_signals, sell_signals):
    """
    Track position state based on buy/sell signals.

    Args:
        buy_signals: Boolean Series of buy signals
        sell_signals: Boolean Series of sell signals

    Returns:
        Series with position state (1 for long, -1 for short, 0 for flat)
    """
    position = pd.Series(0, index=buy_signals.index)

    current_position = 0
    for i in range(len(position)):
        if buy_signals.iloc[i] and current_position <= 0:
            current_position = 1
        elif sell_signals.iloc[i] and current_position >= 0:
            current_position = -1
        position.iloc[i] = current_position

    return position


def calculate_signal_strength(conditions, weights=None):
    """
    Calculate composite signal strength from multiple conditions.
    
    Args:
        conditions: List of pandas Series (conditions/factors)
        weights: Optional list of weights for each condition
        
    Returns:
        Series with signal strength values (0.0 to 1.0)
    """
    if not conditions:
        return pd.Series([0.0] * len(conditions[0]) if conditions else [], 
                        index=conditions[0].index if conditions else None)
    
    if weights is None:
        weights = [1.0 / len(conditions)] * len(conditions)
    
    if len(weights) != len(conditions):
        raise ValueError("Number of weights must match number of conditions")
    
    strength = pd.Series([0.0] * len(conditions[0]), index=conditions[0].index)
    
    for condition, weight in zip(conditions, weights):
        if condition.dtype == bool:
            condition = condition.astype(float)
        strength += condition * weight
    
    return strength.clip(0, 1)


def apply_position_constraints(buy_signals, sell_signals, allow_short=False):
    """
    Apply position constraints to prevent invalid signals.
    
    Args:
        buy_signals: Boolean Series of buy signals
        sell_signals: Boolean Series of sell signals
        allow_short: Whether to allow short positions (currently not implemented)
        
    Returns:
        Tuple of (filtered_buy_signals, filtered_sell_signals)
    """
    position = 0
    filtered_buy = buy_signals.copy()
    filtered_sell = sell_signals.copy()
    
    for i in range(1, len(buy_signals)):
        if buy_signals.iloc[i] and position == 0:
            position = 1
        elif sell_signals.iloc[i] and position == 1:
            position = 0
        else:
            filtered_buy.iloc[i] = False
            filtered_sell.iloc[i] = False
    
    return filtered_buy, filtered_sell


# Additional helper functions that might be needed

def sma(series, period):
    """
    Simple Moving Average.

    Args:
        series: Input pandas Series
        period: Period length

    Returns:
        Series with SMA values
    """
    return series.rolling(window=period).mean()


def ema(series, period):
    """
    Exponential Moving Average.

    Args:
        series: Input pandas Series
        period: Period length

    Returns:
        Series with EMA values
    """
    return series.ewm(span=period, adjust=False).mean()


def rsi(series, period=14):
    """
    Relative Strength Index.

    Args:
        series: Price series
        period: RSI period

    Returns:
        Series with RSI values
    """
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def macd(series, fast=12, slow=26, signal=9):
    """
    MACD (Moving Average Convergence Divergence).

    Args:
        series: Price series
        fast: Fast EMA period
        slow: Slow EMA period
        signal: Signal line EMA period

    Returns:
        Tuple of (macd_line, signal_line, histogram)
    """
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


def bollinger_bands(series, period=20, std_dev=2):
    """
    Bollinger Bands.

    Args:
        series: Price series
        period: Moving average period
        std_dev: Standard deviation multiplier

    Returns:
        Tuple of (upper_band, middle_band, lower_band)
    """
    middle_band = sma(series, period)
    std = series.rolling(window=period).std()
    upper_band = middle_band + (std * std_dev)
    lower_band = middle_band - (std * std_dev)

    return upper_band, middle_band, lower_band