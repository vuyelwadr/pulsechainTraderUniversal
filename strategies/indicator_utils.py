"""Common indicator utilities for strategy implementations."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


def _wilder_smoothing(values: np.ndarray, period: int) -> np.ndarray:
    """Perform Wilder's smoothing (used by ATR/ADX)."""
    result = np.full_like(values, np.nan, dtype=float)
    if len(values) == 0:
        return result
    if period <= 0:
        raise ValueError("period must be positive")
    if len(values) < period:
        return result
    initial_sum = np.nansum(values[:period])
    result[period - 1] = initial_sum
    prev = initial_sum
    for i in range(period, len(values)):
        prev = prev - (prev / period) + values[i]
        result[i] = prev
    return result


def compute_adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.DataFrame:
    """
    Compute Average Directional Index (ADX) along with +DI/-DI.

    Returns a DataFrame with columns: ['adx', 'plus_di', 'minus_di'].
    """
    if period <= 0:
        raise ValueError("period must be positive")
    high = high.astype(float)
    low = low.astype(float)
    close = close.astype(float)

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr_components = np.vstack(
        [
            (high - low).to_numpy(dtype=float),
            np.abs(high - close.shift()).to_numpy(dtype=float),
            np.abs(low - close.shift()).to_numpy(dtype=float),
        ]
    )
    true_range = np.nanmax(tr_components, axis=0)

    tr_smoothed = _wilder_smoothing(true_range, period)
    plus_smoothed = _wilder_smoothing(plus_dm, period)
    minus_smoothed = _wilder_smoothing(minus_dm, period)

    plus_di = np.full_like(true_range, np.nan, dtype=float)
    minus_di = np.full_like(true_range, np.nan, dtype=float)

    valid = tr_smoothed != 0
    plus_di[valid] = 100.0 * (plus_smoothed[valid] / tr_smoothed[valid])
    minus_di[valid] = 100.0 * (minus_smoothed[valid] / tr_smoothed[valid])

    dx = np.full_like(true_range, np.nan, dtype=float)
    di_sum = plus_di + minus_di
    valid_dx = di_sum != 0
    dx[valid_dx] = 100.0 * np.abs(plus_di[valid_dx] - minus_di[valid_dx]) / di_sum[valid_dx]

    adx = _wilder_smoothing(dx, period)
    if period < len(close):
        adx[: period - 1] = np.nan
        plus_di[: period - 1] = np.nan
        minus_di[: period - 1] = np.nan

    return pd.DataFrame(
        {
            'adx': adx,
            'plus_di': plus_di,
            'minus_di': minus_di,
        },
        index=close.index,
    )


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range using Wilder smoothing."""
    tr_components = np.vstack(
        [
            (high - low).to_numpy(dtype=float),
            np.abs(high - close.shift()).to_numpy(dtype=float),
            np.abs(low - close.shift()).to_numpy(dtype=float),
        ]
    )
    true_range = np.nanmax(tr_components, axis=0)
    atr = _wilder_smoothing(true_range, period)
    if period < len(close):
        atr[: period - 1] = np.nan
    return pd.Series(atr, index=close.index)


def compute_multi_timeframe_adx(
    df: pd.DataFrame,
    period: int = 14,
    timeframe_minutes: int = 60,
) -> pd.Series:
    """
    Compute ADX on a higher timeframe and align it with the base dataframe.

    Args:
        df: DataFrame with timestamp, high, low, close columns.
        period: ADX period.
        timeframe_minutes: Higher timeframe in minutes (e.g., 60 for 1H).
    """
    if 'timestamp' not in df.columns:
        raise ValueError("DataFrame must contain 'timestamp' column for multi-timeframe ADX.")

    ohlc = df[['timestamp', 'open', 'high', 'low', 'close']].copy()
    ohlc['timestamp'] = pd.to_datetime(ohlc['timestamp'], utc=True)
    ohlc = ohlc.set_index('timestamp')

    rule = f'{int(timeframe_minutes)}min'
    resampled = ohlc.resample(rule).agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
    if resampled.empty:
        return pd.Series(np.nan, index=df.index)

    adx_htf = compute_adx(resampled['high'], resampled['low'], resampled['close'], period)['adx']
    adx_htf = adx_htf.reindex(ohlc.index, method='ffill')
    return adx_htf.reset_index(drop=True)
