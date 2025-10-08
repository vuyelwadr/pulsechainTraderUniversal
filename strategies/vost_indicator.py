"""Utility helpers for Volatility Optimized SuperTrend (VOST).

All computations are derived strictly from real OHLCV price data provided to
the strategies. No synthetic prices are generated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class VOSTResult:
    """Container for the computed VOST components."""

    line: pd.Series  # trailing stop line that flips with the trend
    trend: pd.Series  # +1 for bullish, -1 for bearish regime
    upper: pd.Series  # adaptive upper band
    lower: pd.Series  # adaptive lower band
    multiplier: pd.Series  # adaptive multiplier applied to ATR
    vol_ratio: pd.Series  # realised volatility vs long-run baseline


def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    """Smoothed ATR using exponential weighting for responsiveness."""

    high = df['high']
    low = df['low']
    close = df['close']
    prev_close = close.shift(1)

    tr_components = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    )
    true_range = tr_components.max(axis=1)
    # Exponential smoothing keeps ATR responsive without introducing lookahead.
    return true_range.ewm(span=max(2, int(period)), adjust=False).mean()


def compute_vost(
    df: pd.DataFrame,
    *,
    atr_period: int = 14,
    base_multiplier: float = 2.3,
    vol_period: int = 48,
    vol_smooth: int = 240,
    vol_ratio_floor: float = 0.6,
    vol_ratio_cap: float = 2.8,
    multiplier_power: float = 1.0,
) -> Optional[VOSTResult]:
    """Compute the Volatility Optimized SuperTrend (VOST).

    Args:
        df: DataFrame with at least ['high', 'low', 'close'] columns.
        atr_period: ATR smoothing window (bars).
        base_multiplier: Baseline multiple applied to ATR.
        vol_period: Lookback window (bars) for realised volatility.
        vol_smooth: EWMA span establishing the long-run volatility baseline.
        vol_ratio_floor: Minimum compression ratio applied to the multiplier.
        vol_ratio_cap: Maximum expansion ratio applied to the multiplier.
        multiplier_power: Exponent that controls how aggressively volatility
            deviations scale the ATR multiplier.

    Returns:
        VOSTResult with adaptive bands, multiplier and trend state. Returns
        ``None`` when the input frame lacks sufficient rows.
    """

    required_cols = {'high', 'low', 'close'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"VOST requires columns {sorted(required_cols)}")

    if len(df) < max(atr_period, vol_period) + 5:
        # Not enough data for stable estimates.
        return None

    price = df['close']
    hl2 = (df['high'] + df['low']) / 2.0

    atr = _atr(df, atr_period).bfill()

    realised_vol = price.pct_change().abs().rolling(max(2, int(vol_period))).mean()
    realised_vol = realised_vol.bfill()

    long_run_vol = realised_vol.ewm(span=max(2, int(vol_smooth)), adjust=False).mean()
    vol_ratio = (realised_vol / long_run_vol).replace([np.inf, -np.inf], np.nan)
    vol_ratio = vol_ratio.fillna(1.0)
    vol_ratio = vol_ratio.clip(lower=vol_ratio_floor, upper=vol_ratio_cap)

    # Adaptive multiplier widens during volatile phases and tightens in calm regimes.
    multiplier = base_multiplier * np.power(vol_ratio, multiplier_power)

    upper = hl2 + multiplier * atr
    lower = hl2 - multiplier * atr

    upper_np = upper.to_numpy(dtype=float, copy=True)
    lower_np = lower.to_numpy(dtype=float, copy=True)
    price_np = price.to_numpy(dtype=float)

    n = len(df)
    line = np.full(n, np.nan, dtype=float)
    trend = np.zeros(n, dtype=float)

    # Seed with bullish assumption once ATR stabilises.
    start_idx = int(np.nanargmin(np.isnan(atr.to_numpy()))) if atr.notna().any() else 0
    start_idx = max(start_idx, atr_period)
    if start_idx >= n:
        return None

    trend[start_idx] = 1.0
    line[start_idx] = lower_np[start_idx]

    for i in range(start_idx + 1, n):
        prev_trend = trend[i - 1] or 1.0
        prev_upper = upper_np[i - 1]
        prev_lower = lower_np[i - 1]

        curr_upper = upper_np[i]
        curr_lower = lower_np[i]
        close_val = price_np[i]

        if close_val > prev_upper:
            curr_trend = 1.0
        elif close_val < prev_lower:
            curr_trend = -1.0
        else:
            curr_trend = prev_trend
            if curr_trend > 0 and curr_lower < prev_lower:
                curr_lower = prev_lower
            if curr_trend < 0 and curr_upper > prev_upper:
                curr_upper = prev_upper

        upper_np[i] = curr_upper
        lower_np[i] = curr_lower
        trend[i] = curr_trend
        line[i] = curr_lower if curr_trend > 0 else curr_upper

    # Forward-fill initial NaNs so callers can align with original frame.
    line_series = pd.Series(line, index=df.index).ffill()
    trend_series = pd.Series(trend, index=df.index)
    trend_series = trend_series.replace(0.0, np.nan).ffill().fillna(1.0)

    return VOSTResult(
        line=line_series,
        trend=trend_series,
        upper=pd.Series(upper_np, index=df.index),
        lower=pd.Series(lower_np, index=df.index),
        multiplier=pd.Series(multiplier, index=df.index),
        vol_ratio=pd.Series(vol_ratio, index=df.index),
    )
