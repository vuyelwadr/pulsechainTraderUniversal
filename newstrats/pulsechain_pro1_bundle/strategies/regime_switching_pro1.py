
"""
strategies/regime_switching_pro1.py

A robust, cost-aware regime-switching strategy for PulseChain HEX/DAI.

Core ideas:
- Detect regime using (a) rolling trend strength via R^2 of log-price regression
  and (b) normalized ATR (volatility).
- In Trend regime: trade a triple-EMA + Donchian breakout trend-following model.
- In Sideways regime: fall back to cautious mean-reversion around EMA.
- Every trade is *gated* by estimated swap costs from swap_cost_cache.json.

Outputs comply with the repository's strategy interface:
    - `calculate_indicators(df)` adds feature columns
    - `generate_signals(df)` adds `buy_signal`, `sell_signal`, `signal_strength`

Signal strength is 0..1 and can be used by the engine for position sizing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from strategies.base_strategy import BaseStrategy  # type: ignore
from utils.cost_model_pro1 import cost_gate

# ---------- small helpers (no external deps) ----------

def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()

def _sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()

def _atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return _sma(tr, window)

def _donchian(df: pd.DataFrame, window: int = 20):
    upper = df["high"].rolling(window=window, min_periods=window).max()
    lower = df["low"].rolling(window=window, min_periods=window).min()
    mid = (upper + lower) / 2.0
    return upper, lower, mid

def _rolling_r2_logprice(close: pd.Series, window: int = 50) -> pd.Series:
    """
    Approximate trend strength by R^2 of a linear fit on log-price.
    """
    y = np.log(close.clip(lower=1e-12))
    x = np.arange(len(y), dtype=float)
    r2 = pd.Series(index=close.index, dtype=float)
    # Use rolling windows
    for i in range(window-1, len(y)):
        ys = y.iloc[i-window+1:i+1].values
        xs = x[i-window+1:i+1]
        # detrend
        xs = (xs - xs.mean())
        ys = (ys - ys.mean())
        denom = (xs**2).sum()
        if denom <= 0:
            r2.iloc[i] = np.nan
            continue
        beta = (xs*ys).sum() / denom
        y_hat = beta * xs
        ss_tot = (ys**2).sum()
        ss_res = ((ys - y_hat)**2).sum()
        r2.iloc[i] = 1.0 - (ss_res / ss_tot if ss_tot > 0 else np.nan)
    return r2

def _zscore(series: pd.Series, window: int = 40) -> pd.Series:
    mean = _sma(series, window)
    std = series.rolling(window=window, min_periods=window).std(ddof=0)
    return (series - mean) / (std.replace(0, np.nan))

# ---------- Strategy ----------

class RegimeSwitchingPro1(BaseStrategy):
    """
    Parameters (good defaults for 5m data; adapt as needed):
      ema_fast=21, ema_slow=55, ema_trend=200
      donchian_n=20, atr_n=14, r2_n=50
      r2_trend_threshold=0.30, atr_norm_threshold=0.9
      mr_band_z=1.25 (entry), mr_exit_z=0.2
      min_edge_multiple=1.15 (relative to round-trip costs)
    """
    def __init__(
        self,
        ema_fast: int = 21,
        ema_slow: int = 55,
        ema_trend: int = 200,
        donchian_n: int = 20,
        atr_n: int = 14,
        r2_n: int = 50,
        r2_trend_threshold: float = 0.30,
        atr_norm_threshold: float = 0.90,
        mr_band_z: float = 1.25,
        mr_exit_z: float = 0.20,
        min_edge_multiple: float = 1.15,
        route_key: str = "HEX/DAI",
        trade_amount_quote: float = 500.0,  # used for cost gating only
    ):
        super().__init__()
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.ema_trend = ema_trend
        self.donchian_n = donchian_n
        self.atr_n = atr_n
        self.r2_n = r2_n
        self.r2_trend_threshold = r2_trend_threshold
        self.atr_norm_threshold = atr_norm_threshold
        self.mr_band_z = mr_band_z
        self.mr_exit_z = mr_exit_z
        self.min_edge_multiple = min_edge_multiple
        self.route_key = route_key
        self.trade_amount_quote = trade_amount_quote

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # Required columns: 'open','high','low','close'
        for col in ("open", "high", "low", "close"):
            if col not in df.columns:
                raise ValueError(f"Missing column: {col}")

        df["ema_fast"] = _ema(df["close"], self.ema_fast)
        df["ema_slow"] = _ema(df["close"], self.ema_slow)
        df["ema_trend"] = _ema(df["close"], self.ema_trend)
        df["atr"] = _atr(df, self.atr_n)
        dc_u, dc_l, dc_m = _donchian(df, self.donchian_n)
        df["dc_upper"], df["dc_lower"], df["dc_mid"] = dc_u, dc_l, dc_m

        # Normalized ATR vs its rolling mean (smooth volatility regimes)
        df["atr_mean"] = _sma(df["atr"], max(50, self.atr_n * 4))
        df["atr_norm"] = df["atr"] / df["atr_mean"]

        df["r2"] = _rolling_r2_logprice(df["close"], self.r2_n)
        df["z_close_ema"] = _zscore(df["close"] - df["ema_slow"], window=max(30, self.ema_slow))

        # Regimes
        df["is_trend"] = (df["r2"] >= self.r2_trend_threshold) & (df["atr_norm"] >= self.atr_norm_threshold)
        df["is_sideways"] = ~df["is_trend"]
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "ema_fast" not in df.columns:
            df = self.calculate_indicators(df)

        close = df["close"]
        atr = df["atr"].replace(0, np.nan)

        # --- Trend regime: breakout + EMA alignment
        trend_long = (
            df["is_trend"] &
            (df["ema_fast"] > df["ema_slow"]) &
            (df["ema_slow"] > df["ema_trend"]) &
            (close > df["dc_upper"].shift(1))  # breakout on prior bar
        )

        trend_exit = (
            df["is_trend"] &
            (close < df["ema_slow"])  # faster exit on loss of momentum
        )

        # --- Sideways regime: mean-reversion bands around EMA
        z = df["z_close_ema"]
        mr_long = df["is_sideways"] & (z <= -self.mr_band_z)
        mr_exit = df["is_sideways"] & (z >= -self.mr_exit_z)

        # Expected edge in bps:
        # - Trend: distance from close to donchian mid scaled by ATR
        # - MR: |z| times ATR distance to EMA
        dc_range = (df["dc_upper"] - df["dc_lower"]).abs()
        expected_move_trend_bps = (dc_range / close * 1e4).clip(lower=0.0)
        expected_move_mr_bps = (z.abs() * (atr / close) * 1e4).clip(lower=0.0)

        # Cost gate
        def _gate(mask: pd.Series, expected_bps: pd.Series) -> pd.Series:
            out = pd.Series(False, index=df.index)
            # vectorized-friendly: evaluate only when mask True
            idx = mask[mask].index
            for i in idx:
                if cost_gate(
                    expected_edge_bps=float(expected_bps.loc[i]),
                    amount_quote=float(self.trade_amount_quote),
                    route_key=self.route_key,
                    min_edge_multiple=self.min_edge_multiple,
                ):
                    out.loc[i] = True
            return out

        long_signal = _gate(trend_long, expected_move_trend_bps) | _gate(mr_long, expected_move_mr_bps)
        exit_signal = _gate(trend_exit, expected_move_trend_bps) | _gate(mr_exit, expected_move_mr_bps)

        df["buy_signal"] = long_signal
        df["sell_signal"] = exit_signal

        # Signal strength: scaled 0..1 by "edge vs cost" multiple
        # Use a simple mapping: (expected_edge_bps / (roundtrip_cost_bps*1.0)).clip(0, 2)/2
        # Compute a blended expected move
        blended_edge = np.where(trend_long | trend_exit, expected_move_trend_bps, expected_move_mr_bps)
        # Avoid IO in loop by fetching a representative round-trip cost once
        from utils.cost_model_pro1 import estimate_trade_cost_bps
        rt_cost = estimate_trade_cost_bps(amount_quote=float(self.trade_amount_quote),
                                          route_key=self.route_key, single_side=False)
        with np.errstate(divide="ignore", invalid="ignore"):
            strength = np.clip((blended_edge / max(rt_cost, 1e-6)), 0.0, 2.0) / 2.0
        df["signal_strength"] = pd.Series(strength, index=df.index).fillna(0.0)

        return df
