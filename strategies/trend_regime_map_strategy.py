"""Trend-regime aware strategy derived from hourly slope analysis.

The strategy mirrors the hourly linear-regression slope logic used by the
`generate_trend_map.py` helper. It classifies the market into uptrend, downtrend
or range regimes and trades only when the uptrend regime aligns with local price
structure and volume confirmation. Positions are protected via trailing ATR
stops and promptly exited on regime deterioration.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .base_strategy import BaseStrategy


@dataclass
class RegimeState:
    """Lightweight container tracking the active position state."""

    in_position: bool = False
    entry_price: float = 0.0
    trailing_stop: float = 0.0


class TrendRegimeMapStrategy(BaseStrategy):
    """Trend following strategy informed by hourly regime analysis."""

    def __init__(self, parameters: Optional[Dict] = None) -> None:
        defaults = {
            "timeframe_minutes": 5,
            "trend_window_hours": 24,
            "slope_threshold": 0.0015,
            "rsquared_threshold": 0.55,
            "ema_fast_period": 36,  # ≈3 hours on 5m bars
            "ema_slow_period": 144,  # ≈12 hours on 5m bars
            "atr_period": 96,  # ≈8 hours on 5m bars
            "atr_trailing_mult": 2.2,
            "max_pullback": 0.012,
            "min_volume_ratio": 0.75,
        }
        if parameters:
            defaults.update(parameters)
        super().__init__("TrendRegimeMapStrategy", defaults)
        self.state = RegimeState()

    # ------------------------------------------------------------------
    # Indicator preparation
    # ------------------------------------------------------------------
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        if data.empty:
            return data

        if "timestamp" not in data.columns:
            raise ValueError("Data must include timestamp column for resampling")

        df = data.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.sort_values("timestamp").reset_index(drop=True)

        hourly = (
            df.set_index("timestamp").resample("1h").agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                }
            ).dropna()
        )
        log_close = np.log(hourly["close"])
        window = max(int(self.parameters["trend_window_hours"]), 6)
        slopes = (
            log_close.rolling(window, min_periods=window, center=True)
            .apply(self._rolling_slope, raw=True)
            .rename("log_slope")
        )
        regime = slopes.apply(self._classify_slope).rename("trend")
        regime_frame = pd.concat([slopes, regime], axis=1).dropna().reset_index()

        df = pd.merge_asof(
            df.sort_values("timestamp"),
            regime_frame.sort_values("timestamp"),
            on="timestamp",
            direction="backward",
        )

        df["ema_fast"] = df["close"].ewm(span=self.parameters["ema_fast_period"], adjust=False).mean()
        df["ema_slow"] = df["close"].ewm(span=self.parameters["ema_slow_period"], adjust=False).mean()

        high = df["high"].fillna(df["close"])
        low = df["low"].fillna(df["close"])
        close = df["close"].fillna(df.get("price", df["close"]))
        prev_close = close.shift(1)
        tr_components = pd.concat(
            [
                (high - low).abs(),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        )
        df["true_range"] = tr_components.max(axis=1)
        df["atr"] = df["true_range"].rolling(window=self.parameters["atr_period"], min_periods=5).mean()

        if "volume" in df.columns:
            df["volume_ma"] = df["volume"].rolling(window=144, min_periods=20).mean()
            df["volume_ratio"] = df["volume"] / df["volume_ma"]
        else:
            df["volume_ratio"] = 1.0

        return df

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        if df.empty or "trend" not in df.columns:
            return df

        df["buy_signal"] = False
        df["sell_signal"] = False
        df["signal_strength"] = 0.0

        slope_threshold = self.parameters["slope_threshold"]
        pullback_limit = self.parameters["max_pullback"]
        atr_mult = self.parameters["atr_trailing_mult"]
        min_volume_ratio = self.parameters["min_volume_ratio"]

        for idx, row in df.iterrows():
            trend = row.get("trend")
            slope = row.get("log_slope", 0.0)
            price = row.get("close", row.get("price"))
            ema_fast = row.get("ema_fast")
            ema_slow = row.get("ema_slow")
            atr = row.get("atr", 0.0)
            volume_ratio = row.get("volume_ratio", 1.0)

            if price is None or ema_fast is None or ema_slow is None:
                continue

            if self.state.in_position and atr > 0:
                new_stop = float(price - atr_mult * atr)
                self.state.trailing_stop = max(self.state.trailing_stop, new_stop)

            entry_conditions = (
                trend == "uptrend"
                and slope is not None
                and slope >= slope_threshold
                and ema_fast > ema_slow
                and price >= ema_fast * (1 - pullback_limit)
                and volume_ratio >= min_volume_ratio
            )
            if entry_conditions:
                if not self.state.in_position:
                    self.state.in_position = True
                    self.state.entry_price = float(price)
                    self.state.trailing_stop = (
                        float(price - atr_mult * atr) if atr > 0 else float(price * (1 - pullback_limit))
                    )
                df.at[idx, "buy_signal"] = True
                strength = min(1.0, max(0.25, slope / (slope_threshold * 2)))
                df.at[idx, "signal_strength"] = strength
                continue

            exit_condition = False
            if self.state.in_position:
                if trend == "downtrend" or slope <= 0:
                    exit_condition = True
                elif atr > 0 and price <= self.state.trailing_stop:
                    exit_condition = True
                elif price < ema_slow:
                    exit_condition = True

            if exit_condition and self.state.in_position:
                df.at[idx, "sell_signal"] = True
                df.at[idx, "signal_strength"] = 1.0
                self.state = RegimeState()

        return df

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _rolling_slope(self, values: np.ndarray) -> float:
        if np.any(np.isnan(values)):
            return np.nan
        x = np.arange(values.size)
        slope, intercept = np.polyfit(x, values, 1)
        fitted = slope * x + intercept
        ss_tot = np.sum((values - values.mean()) ** 2)
        if ss_tot == 0:
            return 0.0
        ss_res = np.sum((values - fitted) ** 2)
        r_squared = 1 - ss_res / ss_tot
        if r_squared < self.parameters["rsquared_threshold"]:
            return 0.0
        return float(slope)

    def _classify_slope(self, slope: float) -> str:
        if slope >= self.parameters["slope_threshold"]:
            return "uptrend"
        if slope <= -self.parameters["slope_threshold"]:
            return "downtrend"
        return "range"
