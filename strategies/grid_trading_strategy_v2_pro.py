
#!/usr/bin/env python3
"""
Regime-Adaptive Grid Trading Strategy V2

Key improvements vs v1:
- Volatility-adaptive step size: grid spacing scales with ATR/volatility so it stays active in quiet markets and
  widens in turbulence to avoid chop.
- Regime-aware skew: in uptrends, more sell levels above center (take profit often); in downtrends, more buy
  levels below center (DCA deeper). Skew is continuous, not on/off.
- Anchor-based center: grid center is an EMA (or price if short history). Optional re-centering when price
  drifts (rebalance_threshold).
- Breakout guard (z-score): when price is statistically far from the anchor, only trade in the direction of the move.
- Cooldown: prevents back-to-back duplicate signals on consecutive bars.
- Fee-aware tolerance: signal window scales with step size and typical range.

Interface parity:
- Same public methods as v1: calculate_indicators(), generate_signals(), get_grid_info().
- Uses self.parameters and BaseStrategy.validate_data() like v1.
- Adds parameter_space() for the runner to read explicit optimization bounds.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

try:
    # Keep relative import so it works inside your repo
    from .base_strategy import BaseStrategy
except Exception:
    # Allow local testing without the package
    class BaseStrategy:
        def __init__(self, name: str, parameters: Dict):
            self.name = name
            self.parameters = parameters
        def validate_data(self, data: pd.DataFrame) -> bool:
            return isinstance(data, pd.DataFrame) and len(data) > 0

logger = logging.getLogger(__name__)

class GridTradingStrategyV2Pro(BaseStrategy):
    """
    Regime-Adaptive Grid Trading Strategy (V2)

    Parameters (defaults are conservative; tune with Bayesian optimizer):
        base_grid_size_percent (float): Base distance between grid levels, before volatility scaling. Default 2.5
        num_grids (int): Maximum grid levels per side (above/below). Default 18
        price_range_percent (float): Hard cap of total grid span from center. Default 24
        min_strength (float): Minimum signal_strength to emit True. Default 0.15
        vol_threshold (float): Minimum realized vol (stdev of returns) to be active. Default 0.0
        rebalance_threshold (float): % move of price from last rebalance to rebuild grid. Default 1.0
        anchor_lookback (int): EMA lookback for grid center. Default 20
        atr_period (int): ATR period for step sizing. Default 14
        step_vol_mult (float): Multiplier for ATR/vol scaling of step size. Default 1.0
        skew_strength (float): 0..1; 0 = symmetric; 1 = fully skew with trend. Default 0.5
        zscore_limit (float): If |z| > limit, only trade with breakout direction. Default 1.2
        cooldown_bars (int): Bars to wait after a signal before another. Default 1
    """

    def __init__(self, parameters: Dict = None):
        default_params = dict(
            base_grid_size_percent=2.5,
            num_grids=18,
            price_range_percent=24.0,
            min_strength=0.15,
            vol_threshold=0.0,
            rebalance_threshold=1.0,
            anchor_lookback=20,
            atr_period=14,
            step_vol_mult=1.0,
            skew_strength=0.5,
            zscore_limit=1.2,
            cooldown_bars=1,
        )
        if parameters:
            default_params.update(parameters)
        super().__init__("GridTradingStrategyV2", default_params)

        # Grid state
        self.grid_levels: List[Dict] = []
        self.grid_center: float = 0.0
        self.last_rebalance_price: float = 0.0
        self._last_signal_index: Optional[int] = None

    # ---- Compatibility: let the runner discover tunable params explicitly ----
    @staticmethod
    def parameter_space() -> Dict[str, Tuple[float, float]]:
        """
        Return skopt-compatible bounds (runner.build_skopt_dimensions will map these).
        Keep the count modest for sample efficiency.
        """
        return {
            "base_grid_size_percent": (0.8, 4.0),   # tighter than v1 in quiet regimes, wider allowed in vol
            "num_grids": (8, 22),                   # integer inferred by runner
            "price_range_percent": (16.0, 40.0),
            "min_strength": (0.05, 0.6),
            "vol_threshold": (0.0, 0.06),           # stdev of returns
            "rebalance_threshold": (0.0, 3.0),
            "anchor_lookback": (12, 50),
            "atr_period": (7, 30),
            "step_vol_mult": (0.5, 2.2),
            "skew_strength": (0.0, 0.9),
            "zscore_limit": (0.6, 2.0),
            "cooldown_bars": (0, 4),
        }

    # ---- Core computations ----
    @staticmethod
    def _ema(x: pd.Series, n: int) -> pd.Series:
        return x.ewm(span=max(1, int(n)), adjust=False, min_periods=1).mean()

    @staticmethod
    def _atr(df: pd.DataFrame, period: int) -> pd.Series:
        # Use proper OHLC if available, else fall back to price changes
        high = df["high"] if "high" in df.columns else df["price"]
        low  = df["low"]  if "low"  in df.columns else df["price"]
        close = df["close"] if "close" in df.columns else df["price"]
        prev_close = close.shift(1).fillna(close.iloc[0])
        tr = pd.concat([
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ], axis=1).max(axis=1)
        atr = tr.rolling(max(1, int(period))).mean()
        return atr.bfill().fillna(0.0)

    @staticmethod
    def _zscore(x: pd.Series, lookback: int) -> pd.Series:
        m = x.rolling(lookback).mean()
        s = x.rolling(lookback).std()
        z = (x - m) / s.replace(0, np.nan)
        return z.fillna(0.0)

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        if not self.validate_data(data):
            return data

        df = data.copy()
        # Normalize column names expected by engine
        if "price" not in df.columns and "close" in df.columns:
            df["price"] = df["close"]
        if "close" not in df.columns and "price" in df.columns:
            df["close"] = df["price"]

        # Volatility metrics
        df["ret"] = df["price"].pct_change().fillna(0.0)
        df["vol"] = df["ret"].rolling(20).std().fillna(0.0)

        # Anchor and trend
        L = max(2, int(self.parameters["anchor_lookback"]))
        df["anchor"] = self._ema(df["price"], L)
        # Fast/slow for regime bias
        df["ema_fast"] = self._ema(df["price"], max(2, L // 2))
        df["ema_slow"] = self._ema(df["price"], max(3, L * 2))
        # Trend bias in [-1, 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            df["trend_bias"] = np.tanh(10.0 * (df["ema_fast"] - df["ema_slow"]) / df["price"].replace(0, np.nan))
        df["trend_bias"] = df["trend_bias"].fillna(0.0)

        # ATR for adaptive spacing
        atr_period = max(2, int(self.parameters["atr_period"]))
        df["atr"] = self._atr(df, atr_period)
        df["atr_pct"] = (df["atr"] / df["price"]).clip(lower=0.0).fillna(0.0)

        # Z-score of price around anchor (use same window as anchor_lookback for stability)
        df["zscore"] = self._zscore(df["price"] - df["anchor"], max(5, L))

        # Range/position context (for opportunity scoring)
        rng_n = max(10, L)
        rolling_min = df["price"].rolling(rng_n).min()
        rolling_max = df["price"].rolling(rng_n).max()
        width = (rolling_max - rolling_min).replace(0, np.nan)
        df["price_pos"] = ((df["price"] - rolling_min) / width).fillna(0.5)

        # Update grid based on the most recent bar only (stateful)
        self._setup_grid_levels(df.iloc[-1], df)

        # Signal opportunity score (used in strength)
        # Prefer: higher vol, mid-range prices, and mild trends (avoid extremes)
        df["grid_opportunity"] = (
            (df["vol"] * 10.0).clip(0, 1.5)
            + (1.0 - (df["price_pos"] - 0.5).abs() * 2.0).clip(0, 1.0)
            + (1.0 - df["zscore"].abs().clip(0, 3.0) / 3.0)
        )

        return df

    def _dynamic_step_pct(self, row: pd.Series) -> float:
        base = float(self.parameters["base_grid_size_percent"]) / 100.0
        mult = float(self.parameters["step_vol_mult"])
        atr_pct = float(row.get("atr_pct", 0.0))
        vol = float(row.get("vol", 0.0))
        # Reference scale: typical volatility proxy; avoid division by zero
        ref = max(1e-6, np.mean([atr_pct, vol]))
        # simple robust scaling: larger than ref -> widen; smaller -> tighten
        scale = 1.0 + mult * ((atr_pct - ref) / max(ref, 1e-6))
        step = base * float(np.clip(scale, 0.5, 2.0))
        return float(step)

    def _setup_grid_levels(self, last: pd.Series, df: pd.DataFrame) -> None:
        """Recompute levels if first time or price drift exceeds threshold."""
        price = float(last["price"])
        center = float(last.get("anchor", price))
        if not np.isfinite(center) or center <= 0:
            center = price

        # Whether to rebuild
        rebalance_thr = float(self.parameters["rebalance_threshold"]) / 100.0
        if (not self.grid_levels) or (self.last_rebalance_price <= 0):
            need_rebuild = True
        else:
            move_pct = abs(price - self.last_rebalance_price) / max(self.last_rebalance_price, 1e-9)
            need_rebuild = move_pct >= rebalance_thr if rebalance_thr > 0 else False

        # Always keep center near anchor (even if not fully rebalancing)
        self.grid_center = center

        if not need_rebuild and self.grid_levels:
            return

        # Determine regime-aware skew
        skew = float(self.parameters["skew_strength"])
        trend = float(last.get("trend_bias", 0.0))  # [-1, 1]
        total_levels = int(self.parameters["num_grids"])
        # Sell levels above, buy levels below
        sell_levels = max(1, min(total_levels - 1, int(round(total_levels * (0.5 + 0.5 * trend * skew)))))
        buy_levels = max(1, total_levels - sell_levels)

        # Dynamic step and range cap
        step_pct = self._dynamic_step_pct(last)
        max_span_pct = float(self.parameters["price_range_percent"]) / 100.0
        # Respect the span cap by limiting the farthest level index
        max_idx = max(1, int(np.floor(max_span_pct / max(step_pct, 1e-9))))
        sell_levels = int(min(sell_levels, max_idx))
        buy_levels = int(min(buy_levels, max_idx))

        # Build fresh levels
        levels: List[Dict] = []
        for i in range(1, buy_levels + 1):
            levels.append({"level": center * (1.0 - step_pct * i), "type": "buy", "distance": i})
        for j in range(1, sell_levels + 1):
            levels.append({"level": center * (1.0 + step_pct * j), "type": "sell", "distance": j})
        levels.sort(key=lambda d: d["level"])
        self.grid_levels = levels
        self.last_rebalance_price = price
        logger.debug(f"[V2] Grid rebuilt: center={center:.6f}, step={step_pct:.5f}, L={buy_levels}/{sell_levels}")

    def _nearest_levels(self) -> Tuple[Optional[Dict], Optional[Dict]]:
        """Return nearest buy level below and sell level above current center; used for quick checks."""
        if not self.grid_levels:
            return None, None
        buys = [l for l in self.grid_levels if l["type"] == "buy"]
        sells = [l for l in self.grid_levels if l["type"] == "sell"]
        return (buys[-1] if buys else None, sells[0] if sells else None)

    def _strength(self, distance: int, row: pd.Series, direction: str) -> float:
        # Depth: deeper levels slightly stronger (encourage DCA / profit taking further from center)
        depth = min(1.0, 0.25 + 0.12 * distance)
        # Volatility fit: ATR relative to step; if ATR >> step, signals are meaningful
        step_pct = self._dynamic_step_pct(row)
        atr_pct = float(row.get("atr_pct", 0.0))
        vol_fit = float(np.clip((atr_pct / max(step_pct, 1e-6)) * 0.6, 0.0, 1.0))
        # Opportunity: mid-range, moderate z, decent vol
        opp = float(np.clip(row.get("grid_opportunity", 0.5) / 3.0, 0.0, 1.0))
        # Regime alignment: in uptrend, boost sells; in downtrend, boost buys
        trend = float(row.get("trend_bias", 0.0))
        if direction == "sell":
            regime = 0.5 + 0.5 * trend
        else:  # buy
            regime = 0.5 - 0.5 * trend
        # Combine
        total = 0.40 * depth + 0.25 * vol_fit + 0.25 * opp + 0.10 * regime
        return float(np.clip(total, 0.0, 1.0))

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        if data is None or len(data) == 0:
            return data

        df = data.copy()
        for col in ("buy_signal", "sell_signal", "signal_strength"):
            if col not in df.columns:
                df[col] = False if col.endswith("_signal") else 0.0

        vol_thr = float(self.parameters["vol_threshold"])
        zlim = float(self.parameters["zscore_limit"])
        cooldown = int(self.parameters["cooldown_bars"])
        min_s = float(self.parameters["min_strength"])

        if not self.grid_levels:
            return df

        # Iterate row-wise to maintain cooldown logic
        for i in range(len(df)):
            row = df.iloc[i]
            price = float(row["price"] if "price" in df.columns else row.get("close", np.nan))
            if not np.isfinite(price):
                continue

            # Gating
            if float(row.get("vol", 0.0)) < vol_thr:
                continue

            # Breakout guard: if far from anchor, only go with the move
            z = float(row.get("zscore", 0.0))
            allow_buy = True
            allow_sell = True
            if abs(z) > zlim:
                if z > 0:   # stretched high
                    allow_buy = False
                else:       # stretched low
                    allow_sell = False

            # Cooldown
            if cooldown > 0 and self._last_signal_index is not None and (i - self._last_signal_index) <= cooldown:
                continue

            # Dynamic tolerance around each level (fraction of step, plus a small atr fraction)
            step_pct = self._dynamic_step_pct(row)
            atr_pct = float(row.get("atr_pct", 0.0))
            tol = float(np.clip(0.25 * step_pct + 0.1 * atr_pct, 0.001, 0.02))

            # Evaluate proximity to levels
            did_signal = False
            strength = 0.0

            # Check buy side
            if allow_buy:
                for lvl in [l for l in self.grid_levels if l["type"] == "buy"]:
                    rel = (price - lvl["level"]) / max(lvl["level"], 1e-9)
                    if -tol <= rel <= tol:  # within window
                        s = self._strength(lvl["distance"], row, "buy")
                        if s >= min_s and s > strength:
                            df.iat[i, df.columns.get_loc("buy_signal")] = True
                            df.iat[i, df.columns.get_loc("signal_strength")] = s
                            did_signal = True
                            strength = s
                            self._last_signal_index = i
                            break  # nearest hit is enough

            # Check sell side (prefer not to overwrite a buy on the same bar)
            if not did_signal and allow_sell:
                for lvl in [l for l in self.grid_levels if l["type"] == "sell"]:
                    rel = (lvl["level"] - price) / max(lvl["level"], 1e-9)
                    if -tol <= rel <= tol:
                        s = self._strength(lvl["distance"], row, "sell")
                        if s >= min_s and s > strength:
                            df.iat[i, df.columns.get_loc("sell_signal")] = True
                            df.iat[i, df.columns.get_loc("signal_strength")] = s
                            did_signal = True
                            strength = s
                            self._last_signal_index = i
                            break

        return df

    def get_grid_info(self) -> Dict:
        buys = [l["level"] for l in self.grid_levels if l.get("type") == "buy"]
        sells = [l["level"] for l in self.grid_levels if l.get("type") == "sell"]
        return {
            "grid_center": self.grid_center,
            "last_rebalance_price": self.last_rebalance_price,
            "num_levels": len(self.grid_levels),
            "buy_levels": buys,
            "sell_levels": sells,
            "parameters": self.parameters,
        }
