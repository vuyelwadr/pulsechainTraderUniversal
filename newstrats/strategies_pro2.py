
"""
newstrats/strategies_pro1.py
---------------------------------
A compact pack of **cost-aware, regime-switching** strategies for PulseX / HEX trading.

Design goals
- Long-only signals for DEX (no shorting).
- Respect realistic trading frictions using swap_cost_cache.json.
- Robustness across regimes (trending vs. range-bound) with adaptive filters.
- Minimal external deps: numpy, pandas only.

Drop-in idea
- You can import `STRATEGIES_PRO1` (a dict of callables), or `EnsemblePro1`.
- Each callable expects a DataFrame with columns: open, high, low, close, (optionally 'volume').
- Returns a DataFrame with columns: 'position' (0/1), 'entry', 'exit', and useful diagnostics.

Author tag
- Produced by: pro1
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd


# =============================
# --------- Cost Model --------
# =============================

@dataclass
class CostEstimate:
    total_bps: float  # total round-trip cost in basis points (fees + expected slippage + gas as bps)
    meta: dict        # debug info (which path, which fields, etc.)


class CostModel:
    """
    Parse and use swap_cost_cache.json with *best-effort* schema detection.

    We support multiple shapes because we don't know the exact JSON schema a priori.
    The heuristics below try very hard to find a reasonable total_bps number by:
      1) Looking for explicit 'total_bps' fields
      2) Else adding 'fee_bps' + 'slippage_bps' (+ gas expressed as bps if size is given)
      3) Falling back to conservative defaults (60 bps round-trip) if nothing is found.
    """

    def __init__(self, cache_path: str = "swap_cost_cache.json", pair_hint: str = "HEX/DAI", default_total_bps: float = 60.0):
        self.cache_path = cache_path
        self.pair_hint = pair_hint
        self.default_total_bps = float(default_total_bps)
        try:
            with open(cache_path, "r") as f:
                self.cache = json.load(f)
        except Exception as e:
            # Fail-safe: honor the "respect costs" principle by being conservative if cache missing
            self.cache = {}
            self._init_error = repr(e)
        else:
            self._init_error = None

    def _as_bps(self, x: Optional[float]) -> Optional[float]:
        if x is None:
            return None
        # if x seems like a fraction (e.g. 0.003), interpret as fraction and scale to bps
        if 0 < x < 1:
            return 10000.0 * x
        return float(x)

    def _extract_any(self, obj: Any, keys: Tuple[str, ...]) -> Optional[float]:
        if not isinstance(obj, dict):
            return None
        for k in keys:
            if k in obj:
                val = obj[k]
                if isinstance(val, (int, float)):
                    return float(val)
                # Sometimes nested
                if isinstance(val, dict):
                    # look for a numeric field inside
                    for kk, vv in val.items():
                        if isinstance(vv, (int, float)):
                            return float(vv)
        return None

    def estimate_total_bps(self, *, size_dai: float = 1000.0, pair: Optional[str] = None) -> CostEstimate:
        """
        Return a conservative estimate for *ROUND-TRIP* trading cost in basis points.
        - size_dai is used only if the cache encodes tiered slippage / gas in absolute units.
        - pair allows overriding pair_hint if your runner sets it differently.
        """
        pair_key = pair or self.pair_hint
        meta = {"pair": pair_key, "cache_path": self.cache_path, "notes": []}

        if not self.cache:
            meta["notes"].append(f"cache_unavailable:{self._init_error}")
            return CostEstimate(self.default_total_bps, meta)

        # Try multiple common shapes
        obj = self.cache

        # 1) Look for exact pair entry directly
        cand_objs = []
        for k in (pair_key, pair_key.replace("/", "_"), pair_key.replace("/", "-"), pair_key.lower(), pair_key.upper()):
            if isinstance(obj, dict) and k in obj and isinstance(obj[k], dict):
                cand_objs.append(obj[k])
        if not cand_objs:
            # Also consider that the top-level may have a list of entries each with a 'pair' field
            if isinstance(obj, list):
                cand_objs = [x for x in obj if isinstance(x, dict) and x.get("pair") in (pair_key, pair_key.replace("/", "_"), pair_key.replace("/", "-"))]

        def _score_node(node: dict) -> Optional[float]:
            # Priority: 'total_bps' else 'round_trip_bps' else sum of fee+slip+gas
            total = self._extract_any(node, ("total_bps", "round_trip_bps"))
            if total is not None:
                return self._as_bps(total)
            fee = self._extract_any(node, ("fee_bps", "fees_bps", "base_fee_bps", "dex_fee_bps", "lp_fee_bps"))
            slip = self._extract_any(node, ("slippage_bps", "slip_bps", "expected_slippage_bps"))
            gas = self._extract_any(node, ("gas_bps", "gas_cost_bps"))
            parts = []
            if fee is not None:
                parts.append(self._as_bps(fee))
            if slip is not None:
                parts.append(self._as_bps(slip))
            if gas is not None:
                parts.append(self._as_bps(gas))
            if parts:
                return float(sum(parts))
            # Sometimes costs are fractions: fee_fraction, slippage_fraction, etc.
            fee = self._extract_any(node, ("fee_fraction",))
            slip = self._extract_any(node, ("slippage_fraction",))
            gas = self._extract_any(node, ("gas_fraction",))
            parts = []
            if fee is not None:
                parts.append(self._as_bps(fee))
            if slip is not None:
                parts.append(self._as_bps(slip))
            if gas is not None:
                parts.append(self._as_bps(gas))
            if parts:
                return float(sum(parts))
            return None

        candidates = []
        # examine candidates
        for node in cand_objs:
            score = _score_node(node)
            if score is not None:
                candidates.append((score, "pair_node"))
            # also scan children if present
            if isinstance(node, dict):
                for v in node.values():
                    if isinstance(v, dict):
                        s2 = _score_node(v)
                        if s2 is not None:
                            candidates.append((s2, "pair_child"))

        # 2) Fallback: search entire document for any cost-like numbers, take *max* as conservative
        if not candidates:
            def walk(d):
                if isinstance(d, dict):
                    val = _score_node(d)
                    if val is not None:
                        yield val
                    for v in d.values():
                        yield from walk(v)
                elif isinstance(d, list):
                    for x in d:
                        yield from walk(x)
            found = list(walk(obj))
            if found:
                candidates.append((max(found), "global_max"))

        if candidates:
            best = max(candidates, key=lambda x: x[0])  # conservative
            bps = float(best[0])
            meta["notes"].append(f"found:{best[1]}")
            return CostEstimate(bps, meta)

        meta["notes"].append("fallback_default")
        return CostEstimate(self.default_total_bps, meta)


# =============================
# ---------- Indicators -------
# =============================

def _ema(x: pd.Series, span: int) -> pd.Series:
    return x.ewm(span=span, adjust=False).mean()

def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()

def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    down = (-delta).clip(lower=0)
    roll_up = up.ewm(alpha=1/n, adjust=False).mean()
    roll_down = down.ewm(alpha=1/n, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    return 100 - (100 / (1 + rs))

def bbands(close: pd.Series, n: int = 20, k: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ma = close.rolling(n, min_periods=n).mean()
    sd = close.rolling(n, min_periods=n).std(ddof=0)
    upper = ma + k * sd
    lower = ma - k * sd
    return ma, upper, lower

def adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    plus_dm = (high.diff()).clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    plus_dm[plus_dm < minus_dm] = 0.0
    minus_dm[minus_dm <= plus_dm] = 0.0

    tr1 = (high - low)
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr_n = tr.rolling(n, min_periods=n).mean()
    plus_di = 100 * (plus_dm.rolling(n, min_periods=n).sum() / (atr_n + 1e-12))
    minus_di = 100 * (minus_dm.rolling(n, min_periods=n).sum() / (atr_n + 1e-12))

    dx = 100 * (plus_di - minus_di).abs() / ((plus_di + minus_di) + 1e-12)
    return dx.rolling(n, min_periods=n).mean()


# =============================
# --------- Base utils --------
# =============================

def _position_to_entries_exits(pos: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """Given a 0/1 position series, derive entries (rising edge) and exits (falling edge)."""
    pos = (pos.fillna(0).astype(int)).clip(0, 1)
    entries = (pos.diff() > 0).astype(int)
    exits = (pos.diff() < 0).astype(int)
    return entries, exits

def _liquidity_filter(df: pd.DataFrame, vol_quantile: float = 0.2) -> pd.Series:
    """Return a boolean mask where volume (if present) is in the top (1 - vol_quantile) quantile."""
    if "volume" not in df.columns:
        return pd.Series(True, index=df.index)
    thresh = df["volume"].quantile(vol_quantile)
    return (df["volume"] >= thresh)

def _edge_bps_from_momo(close: pd.Series, fast: int, slow: int) -> pd.Series:
    """A simple 'edge' proxy: normalized EMA separation in basis points."""
    f = _ema(close, fast)
    s = _ema(close, slow)
    return ((f - s) / (s + 1e-12)) * 10000.0

def _apply_time_stop(pos: pd.Series, bars: int) -> pd.Series:
    """
    Apply a naive time-based exit: close position if it's older than 'bars' since last entry.
    Implemented by counting bars since entry while pos==1; exit when count==bars.
    """
    pos = pos.copy().astype(int)
    ages = pd.Series(0, index=pos.index, dtype=int)
    count = 0
    last = 0
    for i, p in enumerate(pos.values):
        if p == 1:
            if last == 0:
                count = 1
            else:
                count += 1
        else:
            count = 0
        ages.iat[i] = count
        last = p
    forced_exit = ((ages >= bars) & (pos == 1)).astype(int)
    # convert forced_exit into position drop on next bar
    pos2 = pos.copy()
    for i in range(1, len(pos2)):
        if forced_exit.iat[i-1] == 1:
            pos2.iat[i] = 0
    return pos2


# ===================================
# --------- Strategy: TAB -----------
# ===================================

def strategy_TAB_pro1(
    df: pd.DataFrame,
    lookback: int = 55,
    atr_n: int = 14,
    atr_mult_stop: float = 3.0,
    min_hold_bars: int = 6,
    time_stop_bars: int = 96,
    cost_safety_mult: float = 1.2,
    pair_hint: str = "HEX/DAI",
    nominal_trade_size_dai: float = 1000.0,
    vol_quantile_filter: float = 0.2,
) -> pd.DataFrame:
    """
    Trend-Aware Breakout (TAB):
    - Entry: close crosses above Donchian high(lookback) AND normalized edge > cost threshold
    - Exit: trailing ATR stop OR time stop OR close crosses under mid-channel
    - Filters: ADX confirms trend; liquidity filter blocks dead periods

    Returns DataFrame with columns: position, entry, exit, diagnostics...
    """
    out = pd.DataFrame(index=df.index)
    high = df["high"]
    low = df["low"]
    close = df["close"]

    don_high = high.rolling(lookback, min_periods=lookback).max().shift(1)  # confirm on next bar
    don_low = low.rolling(lookback, min_periods=lookback).min().shift(1)
    mid = (don_high + don_low) / 2.0

    adx_v = adx(df, n=max(14, lookback // 4)).fillna(0.0)
    atr_v = atr(df, n=atr_n).fillna(0.0)
    liq_mask = _liquidity_filter(df, vol_quantile=vol_quantile_filter)

    edge_bp = _edge_bps_from_momo(close, fast=max(3, lookback // 6), slow=max(6, lookback // 3)).fillna(0.0)
    cost_model = CostModel(pair_hint=pair_hint)
    cost_est = cost_model.estimate_total_bps(size_dai=nominal_trade_size_dai)
    min_edge_bps = cost_est.total_bps * float(cost_safety_mult)

    # Entry criteria
    breakout = (close > don_high) & (adx_v > 18) & liq_mask
    sufficient_edge = (edge_bp > min_edge_bps)
    entry_sig = (breakout & sufficient_edge).astype(int)

    # Position state (long-only)
    pos = entry_sig.copy()
    pos = pos.where(pos == 1, np.nan).ffill().fillna(0).astype(int)

    # Minimum hold: prevent immediate flip-flop
    pos_min_hold = pos.copy()
    if min_hold_bars > 0:
        # zero-out exits before min_hold
        ages = (pos_min_hold != pos_min_hold.shift()).cumsum()  # segment id
        bar_index_within = pos_min_hold.groupby(ages).cumcount() + 1
        too_soon = (bar_index_within < min_hold_bars) & (pos_min_hold == 1)
        # preserve position until min_hold reached
        pos_min_hold = np.where((pos_min_hold == 1) | too_soon, 1, 0)
        pos_min_hold = pd.Series(pos_min_hold, index=df.index).astype(int)

    # Trailing ATR stop and mid-channel exit
    trail_stop_price = close.copy() * 0.0 + np.nan
    last_entry_price = None

    pos2 = pos_min_hold.copy()
    for i in range(len(pos2)):
        if entry_sig.iat[i] == 1:
            last_entry_price = close.iat[i]
        if pos2.iat[i] == 1 and last_entry_price is not None:
            stop = last_entry_price - atr_mult_stop * atr_v.iat[i]
            trail_stop_price.iat[i] = stop
            if close.iat[i] < stop:
                pos2.iat[i] = 0
                last_entry_price = None
        # mid-channel give-up
        if pos2.iat[i] == 1 and not np.isnan(mid.iat[i]) and close.iat[i] < mid.iat[i]:
            pos2.iat[i] = 0
            last_entry_price = None

    # Time stop
    if time_stop_bars > 0:
        pos2 = _apply_time_stop(pd.Series(pos2, index=df.index), bars=time_stop_bars)

    entries, exits = _position_to_entries_exits(pd.Series(pos2, index=df.index))

    out["position"] = pos2.astype(int)
    out["entry"] = entries
    out["exit"] = exits
    out["diag_adx"] = adx_v
    out["diag_edge_bps"] = edge_bp
    out["diag_min_edge_bps"] = float(min_edge_bps)
    out["diag_cost_bps"] = float(cost_est.total_bps)
    return out


# ===================================
# --------- Strategy: AMAC ----------
# ===================================

def strategy_AMAC_pro1(
    df: pd.DataFrame,
    fast: int = 10,
    slow: int = 45,
    atr_n: int = 14,
    atr_mult_stop: float = 2.5,
    time_stop_bars: int = 96,
    cost_safety_mult: float = 1.15,
    pair_hint: str = "HEX/DAI",
    nominal_trade_size_dai: float = 1000.0,
    slope_confirm: int = 3,
    vol_quantile_filter: float = 0.2,
) -> pd.DataFrame:
    """
    Adaptive Moving Average Crossover (AMAC):
    - EMA(fast) cross above EMA(slow), slope of slow EMA positive over `slope_confirm` bars
    - Edge proxy (EMA separation) must exceed cost threshold * cost_safety_mult
    - Exits via ATR trailing or time-stop
    """
    out = pd.DataFrame(index=df.index)
    close = df["close"]
    f = _ema(close, fast)
    s = _ema(close, slow)
    slope_ok = (s.diff(slope_confirm) > 0).fillna(False)
    edge_bp = ((f - s) / (s + 1e-12)) * 10000.0

    cost_model = CostModel(pair_hint=pair_hint)
    cost_est = cost_model.estimate_total_bps(size_dai=nominal_trade_size_dai)
    min_edge_bps = cost_est.total_bps * float(cost_safety_mult)

    cross_up = (f > s) & (f.shift(1) <= s.shift(1))
    liq_mask = _liquidity_filter(df, vol_quantile=vol_quantile_filter)

    entry_sig = (cross_up & slope_ok & (edge_bp > min_edge_bps) & liq_mask).astype(int)

    # Long-only state
    pos = entry_sig.copy()
    pos = pos.where(pos == 1, np.nan).ffill().fillna(0).astype(int)

    # Exits
    atr_v = atr(df, n=atr_n).fillna(0.0)
    pos2 = pos.copy()
    last_entry = None
    for i in range(len(pos2)):
        if entry_sig.iat[i] == 1:
            last_entry = close.iat[i]
        if pos2.iat[i] == 1 and last_entry is not None:
            stop = last_entry - atr_mult_stop * atr_v.iat[i]
            if close.iat[i] < stop:
                pos2.iat[i] = 0
                last_entry = None

    if time_stop_bars > 0:
        pos2 = _apply_time_stop(pd.Series(pos2, index=df.index), bars=time_stop_bars)

    entries, exits = _position_to_entries_exits(pd.Series(pos2, index=df.index))
    out["position"] = pos2.astype(int)
    out["entry"] = entries
    out["exit"] = exits
    out["diag_edge_bps"] = edge_bp
    out["diag_min_edge_bps"] = float(min_edge_bps)
    out["diag_cost_bps"] = float(cost_est.total_bps)
    return out


# ===================================
# --------- Strategy: RMR -----------
# ===================================

def strategy_RMR_pro1(
    df: pd.DataFrame,
    rsi_lb: int = 14,
    rsi_buy: float = 32,
    bb_n: int = 20,
    bb_k: float = 2.0,
    adx_thresh: float = 18.0,
    atr_n: int = 14,
    atr_mult_stop: float = 2.0,
    time_stop_bars: int = 72,
    cost_safety_mult: float = 1.20,
    pair_hint: str = "HEX/DAI",
    nominal_trade_size_dai: float = 1000.0,
    vol_quantile_filter: float = 0.2,
) -> pd.DataFrame:
    """
    RSI Mean-Reversion (RMR):
    - Regime requires ADX < adx_thresh (sideways)
    - Entry when RSI < rsi_buy and close below lower Bollinger band
    - Edge proxy: distance to middle band in bps must clear min_edge_bps
    """
    out = pd.DataFrame(index=df.index)
    close = df["close"]
    adx_v = adx(df, n=max(14, rsi_lb)).fillna(0.0)
    ma, upper, lower = bbands(close, n=bb_n, k=bb_k)
    rsi_v = rsi(close, n=rsi_lb)

    cost_model = CostModel(pair_hint=pair_hint)
    cost_est = cost_model.estimate_total_bps(size_dai=nominal_trade_size_dai)
    min_edge_bps = cost_est.total_bps * float(cost_safety_mult)

    # Edge proxy: distance back to middle band
    edge_bp = ((ma - close) / (close + 1e-12)) * 10000.0

    regime_ok = (adx_v < adx_thresh)
    liq_mask = _liquidity_filter(df, vol_quantile=vol_quantile_filter)
    entry_sig = ((rsi_v < rsi_buy) & (close < lower) & (edge_bp > min_edge_bps) & regime_ok & liq_mask).astype(int)

    # Long-only state
    pos = entry_sig.copy()
    pos = pos.where(pos == 1, np.nan).ffill().fillna(0).astype(int)

    # Exits: mean reversion to middle band or ATR stop or time stop
    atr_v = atr(df, n=atr_n).fillna(0.0)
    pos2 = pos.copy()
    last_entry = None
    for i in range(len(pos2)):
        if entry_sig.iat[i] == 1:
            last_entry = close.iat[i]
        if pos2.iat[i] == 1:
            # take-profit on touch of middle band
            if not np.isnan(ma.iat[i]) and close.iat[i] >= ma.iat[i]:
                pos2.iat[i] = 0
                last_entry = None
                continue
            # protective ATR stop
            if last_entry is not None:
                stop = last_entry - atr_mult_stop * atr_v.iat[i]
                if close.iat[i] < stop:
                    pos2.iat[i] = 0
                    last_entry = None

    if time_stop_bars > 0:
        pos2 = _apply_time_stop(pd.Series(pos2, index=df.index), bars=time_stop_bars)

    entries, exits = _position_to_entries_exits(pd.Series(pos2, index=df.index))
    out["position"] = pos2.astype(int)
    out["entry"] = entries
    out["exit"] = exits
    out["diag_edge_bps"] = edge_bp
    out["diag_min_edge_bps"] = float(min_edge_bps)
    out["diag_cost_bps"] = float(cost_est.total_bps)
    return out


# ===================================
# --------- Ensemble Pro1 ----------
# ===================================

def EnsemblePro1(
    df: pd.DataFrame,
    params: Optional[Dict[str, Dict[str, Any]]] = None,
    pair_hint: str = "HEX/DAI",
    nominal_trade_size_dai: float = 1000.0,
) -> pd.DataFrame:
    """
    A simple regime-switching ensemble:
    - If ADX >= 18 -> prefer TAB (breakout/trend)
    - Else -> prefer RMR (range-bound)
    - Always require AMAC confirmation before entering

    The function returns a final 'position' plus diagnostics and component positions.
    """
    if params is None:
        params = {}
    # Clone df to ensure we don't mutate caller
    df = df.copy()

    # Compute component strategies
    tab_df = strategy_TAB_pro1(df, pair_hint=pair_hint, nominal_trade_size_dai=nominal_trade_size_dai, **params.get("TAB", {}))
    amac_df = strategy_AMAC_pro1(df, pair_hint=pair_hint, nominal_trade_size_dai=nominal_trade_size_dai, **params.get("AMAC", {}))
    rmr_df = strategy_RMR_pro1(df, pair_hint=pair_hint, nominal_trade_size_dai=nominal_trade_size_dai, **params.get("RMR", {}))

    # Regime from ADX
    adx_v = adx(df, n=14).fillna(0.0)
    trend_regime = (adx_v >= 18)

    # Decision: when trend -> TAB if also AMAC pos==1; else 0
    #          when range -> RMR
    pos = pd.Series(0, index=df.index, dtype=int)

    # Trend regime: need both TAB and AMAC to be 1
    pos_trend = ((tab_df["position"] == 1) & (amac_df["position"] == 1)).astype(int)
    # Range regime: RMR decides
    pos_range = (rmr_df["position"] == 1).astype(int)

    pos = np.where(trend_regime, pos_trend, pos_range)
    pos = pd.Series(pos, index=df.index).astype(int)

    entries, exits = _position_to_entries_exits(pos)

    out = pd.DataFrame(index=df.index)
    out["position"] = pos
    out["entry"] = entries
    out["exit"] = exits
    out["regime_adx"] = adx_v
    out["TAB_pos"] = tab_df["position"]
    out["AMAC_pos"] = amac_df["position"]
    out["RMR_pos"] = rmr_df["position"]
    return out


# Public registry (import this in your runner)
STRATEGIES_PRO1 = {
    "TAB_pro1": strategy_TAB_pro1,
    "AMAC_pro1": strategy_AMAC_pro1,
    "RMR_pro1": strategy_RMR_pro1,
    "EnsemblePro1": EnsemblePro1,
}
