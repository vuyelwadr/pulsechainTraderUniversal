
"""
utils/cost_model_pro1.py

Cost model helper that reads `swap_cost_cache.json` at the repository root
and provides convenience functions to estimate the *total* round-trip
(enter + exit) cost in basis points (bps).

This file is additive and does not modify existing code; strategies can
import it like:

    from utils.cost_model_pro1 import estimate_trade_cost_bps, cost_gate

It was written to be robust to slightly different JSON shapes. If your
`swap_cost_cache.json` has a different schema, adjust the
`_extract_cost_bps_from_cache_entry` function at the bottom.
"""
from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, Optional

_DEFAULT_SINGLE_SIDE_COST_BPS = 35.0  # conservative fallback if cache missing
_MIN_COST_BPS = 5.0                   # never assume below 5 bps single-side


def load_swap_cost_cache(path: str = "swap_cost_cache.json") -> Optional[Dict[str, Any]]:
    """Try to load the swap cost cache from repo root.
    Returns None if not found / invalid.
    """
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def _extract_cost_bps_from_cache_entry(entry: Any) -> Optional[float]:
    """
    Try very hard to extract a *single-side* cost in bps from a cache entry.

    Supported shapes (examples):
      - {"fee_bps": 25, "slippage_bps": 10} -> 35
      - {"avg_bps": 42.1} -> 42.1
      - {"total_bps": 33.2} -> 33.2
      - {"fixed_fee_bps": 15, "variable_slippage_bps": 8} -> 23
      - {"median": {"bps": 28.5}} -> 28.5
      - 31.7 -> 31.7  # raw number
    """
    if entry is None:
        return None

    # raw number
    if isinstance(entry, (int, float)):
        return float(entry)

    if isinstance(entry, dict):
        # Common keys
        for k in ("fee_bps", "avg_bps", "total_bps", "bps", "single_side_bps"):
            if k in entry and isinstance(entry[k], (int, float)):
                return float(entry[k])
        # Combine fee + slippage if present
        fee_like = None
        slip_like = None
        for k in ("slippage_bps", "variable_slippage_bps", "slip_bps"):
            if k in entry and isinstance(entry[k], (int, float)):
                slip_like = float(entry[k])
                break
        for k in ("fixed_fee_bps", "fee_bps", "router_fee_bps"):
            if k in entry and isinstance(entry[k], (int, float)):
                fee_like = float(entry[k])
                break
        if fee_like is not None or slip_like is not None:
            return float((fee_like or 0.0) + (slip_like or 0.0))

        # Nested "median"/"mean" structures
        for top in ("median", "mean", "avg"):
            if top in entry and isinstance(entry[top], dict):
                if "bps" in entry[top] and isinstance(entry[top]["bps"], (int, float)):
                    return float(entry[top]["bps"])

    return None


def estimate_trade_cost_bps(
    amount_quote: float,
    route_key: Optional[str] = None,
    cache: Optional[Dict[str, Any]] = None,
    single_side: bool = True,
) -> float:
    """Estimate single-side or round-trip cost in bps from the cache.
    - amount_quote: trade amount in quote token (e.g., DAI) units. Currently only
      used to allow future size-based curves; if your cache contains size tiers,
      adapt the logic below to choose the appropriate tier.
    - route_key: key inside the JSON for a specific route/pool. If None,
      uses 'default' or a global fallback.
    - cache: pre-loaded JSON dict. If None, we'll load from file.
    - single_side: if False, returns round-trip (enter + exit) cost.

    Returns: cost in bps as float.
    """
    if cache is None:
        cache = load_swap_cost_cache() or {}

    single_side_cost_bps = None

    # Attempt route-specific lookup first
    if route_key and isinstance(cache, dict) and route_key in cache:
        single_side_cost_bps = _extract_cost_bps_from_cache_entry(cache[route_key])

    # Otherwise try common/default keys
    if single_side_cost_bps is None and isinstance(cache, dict):
        for k in ("default", "hex_dai", "HEX/DAI", "router_default", "global"):
            if k in cache:
                single_side_cost_bps = _extract_cost_bps_from_cache_entry(cache[k])
                if single_side_cost_bps is not None:
                    break

    # Fallback
    if single_side_cost_bps is None:
        single_side_cost_bps = max(_DEFAULT_SINGLE_SIDE_COST_BPS, _MIN_COST_BPS)

    if not single_side:
        return single_side_cost_bps * 2.0
    return single_side_cost_bps


def cost_gate(
    expected_edge_bps: float,
    amount_quote: float,
    route_key: Optional[str] = None,
    cache: Optional[Dict[str, Any]] = None,
    min_edge_multiple: float = 1.15,
) -> bool:
    """Return True if expected edge comfortably exceeds costs.
    We require: expected_edge_bps >= min_edge_multiple * (enter + exit costs).

    min_edge_multiple of 1.15 = 15% cushion over raw estimated costs.
    """
    total_cost_bps = estimate_trade_cost_bps(
        amount_quote=amount_quote, route_key=route_key, cache=cache, single_side=False
    )
    return expected_edge_bps >= (min_edge_multiple * total_cost_bps)
