
"""
Cost-aware SMA Reversion (v2) — "C-SMA-Revert"
-----------------------------------------------
Long-only, all-in/all-out. Uses bucketed swap-costs:
- For a notional N, ceil to the next 5k bucket (or whatever "step_notional" is)
- Per-side cost rate = (round-trip loss_rate at bucket) / 2
- Plus side-specific gas from the cache (USD≈DAI assumed)

Entry:  close <= SMA(n_sma) * (1 - entry_drop)  AND  RSI(rsi_n) <= rsi_max
Exit:   close >= SMA(n_sma) * (1 + exit_up)

Defaults (best found in iteration):
  n_sma=576 (≈2 days on 5m bars), entry_drop=0.25, exit_up=0.05, rsi_n=14, rsi_max=30

This module has both:
- A CLI for quick testing over a CSV (5m bars) and swap_cost_cache.json
- A light "adapter()" function returning (name, space, run_fn) for optimizers/runners

Usage (CLI):
  python strategy_c_sma_revert.py \
     --csv /path/to/ohlcv.csv \
     --swap-cost-cache /path/to/swap_cost_cache.json \
     --n-sma 576 --entry-drop 0.25 --exit-up 0.05 --rsi-n 14 --rsi-max 30 \
     --start-cash 1000 --window-days 730 \
     --plot-out /tmp/equity.png

Integration hint (runner):
  from strategies.strategy_c_sma_revert import adapter as c_sma_revert_adapter
  name, space, run_fn = c_sma_revert_adapter()
  # then call run_fn(df, params, swap_cost_cache_path, start_cash) from your runner

Author: ChatGPT (C) 2025
"""

from __future__ import annotations
import argparse, json, math, sys
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List, Optional
from pathlib import Path

import pandas as pd
import numpy as np

# ----------------------------
# Swap-cost model (bucketed)
# ----------------------------
@dataclass
class SwapCostBuckets:
    step: int
    buckets: Dict[int, Dict[str, Any]]  # e.g., 5000 -> entry dict

    @classmethod
    def load(cls, path: str | Path) -> "SwapCostBuckets":
        with open(path, "r") as f:
            data = json.load(f)
        step = int(data["metadata"]["step_notional"])
        buckets = {int(k): v for k, v in data["entries"].items()}
        return cls(step=step, buckets=buckets)

    @property
    def keys_sorted(self):
        return sorted(self.buckets.keys())

    def bucket_for(self, notional: float) -> int:
        n = float(notional)
        for k in self.keys_sorted:
            if n <= k:
                return k
        return self.keys_sorted[-1]

    def per_side_rate(self, notional: float) -> float:
        b = self.bucket_for(notional)
        loss_rate = float(self.buckets[b]["derived"]["loss_rate"])
        return loss_rate / 2.0

    def gas(self, notional: float, side: str) -> float:
        b = self.bucket_for(notional)
        entry = self.buckets[b]
        if side.lower().startswith("b"):
            return float(entry["buy"]["gas_use_estimate_usd"])
        else:
            return float(entry["sell"]["gas_use_estimate_usd"])

# ----------------------------
# Indicators
# ----------------------------
def sma(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n, min_periods=n).mean()

def rsi(series: pd.Series, n: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.rolling(n, min_periods=n).mean()
    ma_down = down.rolling(n, min_periods=n).mean()
    rs = ma_up / (ma_down + 1e-12)
    return 100 - (100 / (1 + rs))

# ----------------------------
# Strategy
# ----------------------------
@dataclass
class CSMARevertParams:
    n_sma: int = 576
    entry_drop: float = 0.25
    exit_up: float = 0.05
    rsi_n: int = 14
    rsi_max: float = 30.0

@dataclass
class Trade:
    entry_time: str
    exit_time: str
    entry_px: float
    exit_px: float
    entry_bucket: int
    exit_bucket: int
    capital_in: float
    proceeds: float
    trade_roi: float

class CSMARevert:
    def __init__(self, params: CSMARevertParams, costs: SwapCostBuckets):
        self.p = params
        self.costs = costs

    def run(self, close: pd.Series, start_cash: float = 1000.0) -> Tuple[float, pd.Series, List[Trade]]:
        s = close.dropna().copy()
        s.index = pd.to_datetime(s.index, utc=True)
        sm = sma(s, self.p.n_sma)
        r = rsi(s, self.p.rsi_n)

        p = s.to_numpy()
        idx = s.index

        cash = start_cash
        tokens = 0.0
        position = 0

        equity_curve = []
        trades: List[Trade] = []

        entry_i = None
        entry_px = None
        capital_in = None
        entry_bucket = None

        for i in range(len(p)):
            px = p[i]
            smi = sm.iloc[i]
            ri = r.iloc[i]
            equity_curve.append(cash + tokens * px)

            if np.isnan(smi) or np.isnan(ri):
                continue

            # Entry
            if position == 0 and px <= smi * (1 - self.p.entry_drop) and ri <= self.p.rsi_max and cash > 0:
                N = cash
                rate = self.costs.per_side_rate(N)
                buy_g = self.costs.gas(N, "buy")
                eff = cash - N * rate - buy_g
                if eff > 0:
                    tokens = eff / px
                    cash -= N
                    position = 1
                    entry_i = i
                    entry_px = px
                    capital_in = N
                    entry_bucket = self.costs.bucket_for(N)

            # Exit
            if position == 1 and px >= smi * (1 + self.p.exit_up):
                gross = tokens * px
                rate = self.costs.per_side_rate(gross)
                sell_g = self.costs.gas(gross, "sell")
                proceed = max(0.0, gross - gross * rate - sell_g)
                cash += proceed
                exit_bucket = self.costs.bucket_for(gross)
                trade_roi = (proceed / capital_in) - 1.0 if capital_in else 0.0
                trades.append(Trade(
                    entry_time=str(idx[entry_i]),
                    exit_time=str(idx[i]),
                    entry_px=float(entry_px),
                    exit_px=float(px),
                    entry_bucket=int(entry_bucket),
                    exit_bucket=int(exit_bucket),
                    capital_in=float(capital_in),
                    proceeds=float(proceed),
                    trade_roi=float(trade_roi)
                ))
                tokens = 0.0
                position = 0
                entry_i = entry_px = capital_in = entry_bucket = None

        # Close any open
        if position == 1 and tokens > 0:
            px = p[-1]
            gross = tokens * px
            rate = self.costs.per_side_rate(gross)
            sell_g = self.costs.gas(gross, "sell")
            proceed = max(0.0, gross - gross * rate - sell_g)
            cash += proceed
            exit_bucket = self.costs.bucket_for(gross)
            trade_roi = (proceed / capital_in) - 1.0 if capital_in else 0.0
            trades.append(Trade(
                entry_time=str(idx[entry_i]) if entry_i is not None else str(idx[-1]),
                exit_time=str(idx[-1]),
                entry_px=float(entry_px) if entry_px is not None else float(px),
                exit_px=float(px),
                entry_bucket=int(entry_bucket) if entry_bucket is not None else int(self.costs.bucket_for(proceed)),
                exit_bucket=int(exit_bucket),
                capital_in=float(capital_in) if capital_in else 0.0,
                proceeds=float(proceed),
                trade_roi=float(trade_roi)
            ))
            tokens = 0.0
            position = 0

        curve = pd.Series(equity_curve, index=idx[:len(equity_curve)], name="equity")
        return float(cash), curve, trades

# ----------------------------
# CSV loader
# ----------------------------
def load_ohlcv_close(csv_path: str | Path) -> pd.Series:
    df = pd.read_csv(csv_path)
    # pick ts column
    ts_col = None
    for c in df.columns:
        if 'time' in c.lower() or 'date' in c.lower():
            ts_col = c; break
    if ts_col is None:
        ts_col = df.columns[0]
    def parse_ts(v):
        try:
            return pd.to_datetime(v, utc=True)
        except Exception:
            try:
                return pd.to_datetime(float(v), unit='s', utc=True)
            except Exception:
                return pd.NaT
    df['ts'] = df[ts_col].apply(parse_ts)
    df = df.dropna(subset=['ts']).sort_values('ts')
    # pick close
    close_col = None
    for c in df.columns:
        if c.lower() == 'close' or 'price' in c.lower():
            close_col = c; break
    if close_col is None:
        # fallback to the second column
        close_col = df.columns[1]
    s = df.set_index('ts')[close_col].astype(float)
    s.index = pd.to_datetime(s.index, utc=True)
    return s

# ----------------------------
# Adapter (for runners)
# ----------------------------
def adapter():
    """
    Returns (name, param_space, run_fn) for external optimizers/runners.

    - name: string identifier for the strategy
    - param_space: dict defining ranges for each param (suggested)
    - run_fn: callable(df:pd.DataFrame/Series, params:dict, swap_cost_json:str, start_cash:float)-> dict
    """
    name = "c_sma_revert"
    space = {
        "n_sma": [480, 576, 720],
        "entry_drop": [0.20, 0.25, 0.30],
        "exit_up": [0.02, 0.05, 0.10],
        "rsi_n": [14],
        "rsi_max": [25, 30, 35],
    }
    def run_fn(close_series: pd.Series, params: Dict[str, Any], swap_cost_json: str, start_cash: float = 1000.0):
        costs = SwapCostBuckets.load(swap_cost_json)
        p = CSMARevertParams(
            n_sma=int(params.get("n_sma", 576)),
            entry_drop=float(params.get("entry_drop", 0.25)),
            exit_up=float(params.get("exit_up", 0.05)),
            rsi_n=int(params.get("rsi_n", 14)),
            rsi_max=float(params.get("rsi_max", 30.0)),
        )
        strat = CSMARevert(p, costs)
        final, curve, trades = strat.run(close_series, start_cash=start_cash)
        return {
            "final_equity": final,
            "equity_curve": curve,
            "trades": [t.__dict__ for t in trades],
            "params": p.__dict__,
            "name": name
        }
    return name, space, run_fn

# ----------------------------
# CLI
# ----------------------------
def main():
    ap = argparse.ArgumentParser(description="Cost-aware SMA Reversion (v2) backtest")
    ap.add_argument("--csv", required=True, help="OHLCV CSV with timestamp + close/price")
    ap.add_argument("--swap-cost-cache", required=True, help="swap_cost_cache.json path")
    ap.add_argument("--n-sma", type=int, default=576)
    ap.add_argument("--entry-drop", type=float, default=0.25)
    ap.add_argument("--exit-up", type=float, default=0.05)
    ap.add_argument("--rsi-n", type=int, default=14)
    ap.add_argument("--rsi-max", type=float, default=30.0)
    ap.add_argument("--start-cash", type=float, default=1000.0)
    ap.add_argument("--window-days", type=int, default=730)
    ap.add_argument("--plot-out", type=str, default=None, help="optional path to save equity curve PNG")

    args = ap.parse_args()

    close = load_ohlcv_close(args.csv)
    if args.window_days and args.window_days > 0:
        cutoff = close.index.max() - pd.Timedelta(days=int(args.window_days))
        close = close[close.index >= cutoff]

    costs = SwapCostBuckets.load(args.swap_cost_cache)
    params = CSMARevertParams(
        n_sma=args.n_sma,
        entry_drop=args.entry_drop,
        exit_up=args.exit_up,
        rsi_n=args.rsi_n,
        rsi_max=args.rsi_max
    )
    strat = CSMARevert(params, costs)
    final, curve, trades = strat.run(close, start_cash=args.start_cash)

    # Print summary
    roi = final/args.start_cash - 1.0
    print(f"Final: {final:.2f}  ROI: {roi*100:.2f}%  Trades: {len(trades)}")

    # Optional plot
    if args.plot_out:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12,5))
        curve.plot(label="Strategy Equity")
        # Simple Buy&Hold curve for comparison
        # Buy once with costs
        N = args.start_cash
        rate = costs.per_side_rate(N)
        buy_g = costs.gas(N, "buy")
        tokens = max(0.0, (args.start_cash - N*rate - buy_g) / float(close.iloc[0]))
        (close * tokens).rename("Buy & Hold Equity").plot(label="Buy & Hold")
        plt.title("Equity Curve — C-SMA-Revert vs Buy & Hold")
        plt.xlabel("Time"); plt.ylabel("Equity (DAI)"); plt.legend(); plt.tight_layout()
        Path(args.plot_out).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(args.plot_out, dpi=160)
        print(f"Saved plot -> {args.plot_out}")

if __name__ == "__main__":
    main()
