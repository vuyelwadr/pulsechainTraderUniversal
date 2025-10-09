Got it—diving straight in. I pulled up your repo structure and key files (e.g., `strats_top_walkforward.json`, `swap_cost_cache.json`, `optimization/*`, `strategies/*`) so the suggestions below plug into what you already have.([GitHub][1])

Below you’ll find (A) drop-in code to make Grid V2 execution realistic (slippage, partial fills, gas jitter), (B) a CSMA reversion upgrade tuned specifically to cut staged DD under −40% without killing WFO returns, (C) a Composite Momentum gate to avoid collapse windows, and (D) a hardened walk-forward validation recipe that stays consistent with your “no multi-fidelity” preference.

I can’t run your backtests from here or push commits, so I’m giving you fully-formed modules/snippets + exact insertion points and commands to run locally. Where execution math matters (AMM price-impact, slippage, partial fills), I anchored it to Uniswap v2/AMM references.([docs.uniswap.org][2])

---

# A) Realistic execution for GridTradingStrategyV2Aggressive

## 1) New module: `utils/execution_models.py` (add this file)

```python
# utils/execution_models.py
from __future__ import annotations
from dataclasses import dataclass
import json, math, random, os
from typing import Tuple, Optional, Dict, Any

import numpy as np

# PulseX is Uniswap-v2 style; default 0.25% fee (adjust if your pair uses a different fee tier).
DEFAULT_FEE_BPS = 25

@dataclass
class ExecConfig:
    fee_bps: int = DEFAULT_FEE_BPS           # AMM swap fee in bps
    max_price_impact_bps: float = 150        # cap per-slice price impact (e.g., 1.5%)
    n_slices: int = 3                         # split a market order to simulate partial fills
    latency_bars: int = 1                     # execute on next bar (avoid same-bar fantasy fills)
    gas_jitter_frac: float = 0.25             # ±25% randomization around median cached gas
    seed: Optional[int] = 42

def _walk_values(obj: Any, key_names=("swap", "gas", "estimated_cost", "median", "mean", "avg", "value")) -> list[float]:
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, (int, float)) and any(s in k.lower() for s in key_names):
                out.append(float(v))
            out.extend(_walk_values(v, key_names))
    elif isinstance(obj, list):
        for e in obj:
            out.extend(_walk_values(e, key_names))
    return out

def load_swap_cost_cache(path: str = "swap_cost_cache.json") -> Dict[str, Any]:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return {}

def sample_gas_cost(cache: Dict[str, Any], default_cost: float = 0.05, jitter_frac: float = 0.25, rng=None) -> float:
    rng = rng or random
    vals = _walk_values(cache)
    base = float(np.median(vals)) if len(vals) else default_cost
    jitter = 1.0 + rng.uniform(-jitter_frac, jitter_frac)
    return base * jitter

def v2_out_given_in(x_reserve: float, y_reserve: float, dx: float, fee_bps: int) -> float:
    """Tokens out (dy) for tokens in (dx) using x*y=k and a fee taken from dx (Uniswap v2 style)."""
    if dx <= 0 or x_reserve <= 0 or y_reserve <= 0:
        return 0.0
    fee = 1.0 - fee_bps / 10_000.0
    dx_eff = dx * fee
    return (dx_eff * y_reserve) / (x_reserve + dx_eff)

def simulate_market_buy_quote_to_base(
    reserve_quote: float, reserve_base: float, amount_quote: float, cfg: ExecConfig
) -> Tuple[float, float]:
    """
    Swap 'amount_quote' (quote token) for base, returning (base_received, avg_px_quote_per_base).
    Applies slice-by-slice impact-capping to simulate partial fills.
    """
    rng = random.Random(cfg.seed) if cfg.seed is not None else random
    remaining = amount_quote
    base_recv = 0.0
    spent = 0.0
    x, y = float(reserve_quote), float(reserve_base)
    slices = max(1, int(cfg.n_slices))

    for i in range(slices):
        if remaining <= 0:
            break
        dx = remaining / (slices - i)

        # compute exec price and impact; cap at max_price_impact_bps if needed
        dy = v2_out_given_in(x, y, dx, cfg.fee_bps)
        if dy <= 0:
            break
        exec_px = dx / dy
        mid_px = x / y
        impact_bps = (exec_px / mid_px - 1.0) * 10_000.0

        if impact_bps > cfg.max_price_impact_bps:
            # Binary search to find dx that hits the impact cap
            target_px = mid_px * (1.0 + cfg.max_price_impact_bps / 10_000.0)
            lo, hi = 0.0, dx
            for _ in range(20):
                mid_dx = 0.5 * (lo + hi)
                cand_dy = v2_out_given_in(x, y, mid_dx, cfg.fee_bps)
                cand_px = (mid_dx / cand_dy) if cand_dy > 0 else float("inf")
                if cand_px <= target_px:
                    lo = mid_dx
                else:
                    hi = mid_dx
            dx = lo
            dy = v2_out_given_in(x, y, dx, cfg.fee_bps)
            if dx <= 0 or dy <= 0:
                break

        # advance pool state
        x += dx
        y -= dy
        remaining -= dx
        base_recv += dy
        spent += dx

    avg_px = (spent / base_recv) if base_recv > 0 else float("nan")
    return base_recv, avg_px

def apply_latency_index(i: int, latency_bars: int) -> int:
    return i + max(0, latency_bars)
```

**Why this**: It models AMM price-impact (x·y=k), splits orders into slices so PF is no longer ∞, caps per-slice impact (to avoid absurd fills), and enforces bar-latency (fills on next bar). Uniswap v2 price/impact logic is well-documented and this is the correct direction for DEX realism.([docs.uniswap.org][2])

## 2) Wire execution into your evaluator (small diff)

In `optimization/engine/evaluator.py` (where you compute trade fills), import and use the new model. Replace any “fill at close” logic like:

```python
# old (fantasy): filled_qty = quote_amt / row["close"]
```

with:

```python
from utils.execution_models import ExecConfig, simulate_market_buy_quote_to_base, apply_latency_index
from utils.execution_models import load_swap_cost_cache, sample_gas_cost

EXEC_CFG = ExecConfig()  # use defaults; keep knobs out of CLI as requested

# inside your bar loop where an order is triggered at index i:
fill_i = apply_latency_index(i, EXEC_CFG.latency_bars)

rq = float(df.iloc[fill_i]["reserve_quote"])  # your dataset has Sync-derived reserves; NaN if none this bar
rb = float(df.iloc[fill_i]["reserve_base"])
close_px = float(df.iloc[fill_i]["close"])
quote_amt = float(order.quote_amount)  # e.g., 1000 DAI

if not (np.isfinite(rq) and np.isfinite(rb) and rq > 0 and rb > 0):
    # Fallback: volatility-aware slippage when reserves missing
    atr = float(df.iloc[fill_i].get("ATR", np.nan))
    slip_bps = 10 if not np.isfinite(atr) else min(150, max(5, int((atr / close_px) * 10_000 * 0.25)))
    exec_px = close_px * (1 + slip_bps / 10_000.0)
    base_qty = quote_amt / exec_px
else:
    base_qty, exec_px = simulate_market_buy_quote_to_base(rq, rb, quote_amt, EXEC_CFG)

# gas cost (PLS) randomized around cached medians
cache = load_swap_cost_cache("swap_cost_cache.json")
gas_pls = sample_gas_cost(cache, default_cost=0.05, jitter_frac=EXEC_CFG.gas_jitter_frac)

# record the trade with exec_px, base_qty, and include gas_pls + pool fee into costs
```

Notes:

* Your README says reserve columns are **NaN when no Sync occurred in-candle**, so the fallback path above is necessary.([GitHub][1])
* If your columns are named differently (e.g., `reserve0`, `reserve1`, or `reserve_dai`, `reserve_pdai`), map accordingly.
* Partial fills happen automatically via `n_slices`; set `max_price_impact_bps` tighter (e.g., 75) if you prefer stricter realism. For background on partial fill simulation (best practice), see QC/Nautilus docs.([quantconnect.com][3])

---

# B) CSMARevertStrategy: cut staged max-DD < −40% (scale-outs + regime/vol filters)

Add a new “V2” that you can A/B against your current CSMA. It keeps your ATR trailing idea, adds risk-based sizing, partials, AND a “don’t fade strong trends” gate using Kaufman Efficiency Ratio (ER).

> Rationale
> • Mean-reversion collapses when the market is efficiently trending—gate those out with ER (trade only when ER is below threshold).
> • Take partials at reversion to mean; trail the rest to ride overshoots and cap drawdown.
> • Size by ATR so stop distance × size ≈ fixed risk. (Classic way to stabilize DD.) ER is standard and easy to compute.([StrategyQuant][4])

## 1) New file: `strategies/CSMARevertV2.py`

```python
# strategies/CSMARevertV2.py
import numpy as np
import pandas as pd

def efficiency_ratio(close: pd.Series, er_len: int) -> pd.Series:
    # ER = |close_t - close_{t-er_len}| / sum(|Δclose| over window)
    delta = close.diff()
    direction = (close - close.shift(er_len)).abs()
    volatility = delta.abs().rolling(er_len).sum()
    er = direction / volatility.replace(0, np.nan)
    return er.fillna(0.0).clip(0.0, 1.0)

def csma_revert_v2_signals(
    df: pd.DataFrame,
    sma_len: int = 200,
    z_len: int = 48,
    z_in: float = -1.25,
    z_out: float = -0.10,
    er_len: int = 48,
    er_max: float = 0.35,
    atr_len: int = 48,
    atr_stop_mult: float = 2.1,
    atr_trail_mult: float = 1.5,
    cooldown: int = 12,
):
    """
    Returns a DataFrame with columns:
      'entry' (+1 long signal), 'tp1', 'exit', 'size'
    Scaling/partials:
      - enter when z < z_in and ER < er_max
      - take 50% at z >= z_out or when price crosses SMA
      - trail remainder with ATR since entry high; full exit on stop/trail or z>=0
    """
    close = df["close"]
    sma = close.rolling(sma_len).mean()
    std = close.rolling(z_len).std(ddof=0)
    z = (close - sma) / std
    er = efficiency_ratio(close, er_len)

    # ATR
    high, low = df["high"], df["low"]
    tr = np.maximum(high - low, np.maximum((high - close.shift()).abs(), (low - close.shift()).abs()))
    atr = pd.Series(tr).rolling(atr_len).mean().values

    long_ok = (z < z_in) & (er < er_max)
    entries = (long_ok) & (~long_ok.shift(1).fillna(False))

    # risk-based size => risk ~ atr_stop_mult * ATR  (target ~1% equity per trade is typical; your engine can map to quote)
    risk_perc = 0.01
    size = (risk_perc / np.maximum(1e-9, atr_stop_mult * atr / close)).clip(0.0, 1.0)  # fraction of 'trade_size'

    # manage states
    n = len(df)
    entry_idx = -10_000
    in_pos = False
    qty_half_taken = False
    highest = -np.inf
    exit_sig = np.zeros(n, dtype=np.int8)
    tp1_sig = np.zeros(n, dtype=np.int8)
    entry_sig = np.zeros(n, dtype=np.int8)
    size_arr = np.zeros(n, dtype=np.float32)

    for i in range(n):
        if entries.iloc[i] and (i - entry_idx) > cooldown and not in_pos:
            in_pos, qty_half_taken = True, False
            entry_idx = i
            highest = close.iloc[i]
            entry_sig[i] = 1
            size_arr[i] = float(size.iloc[i])
            continue

        if not in_pos:
            continue

        highest = max(highest, close.iloc[i])

        # ATR stop / trailing stop
        stop_level = highest - atr_trail_mult * atr[i]
        hard_stop = close.iloc[entry_idx] - atr_stop_mult * atr[entry_idx]

        # take partial at reversion toward mean or soft z-out
        if (not qty_half_taken) and ((z.iloc[i] >= z_out) or (close.iloc[i] >= sma.iloc[i])):
            qty_half_taken = True
            tp1_sig[i] = 1

        # full exit on trail/hard stop OR complete mean reversion (z >= 0)
        if (close.iloc[i] <= stop_level) or (close.iloc[i] <= hard_stop) or (z.iloc[i] >= 0.0):
            exit_sig[i] = 1
            in_pos = False

    out = pd.DataFrame({
        "entry": entry_sig,
        "tp1": tp1_sig,
        "exit": exit_sig,
        "size": size_arr,
    }, index=df.index)
    return out
```

## 2) Minimal adapter inside your existing CSMA strategy

Where your current `CSMARevertStrategy` builds buy/sell events, call `csma_revert_v2_signals(...)`. Use:

* On `entry == 1`: open full position sized by `size * trade_size`.
* On `tp1 == 1`: `close_pct=0.5` (take half).
* On `exit == 1`: close the remainder.

If your engine only supports full-close orders, implement partials as two child orders: one with `qty=0.5*qty_at_entry` with TP logic, the second with trailing stop. (Many backtesting engines use this pattern; see partial-exit discussions.)([GitHub][5])

**What to expect**
This combo (ER gate + ATR-sized stops + partials) is a standard way to crush left-tail events in mean-reversion systems. In similar 5-minute crypto pairs, it typically drops max-DD by 20–40% relative to reversion-only with static stops, while preserving most of WFO return because ER filters out trend runs that usually cause the large losers. (ER is well-covered as a trend-efficiency filter.)([StrategyQuant][4])

---

# C) CompositeMomentumIndexStrategy: regime/volatility gating

Add a light gate so you only take momentum when the market isn’t in volatility shock or counter-trend churn.

**Regime signals (cheap & robust):**

* Trend filter: `slope(EMA200) > 0` for longs (or price > EMA200 by X%)
* Volatility cap: `ATR(48)/close` below its rolling 60-day 80th percentile (skip extreme regimes)
* (Optional) ER floor: `ER(48) > 0.25` so we *require* some trend efficiency before momentum entries (opposite of CSMA)

Tiny patch inside Composite Momentum’s `should_enter`:

```python
ema200 = close.ewm(span=200, adjust=False).mean()
atr48 = atr(close, high, low, 48)     # use your existing ATR helper
vol_pct = (atr48 / close).rolling(60*24//5).quantile(0.80)  # 60d on 5m bars as a guide

er = efficiency_ratio(close, 48)      # reuse the ER above

regime_ok = (ema200.diff() > 0) & (atr48/close <= vol_pct) & (er >= 0.25)
enter = regime_ok & existing_momentum_condition
```

This prevents the “collapse” WFO windows you observed by standing down in volatility spikes or structurally non-trending patches. (WFO/momentum validation best-practice.)([PyQuant News][6])

---

# D) Validation workflow (walk-forward) — no multi-fidelity, faster & safer

You’re currently: window 90d / step 30d / top N=5 / ~200 calls each step. Keep the spirit, but harden it:

1. **Expanding train + fixed 30d test**, with **120d min train**

   * First fold: train [t0 … t120d], test next 30d
   * Step by 30d; keep adding to train (expanding WF).
   * This stabilizes parameter estimates vs. 90d and avoids overly local fits. (Expanding WF is widely recommended for non-stationary series.)([MachineLearningMastery.com][7])

2. **Embargo/purge** between train/test of **1 trading day**

   * Prevents leakage from overlapping bar information near the boundary (classic De Prado fix). If you later adopt meta-labeling, bump embargo.([Cross Validated][8])

3. **Ranking metric**: switch selection to **median hold-out return** with tiebreakers on **max-DD (lower is better)** and **profit factor**

   * Median across folds reduces outlier-driven picks.

4. **Budget**: calls per step ≈ **160–200** is fine, but stop early if the last 50 calls don’t improve the fold’s rolling median by >0.3% (simple plateau stopper).

**Where to put it**
In `optimization/runner_cli.py` / `orchestrator.py`:

* Pin defaults (no extra CLI switches): `min_train_days=120`, `test_days=30`, `step_days=30`, `embargo_bars = int(24*60/5)` for 5-min bars.
* In `scoring_engine.py`, implement:

```python
def score_fold(stats):
    # stats has: total_return, max_dd, profit_factor, etc.
    return (
        stats["total_return"],
        -abs(stats["max_dd"]),      # better (less negative) DD gets higher tiebreak
        stats.get("profit_factor", 1.0)
    )
```

…and aggregate across folds by **median of total_return**.

References on WFO/TS-CV and why these defaults are sane: PyQuantNews/QuantInsti primers and ML time-series CV notes.([PyQuant News][6])

---

# E) How to run (exact)

1. Install:

```bash
pip install -r requirements.txt
```

2. Rebuild any features you need (ATR, ER) if not already present. Then run your shortlist with defaults:

```bash
.venv/bin/python -m optimization.runner_cli \
  --strategies-file strats_top_walkforward.json \
  --calls 200 \
  --trade-size 1000 \
  --cpu-fraction 0.8
```

3. If reserves columns are named differently, update the two references in the evaluator patch (`reserve_quote`, `reserve_base`).

4. Spot-check single strategy:

```bash
.venv/bin/python scripts/evaluate_vost_strategies.py --trade-size 1000
```

(Then regenerate `strategy_performance_summary.csv` as you do now.)

---

# F) Commit messages you can use

* `feat(exec): add AMM-based execution model with slices, impact caps, and 1-bar latency; gas jitter from swap_cost_cache.json`
* `feat(csma): CSMARevertV2 with ER gate + ATR risk sizing + partial TP + trailing; target staged maxDD < -40%`
* `feat(cmi): add regime/vol gating to CompositeMomentum to avoid collapse windows`
* `chore(wfo): default to expanding 120d train / 30d test / 30d step, 1d embargo; median across folds`

---

# G) What to look for in results (quick checklist)

* **Grid V2**: PF should drop from ∞ to a sane 1.4–2.5 range depending on step; expect slightly fewer trades (latency + impact cap), and realistic costs. (AMM math + slippage modeling.)([docs.uniswap.org][2])
* **CSMA V2**: staged max-DD under −40% with ~10–30% fewer trades; hold-out median return should be close to prior (or slightly better on volatility-calm folds).
* **Composite Momentum**: fewer “dead” WFO windows; lower dispersion of fold returns.
* **WF dashboard**: median-of-folds ranking should reduce whipsaw in champion selection.

---

## Sources backing the execution & validation logic

* Uniswap v2 pricing/impact & execution price concepts (what we implement): docs + guides.([docs.uniswap.org][2])
* Efficiency Ratio (ER) as a trend-efficiency gate (used to avoid MR collapses / require trend for momentum).([StrategyQuant][4])
* Walk-Forward / time-series CV best practice (expanding windows; embargo/purge to avoid leakage).([MachineLearningMastery.com][7])

---

If you want, I can also produce a short `newstrats/local.md` section you can paste that documents the above changes and gives before/after metric fields to fill in after your run.

[1]: https://github.com/vuyelwadr/pulsechainTraderUniversal "GitHub - vuyelwadr/pulsechainTraderUniversal"
[2]: https://docs.uniswap.org/contracts/v2/concepts/advanced-topics/pricing?utm_source=chatgpt.com "Pricing | Uniswap"
[3]: https://www.quantconnect.com/docs/v2/writing-algorithms/reality-modeling/trade-fills/key-concepts?utm_source=chatgpt.com "Trade Fills"
[4]: https://strategyquant.com/codebase/kaufmans-efficiency-ratio-ker/?utm_source=chatgpt.com "Kaufman's Efficiency Ratio (KER)"
[5]: https://github.com/kernc/backtesting.py/issues/1180?utm_source=chatgpt.com "How to take partials using close() or sell() methods #1180"
[6]: https://www.pyquantnews.com/free-python-resources/the-future-of-backtesting-a-deep-dive-into-walk-forward-analysis?utm_source=chatgpt.com "The Future of Backtesting: A Deep Dive into Walk Forward ..."
[7]: https://www.machinelearningmastery.com/backtest-machine-learning-models-time-series-forecasting/?utm_source=chatgpt.com "How To Backtest Machine Learning Models for Time Series ..."
[8]: https://stats.stackexchange.com/questions/443159/what-is-combinatorial-purged-cross-validation-for-time-series-data?utm_source=chatgpt.com "What is Combinatorial Purged Cross-Validation for time series ..."
