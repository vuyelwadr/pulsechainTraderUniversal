
# `newstrats/local_pro1.md` — Cost‑Aware Strategy Pack (pro1)

This file documents a new set of **cost-aware, regime-switching strategies** designed to improve *return percentage* while **respecting `swap_cost_cache.json`**.

> Produced by agent: **pro1**

## What’s included

- `strategies_pro1.py` — three robust long-only strategies + a simple ensemble:
  - **TAB_pro1**: Trend‑Aware Breakout using Donchian channels, ADX filter, ATR stops (trend regime).
  - **AMAC_pro1**: Adaptive EMA crossover with slope confirmation and ATR stops (trend follow, whipsaw‑resistant).
  - **RMR_pro1**: RSI/Bollinger mean‑reversion with ADX regime gating (range regime).
  - **EnsemblePro1**: Switches between trend (TAB∧AMAC) and range (RMR) based on ADX.
- `params_pro1.json` — default parameters plus compact grids for walk‑forward search.

All strategies call a common **CostModel** that reads `swap_cost_cache.json` and **blocks entries unless the predicted edge exceeds total round‑trip cost** (plus a safety multiplier).

## Design principles

1. **Edge > Cost**: Each entry requires a predicted move (in bps) larger than `total_bps * cost_safety_mult`.  
2. **Regime awareness**: Use **ADX** to distinguish **trending** vs **range‑bound** periods, reducing bad trades.  
3. **Liquidity gating**: A simple **volume quantile** filter prevents trades in illiquid bars.  
4. **Graceful exits**: ATR‑based stops, time stops, and take‑profits (RMR to middle band).  
5. **Long‑only**: Consistent with DEX spot trading; no shorting.  

## File map

```
newstrats/
  strategies_pro1.py
  params_pro1.json
  local_pro1.md   <-- this file
```

## How it respects `swap_cost_cache.json`

- We added a **`CostModel`** class that:
  - Loads `swap_cost_cache.json` at runtime.
  - Detects several common schema shapes (e.g., `total_bps`, or sum of `fee_bps + slippage_bps (+ gas_bps)`).
  - If structure is unknown, it **falls back conservatively** (60 bps round‑trip) rather than underestimating costs.
- Every strategy computes a **`min_edge_bps = total_bps * cost_safety_mult`** and requires the internal edge proxy to exceed it before entering.

> If your cache encodes tiered slippage/fees, the model uses the **maximum found** as a conservative guard. You can refine the parser to match your exact schema later if desired.

## Quick usage patterns

### 1) Import into your existing validation/backtest

```python
# Example inside your backtest runner
from newstrats.strategies_pro1 import STRATEGIES_PRO1
from newstrats.strategies_pro1 import EnsemblePro1  # optional

# df must have columns: open, high, low, close, (volume optional)
res = STRATEGIES_PRO1["TAB_pro1"](df, **params_TAB)
# -> res[['position','entry','exit']]

# Or run the ensemble with per-component params
resE = EnsemblePro1(df, params={"TAB": {...}, "AMAC": {...}, "RMR": {...}})
```

### 2) Suggested defaults

Load `params_pro1.json` and pass blocks to the callables. For example:

```python
import json
with open("newstrats/params_pro1.json","r") as f:
    P = json.load(f)

tab = STRATEGIES_PRO1["TAB_pro1"](df, **P["TAB_pro1"])
amac = STRATEGIES_PRO1["AMAC_pro1"](df, **P["AMAC_pro1"])
rmr = STRATEGIES_PRO1["RMR_pro1"](df, **P["RMR_pro1"])
ens = EnsemblePro1(df, params=P["EnsemblePro1"])
```

> **Note:** Set `pair_hint` (e.g., `"HEX/DAI"`) and `nominal_trade_size_dai` to match your live/demo sizing so the cost model picks the right tier.

## Backtesting recipe (suggested)

1. **Data**: Use your real on‑chain PulseX data as per repo tools (5‑minute bars are fine).  
2. **Walk‑Forward**: For 2 years of data, try 60‑day training / 15‑day holdout segments and roll forward.  
3. **Parameter search**: Use the compact `grids` in `params_pro1.json` to avoid overfitting. Penalize results by **trade count × total_bps** when ranking.  
4. **Outlier guard**: Require performance robustness: median segment return and worst‑segment drawdown both acceptable.  
5. **Cost sanity**: Log `diag_cost_bps` and `diag_min_edge_bps` from outputs; verify they match `swap_cost_cache.json` expectation.

### Example CLI (illustrative)

```
# If your repository exposes a runner like validate_strategies.py
python validate_strategies.py \
  --strategies newstrats/strategies_pro1.py:STRATEGIES_PRO1 \
  --params newstrats/params_pro1.json \
  --pair "HEX/DAI" --interval 5m \
  --wf-train 60d --wf-test 15d \
  --metric return_pct --penalize-cost \
  --html html_reports/pro1_wf.html
```

> Adjust flags to match your actual runner. If it autoloads strategies, just import `STRATEGIES_PRO1` in it.

## Why these should improve return % across time

- **Lower churn via cost gates**: Many false entries die at the cost threshold.  
- **Regime selection** reduces fighting the tape: trend tools in trends; reversion tools in ranges.  
- **ATR and time stops** cap adverse drift and stagnation.  
- **Liquidity filter** reduces slippage and failed fills in thin bars.

## Next steps

- Plug in your exact `swap_cost_cache.json` structure into `CostModel` for exact tier math (if needed).  
- Run the walk‑forward pipeline and compare **return %** vs existing strategies.  
- If desired, add *position sizing* (e.g., volatility‑scaled size proportional to `edge_bps / min_edge_bps`, clipped).  
- Consider a **daily model selection** wrapper: pick the best of TAB/AMAC/RMR over the last N days *with cost penalty* and lock it for the next day.

---

*pro1 — first pass delivered with conservative cost handling and regime awareness.*
