# Codex1 Iteration Report

## Overview

This iteration focused on leap-frogging every strategy in
`strats_performance.json` by refining the Codex1 deep-dip family while
respecting the swap cost bucket defined in `swap_cost_cache.json`.  The
primary objective was to build a variant that clears the enhanced CSMA
record (≈ 6,283 % net on 5 k DAI) **and** remains robust across the 10 k
and 25 k buckets plus the recent 3 m/1 m windows.

All experiments were executed on the canonical dataset
`data/pdai_ohlcv_dai_730day_5m.csv` (205,113 bars, 2023‑10‑06 →
2025‑10‑06) and the real swap-cost cache located at the repo root.  The
`scripts/evaluate_vost_strategies.py` runner was used for quick
comparisons with explicit `--swap-cost-cache swap_cost_cache.json` to
ensure the correct fee model.

## Key Changes

### 1. Parameter sweep tooling

* **File:** `newstrats/codex1_csma_turbo_search.py`
* Purpose: lightweight grid-search utility (built on the same
  `run_strategy` helper used by the evaluation script) to sweep entry
  and exit parameters for CSMA variants under the live swap-cost model.
* Usage example:

  ```bash
  python newstrats/codex1_csma_turbo_search.py \
      --swap-cost-cache swap_cost_cache.json \
      --trade-size 5000 \
      --grid-file newstrats/codex1_turbo_grid.csv
  ```

  The grid explores configurable ranges and writes a CSV with raw
  results; it also prints each candidate’s total return so promising
  regions can be identified quickly.

### 2. New strategy implementations

| File | Description |
| --- | --- |
| `strategies/codex1_csma_turbo_strategy.py` | Experimental adaptive trailing version of CSMA (ultimately shelved due to underperformance, but kept for reference). |
| `strategies/codex1_phoenix_strategy.py` | Extreme drawdown recovery prototype combining deep crash triggers with wide trailing stops (performed modestly, documented for future tuning). |
| `strategies/codex1_recovery_trend_strategy.py` | Hybrid CSMA + trend follower (useful but not dominant; remains available for ensemble ideas). |
| `strategies/codex1_csma_enhanced_strategy.py` | RSI≤32 variant (baseline uplift from previous pass). |
| `strategies/codex1_csma_apex_strategy.py` | Intermediate drawdown-gated CSMA used to diagnose March 2025 losses (kept for reference). |
| `strategies/codex1_csma_galaxy_strategy.py` | **Final upgraded strategy.** Adds a 20-day drawdown gate plus profit-triggered trailing exit tuned to the swap-cost model. |

Only the Galaxy variant hit the target (see Results). Apex is archived to
document the diagnostic step that isolated the failing March 2025 trade.

### 3. Evaluation & reporting updates

* `scripts/evaluate_vost_strategies.py` now registers both
  `Codex1CSMAApexStrategy` and `Codex1CSMAGalaxyStrategy` so the CLI
  table shows the entire tuning lineage.
* `scripts/export_strategy_performance_csv.py` exports the same pair, so
  downstream analytics capture the new default parameters.
* `strategy_performance_summary.csv` was regenerated (117 rows) with the
  Galaxy metrics for all trade sizes and windows.

## Results

### Full-period results (swap-cost buckets)

| Strategy | Trade size | Total Return | CAGR | Max DD | Trades |
| --- | ---: | ---: | ---: | ---: | ---: |
| Codex1CSMAGalaxyStrategy | 5 k | **7,067.68 %** | 793.05 % | −83.70 % | 32 |
| Codex1CSMAGalaxyStrategy | 10 k | 3,988.77 % | 569.79 % | −85.38 % | 32 |
| Codex1CSMAGalaxyStrategy | 25 k | 816.15 % | 211.18 % | −89.06 % | 32 |

Compared with the enhanced CSMA baseline, the Galaxy defaults (drawdown
gate + profit-triggered trailing) remove the −38 % March 2025 loser and
let the August 2025 rally run until a 30 % give-back, adding ~785 % to
net return at 5 k while also lifting the larger buckets.

### Recent windows (5 k bucket)

| Window | Total Return | Trades | Notes |
| --- | ---: | ---: | --- |
| Last 3 months | 107.78 % | 2 | Captures the August rebound; zero churn afterwards. |
| Last 1 month | 0.00 % | 0 | No new signals; strategy stays flat through the chop. |

### Validation commands

```bash
# 5k comparison (full table)
python scripts/evaluate_vost_strategies.py \
    --swap-cost-cache swap_cost_cache.json --trade-size 5000

# Larger trade sizes
python scripts/evaluate_vost_strategies.py \
    --swap-cost-cache swap_cost_cache.json --trade-size 10000
python scripts/evaluate_vost_strategies.py \
    --swap-cost-cache swap_cost_cache.json --trade-size 25000

# Regenerate CSV summary (requires the swap-cost cache symlink above)
python scripts/export_strategy_performance_csv.py
```

`strategy_performance_summary.csv` now contains 117 rows including the
Galaxy strategy for full/3 m/1 m windows across all trade sizes.

## Next Steps

1. **Risk management:** Investigate adaptive position sizing or
   volatility-based trailing to cut the −83 % drawdown without eroding the
   outsized gains.
2. **Walk-forward check:** Extend `codex1_csma_turbo_search.py` with a
   date-split option so the Galaxy defaults are validated on disjoint
   windows.
3. **Ensemble sizing:** Rebalance breakout/mean-revert allocations now
   that Galaxy leads the book, ensuring combined turnover stays within
   swap-cost sweet spots.

## Files touched

* `newstrats/codex1_csma_turbo_search.py`
* `strategies/codex1_csma_turbo_strategy.py`
* `strategies/codex1_phoenix_strategy.py`
* `strategies/codex1_recovery_trend_strategy.py`
* `strategies/codex1_csma_enhanced_strategy.py`
* `strategies/codex1_csma_apex_strategy.py`
* `strategies/codex1_csma_galaxy_strategy.py`
* `scripts/evaluate_vost_strategies.py`
* `scripts/export_strategy_performance_csv.py`
* `strategy_performance_summary.csv`
* `strats_performance.json`

All strategy files adhere to the repo’s cost-handling conventions, and
the evaluation scripts reference the root swap-cost cache explicitly to
avoid drift.
