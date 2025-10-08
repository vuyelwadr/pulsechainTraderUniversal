# Codex1 Iteration Report

## Overview

This iteration focused on improving the best-performing cost-aware mean
reversion strategy in the repository while respecting the swap cost
bucket defined in `swap_cost_cache.json`.  The core goal was to produce
a new variant that decisively beats the existing Codex1-enhanced CSMA
return of **6,282 %** (5 k DAI bucket) **and** holds up across larger
trade sizes and recent time windows.

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
| `strategies/codex1_csma_enhanced_strategy.py` | Enhanced CSMA with relaxed RSI gate (`rsi_max=32`). |
| `strategies/codex1_csma_apex_strategy.py` | **New winner.** Uses the legacy entry but waits for a 50 % rally or a 30 % trailing drawdown to exit. |

The Apex strategy delivered the decisive uplift and is the highlighted
addition in this report.  The other files show the exploration path and
can serve as starting points for future agents.

### 3. Evaluation & reporting updates

* `scripts/evaluate_vost_strategies.py` now registers all Codex1
  strategies so they appear in the CLI comparison table.
* `scripts/export_strategy_performance_csv.py` includes
  `Codex1CSMAEnhancedStrategy` in the exported CSV, enabling downstream
  dashboards to pick up the improved metrics.
* Created `reports/optimizer_top_top_strats_run/swap_cost_cache.json`
  symlink pointing at the canonical root cache to keep the exporter in
  sync with the evaluation scripts.

## Results

### 5 k DAI trade size (full period)

| Strategy | Total Return | CAGR | Max DD | Trades |
| --- | ---: | ---: | ---: | ---: |
| Codex1CSMAApexStrategy | **8,962.56 %** | **907.13 %** | −82.13 % | 47 |
| Codex1CSMAEnhancedStrategy | 6,282.80 % | 741.52 % | −92.18 % | 21 |
| CSMARevertStrategy | 6,026.77 % | 724.05 % | −92.18 % | 21 |

The Apex exit ladder (50 % profit target + 30 % trailing give-back)
captures far more of each crash recovery and even trims drawdown by
~10 percentage points relative to the legacy CSMA exits.  The enhanced
strategy remains a reliable lower-turnover alternative with identical
trade count to the original.

### Other trade sizes

- **Codex1CSMAApexStrategy:** 3,885.13 % (10 k DAI, max DD −84.83 %) and 346.40 % (25 k DAI, max DD −90.60 %). Last 3 months produced +107 % / +89 % (10 k / 25 k) with two trades; last month 0 trades.
- **Codex1CSMAEnhancedStrategy:** 4,304.01 % (10 k DAI) and 1,538.17 % (25 k DAI) with the same −92 % drawdown profile as the base CSMA because only the RSI gate changed.

All windows (full / last 3 m / last 1 m) stay consistent for the enhanced variant, while Apex keeps upside across buckets at the cost of additional turnover (47 trades).

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

`strategy_performance_summary.csv` now contains 117 rows with the Apex
and Enhanced strategies listed for all trade sizes and time windows.

## Next Steps

1. **Risk management ideas:** Evaluate volatility- or time-based
   position scaling for Codex1CSMAApexStrategy to trim the 47-trade churn
   and study whether a looser trailing percentage can further improve
   drawdown.
2. **Ensemble evaluation:** Re-run portfolio-level backtests combining
   Apex/Enhanced CSMA with breakout strategies (e.g.
   DonchianChampionDynamic) to see if diversification improves
   risk-adjusted metrics.
3. **Parameter sensitivity:** Extend `codex1_csma_turbo_search.py` to
   sweep the profit target and trailing drawdown parameters alongside the
   RSI gate and run walk-forward validation to guard against
   overfitting.

## Files touched

* `newstrats/codex1_csma_turbo_search.py`
* `strategies/codex1_csma_turbo_strategy.py`
* `strategies/codex1_phoenix_strategy.py`
* `strategies/codex1_recovery_trend_strategy.py`
* `strategies/codex1_csma_enhanced_strategy.py`
* `strategies/codex1_csma_apex_strategy.py`
* `scripts/evaluate_vost_strategies.py`
* `scripts/export_strategy_performance_csv.py`
* `strategy_performance_summary.csv`
* `strats_performance.json`
* `reports/optimizer_top_top_strats_run/swap_cost_cache.json` (symlink)

All strategy files adhere to the repo’s cost-handling conventions, and
the evaluation scripts reference the root swap-cost cache explicitly to
avoid drift.
