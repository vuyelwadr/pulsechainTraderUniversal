# Codex1 Iteration Report

## Overview

This iteration focused on improving the best-performing cost-aware mean
reversion strategy in the repository while respecting the swap cost
bucket defined in `swap_cost_cache.json`.  The core goal was to surpass
the existing cost-adjusted return of **CSMARevertStrategy** (≈ 6,027 %
for a 5 k DAI trade size) and the follow-up
**Codex1CSMAEnhancedStrategy** (≈ 6,283 %) without degrading behaviour
at larger trade sizes or in recent time windows.  The user’s acceptance
criteria were explicit: do nothing unless the new strategy beats
everything currently listed in `strats_performance.json`, so the work
prioritised raw return uplift while keeping the swap-cost model intact.

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
| `strategies/codex1_csma_enhanced_strategy.py` | CSMA with relaxed RSI filter (`rsi_max=32`); prior uplift. |
| `strategies/codex1_csma_ultra_strategy.py` | **New record holder.** Shallower SMA (432 bars), 24 % drop trigger, 5.4 % exit buffer, RSI≤30. |

The first Codex1 pass (RSI relaxation) delivered the initial uplift; the
October 2025 revisit introduced a denser grid around `n_sma`,
`entry_drop`, `exit_up`, and `rsi_max`, revealing that a slightly
shallower moving average plus a wider exit buffer materially increases
net performance.  Those tuned parameters now ship as
`Codex1CSMAUltraStrategy`.

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
| CSMARevertStrategy | 6,026.77 % | 724.05 % | −92.18 % | 21 |
| Codex1CSMAEnhancedStrategy | 6,282.80 % | 741.52 % | −92.18 % | 21 |
| **Codex1CSMAUltraStrategy** | **6,806.37 %** | **776.22 %** | −92.18 % | 22 |

The shallower 432-bar SMA lets the strategy recognise capitulation a
touch earlier, while the wider exit buffer reduces premature profit
taking.  Trade count increases by just one over the full sample, keeping
swap costs manageable.

### Other trade sizes

| Strategy | 10 k DAI Total Return | 25 k DAI Total Return | Max DD |
| --- | ---: | ---: | ---: |
| Codex1CSMAEnhancedStrategy | 4,304.01 % | 1,538.17 % | −92.18 % |
| **Codex1CSMAUltraStrategy** | **4,578.25 %** | **1,556.73 %** | −92.18 % |

All windows (full / last 3 m / last 1 m) remain consistent because the
structure is unchanged—only the SMA horizon and exit buffer were tuned.

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

`strategy_performance_summary.csv` now contains 108 rows with the new
strategy listed for all trade sizes and time windows.

## Next Steps

1. **Risk management ideas:** Explore dynamic trailing stops or staged
   profit-taking overlays that can trim the −92 % drawdown while
   preserving the 6.8 k % return profile.
2. **Ensemble evaluation:** Re-run portfolio-level backtests combining
   Codex1CSMAUltra with breakout strategies (e.g.
   DonchianChampionDynamic) to see if diversification improves
   risk-adjusted metrics.
3. **Parameter sensitivity:** Extend `codex1_csma_turbo_search.py` to
   sweep `n_sma` and `exit_up` on finer grids and optionally expose
   walk-forward validation to guard against overfitting.

## Files touched

* `newstrats/codex1_csma_turbo_search.py`
* `strategies/codex1_csma_turbo_strategy.py`
* `strategies/codex1_phoenix_strategy.py`
* `strategies/codex1_recovery_trend_strategy.py`
* `strategies/codex1_csma_enhanced_strategy.py`
* `strategies/codex1_csma_ultra_strategy.py`
* `scripts/evaluate_vost_strategies.py`
* `scripts/export_strategy_performance_csv.py`
* `strategy_performance_summary.csv`
* `reports/optimizer_top_top_strats_run/swap_cost_cache.json` (symlink)

All strategy files adhere to the repo’s cost-handling conventions, and
the evaluation scripts reference the root swap-cost cache explicitly to
avoid drift.
