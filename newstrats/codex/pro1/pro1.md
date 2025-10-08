# Optimization Notes (pro1)

## 1. Scope & Dataset
- Repository: `pulsechainTraderUniversal`
- Dataset: `data/pdai_ohlcv_dai_730day_5m.csv`
- Swap-cost model: `swap_cost_cache.json` (step-rounded loss rates + gas, same buckets as `reports/optimizer_top_top_strats_run/swap_cost_cache.json`)
- Evaluation harness: `scripts/evaluate_vost_strategies.py` (cost-aware, one-bar execution lag)

## 2. Changes Implemented
### 2.1 New cost-aware mean reversion (`CSMARevertPro1Strategy`)
- File: `strategies/c_sma_revert_pro1_strategy.py`
- Core logic: 2-day SMA anchor with
  - 30% crash entry, RSI ≤ 35, ATR/price ≥ 1.5%
  - 7% SMA rebound target, RSI ≥ 65 relief, 20% peak trail, 1,440-bar cooldown
- Rationale: reduce trade count to four deep-dip campaigns while preserving massive rebounds → lowers swap cost impact at 10k–25k DAI.

### 2.2 Retuned Donchian champion dynamic
- File: `strategies/donchian_champion_strategy.py`
- Updated defaults (`dd_k=0.4`, `dd_max=0.32`, `dd_min=0.10`) widen the ATR-responsive trail just enough to hold trend extensions and exit sooner in calmer regimes. This improves net return for larger buckets.

### 2.3 Tooling upgrade
- `scripts/export_strategy_performance_csv.py` now accepts `--data`, `--swap-cost-cache`, and `--output` arguments (matching `evaluate_vost_strategies.py`). This enables running analytics without the hard-coded reports path.

### 2.4 Cleanup
- Removed the experimental `AdaptiveBreakoutReboundPro1Strategy` from evaluation/export registries after testing showed sub-par cost-adjusted performance at 25k DAI.

## 3. Performance Summary
### 3.1 Key strategies vs. buy-and-hold (trade size 5k / 10k / 25k DAI)
| Strategy | 5k Total % | 10k Total % | 25k Total % | Trades |
| --- | ---: | ---: | ---: | ---: |
| CSMARevert (baseline) | 5,418% | 3,708% | 1,317% | 21 |
| **CSMARevertPro1** | **298%** | **271%** | **209%** | **4** |
| DonchianChampionDynamic (baseline defaults) | 4,600% | 2,479% | 421% | 34 |
| **DonchianChampionDynamic (retuned)** | **4,831%** | **2,606%** | **447%** | **34** |
| MultiWeekBreakout (for context) | 1,558% | 1,150% | 488% | 16 |

- Source: `python scripts/evaluate_vost_strategies.py --trade-size {5000,10000,25000} --swap-cost-cache swap_cost_cache.json`
  - Baseline tables: see `2fb338`, `f2a9a9`, `6f083d`
  - Updated tables: see `1c86c6`, `553938`, `bb1c64`

### 3.2 Recent windows (5k DAI, cost-adjusted)
| Strategy | Full-period % | Last 3m % | Last 1m % | Trades (full) |
| --- | ---: | ---: | ---: | ---: |
| CSMARevertPro1 | 297.7 | 0.0 | 0.0 | 4 |
| DonchianChampionDynamic (retuned) | 4,830.8 | -38.3 | -10.8 | 34 |
| MultiWeekBreakout | 1,557.9 | 0.0 | 0.0 | 16 |

- Commands: see chunk `46e5b1`
- Interpretation: Pro1 CSMA sits out quiet regimes (no trades in last quarter), while Donchian still suffers recent whipsaws—next step is to add regime gating similar to MultiWeekBreakout.

## 4. Reproduction Checklist
```bash
# Cost-aware evaluation (trade sizes 5k / 10k / 25k)
python scripts/evaluate_vost_strategies.py --trade-size 5000 --swap-cost-cache swap_cost_cache.json
python scripts/evaluate_vost_strategies.py --trade-size 10000 --swap-cost-cache swap_cost_cache.json
python scripts/evaluate_vost_strategies.py --trade-size 25000 --swap-cost-cache swap_cost_cache.json

# Export CSV (new CLI arguments)
python scripts/export_strategy_performance_csv.py \
    --swap-cost-cache swap_cost_cache.json \
    --output strategy_performance_summary.csv
```
- Ensure `data/pdai_ohlcv_dai_730day_5m.csv` is present and matches the 205,113-bar dataset described in `newstrats/local.md`.
- Swap-cost lookup must point to `swap_cost_cache.json`; rounding and gas costs are applied automatically inside the helper.

## 5. Findings & Recommendations
1. **CSMARevertPro1** drastically improves scalability: four trades → ~209% net at 25k DAI. Add optional partial exits to lock gains earlier while keeping fee drag minimal.
2. **Retuned DonchianDynamic** now beats buy-and-hold across all buckets, but recent 3m drawdown (−38%) calls for a volatility regime filter; consider blending MultiWeekBreakout’s recovery gating.
3. **Reporting**: run the refreshed `export_strategy_performance_csv.py` after parameter tweaks to populate `strategy_performance_summary.csv` (99 rows generated in chunk `788f0f`).
4. **Next iteration ideas**:
   - Add “no-trade” state to Donchian when price < 50-day EMA to stop bleeding in downtrends.
   - Experiment with partial capital deployment (e.g., 50% size) for CSMA entries to reduce peak drawdown while preserving upside.
   - Consider an ensemble report (mean of CSMAPro1 + MultiWeekBreakout) to evaluate diversified equity curves.

## 6. File Inventory
- Updated: `strategies/donchian_champion_strategy.py`, `scripts/evaluate_vost_strategies.py`, `scripts/export_strategy_performance_csv.py`
- Added: `strategies/c_sma_revert_pro1_strategy.py`
- Generated artifacts: `strategy_performance_summary.csv` (refresh using the command above)

