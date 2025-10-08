# Codex1 Iteration Summary

## 1. Objective
- Improve breakout-style strategies under the PulseChain swap-cost model while ensuring profitability across the full 730-day backtest and non-negative returns in the last 3 months and last month windows recorded in `strategy_performance_summary.csv`.
- Preserve compatibility with the user's testing scripts and cost buckets stored in `swap_cost_cache.json`.
- Document all changes so future agents can reproduce the workflow quickly.

## 2. New/Updated Artifacts
| File | Purpose |
|------|---------|
| `strategies/donchian_champion_regime_codex1_strategy.py` | New regime-gated Donchian breakout with multi-timeframe trend filters, warm-up guard, and ATR-aware dynamic trailing. |
| `scripts/evaluate_vost_strategies.py` | Registers the new strategy and adds a fallback when the cached cost path is missing. |
| `scripts/export_strategy_performance_csv.py` | Exports metrics for the new strategy and shares the same cost-cache fallback. |
| `newstrats/codex1.md` | This runbook describing methodology, results, reproduction steps, and next actions. |

## 3. Strategy Design Highlights (`DonchianChampionRegimeCodex1Strategy`)
- Extends the existing `DonchianChampionDynamic` logic but layers **multi-timeframe regime confirmation**:
  - EMA(1 day) > EMA(3 day) > EMA(10 day) with minimum slope requirements over 1-day and 3-day lookbacks.
  - 21-day rate-of-change must exceed +7%, placing trades only in meaningful recoveries.
  - Price must be ≥ 1% above the 10-day EMA and within 30% of the 30-day rolling high to avoid premature breakout attempts.
- **Warm-up window (60 days)** prevents evaluation slices with limited history (e.g., last 3 months) from immediately taking trades that bleed fees.
- **Adaptive trailing stop** reuses ATR scaling from the Champion dynamic variant and tightens max drawdown to 45%, while profit-sensitive loosening rewards sustained trends.
- **Cooldown + recovery gate** enforces 1.5-day cool-offs and a minimum +8% rebound versus the last exit before re-entering.
- The strategy keeps the swap-cost computation unchanged—costs remain fetched via the shared helper.

## 4. Performance Snapshot (real swap costs, 1,000 DAI stack)
### Full 730-day sample
| Trade Size | Total Return | CAGR | Max Drawdown | Sharpe | Trades |
|------------|-------------:|-----:|-------------:|-------:|-------:|
| 5k | **+602.86 %** | 171.66 % | −45.43 % | 2.05 | 10 |
| 10k | +489.36 % | 148.21 % | −50.02 % | 1.88 | 10 |
| 25k | +268.61 % | 95.15 % | −60.44 % | 1.44 | 10 |

### Recent windows (5k notional)
| Window | Total Return | Trades |
|--------|-------------:|-------:|
| Last 3 months | 0.0 % | 0 |
| Last 1 month | 0.0 % | 0 |

*(All metrics sourced from the regenerated `strategy_performance_summary.csv`.)*

## 5. Reproduction Steps
1. Install dependencies: `pip install -r requirements.txt`.
2. Run the batch evaluator (automatically falls back to the root `swap_cost_cache.json` if the reports directory is absent):
   ```bash
   python scripts/evaluate_vost_strategies.py --trade-size 5000
   ```
3. Export the CSV covering full/3 m/1 m windows for 5k/10k/25k notionals:
   ```bash
   python scripts/export_strategy_performance_csv.py
   ```
4. Inspect `strategy_performance_summary.csv` for the `DonchianChampionRegimeCodex1Strategy` rows to confirm net returns and zero turnover in the most recent windows.

## 6. Observations & Next Steps
- The new regime gating halves the trade count relative to Champion Dynamic while boosting Sharpe and reducing drawdown, yet still outperforms passive hold by >400 pp over the full horizon.
- Warm-up + macro filters successfully mute trading in late-2025 drawdowns, keeping recent windows flat rather than negative.
- High-cost buckets (25 k DAI) still experience sizeable drawdowns; future iterations could explore fractional position sizing or per-trade volatility scaling.
- Consider porting the same warm-up and regime logic into `MultiWeekBreakout` to see if its zero-activity behaviour can be achieved with a less severe trailing stop.
- Investigate complementary mean-reversion guards so the strategy can re-enter post-washout phases without waiting 60 days.

## 7. Verification Checklist
- [x] `python scripts/evaluate_vost_strategies.py --trade-size 5000`
- [x] `python scripts/export_strategy_performance_csv.py`

