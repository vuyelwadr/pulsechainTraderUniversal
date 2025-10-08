# Codex1 Iteration Report – Adaptive Donchian & CSMA Enhancements

## 1. Objective
- Analyse existing HEX PDAI/DAI strategies with the full swap-cost model.
- Deliver materially higher all-period returns (primary metric: total % return) without violating cost assumptions.
- Add clear documentation so future agents can reproduce the improvements using in-repo tooling.

## 2. Changes Implemented
1. **New adaptive mean-reversion module** – `strategies/csma_revert_dynamic_codex1_strategy.py`
   - Deep-dip entry identical to legacy CSMA (`price ≤ SMA(576) × (1−24 %)` and `RSI≤32`).
   - Adds volatility gating (`ATR_ratio ≥ 0.10`), a 35 % gain-triggered 18 % trail, SMA+6 % momentum exit, and −48 % panic stop.
   - Reduces drawdown ~24 pp while keeping trade count minimal (7 trades across the sample).
2. **Codex1 Donchian preset** – `DonchianChampionSupremeCodex1Strategy` inside `strategies/donchian_champion_strategy.py`
   - Tuned defaults (`dd_base=0.14`, `dd_k=0.40`, `gain_weight=0.12`, `dd_min=0.08`, `dd_max=0.45`) discovered via a coarse grid search using the real swap costs.
   - Retains the 11-day breakout/2-day exit logic but clamps early losses and loosens trailing once trades are profitable.
   - Outperforms the previous champion by ~1,800 pp on the full data while trimming drawdown.
3. **Evaluation harness update** – `scripts/evaluate_vost_strategies.py`
   - Registers the new strategies so they are automatically benchmarked with cost-aware returns.
4. **Knowledge base refresh** – `newstrats/local.md`
   - Added detailed documentation for both codex1 strategies, refreshed the comparison table, and updated recent-period commentary.
5. **This report** – `newstrats/codex1.md` to capture methodology, metrics, and reproduction steps.

## 3. Performance Summary (swap-cost inclusive)
| Strategy | 5 k DAI Total Return | 10 k DAI Total Return | 25 k DAI Total Return | Max DD (5 k) | Trades |
|----------|---------------------:|----------------------:|----------------------:|-------------:|-------:|
| CSMARevertDynamicCodex1 | +1,023.7 % | +897.0 % | +624.9 % | −68.4 % | 7 |
| DonchianChampionSupremeCodex1 | **+7,231.8 %** | +4,068.3 % | +825.7 % | −47.0 % | 32 |
| DonchianChampionDynamic (previous best) | +5,434.7 % | +3,046.6 % | +598.8 % | −49.4 % | 32 |

*All numbers taken from `scripts/evaluate_vost_strategies.py` runs using `swap_cost_cache.json`.*

## 4. Validation Commands
Run from repository root (already executed during this iteration):
```bash
python scripts/evaluate_vost_strategies.py --trade-size 5000 --swap-cost-cache swap_cost_cache.json
python scripts/evaluate_vost_strategies.py --trade-size 10000 --swap-cost-cache swap_cost_cache.json
python scripts/evaluate_vost_strategies.py --trade-size 25000 --swap-cost-cache swap_cost_cache.json
```
These commands confirm cost-aware performance across multiple position sizes.

## 5. Files Created / Modified
- `strategies/csma_revert_dynamic_codex1_strategy.py` (new strategy module)
- `strategies/donchian_champion_strategy.py` (added codex1 preset & restored dynamic override)
- `scripts/evaluate_vost_strategies.py` (registered new strategies)
- `newstrats/local.md` (documentation updates)
- `newstrats/codex1.md` (this report)

## 6. Next Steps / Ideas
1. **Regime filter for DonchianSupreme:** Combine ATR drawdown with a slow EMA regime check to mute trades during persistent bear markets.
2. **Position stacking:** Explore fractional re-entries for CSMA variants once the first tranche exits successfully.
3. **Portfolio construction:** Evaluate ensemble allocations (e.g., 60 % DonchianSupreme, 20 % CSMA classic, 20 % CSMA Dynamic) using the CSV exporter to judge correlation and drawdown overlap.
4. **Automation:** Wrap the evaluation commands plus CSV export into a `make analyze` target for quicker validation.
5. **Recent-window tuning:** Re-run the parameter grid quarterly to ensure the codex1 defaults remain dominant as new data accrues.

This document, together with `newstrats/local.md`, should be sufficient for any contributor to reproduce the tuning process and understand the rationale behind the codex1 presets.
