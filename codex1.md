# PulseChain HEX Strategy Iteration — codex1

## Objective

Improve the repository's cost-aware HEX/DAI trading strategies using the
documented swap-cost buckets while leaving a reproducible paper trail of the
changes introduced by the `codex1` agent. All work references the 5 minute
`pdai_ohlcv_dai_730day_5m.csv` dataset and the fee model stored in
`swap_cost_cache.json` as described in `newstrats/local.md`.

## Key Changes

### 1. Regime-aware Donchian breakout (new)

* Added `strategies/donchian_champion_regime_codex1_strategy.py`, a
  Donchian-breakout derivative that layers three risk filters on top of the
  existing Champion family:
  * fast/slow EMA regime check with slope gating,
  * macro EMA guardrail (14‑day) to avoid structurally bearish phases,
  * ATR-ratio based trailing stop with post-loss cooling-off period (10 days).
  The implementation inherits the bucketed swap-cost semantics enforced by the
  evaluation harness, so results remain compatible with
  `swap_cost_cache.json`.【F:strategies/donchian_champion_regime_codex1_strategy.py†L1-L162】

### 2. Evaluation & reporting updates

* Registered the new strategy with the quick evaluation runner so the
  aggregated console table now highlights the `RegimeCodex1` variant alongside
  existing Champion strategies.【F:scripts/evaluate_vost_strategies.py†L21-L31】【F:scripts/evaluate_vost_strategies.py†L116-L120】
* Extended the CSV export utility with CLI arguments (`--data`, `--swap-cost`
  and `--output`) and appended the new strategy to the structured summary so it
  surfaces automatically in downstream analytics.【F:scripts/export_strategy_performance_csv.py†L1-L58】【F:scripts/export_strategy_performance_csv.py†L68-L108】

## Performance Snapshot (swap-cost adjusted)

All metrics generated with:

```bash
python scripts/evaluate_vost_strategies.py --swap-cost-cache swap_cost_cache.json --trade-size 5000
python scripts/export_strategy_performance_csv.py --swap-cost-cache swap_cost_cache.json
```

| Strategy | Trade Size (DAI) | Total Return | Max DD | Trades | Notes |
|----------|-----------------:|-------------:|-------:|-------:|-------|
| CSMARevert | 5 000 | +5 418 % | −92.2 % | 21 | Still the blow-off crash catcher; huge upside, extreme risk.【db0475†L6-L9】 |
| DonchianChampionDynamic | 5 000 | +4 600 % | −49.0 % | 34 | Prior best breakout baseline (Champion v4).【db0475†L6-L9】 |
| **DonchianChampionRegimeCodex1** | 5 000 | **+403 %** | **−40.9 %** | 22 | New regime-gated variant; trades far less during bearish regimes.【db0475†L6-L10】【F:strategy_performance_summary.csv†L20-L22】 |
| DonchianChampionRegimeCodex1 | 10 000 | +241 % | −49.5 % | 22 | Maintains positive net even with heavier fee bucket.【F:strategy_performance_summary.csv†L21-L22】 |
| DonchianChampionRegimeCodex1 | 25 000 | +21 % | −66.9 % | 22 | Largest trade bucket remains profitable after costs; losses capped by tighter trail.【F:strategy_performance_summary.csv†L22-L22】 |
| DonchianChampionRegimeCodex1 (last 3 mo) | 5 000 | −24.7 % | −28.4 % | 2 | Only two trades triggered; post-loss cooldown prevented further churn in the persistent downtrend.【F:strategy_performance_summary.csv†L53-L55】 |
| DonchianChampionRegimeCodex1 (last 1 mo) | 5 000 | 0 % | 0 % | 0 | Macro filter kept the strategy flat during the most recent selloff.【F:strategy_performance_summary.csv†L86-L88】 |

> **Observation.** The new cooldown logic materially reduces the number of
> post-loss re-entries. Compared with the raw Champion variants it sacrifices
> headline total return but slashes drawdown and recent-period fee bleed.

## How to Reproduce

1. Install dependencies (`pip install -r requirements.txt`).
2. Ensure `data/pdai_ohlcv_dai_730day_5m.csv` and `swap_cost_cache.json`
   are present (default locations).
3. Run the evaluation commands shown above to refresh the console summary and
   CSV report. The CSV now contains 99 rows, including full/3 mo/1 mo slices for
   the new strategy across 5k/10k/25k buckets.【F:strategy_performance_summary.csv†L1-L99】

## Next Iteration Ideas

* Investigate a hybridised approach that mixes CSMA rebounds with RegimeCodex1
  trend legs to achieve positive returns across all rolling windows while
  keeping max drawdown below 50 %.
* Explore adaptive position sizing based on `atr_ratio` so the strategy scales
  down automatically in turbulent regimes, mitigating the remaining 25 k DAI
  drawdown sensitivity.
* Consider layering a macro economic filter (e.g. HEX on-chain inflow or PLS
  strength) to further reduce the −25 % three-month slide without starving the
  strategy of entries during broad market recoveries.

