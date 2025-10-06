# Aggressive Grid Progress Notes

## 2025-09-19
- Added optimizer warm-start support (`--initial-sample`, `--acq-funcs`) so seeded restarts can reuse champion parameter vectors and alternate acquisition functions without hacking the GP internals.
- Dropped `scripts/evolutionary_perturbation.py` to spawn Gaussian jitters around top configurations; first 1-generation run (sigma 0.05, 24 children) immediately surfaced a new best child at **18,511.03 DAI**.
- Logged the winning child parameters under `reports/gridTradeAggressive/evo_20250919_120557/` and promoted them to `best_current/`.
- Added surrogate seeding utility (`scripts/surrogate_seed.py`) to train a random forest on historical runs and export high-confidence warm-start vectors; first pass (6k samples, RF depth 12) produced limited lift (train R²≈0.79, test R²≈-0.46) but yielded candidate lists for follow-up warm starts.
- Next actions: run focused 1y sweeps (~300–600 calls) seeded with `warm_start_candidates.json` from the evolutionary batch + surrogate predictions, compare EI/PI/LCB schedules, and update docs after each stage.
- Sanity-checked the pipeline with a 90-call 1y sweep (`--acq-funcs cycle`, warm-start list) — no better balance yet, but confirms the seeded run infrastructure behaves as expected.
- Expanded the evolutionary search (σ=0.04, 48 children × 2 generations) and hit a new record **25,495.17 DAI** (16 trades, max DD ~62%), logged under `evo_20250919_133714/` and promoted to `best_current/`.
- Pushed sigma lower (0.03 with decay 0.55) and ran 3 generations × 60 children, uncovering **29,086.44 DAI** (still 16 trades, DD ~62%). Winner saved in `evo_20250919_134128/` and now defines the warm-start set.
- Tightened again (σ=0.02 → 0.0025) with four generations/72 children and found **40,354.95 DAI** (28 trades, DD ~57%)—notable that the run flipped `allow_bear_breakout_sell` on. Parameters archived in `evo_20250919_134443/` and promoted to `best_current/`.
- Another refinement (σ=0.015 start, 4 generations × 80 children) nudged the ridge higher to **41,649.05 DAI** (28 trades, DD ~57%), keeping the bear breakout enabled and slightly increasing grids.
- Continued annealing (σ=0.012 with 5 generations × 90 children, shrinking to 0.00075) delivered **45,335.16 DAI** with drawdown dropping to ~50% and a modest rise in grids/hold bars.
- Sixth-round anneal (σ=0.008 → 0.001 across 6 generations × 110 children) eked out **45,497.77 DAI**, reinforcing the bear-breakout-on configuration while holding drawdown near 50%.
- Manual refinement pass (trimmed `trail_atr_mult` to 0.7 and `trend_trail_pct` to 0.12) extended trend holds and pushed the record to **49,073.10 DAI** (26 trades, max DD ~50%).
- New evolutionary batch (σ=0.01, 4 generations, 160 children; parent = latest manual vector) discovered **51,093.86 DAI** with similar drawdown, slightly lower atr multiplier (~0.76) and shorter breakout hold (10 bars).

## 2025-09-27
- Added a lightweight bounds-override path to the optimizer (`--bounds-file`). The runner now reads JSON ranges and applies them before constructing the skopt dimensions, so we can deliberately tighten (Stage A) or selectively widen (Stage B) the search box without editing strategy code.
- Dropped the first two profiles under `optimization/bounds/`:
  - `grid_trade_aggressive_stage_a.json` → narrow box around the 40–50 k ridge (min_step_pct 0.03–0.08, breakout_window 70–150, trend_trail_pct 0.10–0.18, etc.).
  - `grid_trade_aggressive_stage_b.json` → builds on Stage A but only widens the breakout/strong-bull controls (hold bars up to 24, slope/gap windows expanded) for the second sweep.
- Usage (single-stage 1y sweep, seeded restart):
  ```bash
  python optimization/runner.py \
    --stage 1y \
    --workers 14 \
    --timeframes 1d \
    --strategies-file optimization/targets/grid_trade_aggressive.json \
    --calls 600 \
    --objective final_balance \
    --initial-sample best_current \
    --bounds-file optimization/bounds/grid_trade_aggressive_stage_a.json \
    --out-dir reports/gridTradeAggressive/stageA_$(date +%Y%m%d_%H%M%S)
  ```
  Repeat with the Stage B JSON once Stage A saturates (and log both runs under `reports/gridTradeAggressive/stageB_*`).
- The warm-start path accepts multiple parameter dictionaries, so we can mix the champion + top children from the latest evolutionary batch when setting up Stage B warm restarts.
- Discovered that a previous quick-and-dirty shell snippet wrote run artefacts into a literal `$base_dir/seed_*` directory (because the here-doc quoted `$base_dir`). These folders are safe to keep as historical outputs but are not consumed by the tooling; new runs land in `reports/gridTradeAggressive/...`.
- Next queue:
  1. Execute Stage A sweeps across ≥3 seeds (`--random-state` varied) and update `docs/aggressive_grid_progress.json` + `best_current/` if 51 k is beaten.
  2. Promote the top Stage A vectors into a manual perturbation batch (reuse `scripts/evolutionary_perturbation.py` with tighter σ) before Stage B.
  3. Run Stage B sweeps (same command, swap bounds JSON) using the best Stage A outputs as warm starts.
  4. If both stages stall < new record, fall back to a brute-force long sweep (≥1200 calls) with `--bounds-file` unset to re-open the wider search.
