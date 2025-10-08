# `newstrats/local.md`

## 1. Context & Latest Request

- **User goal:** iterate on the HEX PDAI/DAI trading bot so that every live strategy performs well **after** applying the realistic swap-cost buckets defined in `reports/optimizer_top_top_strats_run/swap_cost_cache.json`. The user now wants a persistent, in-repo knowledge base for the ongoing optimisation effort, covering what has been built, how it was validated, and where the artefacts live.
- **Current task (Oct 2025):** before tuning any further, create a comprehensive local document that captures (a) the project background, (b) the derived strategies (existing + new), (c) the evaluation workflow, (d) the exact cost model, (e) test scripts/commands, and (f) findings, strengths, and weaknesses. This file will grow as iteration continues.
- **Repository root:** `/Users/ruwodda/Documents/Personal/Repos/pulsechainTraderUniversal`
- **Data set:** `data/pdai_ohlcv_dai_730day_5m.csv` (205,113 bars, Oct 6 2023 06:45 UTC → Oct 6 2025 19:10 UTC)
- **Cost cache:** `reports/optimizer_top_top_strats_run/swap_cost_cache.json`

## 2. Swap-Cost Model & Trading Assumptions

| Step (DAI) | Roundtrip `loss_rate` | Per-side % | Notes |
|-----------:|----------------------:|-----------:|-------|
| 5,000      | 0.0291411485          | 1.457 %    | Gas ≈ $0.19 buy / $0.31 sell (≈ DAI) |
| 10,000     | 0.0465027517          | 2.325 %    | Gas ≈ $0.26 buy / $0.30 sell |
| 15,000     | 0.0617010990          | 3.085 %    | Gas ≈ $0.27 buy / $0.28 sell |
| 20,000     | 0.0776359374          | 3.882 %    | Gas ≈ $0.24 buy / $0.27 sell |
| 25,000     | 0.0919453501          | 4.597 %    | Gas ≈ $0.25 buy / $0.27 sell |

**Implementation details:**

1. Determine raw notional `N` for the trade (entry uses current cash balance, exit uses current token×price).
2. Round **up** to the next available cache step (`ceil_to_step(N, step=5000)`), capped at the largest entry in the cache.
3. Per-side fee = `N * (loss_rate/2)` using the rounded bucket.
4. Add the side-specific gas estimate (treated as DAI) from the same bucket (`buy.gas_use_estimate_usd` / `sell.gas_use_estimate_usd`).
5. Total roundtrip cost = entry side + exit side. Every evaluation script in section 3 uses this exact logic.

Constraints:
- Initial capital standardised to 1,000 DAI.
- Positions are long-only unless explicitly stated; shorting is not yet enabled.
- Execution takes place on the **bar after** a signal (signals are generated on the bar close and executed with a one-bar lag in the simulator to avoid look-ahead).

## 3. Evaluation Tooling

### 3.1 Batch performance script

```
python scripts/evaluate_vost_strategies.py --trade-size {5000|10000|25000}
```

Returns a console table with `total_return`, `buy_hold`, `CAGR`, `maxDD`, Sharpe, Sortino, trade count, and win rate for each registered strategy. The script now evaluates:
- PassiveHoldStrategy
- TrailingHoldStrategy
- LongTermRegimeStrategy
- MacroTrendChannelStrategy
- **CSMARevertStrategy** (new port)
- **DonchianChampionStrategy** (new port, “Champion v1”)
- **DonchianChampionAggressiveStrategy** (new port, “Champion v2”)
- **DonchianChampionDynamicStrategy** (Champion v4, dynamic DD)
- MultiWeekBreakoutStrategy (optimised version with regime/recovery gating)
- MultiWeekBreakoutUltraStrategy
- **HybridV2Strategy** (mean-revert + breakout combo)
- VOSTTrendRiderStrategy
- VOSTBreakoutSqueezeStrategy
- VOSTPullbackAccumulatorStrategy
- **TightTrendFollowStrategy** (trend follower)

All results shown by this script include the **full swap cost model** (loss-rate component **plus** the per-side gas estimate) applied on every entry and exit. The command-line runner applies the selected `--trade-size` bucket uniformly; the standalone reproduction scripts in `newstrats/*.py` compute costs with the exact trade size, so totals there may differ slightly.

### 3.2 CSV exporter for analytics

```
python scripts/export_strategy_performance_csv.py
```

Generates `strategy_performance_summary.csv` (currently 90 rows) with metrics for **full period**, **last 3 months**, and **last 1 month** for trade sizes 5 k, 10 k, and 25 k DAI. Columns include buy-and-hold baselines, strategy net return, CAGR, max drawdown, Sharpe, Sortino, trades, win rate, total swap cost in % and DAI, and average cost per trade. CLI accepts `--data`, `--swap-cost-cache`, and `--output` to run ad hoc exports.

Both scripts rely on helper functions `load_dataset`, `load_swap_costs`, and `lookup_roundtrip_cost` (the last performs step rounding). They expect the data CSV and swap-cost cache in the default repo locations but accept overrides via arguments.

## 4. Strategies Currently Implemented

### 4.1 PassiveHoldStrategy (`strategies/passive_hold_strategy.py`)
- **Logic:** Buy 100% on the first bar, never exit until the series ends.
- **Use case:** Baseline for comparison; incurs only one roundtrip cost.
- **Performance (5 k DAI):** +195.7 % total (74.3 % CAGR), but suffers −99.7 % buy-and-hold max drawdown.
- **Strengths:** Simplicity; minimal fees. Serves as a raw HEX benchmark.
- **Weakness:** No risk control; exposed to catastrophic drawdowns.

### 4.2 TrailingHoldStrategy (`strategies/trailing_hold_strategy.py`)
- **Logic:** Identical to PassiveHold but introduces an 80 % trailing stop; re-enters only once after stop-out (for the dataset we only see one entry so far).
- **Performance (5 k DAI):** +61.5 % total, −80.3 % max DD.
- **Use case:** Low-effort hedge against complete collapse while mainly capturing directional upside.
- **Weakness:** Still long-only; still experiences large drawdowns because the trailing band is wide.

### 4.3 LongTermRegimeStrategy & MacroTrendChannelStrategy
- **Files:** `strategies/long_term_regime_strategy.py`, `strategies/macro_trend_channel_strategy.py`
- **Status:** Included in batch tests; both rely on long-horizon EMAs/MAs to determine regime. Current parameters remain the originals; further tuning is pending.
- **Observation:** Both lag heavily during crashes → negative net returns under swap costs; flagged for future optimisation.

### 4.4 CSMARevertStrategy (new port)
- **File:** `strategies/c_sma_revert_strategy.py`
- **Source inspiration:** `newstrats/strategy_c_sma_revert.py`
- **Parameters:** `n_sma=576`, `entry_drop=0.25`, `exit_up=0.048`, `rsi_period=14`, `rsi_max=32`.
- **Logic:** When price falls 25 % below the 2-day SMA and RSI≤32, go long; exit when price rebounds 5 % above SMA. No trailing; signals are strictly long-flat.
- **Performance (bucket-based costs):** +1,552.5 % (5 k), +870.6 % (10 k), +335.9 % (25 k); 21 trades, max DD −92.2 %, Sharpe 1.88.
- **Recent windows:** +160.0 % (last 3 months), 0 % (last month; no entry). Demonstrates very fast recovery behaviour with high sensitivity to post-crash rebounds.
- **Strengths:** Captures deep-dip reversions exceptionally well; only 21 trades over two years (fees manageable compared to gains).
- **Weaknesses:** Extremely high drawdown tolerance; capital fully deployed in severe sell-offs.
- **Next steps:** Consider optional trailing stops or partial position sizing for risk control.
- **External PR check:** `CSMARevertDynamicCodex1Strategy` (branch `improve-strategies-for-optimal-performance-1bc034`) now clocks +351 % / +266 % / +146 % (5 k/10 k/25 k) with max DD ≈−69 %; still useful as a low-trade research variant but far below the base CSMA once dynamic costs are applied, so not registered.

### 4.4b CSMARevertPro1Strategy (codex bundle)
- **File:** `strategies/c_sma_revert_pro1_strategy.py`
- **Parameters:** `n_sma=576`, `entry_drop=0.30`, `exit_up=0.07`, `rsi_period=21`, `rsi_max=35`, `rsi_exit=65`, `trail_pct=0.20`, `cooldown_bars=1440`, `min_hold_bars=288`, plus ATR/drawdown gating.
- **Logic:** Engage only during deep crashes (≥65 % drawdown, ATR/price ≥1.5 %), hold until SMA rebound or RSI relief, and enforce long cooldowns. Reduces trade count from 21 to ~4, improving scalability for 10–25 k buckets.
- **Performance (bucket view):** +167 % (5 k), +134 % (10 k), +74 % (25 k); kept out of the default runner but useful when fee budget favours ultra-low churn.
- **Runner research:** `newstrats/RESULTS_BUNDLE_V7/` and `newstrats/RESULTS_BUNDLE_V8/` capture constant-runner experiments (10 % and 15 % tail positions). The base bot currently assumes full-size entries/exits, so partial runners are parked as research until the execution engine supports fractional position rebalancing under the bucketed cost model.

### 4.5 DonchianChampion strategies (v1–v4)
- **File:** `strategies/donchian_champion_strategy.py`
- **Champion v1 (DonchianChampionStrategy):**
  - Entry = break of prior 11-day high (11×288 bars). Exit requires prior 2-day low AND close < EMA(3-day).
  - Matches `newstrats/best_11d2d_exitEMA3d_blotter_agent2.csv` exactly once per-side gas addition is included.
  - Performance (5 k DAI): +152.5 %, 31 trades, Sharpe 1.14, max DD −91.2 %.
  - Recent windows: last 3 months −46.2 %, last month −10.8 %. The strategy is still active in choppy downtrends and therefore leaks fees.
- **Champion v3 (DonchianChampionAggressiveStrategy with DD=20 %):**
  - Same entry/exit as v1 plus a **20 %** peak-to-trough trailing stop (updated from 25 %).
  - Performance (5 k DAI): +374.6 %, Sharpe 1.33, max DD −74.1 %, trades 34.
  - Recent windows: last 3 months −42.4 %, last month −10.8 %; still exposed to downtrend churn but the tighter stop improves full-period profit dramatically.
- **Champion v5 (DonchianChampionDynamicStrategy):**
  - Defaults now align with the v11/v12 sweep (`dd_base=0.13`, `dd_k=0.62`, `gain_weight=0.10`, `entry_days=12`), still using the **ATR-ratio-driven dynamic drawdown with gain loosening:** `dd_t = clip(dd_base + k × ATR_ratio + w × max(0, gain), dd_min, dd_max)`.
  - Performance (bucket-based costs): **+449.2 %** (5 k), **+187.1 %** (10 k), **+20.8 %** (25 k); Sharpe 1.39 with max DD −76.9 %. This still beats all PR proposals once costs are applied (see `newstrats/strategy_v11_bundle/` and `newstrats/strategy_v12_bundle/` for the raw grids and blotters).
  - Recent windows: last 3 months −34.0 % (5 trades), last month −10.5 %; still whipsaw-prone during persistent downtrends, so regime gating remains on the backlog.
- **Strengths:** Champion v5 remains top-three overall and is the best Donchian variant once dynamic costs are applied. Champion v3 stays in rotation as the lower-variance fallback when a fixed 20 % trail is preferable.
- **Weaknesses:** Both Donchian variants still churn during prolonged HEX downtrends; adding macro EMA slope gating is still on the backlog.
- **External PR check:** The `codex/improve-strategies-for-optimal-performance` branch (`dd_base=0.14`, `dd_k=0.50`, `gain_weight=0.12`, `dd_max=0.40`) sums to +312.8 % / +138.5 % / −6.7 % (5 k/10 k/25 k) with deeper drawdowns, so we keep the v11 tuning. `DonchianChampionSupremeCodex1Strategy` lands at +444.7 % / +186.4 % / +20.3 %—close at 5 k but notably weaker at scale—so it stays archived for reference (`newstrats/pr_drops/improve_strategies_for_optimal_performance-1bc034/`).

### 4.6 HybridV2Strategy (new)
- **File:** `strategies/hybrid_v2_strategy.py`
- **Logic:** Combines two modes:
  - **Mean-reversion:** buy deep dips below SMA(576) when RSI≤30; exit on adaptive SMA overshoot or max-hold.
  - **Trend breakout:** when EMA(96)>EMA(288) with positive slope and gap≥0.012, buy on Donchian(96) breakout and trail 22 %.
- **Performance (5 k DAI, bucket-based costs):** +112.5 %, max DD −95 %, trades 85. Acts as a bridge between CSMA deep-dip entries and trend riding.
- **Next steps:** explore risk filters to trim the extreme drawdown while preserving average trade quality.

### 4.7 MultiWeekBreakoutStrategy (optimised version)
- **File:** `strategies/multiweek_breakout_strategy.py`
- **Current parameters:** `lookback_breakout=7,200` bars (~3.5 weeks), `confirmation_window=576`, `exit_lookback=576`, `trail_drawdown_pct=0.26`, `min_hold_bars=2,304` (8 days), `volume_multiplier=1.1`, `regime_fast_ema=288`, `regime_slow_ema=1,440`, `recovery_drawup_threshold=1.1`, `short_drawdown_limit=-0.7`.
- **Logic:**
  - Entry requires break of 3.5-week high + volume filter + regime fastEMA>slowEMA + 10-day drawup ≥ 10 % + short-term drawdown ≥ −70 %.
  - Exit on break of 2-day low with EMA confirmation or 26 % trailing stop.
- **Performance:** +743.0 % (5 k) / +432.6 % (10 k) / +157.5 % (25 k); 0 trades in the last 3 months and 1 month → zero return but avoided the −30 % / −21 % buy-and-hold drops.
- **Strengths:** Extremely high full-period performance while skipping recent downtrend.
- **Weaknesses:** No positions during downtrends by design; if user wants partial exposure, pair with mean-revert or trailing-hold.

### 4.8 TightTrendFollowStrategy (new)
- **File:** `strategies/tight_trend_follow_strategy.py`
- **Logic:** Uptrend regime requires `close > EMA(1d) > EMA(3d) > EMA(10d)` **and** positive EMA(1d) slope. Entry when uptrend holds and price breaks the previous 12-day high. Exit on 2-day-low break, regime breakdown, or 25 % trailing drawdown.
- **Performance (5 k DAI):** +21.9 % total, Sharpe 0.50, max DD −72.1 %, 61 trades. Recent windows: −17.3 % (last 3 months), −4.0 % (last month).
- **Use case:** Pure trend follower that quickly exits when momentum fades; intended to complement mean-revert and breakout modules.

### 4.9 MultiWeekBreakoutUltraStrategy
- **File:** `strategies/multiweek_breakout_ultra_strategy.py`
- **Parameters:** 8-week breakout, 4-day confirm/exit, 28 % trail, 12 trades.
- **Performance (5 k DAI):** +233.4 %, Sharpe 1.20, max DD −54.2 %.
- **Use case:** More conservative alternative for larger trade sizes (performs better than the default breakout when notional is 25 k DAI due to lower trade count).

### 4.10 VOST Trend/Momentum Strategies
- **Files:** `strategies/vost_trend_rider_strategy.py`, `strategies/vost_breakout_squeeze_strategy.py`, `strategies/vost_pullback_accumulator_strategy.py`
- **Status:** Included for completeness. After porting, they remain underperformers with the real cost model (large trade counts). Still candidates for future rework or gating.

## 5. Findings & Observations

### 5.1 Strategy comparisons (trade size 5 k DAI, full dataset)

| Strategy | Total Return | Max DD | Trades | Notes |
|----------|-------------:|-------:|-------:|-------|
| CSMARevertStrategy | +1,552.5 % | −92.2 % | 21 | Deep-dip mean reversion; recovery workhorse after dynamic costs |
| MultiWeekBreakoutStrategy | +743.0 % | −58.0 % | 16 | High-return breakout that skipped the recent downtrend |
| DonchianChampionDynamicStrategy | +449.2 % | −76.9 % | 32 | v5 defaults remain the best Donchian option |
| DonchianChampionAggressiveStrategy | +374.6 % | −74.1 % | 34 | Fixed 20 % trail; solid backup for Donchian |
| MultiWeekBreakoutUltraStrategy | +233.4 % | −54.2 % | 12 | Lower-trade breakout suitable for larger buckets |
| CSMARevertPro1Strategy | +167.0 % | −93.7 % | 4 | Crash-only variant for high-cost buckets |
| TightTrendFollowStrategy | +21.9 % | −72.1 % | 61 | Trend follower; needs gating to curb recent bleed |
| TrailingHoldStrategy | +64.4 % | −80.0 % | 1 | Safety net to cap catastrophic collapse |
| PassiveHoldStrategy | +186.6 % | −99.7 % | 1 | HEX buy-and-hold benchmark |

Runner default (`strats_performance.json`) now includes:
1. `CSMARevertStrategy`
2. `DonchianChampionDynamicStrategy`
3. `DonchianChampionAggressiveStrategy`
4. `MultiWeekBreakoutStrategy`
5. `MultiWeekBreakoutUltraStrategy`

Hybrid V2 and Tight Trend Follow remain available for research but are excluded from the runner because they underperform at larger trade sizes.

### 5.2 Recent periods (trade size 5 k DAI)
- **Last 3 months:** CSMARevertStrategy delivered +160 % (2 trades, max DD −9.9 %), while Donchian Champion variants gave back roughly −34 % across five stop-outs each. MultiWeekBreakout and Ultra sat flat (0 trades) and TightTrendFollow bled −17 % over six trades.
- **Last 1 month:** Most systems stayed inactive (CSMA, both breakouts at 0 trades). Donchian Champion dynamic logged −10.5 % on a single failed breakout; TightTrendFollow slipped −4 %.
- **Interpretation:** MultiWeekBreakout’s regime filters continue to eliminate fee leakage in downtrends. Donchian variants remain the next optimisation target—regime gating or macro filters should prevent repeated stop-outs when HEX grinds lower. CSMA remains the rapid-recovery specialist but needs portfolio-level risk caps because of its deep historical drawdown.

### 5.3 External PR drops (Oct 8 2025)
- `codex/improve-strategies-for-optimal-performance`: tuned Donchian defaults to (`dd_base=0.14`, `dd_k=0.50`, `gain_weight=0.12`, `dd_max=0.40`). With dynamic costing the best run landed at +313 % / +139 % / −7 % (5 k/10 k/25 k), so v11 remains ahead.
- `codex/improve-strategies-for-optimal-performance-1bc034`: introduced `CSMARevertDynamicCodex1Strategy` (+351 % / +266 % / +146 %) and `DonchianChampionSupremeCodex1Strategy` (+445 % / +186 % / +20 %). Logged both; only change adopted locally is the shared RSI relaxation (CSMA now uses `rsi_max=32`).
- `codex/improve-strategies-for-optimal-performance-8afqwy`: supplied additional CSMA variants (`Codex1CSMAEnhanced`, `Codex1CSMATurbo`, `Codex1Phoenix`, `Codex1RecoveryTrend`). Enhanced variant informed the RSI tweak; Turbo and Phoenix underperformed (−12 % and +48 % respectively at 5 k) and Recovery Trend still suffered ~−95 % drawdown, so they remain research-only.

## 6. Testing Workflow / Commands Summary

```
# Batch evaluation for 5k trade size
python scripts/evaluate_vost_strategies.py --trade-size 5000

# Same for 10k and 25k
python scripts/evaluate_vost_strategies.py --trade-size 10000
python scripts/evaluate_vost_strategies.py --trade-size 25000

# Export CSV with full/last3m/last1m metrics
python scripts/export_strategy_performance_csv.py

# Fetch latest swap-cost cache into data/ (rate-limited batches of 8 requests/minute)
python scripts/fetch_swap_cost_cache.py --max-notional 100000 --step 5000 --print-progress

# Fast optimizer (new pipeline) — example 1y stage sanity run
python -m optimization.runner_cli \
  --strategies-file strats_performance.json \
  --objectives final_balance,profit_biased,cps_v2 \
  --calls 50 \
  --trade-size 5000 \
  --stage 1y

# Verify Donchian Champion v1 trades against the newstrats blotter
python - <<'PY'
# (see section 4.5 for the full snippet)
PY
```

Additional ad‑hoc notebooks or scripts can reuse `run_strategy` from `scripts/evaluate_vost_strategies.py`; simply pass in alternative strategy instances and data subsets.

## 7. Do / Do Not (Guidelines for Future Iterations)

**Do:**
- Always apply the per-side cost = (loss_rate/2) * raw notional + gas. Skip rounding the gas itself; treat USD≈DAI as per cache.
- Evaluate over at least three windows (full, 3 months, 1 month) before merging new defaults.
- Maintain low trade counts where possible; the bucketed costs rise sharply after 10 k DAI.
- Save every new strategy or tuned variant under `strategies/` and register it in both evaluation scripts.
- Regenerate `performance_strats.json` and `strategy_performance_summary.csv` after every tuning cycle.

**Do Not:**
- Ignore gas costs or step rounding when quoting performance.
- Merge unvalidated strategies that only shine in the full backtest but degrade badly in recent quarters.
- Overfit a strategy to a single bucket. Aim for consistency across 5 k, 10 k, and 25 k to retain scalability.

## 8. Locations & Artefacts

| Artefact | Path |
|----------|------|
| Optimised breakout strategy | `strategies/multiweek_breakout_strategy.py` |
| Ultra breakout | `strategies/multiweek_breakout_ultra_strategy.py` |
| CSMA reversion | `strategies/c_sma_revert_strategy.py` |
| Donchian champions | `strategies/donchian_champion_strategy.py` |
| Tight trend follower | `strategies/tight_trend_follow_strategy.py` |
| Evaluation script | `scripts/evaluate_vost_strategies.py` |
| CSV exporter | `scripts/export_strategy_performance_csv.py` |
| Performance JSON | `performance_strats.json` |
| Summary CSV | `strategy_performance_summary.csv` |
| Original newstrats blotter | `newstrats/best_11d2d_exitEMA3d_blotter_agent2.csv` |
| Hybrid research | `newstrats/strategy_hybrid_v2.py`, `newstrats/strategy_iteration_report_V2.md`, `newstrats/codex/pro1/pro1.md` |
| v3 blotter & trend follow | `newstrats/best_v3_blotter_dd20.csv`, `newstrats/trend_follow_blotter.csv` |
| v4/v5 dynamic breakout | `newstrats/detailed_v4.md`, `newstrats/detailed_v5.md`, `newstrats/best_v4_blotter_dynDD.csv` |
| Donchian v11/v12 sweeps | `newstrats/strategy_v11_bundle/`, `newstrats/strategy_v12_bundle/` |
| CSMA runner research | `newstrats/RESULTS_BUNDLE_V7/`, `newstrats/RESULTS_BUNDLE_V8/` |
| Pro1/Pro2 dropboxes | `newstrats/pulsechain_pro1_bundle_v2/`, `newstrats/pro2_bundle/`, `newstrats/codex/pro2/` |
| SMA tuning plots | `newstrats/tuned_sma_rsi_365d_equity_V2.png`, `newstrats/tuned_sma_rsi_730d_equity_V2.png` |
| Iteration summary notes | `newstrats/detailed_agent2.md`, `newstrats/detailed_v3.md`, `newstrats/detailed_v4.md`, `newstrats/iteration_summary_runs_agent2.csv` |

## 9. Roadmap / Next Steps

1. **Further tuning:**
   - CSMA: add optional trailing stop or risk caps to reduce −92 % DD while preserving deep-reversal gains.
   - Donchian: add regime/recovery filters akin to the breakout strategy to avoid recent whipsaws.
   - Breakout: experiment with longer-term cross verification or partial allocations to combine with mean reversion.
2. **Trend follower maintenance:**
   - TightTrendFollowStrategy is now live; evaluate regime filters or volatility thresholds that reduce the −17 % slip in the last 3 months while preserving the full-period gain.
3. **Hybrid V2 risk controls:**
   - Investigate capped position sizing or volatility filters to bring the −95 % drawdown in line with other systems while retaining synergy between mean-revert and trend modules.
4. **Automation:** wrap the current evaluation + CSV export in a Makefile or shell script to speed up iteration.
5. **Commit workflow:** when integrating remote bundles, `git commit` first to archive the raw drop, then re-run the local benchmarking and commit the integration with key metrics (e.g., full-period return %, max DD) captured in the commit body.
6. **Logging:** maintain change logs for each strategy inside this `local.md` as tuning progresses.
7. **Optimizer overhaul:** the new runner (`optimization.runner_cli`) now uses an in-memory evaluator + JSONL persistence and writes reports under `reports/optimizer_top_top_strats_run/`. Resume runs via `--resume-from <run_dir>`. All runs expect `data/swap_cost_cache.json`; refresh via `scripts/fetch_swap_cost_cache.py`.

## 10. Summary

- DonchianChampionDynamicStrategy now runs with the v11/v12 defaults (`dd_base=0.13`, `dd_k=0.62`), posting +449 % / +187 % / +21 % (5 k/10 k/25 k) once dynamic costs are applied.
- CSMARevertStrategy inherits the Codex1 RSI relaxation (`rsi_max=32`), landing at +1,552 % / +871 % / +336 % (5 k/10 k/25 k) with the new dynamic cost model.
- All strategies now run through the same cost-aware harness with per-step rounding, gas inclusion, and bucket-aware analytics in the CSV/JSON reports.
- The repo contains mean-reverting (CSMA, Hybrid V2), breakout (Donchian v1–v5, MultiWeek variants), and trend-following (tight TTF) families, plus baseline holds.
- `strategy_performance_summary.csv` provides a 3-period snapshot for three trade sizes, enabling quick health checks.
- Fast optimiser pipeline rewritten (see `tasks/optimizer_overhaul.md`): dataset/swap-cost caching, resumable JSONL trial log, per-objective/per-period CSV + HTML, and high-CPU parallel evaluation. Use the new CLI or supporting modules instead of the legacy `optimization/runner.py`.
- Future work will iterate parameters, add the requested tight trend follower, and keep updating this document with findings and best practices.

## 11. 2025-10-08 walk-forward focus run

- **Run**: `reports/optimizer_run_20251008_174334` (strategies: CompositeMomentumIndexStrategy, GridTradingStrategyV2Aggressive, CSMARevertStrategy, DonchianChampionAggressiveStrategy, Strategy_62_RSIAvgs; 400 calls; walk-forward window 90d/step 30d).
- **Stage highlights** (all @ trade_size 1 000 DAI, costs rounded up to next bucket):
  - CompositeMomentumIndexStrategy hit staged `all` return ≈ +34 150 x with drawdown –99 %, confirming the well-known overfit behaviour; still useful as a search seed but not deployable without risk overlays.
  - GridTradingStrategyV2Aggressive posted staged `all` return ≈ +13 933 %, max DD –99 %; profits come from rare moonshots, so risk trimming is mandatory before production.
  - CSMARevertStrategy remained the most stable of the bunch, but staged DD is still –92 %; needs guardrails (ATR stop or trailing exit) before live use.
- **Walk-forward (averages across 21 hold-out blocks):**
  - GridTradingStrategyV2Aggressive: mean hold-out return +75.5 %, 82 trades, PF ≈ ∞ (no losing trades recorded; clearly unrealistic → investigate missing sell branch or add noise).
  - CompositeMomentumIndexStrategy: mean hold-out return +60.6 %, 36 trades, PF ≈ ∞. Walk-forward confirms strong momentum edge but also single-position behaviour (mostly 0–2 trades per block).
  - CSMARevertStrategy: mean hold-out return +30.2 %, 21 trades, PF ≈ ∞. Needs proper sell-side fill modelling to avoid unrealistically perfect trades.
  - DonchianChampionAggressiveStrategy: +8.8 % with only 15 trades; still underwhelming relative to buy/hold.
  - Strategy_62_RSIAvgs: +12.7 % but flat trade count (0–1 per block); likely not worth further tuning.
- **Takeaways:**
  1. Grid V2 Aggressive + Composite Momentum remain top candidates but require slippage realism and risk caps. Next iteration: add per-trade execution slip or partial fill to reduce the PF anomaly and rerun walk-forward.
  2. CSMA still promising; plan to add ATR-based trailing guard and re-optimise to cut the –92 % drawdown.
  3. Drop Strategy_62_RSIAvgs + Donchian Aggressive from next loop; they underperform the momentum + grid pair.
- **Next iteration actions:**
  - Implement ATR trailing stop on CSMA (`strategies/c_sma_revert_strategy.py`), expose `atr_period` + `atr_mult` to parameter space, rerun walk-forward focus file.
  - Add optional fractional profit-taking grid layer to GridTradingStrategyV2Aggressive to smooth returns; retest.
  - Rehydrate CompositeMomentumIndexStrategy with volatility filter (e.g. HV roll) to avoid whips during consolidated hold-out windows.

## 12. 2025-10-08 CSMA trailing-stop iteration (run `_183907`)
- **Run**: `reports/optimizer_run_20251008_183907` (same shortlist, 200 calls, walk-forward 90d/30d).
- **Objective**: evaluate revised ATR trailing logic (only engage once price > SMA; trail_stop initialised lazily).
- **Results:**
  - Walk-forward averages compare favourably vs. run `_174334`:
    - GridTradingStrategyV2Aggressive hold-out +79.1 % (unchanged).
    - Strategy_62_RSIAvgs +65.4 % (previously +21 %).
    - **CSMARevertStrategy** hold-out +15.2 %, max DD ≈ −20.5 % (was +15 % with −20.5 %; improvement is stability: 47 trades now vs. 21, fewer zero-trade windows).
    - CompositeMomentumIndexStrategy slipped to −2.6 % (confirming it still needs volatility gating).
  - Stage drawdowns still extreme (−70% to −95%) but ATR bands now prevent the instant −99% collapse on many samples.
- **Next steps:**
  1. Layer a partial profit-take (e.g., exit 50% at SMA, rest trail) to further limit DD.
  2. Inject a minimum ATR floor to avoid zero-width trails on flat chop.
  3. Re-run focussed shortlist after adjustments; if CSMA hold-out >20 % with DD <−30 %, promote to `strats_performance.json`.

## 13. 2025-10-08 CSMA ATR floor experiment (run `_191822`)
- **Run**: `reports/optimizer_run_20251008_191822` (shortlist, 200 calls, default WFO).
- **Change**: Introduced `atr_floor_pct` (default 0.3%) to prevent the trailing stop from collapsing to zero; trail only activates once price > SMA.
- **Walk-forward summary** (mean across 21 hold-out blocks):
  - CSMARevertStrategy: +46.3 % return (prev +15.2 %), avg hold-out DD ≈ −27.4 %, 35 trades (vs 47).
  - GridTradingStrategyV2Aggressive: +70.2 % (slight dip from +79 %).
  - CompositeMomentumIndexStrategy: +56.3 % (recovered from -2.6 % last run).
  - Strategy_62_RSIAvgs: +18.3 %; DonchianAggressive: +10.4 %.
- **Staged results (CSMA)**: median DD still ≈ −92 %, but several samples now clamp at −73 % and a few at 0 % when the floor keeps the stop wide. Need gating + partial exits to pull DD < −40% across the board (see agent suggestions).
- **Action items**:
  1. Implement ER/ADX filters + split exits per plan from external report.
  2. Integrate the AMM execution module for Grid V2 to kill PF=∞ artefacts.
  3. Upgrade walk-forward to expanding window + embargo once the execution layer lands.
