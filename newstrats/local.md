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
- **Performance (5 k DAI, bucket-based costs):** +6,282.8 % total, 21 trades, max DD −92.2 %, Sharpe 2.32. (`Codex1` RSI relax in PR `improve-strategies-for-optimal-performance-8afqwy` added ~256 pp net gain across all buckets.)
- **Recent windows:** +213.4 % (last 3 months), 0 % (last month; no entry). Demonstrates very fast recovery behaviour with high sensitivity to post-crash rebounds.
- **Strengths:** Captures deep-dip reversions exceptionally well; only 21 trades over two years (fees manageable compared to gains).
- **Weaknesses:** Extremely high drawdown tolerance; capital fully deployed in severe sell-offs.
- **Next steps:** Consider optional trailing stops or partial position sizing for risk control.
- **External PR check:** `CSMARevertDynamicCodex1Strategy` (branch `improve-strategies-for-optimal-performance-1bc034`) posted +1,023 % / +897 % / +625 % with max DD ≈−69 %; useful as a low-trade research variant but still far below the base CSMA and Donchian leaders, so not registered.

### 4.4b CSMARevertPro1Strategy (codex bundle)
- **File:** `strategies/c_sma_revert_pro1_strategy.py`
- **Parameters:** `n_sma=576`, `entry_drop=0.30`, `exit_up=0.07`, `rsi_period=21`, `rsi_max=35`, `rsi_exit=65`, `trail_pct=0.20`, `cooldown_bars=1440`, `min_hold_bars=288`, plus ATR/drawdown gating.
- **Logic:** Engage only during deep crashes (≥65 % drawdown, ATR/price ≥1.5 %), hold until SMA rebound or RSI relief, and enforce long cooldowns. Reduces trade count from 21 to ~4, improving scalability for 10–25 k buckets.
- **Performance (bucket view):** +298 % (5 k), +271 % (10 k), +209 % (25 k); kept out of the default runner but useful when fee budget favours ultra-low churn.
- **Runner research:** `newstrats/RESULTS_BUNDLE_V7/` and `newstrats/RESULTS_BUNDLE_V8/` capture constant-runner experiments (10 % and 15 % tail positions). The base bot currently assumes full-size entries/exits, so partial runners are parked as research until the execution engine supports fractional position rebalancing under the bucketed cost model.

### 4.5 DonchianChampion strategies (v1–v4)
- **File:** `strategies/donchian_champion_strategy.py`
- **Champion v1 (DonchianChampionStrategy):**
  - Entry = break of prior 11-day high (11×288 bars). Exit requires prior 2-day low AND close < EMA(3-day).
  - Matches `newstrats/best_11d2d_exitEMA3d_blotter_agent2.csv` exactly once per-side gas addition is included.
  - Performance (5 k DAI): +587.9 %, 31 trades, Sharpe 1.59, max DD −88.0 %.
  - Recent windows: last 3 months −46.2 %, last month −10.8 %. The strategy is still active in choppy downtrends and therefore leaks fees.
- **Champion v3 (DonchianChampionAggressiveStrategy with DD=20 %):**
  - Same entry/exit as v1 plus a **20 %** peak-to-trough trailing stop (updated from 25 %).
  - Performance (5 k DAI): +3,668.3 %, Sharpe 2.54, max DD −49.4 %, trades 34.
  - Recent windows: last 3 months −42.4 %, last month −10.8 %; still exposed to downtrend churn but the tighter stop improves full-period profit dramatically.
- **Champion v5 (DonchianChampionDynamicStrategy):**
  - Defaults now align with the v11/v12 sweep (`dd_base=0.13`, `dd_k=0.62`, `gain_weight=0.10`, `entry_days=12`), still using the **ATR-ratio-driven dynamic drawdown with gain loosening:** `dd_t = clip(dd_base + k × ATR_ratio + w × max(0, gain), dd_min, dd_max)`.
  - Performance (bucket-based costs): **+7,288.5 %** (5 k), **+4,099.8 %** (10 k), **+832.2 %** (25 k); Sharpe improves to 2.90 and full-period max DD tightens to −44.3 %. This is a +34 % lift over the prior defaults at every bucket (see `newstrats/strategy_v11_bundle/` and `newstrats/strategy_v12_bundle/` for the raw grids and blotters).
  - Recent windows: last 3 months −32.2 % (5 trades), last month −11.0 %; still whipsaw-prone during persistent downtrends, so regime gating remains on the backlog.
- **Strengths:** Champion v5 now tops every bucket (5 k/10 k/25 k) while keeping drawdown under 45 %, so it replaces the older defaults in both the runner and analytics. Champion v3 stays in rotation as the lower-variance fallback when a fixed 20 % trail is preferable.
- **Weaknesses:** Both Donchian variants still churn during prolonged HEX downtrends; adding macro EMA slope gating is still on the backlog.
- **External PR check:** The `codex/improve-strategies-for-optimal-performance` branch proposed `dd_base=0.14`, `dd_k=0.50`, `gain_weight=0.12`, `dd_max=0.40`. Cost-aware replays showed +7,148 % / +4,021 % / +815 % (5 k/10 k/25 k) with deeper drawdowns (−47 % to −62 %) versus our v11 defaults (+7,289 % / +4,100 % / +832 %, max DD −44 %). Logged the drop in `newstrats/pr_drops/improve_strategies_for_optimal_performance/` and kept the superior v11 tuning.
- **External PR check:** `DonchianChampionSupremeCodex1Strategy` (branch `improve-strategies-for-optimal-performance-1bc034`) matches the first PR’s performance (≈+7,232 % at 5 k) but still trails our v11 defaults by 40–60 pp across buckets, so it remains archived for reference only (`newstrats/pr_drops/improve_strategies_for_optimal_performance-1bc034/`).

### 4.6 HybridV2Strategy (new)
- **File:** `strategies/hybrid_v2_strategy.py`
- **Logic:** Combines two modes:
  - **Mean-reversion:** buy deep dips below SMA(576) when RSI≤30; exit on adaptive SMA overshoot or max-hold.
  - **Trend breakout:** when EMA(96)>EMA(288) with positive slope and gap≥0.012, buy on Donchian(96) breakout and trail 22 %.
- **Performance (5 k DAI, bucket-based costs):** +620.3 %, max DD −95 %, trades 85. Acts as a bridge between CSMA deep-dip entries and trend riding.
- **Next steps:** explore risk filters to trim the extreme drawdown while preserving average trade quality.

### 4.7 MultiWeekBreakoutStrategy (optimised version)
- **File:** `strategies/multiweek_breakout_strategy.py`
- **Current parameters:** `lookback_breakout=7,200` bars (~3.5 weeks), `confirmation_window=576`, `exit_lookback=576`, `trail_drawdown_pct=0.26`, `min_hold_bars=2,304` (8 days), `volume_multiplier=1.1`, `regime_fast_ema=288`, `regime_slow_ema=1,440`, `recovery_drawup_threshold=1.1`, `short_drawdown_limit=-0.7`.
- **Logic:**
  - Entry requires break of 3.5-week high + volume filter + regime fastEMA>slowEMA + 10-day drawup ≥ 10 % + short-term drawdown ≥ −70 %.
  - Exit on break of 2-day low with EMA confirmation or 26 % trailing stop.
- **Performance:** +1,560.6 % (5 k) / +1,150.8 % (10 k) / +488.5 % (25 k), with 0 trades in last 3 months and 1 month → zero return but avoided the −30 % / −21 % buy-and-hold drops.
- **Strengths:** Extremely high full-period performance while skipping recent downtrend.
- **Weaknesses:** No positions during downtrends by design; if user wants partial exposure, pair with mean-revert or trailing-hold.

### 4.8 TightTrendFollowStrategy (new)
- **File:** `strategies/tight_trend_follow_strategy.py`
- **Logic:** Uptrend regime requires `close > EMA(1d) > EMA(3d) > EMA(10d)` **and** positive EMA(1d) slope. Entry when uptrend holds and price breaks the previous 12-day high. Exit on 2-day-low break, regime breakdown, or 25 % trailing drawdown.
- **Performance (5 k DAI):** +320.1 % total, Sharpe 1.41, max DD −47.9 %, 61 trades. Recent windows: −4.0 % (last 3 months), 0 % (last month; no entry).
- **Use case:** Pure trend follower that quickly exits when momentum fades; intended to complement mean-revert and breakout modules.

### 4.9 MultiWeekBreakoutUltraStrategy
- **File:** `strategies/multiweek_breakout_ultra_strategy.py`
- **Parameters:** 8-week breakout, 4-day confirm/exit, 28 % trail, 12 trades.
- **Performance (5 k DAI):** +354.3 %, Sharpe 1.42, max DD −53.9 %.
- **Use case:** More conservative alternative for larger trade sizes (performs better than the default breakout when notional is 25 k DAI due to lower trade count).

### 4.10 VOST Trend/Momentum Strategies
- **Files:** `strategies/vost_trend_rider_strategy.py`, `strategies/vost_breakout_squeeze_strategy.py`, `strategies/vost_pullback_accumulator_strategy.py`
- **Status:** Included for completeness. After porting, they remain underperformers with the real cost model (large trade counts). Still candidates for future rework or gating.

## 5. Findings & Observations

### 5.1 Strategy comparisons (trade size 5 k DAI, full dataset)

| Strategy | Total Return | Max DD | Trades | Notes |
|----------|-------------:|-------:|-------:|-------|
| DonchianChampionDynamicStrategy | +7,288.5 % | −44.3 % | 32 | v5 defaults (`dd_base=0.13`, `dd_k=0.62`) now lead every bucket |
| CSMARevertStrategy | +6,282.8 % | −92.2 % | 21 | Deep-dip mean reversion; RSI≤32 tweak adds ~256 pp without altering drawdown |
| DonchianChampionAggressiveStrategy | +3,668.3 % | −49.4 % | 34 | Fixed 20 % trail; slightly lower return but steadier than v1 |
| MultiWeekBreakoutStrategy | +1,557.9 % | −60.6 % | 16 | High-return breakout that fully avoids recent downtrend fees |
| HybridV2Strategy | +620.3 % | −95.0 % | 85 | Research combo of reversion + breakout; keep out of runner for now |
| MultiWeekBreakoutUltraStrategy | +353.7 % | −53.9 % | 12 | Low-trade breakout, better scaling for 25 k bucket |
| DonchianChampionStrategy | +587.9 % | −88.0 % | 31 | Legacy v1 baseline; high drawdown and fee leakage in chop |
| TightTrendFollowStrategy | +320.1 % | −47.9 % | 61 | Trend follower; flat-to-negative for larger trade sizes |
| TrailingHoldStrategy | +61.5 % | −80.3 % | 1 | Safety net to cap catastrophic collapse |
| PassiveHoldStrategy | +195.7 % | −99.7 % | 1 | HEX buy-and-hold benchmark |

Runner default (`strats_performance.json`) now includes:
1. `CSMARevertStrategy`
2. `DonchianChampionDynamicStrategy`
3. `DonchianChampionAggressiveStrategy`
4. `MultiWeekBreakoutStrategy`
5. `MultiWeekBreakoutUltraStrategy`

Hybrid V2 and Tight Trend Follow remain available for research but are excluded from the runner because they underperform at larger trade sizes.

### 5.2 Recent periods (trade size 5 k DAI)
- **Last 3 months:** CSMARevertStrategy delivered +211 % (2 trades, max DD −9.9 %), while Donchian Champion variants gave back −32 % to −39 % across five stop-outs each. MultiWeekBreakout and Ultra sat flat (no trades) and TightTrendFollow bled −17 % over six trades.
- **Last 1 month:** Most systems stayed inactive (CSMA, both breakouts at 0 trades). Donchian Champion dynamic logged −11 % on a single failed breakout; TightTrendFollow slipped −4 %.
- **Interpretation:** MultiWeekBreakout’s regime filters continue to eliminate fee leakage in downtrends. Donchian variants remain the next optimisation target—regime gating or macro filters should prevent repeated stop-outs when HEX grinds lower. CSMA remains the rapid-recovery specialist but needs portfolio-level risk caps because of its deep historical drawdown.

### 5.3 External PR drops (Oct 8 2025)
- `codex/improve-strategies-for-optimal-performance`: tuned Donchian defaults to (`dd_base=0.14`, `dd_k=0.50`, `gain_weight=0.12`, `dd_max=0.40`). Archived metrics (+7,148 % at 5 k) but kept v11 defaults that deliver +7,289 % and marginally tighter drawdowns.
- `codex/improve-strategies-for-optimal-performance-1bc034`: introduced `CSMARevertDynamicCodex1Strategy` (+1,024 % at 5 k, −68 % DD) and `DonchianChampionSupremeCodex1Strategy` (+7,232 % at 5 k). Logged both; only change adopted locally is the shared RSI relaxation (CSMA now uses `rsi_max=32`).
- `codex/improve-strategies-for-optimal-performance-8afqwy`: supplied additional CSMA variants (`Codex1CSMAEnhanced`, `Codex1CSMATurbo`, `Codex1Phoenix`, `Codex1RecoveryTrend`). Enhanced variant informed the RSI tweak; Turbo and Phoenix underperformed (−12 % and +48 % respectively at 5 k) and Recovery Trend suffered −95 % drawdown, so they remain research-only.

## 6. Testing Workflow / Commands Summary

```
# Batch evaluation for 5k trade size
python scripts/evaluate_vost_strategies.py --trade-size 5000

# Same for 10k and 25k
python scripts/evaluate_vost_strategies.py --trade-size 10000
python scripts/evaluate_vost_strategies.py --trade-size 25000

# Export CSV with full/last3m/last1m metrics
python scripts/export_strategy_performance_csv.py

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

## 10. Summary

- DonchianChampionDynamicStrategy now runs with the v11/v12 defaults (`dd_base=0.13`, `dd_k=0.62`), improving net return by ~34 % across all trade-size buckets (7,288 % @ 5 k).
- CSMARevertStrategy inherits the Codex1 RSI relaxation (`rsi_max=32`), lifting the 5 k bucket to +6,283 % (previously +6,027 %) while leaving trade count and drawdown unchanged.
- All strategies now run through the same cost-aware harness with per-step rounding, gas inclusion, and bucket-aware analytics in the CSV/JSON reports.
- The repo contains mean-reverting (CSMA, Hybrid V2), breakout (Donchian v1–v5, MultiWeek variants), and trend-following (tight TTF) families, plus baseline holds.
- `strategy_performance_summary.csv` provides a 3-period snapshot for three trade sizes, enabling quick health checks.
- Future work will iterate parameters, add the requested tight trend follower, and keep updating this document with findings and best practices.
