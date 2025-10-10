# SegmentTrendHold Strategy Iteration Log  
_Updated: 2025-10-09_

This document tracks the ongoing effort to make `SegmentTrendHoldStrategy` consistently outperform buy-and-hold across all walk-forward windows. It records the tooling that powers fast iteration, the parameter sweeps we have already executed, the configurations under review, and the open items that still block a definitive commit.

---

## 1. Tooling & Fast-Iteration Workflow

### 1.1 New Utilities

- `scripts/debug_segment_window.py`  
  Runs a single strategy over an arbitrary `[start, end)` window from the trend-state datasets. It prints summary metrics, generates a price chart with buy/sell markers plus EMA overlays, and (optionally) saves the full signals/NAV history as CSV.  
  Example:
  ```bash
  python scripts/debug_segment_window.py \
    --strategy SegmentTrendHoldStrategy \
    --timeframe 1h \
    --start 2025-05-01T00:00:00Z \
    --end   2025-10-01T00:00:00Z \
    --cost 0.015 \
    --params '{"confirm_grace_bars":7, ...}' \
    --save-signals
  ```
- `scripts/random_search_segment_hold.py`  
  Lightweight random-sweep runner that reuses the simple simulator. It evaluates hundreds of parameter combinations across user-specified windows and dumps the results to CSV for quick filtering.

### 1.2 Data & Window Conventions

- Base dataset: `analysis/trend_states_1h.csv` (trend-aligned OHLCV + regime labels).
- Swap-cost anchor: fixed 1.5 % round-trip cost (slippage + gas).
- Windows used to date:
  - **Legacy span** (4 windows):  
    `2024-07-01:2024-12-01`, `2024-12-01:2025-04-01`, `2025-04-01:2025-07-15`, `2025-07-15:2025-10-01`
  - **Recent focus** (4 windows):  
    `2024-10-01:2025-02-01`, `2024-12-01:2025-04-01`, `2025-02-01:2025-06-01`, `2025-04-01:2025-08-30`
  - **Union (mixed)** (7 windows):  
    `2024-07-01:2024-12-01`, `2024-10-01:2025-02-01`, `2024-12-01:2025-04-01`, `2025-02-01:2025-06-01`, `2025-04-01:2025-07-15`, `2025-04-01:2025-08-30`, `2025-07-15:2025-10-01`

All sweeps produce CSVs under `reports/`, e.g. `reports/segment_hold_random_search.csv`, `reports/segment_hold_random_search_recent.csv`, and `reports/segment_hold_random_search_union.csv`.

---

## 2. Parameter Sweeps & Findings

| Sweep (CSV) | Window Count | Iterations | Best Mean Excess | Best Min Excess | Notes |
|-------------|--------------|------------|------------------|-----------------|-------|
| `segment_hold_random_search.csv` | 4 | 300 | +99.8 pp | +29.1 pp | Strong on legacy mix; first hint that low entry threshold + higher trailing ATR works. |
| `segment_hold_random_search_recent.csv` | 4 | 400 | +228.4 pp | +88.97 pp | All sampled configs with `confirm_grace_bars ∈ [2,17]`, `trailing_atr_mult ≥ 1.0` produced positive min excess. |
| `segment_hold_random_search_union.csv` | 7 | 250 | +168.0 pp | +29.15 pp | Blend of legacy and recent windows; highlighted the configuration that is now the updated default. |
| `segment_hold_random_search_recent_focus.csv` | 3 | 300 | +126.2 pp | +40.83 pp | Narrow sampling around high momentum + high confirmation; produces the “focus” config below. |
| `segment_hold_random_search_fullspan.csv` | 6 | 250 | +104.7 pp | _negative_ | No positive min excess; confirms need for additional risk-off rules pre-2024. |
| `segment_hold_random_search_recent_focus.csv` | 3 | 300 | +126.2 pp | +40.83 pp | Narrow sampling around high momentum + high confirmation; produces the “focus” config below. |
| `segment_hold_random_search_fullspan.csv` | 6 | 250 | +104.7 pp | _negative_ | No positive min excess; confirms need for additional risk-off rules pre-2024. |

No configuration from the **full-span sweep** (including early 2024 crash windows) achieved a positive minimum excess return; additional regime/risk filters are required there.

---

## 3. Configurations Under Evaluation

### 3.1 “Recent Momentum” (Historical Best on Last-Year Slice)
From `segment_hold_random_search_recent.csv`.  
**Parameters** (`tmp/segment_params_recent.json`):
```json
{
  "allow_early_entry": true,
  "confirm_grace_bars": 7,
  "min_confirm_strength": 0.06461935848319866,
  "early_momentum_z": 0.4654094912346972,
  "early_pullback_max": 0.04034816058056017,
  "entry_strength_threshold": 0.0936550281387982,
  "exit_strength_threshold": -0.051533027162861136,
  "trailing_atr_mult": 1.3503937010960079,
  "trendline_buffer_pct": 0.049987492364156255,
  "trendline_atr_factor": 0.5731618379809035,
  "reentry_cooldown_bars": 6
}
```
- **Walk-forward (last 9 windows, Oct 2024→Sep 2025)**: wins 8/9, mean excess ≈ +124 pp, min excess ≈ –36.8 pp.
- **Full 21-window walk-forward**: wins 14/21, mean excess ≈ –48.6 pp (dragged by early-2024 regime).
- CLI optimizer (analysis data, stages 30 d / 90 d / 1 y, `--disable-walk-forward`) mirrors the full-run stats and confirms strong recent-stage performance.

### 3.2 “Blended Default” (Current `SegmentTrendHoldStrategy` defaults)
From `segment_hold_random_search_union.csv`.  
**Parameters** (now baked into the strategy defaults):
```json
{
  "allow_early_entry": true,
  "confirm_grace_bars": 11,
  "min_confirm_strength": 0.19967431707055017,
  "early_momentum_z": 0.25108415845293786,
  "early_pullback_max": 0.04639476062932798,
  "entry_strength_threshold": 0.011944479278776855,
  "exit_strength_threshold": -0.23271031552437793,
  "trailing_atr_mult": 1.954622467121972,
  "trendline_buffer_pct": 0.021965659628880856,
  "trendline_atr_factor": 0.22622007515703066,
  "reentry_cooldown_bars": 7
}
```
- **Walk-forward (last year)**: matches the “recent momentum” results (win rate 8/9, mean excess ≈ +124 pp).
- **Full walk-forward**: currently unchanged from the recent configuration (wins 14/21, mean excess ≈ –48.6 pp). The longer grace period and lower entry threshold did not yet fix the early-2024 drawdown.
- CLI optimizer: see `reports/optimizer_run_20251009_200647/` for stage-by-stage outputs.

### 3.3 “Recent Focus” (High-Momentum Confirmation)
From `segment_hold_random_search_recent_focus.csv`.  
**Parameters** (`tmp/segment_params_focus.json`):
```json
{
  "allow_early_entry": true,
  "confirm_grace_bars": 17,
  "min_confirm_strength": 0.38731532051667106,
  "early_momentum_z": 0.6337100946049572,
  "early_pullback_max": 0.05585262637073504,
  "entry_strength_threshold": 0.04879832036102522,
  "exit_strength_threshold": -0.48090651871652795,
  "trailing_atr_mult": 1.7590285353169286,
  "trendline_buffer_pct": 0.01039407357930486,
  "trendline_atr_factor": 0.3056873478333785,
  "reentry_cooldown_bars": 4
}
```
- **Walk-forward (last year)**: wins 8/9 windows (mean excess ≈ +140 pp; min ≈ –124 pp on Aug‑Nov 2025 window).
- **Full history**: wins 14/21 windows (mean excess ≈ –54 pp). Early 2024 still draws down hard without additional risk-off logic.
- Additional 2024-only run (`reports/wf_segment_focus_2024.csv`) shows mean excess ≈ +50 pp but min ≈ –685 pp, reinforcing the need to cap exposure during crash regimes.

All underlying artefacts (plots, CSVs, HTML reports) are preserved under `reports/` for auditing.

---

## 4. Visual Diagnostics

- Debug charts with signal overlays:  
  - `reports/debug_segment/SegmentTrendHoldStrategy_1h_trades.png` (May–Oct 2025, blended config).  
  - `reports/debug_segment_recent/SegmentTrendHoldStrategy_1h_trades.png` (Apr–Sep 2025, recent config).

These confirm that newer parameter sets enter near the emerging rallies and exit before the worst of the spikes, in contrast to the earlier strategy which lagged both entry and exit.

---

## 5. Open Issues & Next Experiments

1. **Full-span underperformance**  
   - No parameter set has positive min excess when early-2024 crash windows are included. Need a “risk-off” gate: e.g., require `trend_strength_score > 0` for fast timeframe or enforce capital scaling inversely with trailing volatility.
2. **Regime-aware capital throttling**  
   - Integrate high-level regime state (e.g., `analysis/trend_segments_*`) so positions are capped during prolonged downtrends.
3. **Automated staging**  
   - Extend `optimization/runner_cli.py` to accept per-stage parameter overrides (seeded from the best configs) to reduce duplication when sweeping across multiple time horizons.
4. **Documentation cadence**  
   - Continue updating this log after each sweep/major run. The current results cover data through 2025-10-01.

---

## 6. Artifacts & References

- Walk-forward CSVs:
- `reports/wf_segment_union_lastyear.csv` (blended default, last year slice).
- `reports/wf_segment_union_full.csv` (blended default, full 21 windows).
- `reports/wf_segment_recent_lastyear.csv` (recent momentum config).
- Trend-state comparison tables:
- `reports/wf_segment_baseline_vs_v2.csv`
- `reports/wf_mtt_baseline_vs_v2.csv`
- Walk-forward trade sizing comparison:
- `reports/wf_segment_v2_trade_comparison.csv`

## 7. Trend-State Generator Comparison (Baseline vs V2)

We regenerated the trend states with the adaptive multi-timeframe pipeline (`analysis_v2/`) and reran the 120-day walk-forward harness using a fixed `--cost 0.015` (1.5 % round-trip swap cost + fees). That rate is now the global assumption for every benchmark; deviate only if real PulseX execution proves cheaper.

- **SegmentTrendHoldStrategy (`reports/wf_v2_segment.csv`)**  
  - Aggregate: mean excess moved from −46.01 pp (baseline) to −45.49 pp (v2), but the per-window win rate collapsed from 14/21 (67 %) to 4/21 (19 %).  
  - Improvements: the crash window starting 2024-08-01 tightened the drawdown from −1277 pp to −70 pp (+1207 pp delta) while cutting trades from 12 to 2.  
  - Regressions: the window starting 2024-09-30 flipped from +327 pp to −367 pp excess (−694 pp delta) as the stricter filters skipped the early breakout sequence.  
  - Takeaway: the new trend gating removes catastrophic downside in one segment but disengages from several profitable bursts; we need refined re-entry logic before making it default.

- **MultiTimeframeTrendStrategy (`reports/wf_v2_mtt.csv`)**  
  - Aggregate: mean excess slid from −101.67 pp to −310.43 pp with every window showing lower excess (delta range −39 pp → −611 pp). Win rate fell from 13/21 (62 %) to 6/21 (29 %).  
  - Volume: total trades roughly doubled in most folds (e.g., 36 → 79 in the first window) yet NAV still lagged, signalling that the new micro filters fire frequently but without enough edge.
  - Takeaway: this strategy regresses outright under the v2 states; defer any rollout until we throttle the fast-timeframe noise triggers.

Action items created from this comparison:
1. Restore a positive win rate for SegmentTrendHold by reintroducing selective early entries when parent regimes are trending but child slopes are flat.  
2. Add volatility-aware gating for MultiTimeframeTrendStrategy so the higher trade count translates into quality signals rather than churn.  
3. Keep v2 data side-by-side with `analysis_baseline/` until both strategies beat buy-and-hold across 30/90/180/365-day slices.
- `reports/wf_segment_focus_lastyear.csv` / `wf_segment_focus_full.csv` (recent-focus sweep top config).
- Optimizer runs:
  - `reports/optimizer_run_20251009_195610/` (stages 30 d, 90 d, 1 y with blended defaults).
  - `reports/optimizer_run_20251009_195501/` (single-stage validation using trend-state data feed).
- Parameter sweep outputs:
  - `reports/segment_hold_random_search.csv`
  - `reports/segment_hold_random_search_recent.csv`
  - `reports/segment_hold_random_search_union.csv`

Future updates to this document should append new sections or tables so the iteration history remains auditable end-to-end.

## 8. Tuned SegmentTrendHoldStrategy (analysis trend set, trade 137 %)

- Trend-state generator: slope/net thresholds relaxed across 1 h/2 h/4 h while keeping the panic-loss guard. The regenerated dataset lives in `analysis/`.
- Parameter bundle from the latest 400-sample sweep (`reports/segment_hold_analysis_best_params.json`):
  ```json
  {
    "allow_early_entry": true,
    "confirm_exit": true,
    "confirm_grace_bars": 9,
    "confirm_timeframe_minutes": 240,
    "early_momentum_z": 0.17362167418726715,
    "early_pullback_max": 0.10317802408750791,
    "entry_strength_threshold": 0.009400508241926309,
    "exit_delay_bars": 6,
    "exit_strength_threshold": -0.43061834828609735,
    "min_confirm_strength": 0.15018875271979584,
    "reentry_cooldown_bars": 5,
    "require_confirm": false,
    "timeframe_minutes": 60,
    "trade_amount_pct": 1.373680908273228,
    "trailing_atr_mult": 2.05336449700627,
    "trendline_atr_factor": 0.595648418859912,
    "trendline_buffer_pct": 0.05980493702293581
  }
  ```
- 120-day walk-forward comparison (`reports/wf_segment_analysis_candidate_best.csv` vs. `reports/wf_baseline_segment.csv`):
  - Mean excess: **+84.7 pp** (baseline −46.0 pp)
  - Min excess: **−38.9 pp** (baseline −1,277 pp)
  - Win rate: 61.9 % (baseline 66.7 %), mean trades ≈ 2 (baseline 3.6)
  - Highlight: the Aug 1 → Nov 29 2024 window jumps from **974 pp** to **2,217 pp** total return (Δ +1,242 pp). See `reports/wf_segment_analysis_candidate_vs_baseline.csv` for the full 21-window table.
- Multi-horizon ladders (`reports/wf_segment_analysis_ladders_summary.csv`):

  | Window profile | Windows | Mean excess % | Min excess % | Max excess % | Win rate | Mean trades |
  | -------------- | ------- | ------------- | ------------ | ------------ | -------- | ----------- |
  | 30 d / 7 d step | 101 | +18.6 | −56.1 | +129.6 | 0.525 | 1.94 |
  | 90 d / 15 d step | 43 | +54.8 | −67.2 | +184.0 | 0.535 | 2.00 |
  | 180 d / 30 d step | 19 | +48.4 | −110.9 | +159.5 | 0.421 | 2.00 |
  | 365 d / 30 d step | 13 | +177.0 | −120.7 | +5,520.7 | 0.692 | 2.00 |

- Remaining items:
  1. Trim the −100 pp to −200 pp consolidation windows (Nov 2024 → Jun 2025) by refining exit throttles now that the catastrophic dip is already fixed.
  2. Fold the tuned profile into the CLI / optimizer pipeline (30 d → 90 d → 1 y) for regression safety.
  3. Add volatility-aware sizing so the 1.37× clip scales back automatically during high ATR regimes.

### Artefacts

- `reports/wf_segment_analysis_candidate_vs_baseline.csv` — window-by-window deltas
- `reports/wf_segment_analysis_profiles_summary.csv` — aggregate comparison vs old default
- `reports/wf_segment_analysis_ladders_summary.csv` — multi-horizon statistics
- `reports/segment_hold_analysis_best_params.json` — canonical parameters

## 9. Cross-Strategy Sweep (analysis trend set)

- Re-ran the full catalog in `strats_all.json` with the refreshed trend states and the best parameters recorded by the Oct 8 optimizer run. Summary lives in `reports/wf_all_strategies_analysis_summary.csv`.
  - 84 strategies produced results; 7 failed to load or simulate (`reports/wf_all_strategies_analysis_errors.csv` lists them).
  - CompositeMomentumIndexStrategy now leads with mean excess ≈ +31 pp and a 62 % win rate.
  - Several grid/range strategies remain negative; they will need targeted retuning before inclusion in any “core” bundle.
- Raw per-window outputs are stored in `reports/wf_all_strategies_analysis_windows.csv` for deeper inspection.
