
# Cost-Aware Strategy — Iteration Report

**Dataset:** `/mnt/data/pdai_ohlcv_dai_730day_5m.csv`  
**Swap-cost cache:** `/mnt/data/swap_cost_cache.json` (ceil to 5k buckets; per-side rate = loss_rate/2; USD≈DAI).

## Objective
Maximize **total return %** from **1,000 DAI** using cost-aware execution (bucketed slippage+gas).

## Quick Timeline
1. **Baseline checks**
   - Build cost model (ceil-to-5k buckets).
   - Compute Buy & Hold as reference (**cost-aware** buy/sell).

2. **Iteration 1 — Donchian (trend breakout)**
   - `n_entry {96,144,288}`, `n_exit {24,48}`, buffer `{0.5%,1%}`, trail `{5%,10%}`.
   - **Result:** Mostly no trades in 90d dev window → 0%. Discarded.

3. **Iteration 2 — SMA mean reversion (all-in/out)**
   - Grid over `SMA {144,288,576}`, entry drop `{15%,25%,35%,50%}`, exit buffer `{0,2%,5%}`, trail `{none,10%,20%}`.
   - **Best v1:** `SMA=576`, **entry ≤ SMA×(1−25%)**, **exit ≥ SMA**, no trail.
   - **ROI (net):** +191% (90d), +1,250% (365d), +2,935% (730d).

4. **Iteration 3 — Z-score, BB, ATR variants**
   - Tested Z-score revert, Bollinger revert, ATR distance to SMA, multi-scale scaling.
   - All underperformed **v1** given real costs.

5. **Iteration 4 — NEW BEST (v2)**
   - **Exit improvement:** **SMA + 5%** take-profit to catch overshoot.
   - **Noise gate:** **RSI(14) ≤ 30** required at entry.
   - **ROI (net):**
     - **90d:** +212.5% → 3,125 DAI
     - **365d:** **+1,575%** → 16,754 DAI (**new best**)
     - **730d:** **+3,973%** → 40,733 DAI (**new best**)
   - Trades: 12 (365d), 21 (730d).

## Equity Curve
![Equity Curve — NEW BEST vs Buy & Hold](sandbox:/mnt/data/equity_curve_new_best_vs_bh.png)

## Downloadables
- 90d v1 trades: [sma_revert_best_90d_trades.csv](sandbox:/mnt/data/sma_revert_best_90d_trades.csv)
- 365d v1 trades: [sma_revert_best_365d_trades.csv](sandbox:/mnt/data/sma_revert_best_365d_trades.csv)
- 730d v1 trades: [sma_revert_best_730d_trades.csv](sandbox:/mnt/data/sma_revert_best_730d_trades.csv)
- 365d v2 trades: [best_iter_sma_rsi_365d_trades.csv](sandbox:/mnt/data/best_iter_sma_rsi_365d_trades.csv)
- 730d v2 trades: [best_iter_sma_rsi_730d_trades.csv](sandbox:/mnt/data/best_iter_sma_rsi_730d_trades.csv)

## Current Best Parameters
- `n_sma=576`, `entry_drop=0.25`, `exit_up=0.05`, `rsi_n=14`, `rsi_max=30`
- Long-only, all-in/out, cost-aware with bucketed slippage + gas.

## Runner Integration (minimal)
1. **Drop in the strategy module:**
   - `strategies/strategy_c_sma_revert.py` — use the file provided:  
     [Download the module](sandbox:/mnt/data/strategy_c_sma_revert.py)

2. **Register in your runner:**
   ```python
   # optimization/runner.py (example snippet)
   from strategies.strategy_c_sma_revert import adapter as c_sma_revert_adapter

   STRATS = {}
   name, space, run_fn = c_sma_revert_adapter()
   STRATS[name] = {"space": space, "run_fn": run_fn}
   ```

3. **Call your runner (example):**
   ```bash
   python -m optimization.runner      --strategy c_sma_revert      --swap-cost-cache /path/to/swap_cost_cache.json      --csv /path/to/PAIR_5m.csv      --start-cash 1000      --params '{"n_sma":576,"entry_drop":0.25,"exit_up":0.05,"rsi_n":14,"rsi_max":30}'
   ```

4. **Or use the module’s CLI directly:**
   ```bash
   python /path/to/strategies/strategy_c_sma_revert.py      --csv /path/to/PAIR_5m.csv      --swap-cost-cache /path/to/swap_cost_cache.json      --n-sma 576 --entry-drop 0.25 --exit-up 0.05 --rsi-n 14 --rsi-max 30      --start-cash 1000 --window-days 730      --plot-out ./equity_curve.png
   ```

## Next ideas to try
- Adaptive `exit_up` from local vol (zscore/ATR).  
- Time-in-trade guard (X hours) to recycle capital if target not reached.  
- RSI-of-RSI / slope filter to capture fresh flushes only.  
- Session filter to avoid low-liquidity chop hours.

---

*All results shown are **net** of your swap-cost cache model (ceil-to-bucket per side slippage + side gas).*

