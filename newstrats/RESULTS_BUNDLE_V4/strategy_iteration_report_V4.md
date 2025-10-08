
# Iteration Report — V4 (Deepening the Edge, Cost‑Aware)

**Objective.** Maximize **total return %** from **1,000 DAI** using your swap-cost cache exactly (ceil notional to 5k buckets; per‑side loss_rate/2 + side gas; USD≈DAI).  

**Champion carried forward.** The simple SMA‑reversion with RSI gate, exiting at **SMA + 4.8%** remains the best across the full 730‑day sample and improves on the V3 result.

---

## Data & Cost Model
- Dataset: `/mnt/data/pdai_ohlcv_dai_730day_5m.csv` (≈ 2 years of 5‑minute bars).
- Costs: `/mnt/data/swap_cost_cache.json`  
  - Bucket step: 5,000 DAI.  
  - Per‑side cost rate: `(round‑trip loss_rate at bucket) / 2`.  
  - Fixed gas per side from cache (USD≈DAI).  
  - At trade time, **ceil(N)** to bucket (e.g., 21k → 25k).

---

## Strategies Tested This Round (all cost-aware)
1. **Tuned Baseline (Champion)** — _SMA(576) dip‑buy + RSI(14) ≤ 30; exit at SMA + **4.8%**._  
   - Saved artifacts (V2 naming):
     - 30d: [CSV](sandbox:/mnt/data/tuned_sma_rsi_30d_trades_V2.csv), [PNG](sandbox:/mnt/data/tuned_sma_rsi_30d_equity_V2.png)
     - 90d: [CSV](sandbox:/mnt/data/tuned_sma_rsi_90d_trades_V2.csv), [PNG](sandbox:/mnt/data/tuned_sma_rsi_90d_equity_V2.png)
     - 365d: [CSV](sandbox:/mnt/data/tuned_sma_rsi_365d_trades_V2.csv), [PNG](sandbox:/mnt/data/tuned_sma_rsi_365d_equity_V2.png)
     - 730d: [CSV](sandbox:/mnt/data/tuned_sma_rsi_730d_trades_V2.csv), [PNG](sandbox:/mnt/data/tuned_sma_rsi_730d_equity_V2.png)

2. **Dynamic Exit (z‑aware)** — exit_up = min(cap, base + γ·z⁺), z from SMA(576).  
   - Grid (γ ∈ {0.006, 0.008, 0.010, 0.012, 0.015}, base ∈ {4.6%, 4.8%, 5.0%}).  
   - Best 730d: **42,056 DAI** (base=5.0%, γ=0.006).  
   - CSV: [v4_dynamic_exit_grid_730d_V2.csv](sandbox:/mnt/data/v4_dynamic_exit_grid_730d_V2.csv).  
   - **Conclusion:** helpful but **did not beat** fixed 4.8% (42,806 DAI).

3. **Two‑Tier Take‑Profit + Runner** — sell 90–95% at TP1, keep a 5–10% runner with vol‑style trailing or TP2.  
   - Grid (keep ∈ {5%, 10%}, trail ∈ {25%, 30%, 35%}, TP2Δ ∈ {5%, 8%, 10%}).  
   - Best 730d: **40,815 DAI** (keep=10%, trail=30%, TP2Δ=10%).  
   - CSV: [v4_runner_grid_730d_V2.csv](sandbox:/mnt/data/v4_runner_grid_730d_V2.csv).  
   - **Conclusion:** strong but still below the tuned single‑shot TP at 4.8%.

4. **Entry/Exit micro‑sweeps** — confirmed **SMA(576)**, **drop=25%**, **RSI≤30**, **TP=4.8%** is the local optimum on 730d.  
   - (512/540/560/600/640) windows underperform; TP micro‑sweep 4.6%–5.0% in 0.05% steps peaks at **4.8%**.

---

## Performance Summary (start 1,000 DAI; net of costs)

| Strategy | 30d | 90d | 365d | 730d |
|---|---:|---:|---:|---:|
| Tuned SMA+RSI (TP 4.8%) | 1,000 | **3,106** | **16,200** | **42,806** |
| Dynamic Exit (best) | — | — | — | 42,056 |
| 2‑Tier TP + Runner (best) | — | — | — | 40,815 |

(Buy & Hold for reference at these endpoints: see V2 & V3 figures in previous reports.)

---

## Ready‑to‑run Module (default TP=4.8%)
- **[strategy_c_sma_revert_v2.py](sandbox:/mnt/data/strategy_c_sma_revert_v2.py)**

Example:
```bash
python -m optimization.runner   --strategy c_sma_revert_v2   --swap-cost-cache /path/to/swap_cost_cache.json   --csv /path/to/PAIR_5m.csv   --start-cash 1000   --params '{"n_sma":576,"entry_drop":0.25,"exit_up":0.048,"rsi_n":14,"rsi_max":30}'
```

---

## What’s next (queued for V5)
1. **Vol‑adaptive TP around 4.8% but with a hard clamp to keep cycles:** TP = 4.8% + β·clip(z⁺, 0, z_cap), β≤0.01, z_cap≤1 — combined with **ATR‑clock timeout** to bail if not hit after X ATR‑hours.  
2. **Micro‑runner (5%) only during confirmed uptrends** (EMA96>EMA288 & slope>0), with **EMA‑cross or k·ATR trailing**; otherwise no runner.  
3. **Session filter**: suppress entries during historically low‑edge UTC hours (will profile by hour‑of‑day using this dataset).

---

## Files bundled in V4 ZIP
A curated pack including: code modules, V2 & V3/V4 reports, tuned strategy CSVs/PNGs, and the new V4 grids. (See link below.)
