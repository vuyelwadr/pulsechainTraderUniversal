
# Iteration Report — V7 (Constant Micro‑Runner, Cost‑Aware)

**Big change:** At TP1 = **SMA + 4.8%**, if the regime is an **uptrend** (EMA96 > EMA288 and EMA slope > 0),
we now keep a **constant runner of 10%** of the position (previously 5%). Runner exits on **EMA cross‑down or 30% trailing**.
Entry remains SMA(576) dip‑buy + RSI(14) ≤ 30. All trades pay your bucketed costs (ceil to 5k, per‑side loss_rate/2 + side gas).

## Results (net of costs; start 1,000 DAI)
- **30d:** 1,000 DAI (no trades) — [trades](sandbox:/mnt/data/v7_runner10_30d_trades_V2.csv), [equity](sandbox:/mnt/data/v7_runner10_30d_equity_V2.png)
- **90d:** 3,106 DAI (+210.6%) — [trades](sandbox:/mnt/data/v7_runner10_90d_trades_V2.csv), [equity](sandbox:/mnt/data/v7_runner10_90d_equity_V2.png)
- **365d:** 15,633 DAI (+1,463.3%) — [trades](sandbox:/mnt/data/v7_runner10_365d_trades_V2.csv), [equity](sandbox:/mnt/data/v7_runner10_365d_equity_V2.png)
- **730d:** **43,433 DAI (+4,343.3%)** — **new high watermark** — [trades](sandbox:/mnt/data/v7_runner10_730d_trades_V2.csv), [equity](sandbox:/mnt/data/v7_runner10_730d_equity_V2.png)

## Module
Drop‑in module (adapter + CLI stub): **[strategy_runner_v7.py](sandbox:/mnt/data/strategy_runner_v7.py)**  
Default params: `exit_up=0.048`, `runner_frac=0.10`, `runner_trail=0.30`.

Example (runner):
```bash
python -m optimization.runner   --strategy runner_v7   --swap-cost-cache /path/to/swap_cost_cache.json   --csv /path/to/PAIR_5m.csv   --start-cash 1000   --params '{"n_sma":576,"entry_drop":0.25,"exit_up":0.048,"rsi_n":14,"rsi_max":30,"runner_frac":0.10,"runner_trail":0.30}'
```

## Why this helps
- The 4.8% TP remains the **cycle-maximizing sweet spot** under real costs; we keep it.  
- A slightly larger runner (10%) monetizes extended uptrend legs while limiting cost exposure on routine cycles.

## Next (V8 ideas)
- Combine V6’s ATR‑clock with V7 runner (but only in non‑uptrend regimes).  
- Hour‑of‑day entry heatmap + session filter to avoid low-edge hours.  
- Very narrow TP wiggle (±0.2%) driven by near-term z⁺ but **clamped** to preserve cycle frequency.
