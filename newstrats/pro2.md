
# pro1 — Strategy upgrade pack for PulseChain Trader Universal

**Goal:** Maximize *return percentage* with strategies that are **robust across time** and **strictly cost‑aware** (respecting `swap_cost_cache.json`).

**What I changed / added (all files tagged with `_pro1`)**

- `newstrats/strategies_pro1.py` — three improved strategies + an ensemble with regime switching.
- `newstrats/params_pro1.json` — defaults and compact parameter grids for walk‑forward optimization.
- `newstrats/local_pro1.md` — usage notes focused on the `newstrats` folder.
- This `pro1.md` — full rationale, setup, and next steps.

Your repository already includes a modular strategy system and backtesting pipeline using **real on‑chain PulseX data** (per the README), with demo/live modes and HTML reporting. These additions plug into that system without changing core assumptions. [repo overview, README] citeturn0view0

---

## 1) Strategy design overview

### A. **TAB_pro1** (Trend‑Aware Breakout)
- Donchian breakout + ADX filter to avoid chop.
- Entry requires an **edge proxy** (EMA separation in bps) larger than `total_bps * cost_safety_mult`.
- Exits: ATR trailing stop, mid‑channel give‑up, and optional time stop.

### B. **AMAC_pro1** (Adaptive EMA Crossover)
- EMA(fast)↑ over EMA(slow) with **slope confirmation**.
- Edge gate: EMA separation in bps must clear cost threshold.
- Exits: ATR trailing + time‑stop.

### C. **RMR_pro1** (Range Mean‑Reversion)
- ADX below threshold ⇒ sideways regime.
- Buy when **RSI** oversold and price below **lower Bollinger Band**.
- Take‑profit on revert to the middle band; ATR stop + time stop protection.

### D. **EnsemblePro1**
- **Regime switch** via ADX: trend ⇒ (TAB ∧ AMAC), range ⇒ RMR.
- Long‑only, appropriate for DEX spot trading.

**All** strategies use a common **CostModel** that reads `swap_cost_cache.json` to estimate total round‑trip **bps** and blocks trades without sufficient edge.

---

## 2) Respecting `swap_cost_cache.json`

- The parser tolerates different schemas (`total_bps`, or `fee_bps + slippage_bps + gas_bps`, or fractional forms).  
- When ambiguous, it **falls back conservatively** (60 bps round‑trip) so we avoid over‑trading.  
- You can refine the mapping once we confirm your exact JSON fields (see code comments).

> This is critical for realistic backtests and is explicitly called out as a key constraint of your bot design. [real‑data & fee modeling noted in README] citeturn0view0

---

## 3) How to plug in and test

1. **Place files** under `newstrats/` and keep `swap_cost_cache.json` at repo root (or pass a path).  
2. In your backtest runner (e.g., `validate_strategies.py` or `final_analysis.py`), import:

```python
from newstrats.strategies_pro1 import STRATEGIES_PRO1, EnsemblePro1
```

3. Load params and run a walk‑forward backtest across your 2‑year dataset:

```
python validate_strategies.py \
  --strategies newstrats/strategies_pro1.py:STRATEGIES_PRO1 \
  --params newstrats/params_pro1.json \
  --pair "HEX/DAI" --interval 5m \
  --wf-train 60d --wf-test 15d \
  --metric return_pct --penalize-cost
```

4. Compare **return %**, trade count, and drawdown vs your existing sets. The repo’s **HTML reports** should capture this if wired (see README). citeturn0view0

> If your runner auto-loads strategies, just add an import line and include `STRATEGIES_PRO1` in its registry.

---

## 4) Results (template for your run)

*I couldn’t execute your local runner from here, so below is a template you can fill after running on your machine.*

| Segment | Train Window | Test Window | Strategy       | Return % | Max DD % | Trades | Costs (bps) |
|--------:|:-------------|:------------|:---------------|---------:|---------:|------:|------------:|
| 1       | 60d          | 15d         | EnsemblePro1   |    TBD   |    TBD   |  TBD  |      (log)  |
| …       | …            | …           | TAB_pro1       |    TBD   |    TBD   |  TBD  |      (log)  |
| …       | …            | …           | AMAC_pro1      |    TBD   |    TBD   |  TBD  |      (log)  |
| …       | …            | …           | RMR_pro1       |    TBD   |    TBD   |  TBD  |      (log)  |

**Acceptance checks**
- Positive median test return across segments
- Worst‑segment drawdown within tolerance
- Trade‑count reasonable after cost gate
- Costs (logged) align with `swap_cost_cache.json`

---

## 5) Why this should be “good across time periods”

- **Regime switching** prevents a single‑mode strategy from dominating.
- **Cost gating** reduces churn in high‑fee/slippage eras.
- **ATR/time exits** protect both low‑vol and high‑vol regimes.
- **Compact parameter grids** encourage stability over curve‑fit.

---

## 6) Next steps / options to push further

- **Refine CostModel** to your exact `swap_cost_cache.json` schema (tiered sizes, pool routes).  
- **Auto‑sizing**: position size proportional to `(edge_bps / min_edge_bps)` with hard caps.  
- **Daily model selection**: rolling tournament between TAB/AMAC/RMR with cost‑penalized returns.  
- **Add liquidity regime** from reserves/TVL to sharpen filters.  
- **Hedged exits**: partial scale‑outs at 1×/2× ATR.

---

### Files delivered (all tagged `_pro1`)

- `newstrats/strategies_pro1.py`  
- `newstrats/params_pro1.json`  
- `newstrats/local_pro1.md`  
- `pro1.md` (this report)

> If you’d like, I can also prep a minimal patch to your `validate_strategies.py` to auto‑register `STRATEGIES_PRO1` in the same style as your existing strategies.

— **pro1**
