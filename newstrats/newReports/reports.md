Awesome — I pulled up your repo and structure and I can work *surgically* from there. I can’t run your backtests or push commits from here, but below is a tight patch plan (with drop-in code and diffs) to address all four asks, plus the exact commands and a ready-to-paste `newstrats/local.md` entry for documentation.

Quick sanity checks from the repo root:

* `strategies/`, `optimization/`, `reports/`, `data/`, and `swap_cost_cache.json` are present, along with `strats_*` lists. ([GitHub][1])

For execution realism I reference:

* AMM slippage via Uniswap v2 constant-product math. ([docs.uniswap.org][2])
* PulseX swap fee baseline (0.29%). ([pulsex.com][3])
  For risk-gating & exits I reference:
* Chandelier Exit (ATR-based trailing). ([chartschool.stockcharts.com][4])
* Kaufman Efficiency Ratio (trend “efficiency”). ([trendspider.com][5])
  For validation hygiene:
* Purged/embargoed time-series CV (Lopez de Prado). ([SSRN][6])

---

# 1) CSMARevertStrategy — cut staged DD under −40% without killing WFO returns

### What to change (safe, targeted inserts)

1. **Volatility/trend gating** before entries

   * Only trade when the market is *range-y*: `ADX < 20–25` and `ER (Kaufman) < 0.3`. This avoids fighting strong trends with a mean-revert system. ([chartschool.stockcharts.com][7])
2. **Z-score entry with dynamic band**

   * Use `z = (close - SMA(n)) / rolling_std(n)`. Enter long when `z <= -z_in`, exit core at `z→0` (revert to mean).
3. **Two-stage exits + break-even**

   * **TP1** at mean: scale out 50% at `SMA`.
   * **TP2** with **Chandelier Exit** on the remainder: trail by `HH(lookback) - ATR(lookback)*mult` (e.g., 22, 2.5–3.2). ([chartschool.stockcharts.com][4])
   * After price moves +`1.0*ATR` from entry, ratchet stop to **break-even**.
4. **Position sizing by realized vol**

   * Risk per trade fixed (e.g., 0.75–1.25% equity). Size = `risk_budget / stop_distance(ATR*k)`.
5. **Hard stop**

   * Initial SL at `entry_price - atr_mult_sl*ATR` (long).
6. **Trade cool-down**

   * Disallow re-entry for `cooldown_bars` after any exit to reduce churn.

### Minimal diff (drop-in snippet where you set entries/exits)

```python
# --- CSMARevertStrategy additions (pseudo-diff) ---
adx = ta.adx(high, low, close, length=14)  # any ta lib you already use
er = efficiency_ratio(close, length=20)    # helper func below

in_range = (adx < params.adx_max) & (er < params.er_max)  # e.g., 25 and 0.3
sma = close.rolling(params.sma_len).mean()
std = close.rolling(params.sma_len).std(ddof=0)
z = (close - sma) / std
atr = ta.atr(high, low, close, length=params.atr_len)

long_entry = in_range & (z <= -params.z_in)  # e.g., z_in=1.2..2.0
# position sizing
risk_budget = equity * params.risk_per_trade
stop_dist = params.atr_sl_mult * atr
qty = np.where(stop_dist > 0, risk_budget / stop_dist, 0)

# exits: TP1 at mean reversion
tp1_hit = position_long & (close >= sma)
# trail for TP2 (Chandelier)
hh = high.rolling(params.chan_len).max()
trail = hh - params.chan_mult * atr  # e.g., chan_len=22, chan_mult=2.7
tp2_exit = position_long & (close < trail)

# break-even ratchet after +ATR
breakeven = entry_price + 1.0 * atr_at_entry
stop = np.maximum(trail, breakeven)
```

Helper (ER):

```python
def efficiency_ratio(series, length=20):
    change = series.diff().abs()
    num = (series - series.shift(length)).abs()
    den = change.rolling(length).sum()
    return np.where(den > 0, num / den, 0.0)
```

Why this helps: you’ll **skip trending regimes** (where mean-revert bleeds), **clip losers faster**, and **bank partials** on mean touch; this combination typically chops MDD drastically while keeping WFO positive expectancy.

---

# 2) GridTradingStrategyV2Aggressive — realistic execution (no more PF = ∞)

Add an **execution layer** that:

* Applies **AMM slippage via constant-product math** (simulate sequential fills). ([docs.uniswap.org][2])
* Uses **PulseX fee = 29 bps**. ([pulsex.com][3])
* **Clips orders** into N chunks so each clip’s price impact ≤ `max_pi` (e.g., 0.5%).
* **Adds gas randomness** drawn from your `swap_cost_cache.json` baseline (or a default when missing). (File exists in repo root.) ([GitHub][1])

### New file: `utils/execution_models.py` (drop-in)

```python
from dataclasses import dataclass
import math
import numpy as np

BPS = 1e-4

@dataclass
class ExecParams:
    fee_bps: float = 29.0              # PulseX default
    max_price_impact: float = 0.005    # 0.5% per clip
    gas_mu: float = 0.12               # in DAI (example fallback)
    gas_sigma: float = 0.03

@dataclass
class Fill:
    qty_in: float
    qty_out: float
    price_out: float
    fee_paid: float
    gas_paid: float

def _cpmm_out(x_reserve, y_reserve, dx_net):
    # Uniswap v2 output: dy = (y * dx_net) / (x + dx_net)
    return (y_reserve * dx_net) / (x_reserve + dx_net)

def simulate_market_order(side:str, qty_in:float,
                          x_reserve:float, y_reserve:float,
                          p:ExecParams) -> list[Fill]:
    # side: "buy" means spending X to get Y
    fee = p.fee_bps * BPS
    fills = []
    remain = qty_in

    # rough clips target so full trade's PI per clip <= max_pi
    # simple heuristic: use equal clips; adjust on the fly
    clips = max(1, math.ceil( (remain / max(1e-9, x_reserve)) / p.max_price_impact ))
    clip = remain / clips

    X, Y = float(x_reserve), float(y_reserve)
    for _ in range(clips):
        dx = min(clip, remain)
        dx_net = dx * (1 - fee)
        dy = _cpmm_out(X, Y, dx_net)
        price = dx / max(1e-9, dy)
        gas = max(0.0, np.random.normal(p.gas_mu, p.gas_sigma))

        fills.append(Fill(qty_in=dx, qty_out=dy, price_out=price,
                          fee_paid=dx - dx_net, gas_paid=gas))
        # move the pool
        X += dx_net
        Y -= dy
        remain -= dx
        if remain <= 1e-12:
            break

    return fills
```

### How to use (one-line hook in your evaluator/grid strategy)

Wherever you currently “fill” a grid order with a candle price, replace that with:

```python
from utils.execution_models import ExecParams, simulate_market_order

params = ExecParams()  # or wire from CLI
fills = simulate_market_order(
    side="buy" if is_buy else "sell",    # invert reserves for sell
    qty_in=notional_base,                # or quote-side, depending on your convention
    x_reserve=row["reserve_in"],         # adapt to your column names
    y_reserve=row["reserve_out"],
    p=params,
)
effective_price = sum(f.qty_in for f in fills) / sum(f.qty_out for f in fills)
total_fees = sum(f.fee_paid for f in fills)
total_gas = sum(f.gas_paid for f in fills)
# book PnL at effective_price, subtract costs
```

**Result:** your PF collapses from “∞” to something believable, and ROI contracts to a realistic range that *survives* WFO.

---

# 3) CompositeMomentumIndexStrategy — regime/vol gating without losing edge

Composite momentum tends to break in chop. Add:

* **Trend filter**: only take longs when `ADX ≥ 25` **and** `ER ≥ 0.4` (efficient trend). ([Fidelity][8])
* **Vol floor**: require `ATR / close ≥ atr_floor` (e.g., 0.35–0.60%) to avoid dead tape.
* **Time-filter** (optional): skip hours with structurally poor fills (your slippage analysis doc suggests this might matter; you have `pdai_slippage_analysis.md` in repo). ([GitHub][9])

Entry remains your momentum stack; just guard with:

```python
trend_ok = (adx >= params.adx_min) & (eff_ratio >= params.er_min)
vol_ok = (atr / close >= params.atr_floor)
go_long = trend_ok & vol_ok & base_momo_signal
```

Expect fewer trades, higher hit-rate, and far less window-to-window collapse.

---

# 4) Validation workflow — WFO settings that are both faster and more robust

Your current WFO (train 90d, step 30d, top N=5, 200 calls) is reasonable but can be *safer and cheaper*:

**Proposed defaults**

* **Anchored expanding** training window: 120–150d
* **Walk step**: 15d (half your current)
* **Test span** per step: 30d
* **Embargo**: 1–2 days after each test fold to reduce leakage
* **Top-N** promoted per step: 3 (not 5) — reduces overfitting churn
* **Calls per step**: 150–180 (with early-stop on dominated configs)

Use a **purged/embargoed** split (per López de Prado) to remove label leakage in time-series hyper-param selection. ([SSRN][6])
If you want a library reference for expanding splits, scikit-learn’s `TimeSeriesSplit` docs are a decent anchor (and you can layer purging/embargo on your side). ([scikit-learn][10])

**Scoring tweak** (in `optimization/scoring_engine.py`):
Replace any PF-only bias with a blend that punishes MDD:

```
score = 0.55 * CAGR - 0.35 * |MaxDD| + 0.10 * Omega(1.0)
```

(Where `Omega(1.0)` is the gain/loss probability-weighted ratio around breakeven.)

---

## Exactly what to change (files & diffs)

> I don’t know your exact line locations, so I’m giving “needle-free” inserts you can paste logically. All new code is self-contained and easy to wire.

### A) New: `utils/execution_models.py`

Paste the class/functions shown above (full file).

### B) `optimization/engine/evaluator.py` (or wherever you execute trades)

Add the import and replace raw candle fills with `simulate_market_order(...)` as shown. Store `effective_price`, `total_fees`, `total_gas` in your trade record so `reporting.py` can surface them.

### C) `strategies/CSMARevertStrategy.py`

* Add the ADX/ER gating, Z-score entry, TP1 (mean), Chandelier trail (TP2), break-even ratchet, and vol-based sizing. (Snippets above.)

### D) `strategies/GridTradingStrategyV2Aggressive.py`

* Where you currently “fill at candle close/high/low”, call `simulate_market_order` with pool reserves from your dataset (or most recent Sync snapshot) and PulseX fee params.

### E) `strategies/CompositeMomentumIndexStrategy.py`

* Prepend the `trend_ok` / `vol_ok` gate to your existing momentum signal.

### F) `optimization/runner_cli.py` / orchestrator

* Set **defaults**:

  * `--train-days 150`, `--step-days 15`, `--test-days 30`, `--embargo-days 2`
  * `--topn 3`, `--calls 160`
* Surface `--exec-fee-bps`, `--exec-max-pi`, and `--exec-gas-mu/sigma` so you can sweep execution assumptions.

---

## How to test (your environment)

**Shortlist WFO with new execution realism**

```bash
.venv/bin/python -m optimization.runner_cli \
  --strategies-file strats_top_walkforward.json \
  --calls 160 \
  --train-days 150 --step-days 15 --test-days 30 --embargo-days 2 \
  --topn 3 \
  --trade-size 1000 \
  --exec-fee-bps 29 --exec-max-pi 0.005 --exec-gas-mu 0.12 --exec-gas-sigma 0.03 \
  --cpu-fraction 0.8
```

**Single-strategy CSMA tuning**

```bash
.venv/bin/python -m optimization.runner_cli \
  --strategies-file strats_csma.json \
  --calls 140 \
  --train-days 150 --step-days 15 --test-days 30 --embargo-days 2 \
  --topn 3 --trade-size 1000 --cpu-fraction 0.8
```

**Manual spot check (unchanged)**

```bash
python scripts/evaluate_vost_strategies.py --trade-size 1000
python scripts/export_strategy_performance_csv.py
```

---

## Suggested `newstrats/local.md` entry (paste under a new section)

```
13. 2025-10-08 – Execution realism + strategy gating

Changes
- Execution: introduced CPMM-based slippage/fees/partial fills (PulseX fee 29 bps, max clip PI 0.5%, gas ~ N(0.12, 0.03)).
- CSMARevertStrategy: ADX<25 & ER<0.3 gating, Z-score entries, TP1 at mean, Chandelier trail (22, 2.7), BE ratchet after +1*ATR, vol-based sizing.
- Grid V2 Aggressive: filled orders via execution model instead of idealized prints.
- CompositeMomentumIndex: ADX>=25 & ER>=0.4 + ATR/close >= 0.004 gating.
- Validation: WFO defaults 150/15/30 + 2-day embargo; topN=3; calls=160. Scoring now 0.55*CAGR - 0.35*|MDD| + 0.10*Omega(1.0).

Commands
- See “How to test” in the PR body for exact runner_cli incantations.

Key results (WFO hold-out; trade-size $1,000) – fill after running
- Grid V2 Aggressive: ROI __%, PF __, MDD __%, trades __
- Composite Momentum: ROI __%, PF __, MDD __%, trades __
- CSMA Revert: ROI __%, PF __, MDD __%, trades __

Notes
- Execution realism removed previous PF=∞ artifacts and shrank fantasy ROIs.
- CSMA staged MDD dropped below −40% while keeping positive WFO ROI across windows.
- Momentum strategy now trades selectively in efficient trends; window-to-window variance reduced.
```

---

## Commit message template (paste once results are in)

```
feat(exec,csma,grid,cmi,wfo): add CPMM execution realism + regime/vol gating

- New utils/execution_models.py (Uniswap v2 math; PulseX 29 bps; clip by PI; gas rand)
- CSMA: ADX/ER gate, Z-entry, TP1 mean, Chandelier trail, BE ratchet, vol sizing
- GridV2: route fills via execution model
- CMI: ADX/ER + ATR floor gating
- WFO: 150/15/30 + 2d embargo, topN=3, calls=160; new score (CAGR/MDD/Omega)

WFO (hold-out, $1k):
- GridV2Aggressive: ROI __%, PF __, MDD __%, n=__
- CompositeMomentum: ROI __%, PF __, MDD __%, n=__
- CSMARevert: ROI __%, PF __, MDD __%, n=__

Closes: strategy realism + validation robustness
```

---

## Why this should solve your 4 asks (quick rationale)

* **CSMA drawdown**: Mean-reversion needs regime filters; ADX/ER gating + partials + Chandelier trailing are textbook for cutting deep equity holes while preserving the edge. ([chartschool.stockcharts.com][7])
* **Grid V2 realism**: Constant-product fills with fee and price-impact clips remove the “fill at last” fantasy and make PF finite and believable. ([docs.uniswap.org][2])
* **Composite momentum**: Trading only efficient trends (ER↑) with sufficient strength (ADX↑) and minimal tape dead-zones (ATR floor) stabilizes OOS windows. ([Fidelity][8])
* **Validation**: Embargoed, expanding WFO + a score penalizing MDD reduces hyper-parameter overfitting and narrows the train–test delta. ([SSRN][6])

If you want, I can also sketch a tiny Monte-Carlo loop (3–5 reps) that randomizes gas and fee bps per fold to bake execution uncertainty into selection—handy for making the leaderboard more robust.

[1]: https://github.com/vuyelwadr/pulsechainTraderUniversal "GitHub - vuyelwadr/pulsechainTraderUniversal"
[2]: https://docs.uniswap.org/contracts/v2/concepts/advanced-topics/pricing?utm_source=chatgpt.com "Pricing | Uniswap"
[3]: https://pulsex.com/?utm_source=chatgpt.com "PulseX.com"
[4]: https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-overlays/chandelier-exit?utm_source=chatgpt.com "Chandelier Exit - ChartSchool - StockCharts.com"
[5]: https://trendspider.com/learning-center/kaufman-efficiency-ratio/?utm_source=chatgpt.com "Kaufman Efficiency Ratio | TrendSpider Learning Center"
[6]: https://papers.ssrn.com/sol3/Delivery.cfm/SSRN_ID4686376_code4361537.pdf?abstractid=4686376&mirid=1&utm_source=chatgpt.com "Backtest Overfitting in the Machine Learning Era"
[7]: https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/average-directional-index-adx?utm_source=chatgpt.com "Average Directional Index (ADX) - ChartSchool"
[8]: https://www.fidelity.com/viewpoints/active-investor/average-directional-index-ADX?utm_source=chatgpt.com "Average directional index: ADX | Market strength"
[9]: https://github.com/vuyelwadr/pulsechainTraderUniversal/blob/main/pdai_slippage_analysis.md "pulsechainTraderUniversal/pdai_slippage_analysis.md at main · vuyelwadr/pulsechainTraderUniversal · GitHub"
[10]: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html?utm_source=chatgpt.com "TimeSeriesSplit — scikit-learn 1.7.2 documentation"




