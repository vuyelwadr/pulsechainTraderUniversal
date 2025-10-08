# Swap‑Cost‑Aware Breakout Strategy — Detailed Design & Reproduction Guide

**Files used**
- OHLCV (5m): `/mnt/data/pdai_ohlcv_dai_730day_5m.csv`
- Swap cost cache: `/mnt/data/swap_cost_cache.json`

**Objective:** Maximize **total return %** starting from **1,000 DAI**, net of realistic swap costs that scale with trade notional by **rounding up to the nearest 5,000 DAI step** (capped to the largest step present in the cache).

---

## 1) Data & Environment

- Bars: 5‑minute OHLCV in UTC, columns: `timestamp, open, high, low, close, volume, price`.
- Instrument: the `close` column is used as the tradable price.
- Test span: full file range.
- Initial capital: 1,000 DAI. Always **full‑capital** when entering a position.
- Execution: trades occur at the **bar close** when a signal changes (flat→long to buy, long→flat to sell).
- End-of-test: force liquidation at the final bar so equity is comparable across runs.

---

## 2) Swap cost model (from `swap_cost_cache.json`)

Let `N` be the raw notional for the order. We compute:

1. **Step rounding:**  
   `N_step = ceil(N / 5000) * 5000` and **clamp** to the cache range:  
   `N_step = min(max_step_in_cache, N_step)`; also never below the min step.

2. **Per‑side cost (DAI):**  
   Each cache entry provides a **round‑trip loss rate** `loss_rate` (fraction) and per‑side gas estimates in USD.  
   We charge per side:
   ```text
   side_cost = N * (loss_rate / 2) + gas_usd_side
   ```
   So the charged cost depends on the *raw* notional `N` but uses the **loss_rate** associated with `N_step`.

3. **Round‑trip cost:** buy side + sell side at their respective notionals at the time of each trade.

**Example effective rates (assuming notional ≈ step):**

| Step (DAI) | Roundtrip loss% | Eff. buy-side % | Eff. sell-side % |
|---:|---:|---:|---:|
| 5,000 | 2.914% | 1.461% | 1.463% |
| 10,000 | 4.650% | 2.328% | 2.328% |
| 15,000 | 6.170% | 3.087% | 3.087% |
| 20,000 | 7.764% | 3.883% | 3.883% |
| 25,000 | 9.195% | 4.598% | 4.598% |

> Interpretation: at the 5k step, the per‑side friction is ~1.46–1.49%. At 25k, it rises to ~4.6% per side, etc. The model penalizes churn and rewards holding trends.

---

## 3) Strategy logic

We explored **Donchian breakout** long/flat systems with costs applied as above, then layered exits/constraints that improved **total return %** on the full data.

### 3.1 Base Donchian signals (no look‑ahead)
- **Entry:** close breaks the **previous** *E‑day* high (e.g., 12‑day high).  
- **Exit:** close breaks the **previous** *X‑day* low (e.g., 2‑day low).  
- Signals change only at bar close. “Previous” means the rolling extreme **shifted by 1 bar** to avoid look‑ahead.

**Best base configs on full dataset:**
- **12/2:** +376.24% (31 round trips)
- **11/2:** +375.12% (32 round trips)
- **10/2:** +293.68% (34 round trips)
- Buy & Hold (net costs): +191.33%

### 3.2 Champion v1 — Add an EMA exit confirmation
**Rule:** Keep the base **11/2** but require **price < EMA(3‑day)** at exit in addition to the 2‑day‑low break.  
This removes many premature exits in noisy uptrends and cut a lot of “pay two swaps for nothing” cases.

- **Performance:** **+423.76%** (final equity ≈ **5237.55 DAI**), **31** round trips.

### 3.3 Champion v2 (aggressive) — Add a drawdown stop
**Rule:** Start from **Champion v1** and add an *in‑position peak* trailing stop:
> If while long, close ≤ `peak_since_entry × (1 − DD)`, then exit immediately (paying swap costs).
- Working setting: **DD = 25%**.

- **Performance:** **+1282.56%** (final equity ≈ **13825.61 DAI**), **32** round trips.  
  This variant rides large trends and exits quickly once a multi‑day surge rolls over. It’s more aggressive and will be more path‑sensitive.

> Your priority was *maximize return%*. If you care about stability later, prefer **Champion v1**.

---

## 4) Reproduction — minimal, runnable reference

Save as `reproduce_strategy.py` (or run in a notebook). It expects both data files at the paths shown above.

```python
import pandas as pd, json, math, numpy as np

DATA = "/mnt/data/pdai_ohlcv_dai_730day_5m.csv"
SWAP = "/mnt/data/swap_cost_cache.json"

# ----- Load -----
df = pd.read_csv(DATA, parse_dates=["timestamp"]).sort_values("timestamp")
df = df.set_index("timestamp")
closes, highs, lows = df["close"], df["high"], df["low"]
bars_per_day = int(24*60/5)

# ----- Swap-cost model -----
sc = json.load(open(SWAP))
STEP = int(sc["metadata"]["step_notional"])
ENTRIES = {int(k): v for k,v in sc["entries"].items()}
MAX_STEP, MIN_STEP = max(ENTRIES), min(ENTRIES)

def ceil_to_step(notional: float) -> int:
    step = int(math.ceil(max(notional, MIN_STEP)/STEP) * STEP)
    return min(step, MAX_STEP)

def per_side_cost(notional: float, side: str) -> float:
    step = ceil_to_step(notional)
    e = ENTRIES[step]
    loss_rate = float(e["derived"]["loss_rate"])         # round‑trip fraction
    gas = float(e[side]["gas_use_estimate_usd"])
    return notional * (loss_rate/2.0) + gas

# ----- Backtester -----
def simulate(closes: pd.Series, signals: pd.Series, initial=1000.0):
    cap = initial
    tokens = 0.0
    in_pos = False
    trades = []
    for t, p, s in zip(closes.index, closes.values, signals.values):
        if not in_pos and s == 1:
            cost = per_side_cost(cap, "buy")
            if cap <= cost: continue
            cap_after = cap - cost
            tokens = cap_after / p
            cap = 0.0
            in_pos = True
            trades.append(dict(time=t, side="BUY", price=p, notional=cap_after+cost, cost=cost))
        elif in_pos and s == 0:
            notional = tokens * p
            cost = per_side_cost(notional, "sell")
            cap = notional - cost
            tokens = 0.0
            in_pos = False
            trades.append(dict(time=t, side="SELL", price=p, notional=notional, cost=cost))
    if in_pos:
        p = closes.iloc[-1]
        notional = tokens * p
        cost = per_side_cost(notional, "sell")
        cap = notional - cost
        trades.append(dict(time=closes.index[-1], side="SELL", price=p, notional=notional, cost=cost))
    ret_pct = (cap/initial - 1.0)*100.0
    return dict(final_equity=cap, return_pct=ret_pct, trades=trades)

# ----- Donchian helpers (no look‑ahead) -----
def donchian_signals(closes: pd.Series, e_days:int, x_days:int) -> pd.Series:
    e, x = e_days*bars_per_day, x_days*bars_per_day
    hi = closes.rolling(e, min_periods=e).max().shift(1)
    lo = closes.rolling(x, min_periods=x).min().shift(1)
    sig = pd.Series(0, index=closes.index, dtype=int)
    long = False
    for i in range(len(closes)):
        c = closes.iloc[i]
        if not long:
            if i>=e and c >= hi.iloc[i]: sig.iloc[i]=1; long=True
        else:
            sig.iloc[i]=1
            if i>=x and c <= lo.iloc[i]: sig.iloc[i]=0; long=False
    return sig

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

# Champion v1: 11/2 with exit confirm under EMA(3d)
def signals_11_2_exit_ema3d(closes: pd.Series) -> pd.Series:
    e, x = 11*bars_per_day, 2*bars_per_day
    hi = closes.rolling(e, min_periods=e).max().shift(1)
    lo = closes.rolling(x, min_periods=x).min().shift(1)
    ema_exit = ema(closes, 864)  # 3d on 5m bars
    sig = pd.Series(0, index=closes.index, dtype=int)
    long=False
    for i in range(len(closes)):
        c = closes.iloc[i]
        if not long:
            if i>=e and c >= hi.iloc[i]: sig.iloc[i]=1; long=True
        else:
            sig.iloc[i]=1
            if i>=x and (c <= lo.iloc[i]) and (c < ema_exit.iloc[i]): sig.iloc[i]=0; long=False
    return sig

# Champion v2: add 25% drawdown stop since entry-peak
def signals_11_2_exit_ema3d_dd25(closes: pd.Series) -> pd.Series:
    e, x = 11*bars_per_day, 2*bars_per_day
    hi = closes.rolling(e, min_periods=e).max().shift(1)
    lo = closes.rolling(x, min_periods=x).min().shift(1)
    ema_exit = ema(closes, 864)
    sig = pd.Series(0, index=closes.index, dtype=int)
    long=False; peak=None
    for i in range(len(closes)):
        c = closes.iloc[i]
        if not long:
            if i>=e and c >= hi.iloc[i]: sig.iloc[i]=1; long=True; peak=c
        else:
            sig.iloc[i]=1; peak = max(peak, c) if peak is not None else c
            cond_exit = (i>=x and c <= lo.iloc[i] and c < ema_exit.iloc[i]) or (peak is not None and c <= 0.75*peak)
            if cond_exit: sig.iloc[i]=0; long=False; peak=None
    return sig

if __name__ == "__main__":
    # Baseline comparisons
    s_12_2 = donchian_signals(closes, 12, 2)
    r_12_2 = simulate(closes, s_12_2)
    s_c1 = signals_11_2_exit_ema3d(closes)
    r_c1 = simulate(closes, s_c1)
    s_c2 = signals_11_2_exit_ema3d_dd25(closes)
    r_c2 = simulate(closes, s_c2)
    print("12/2:", r_12_2)
    print("Champion v1 (11/2 + exit<EMA3d):", r_c1)
    print("Champion v2 (+25% DD stop):", r_c2)
```

---

## 5) Iteration log (why these choices)

1. **Start:** Donchian grid around trend horizons that match your swap friction. On the dev window (last ~180 days), 10/2 looked best, and it held up on the full set with **+293.68%**.
2. **Cost reality check:** Costs at 5k/10k steps (~1.5–2.3% per side) make frequent exits expensive. So we pushed to **12/2** and **11/2** to cut churn → **+376.24%** and **+375.12%** respectively.
3. **Exit quality:** Pure 2‑day lows can eject mid‑trend. Requiring **price < EMA(3d)** on exit avoids many tiny whipsaws → **Champion v1: +423.76%**.
4. **Aggressive layer:** To squeeze more upside in the massive rallies, add a **25% drawdown stop** while long (exit if drop from peak ≥25%). This produced **Champion v2: +1282.56%**. It’s more path‑sensitive but respects your “maximize % return” goal.
5. Other tweaks tried and discarded: ATR trailing‑stops (too jumpy here), entry EMA gates (reduced wins), extra exit buffers/cooldowns (mixed results).

---

## 6) Results snapshot (full data, net of costs)

| Strategy | Params | Round trips | Final equity (DAI) | Return % |
|---|---|---:|---:|---:|
| Champion v2 (aggr.) | 11/2 + exit<EMA(3d) + 25% DD stop | 32 | 13825.61 | **1282.56%** |
| Champion v1 | 11/2 + exit<EMA(3d) | 31 | 5237.55 | **423.76%** |
| Base | 12/2 | 31 | 4762.41 | 376.24% |
| Base | 10/2 | 34 | 3936.83 | 293.68% |
| Buy & Hold | — | 1 | 2913.33 | 191.33% |

**Artifacts you can open:**
- Iteration summary (CSV): `/mnt/data/iteration_summary_runs.csv`
- Trade blotter for Champion v1 (CSV): `/mnt/data/best_11d2d_exitEMA3d_blotter.csv`

---

## 7) Practical notes & guardrails
- **No look‑ahead:** Rolling highs/lows are shifted by 1 bar. Exits evaluate with the same rule. Stops use only information available as of each bar.
- **Execution model:** Fill at bar close (no slippage beyond the loss_rate in the cache).
- **Costs:** The per‑side cost uses the *current* notional and the **loss_rate** for the **rounded step** from the cache; if `N` exceeds the largest step in the cache, we clamp to that max step’s loss_rate.
- **Sensitivity:** The aggressive drawdown variant is powerful in big uptrends but will underperform in range‑bound regimes. If you later care about stability, start with Champion v1.

---

## 8) How to extend
- Re‑run the grid on rolling sub‑periods or different start dates to check stability.
- Add time‑of‑day filters (avoid low‑liquidity hours) to cut bad breaks.
- Port the logic into your repo’s runner; the functions above are self‑contained and mirror what I used here.

*Generated: 2025-10-08T10:30:04.870841Z*
