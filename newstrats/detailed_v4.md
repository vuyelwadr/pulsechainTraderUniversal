# Swap-Cost-Aware Breakout — v4 (Dynamic Drawdown) — Full Reproduction Guide
**Goal:** Maximize **total return %** from **1,000 DAI** on 5-minute data, net of realistic swap costs. This file is self-contained: copy the code block below into a Python file and run it to reproduce results end-to-end.
## Data
- OHLCV (5m): `/mnt/data/pdai_ohlcv_dai_730day_5m.csv`
- Swap costs: `/mnt/data/swap_cost_cache.json` (contains step-based roundtrip loss rates and gas estimates; we **round notional up** to the nearest step)
## Cost model summary
| Step (DAI) | Roundtrip loss% | Eff. buy-side % | Eff. sell-side % |
|---:|---:|---:|---:|
| 5,000 | 2.914% | 1.461% | 1.463% |
| 10,000 | 4.650% | 2.328% | 2.328% |
| 15,000 | 6.170% | 3.087% | 3.087% |
| 20,000 | 7.764% | 3.883% | 3.883% |
| 25,000 | 9.195% | 4.598% | 4.598% |

---

## Exact strategy rules

### BEST v4 (Dynamic-DD Breakout)
- **Entry:** Close > previous **11-day high** (no look-ahead: rolling high is shifted by 1 bar).
- **Exit:** Exit when **any** of these hit:
  1) Close < previous **2-day low** **AND** Close < **EMA(3-day)**, or  
  2) **Dynamic drawdown** from the running peak since entry exceeds `DD_t`, where  
     `DD_t = clip(dd_base + k × ATRratio_1d, dd_min, dd_max)`
     - In this run: `dd_base = 0.18`, `k = 0.5`, bounds `[0.12, 0.30]`
     - `ATRratio_1d = ATR(1d) / Close`, ATR is the 1-day average true range.
- **Positioning:** Long/flat, invest full capital on entry; apply swap costs on each side with step-rounding.

### Tight Trend-Follower (for comparison)
- **Regime (uptrend):** Close > EMA(1d) > EMA(3d) > EMA(10d) and EMA(1d) slope > 0.
- **Entry:** In uptrend **and** Close > previous **12-day high**.
- **Exit:** 2-day-low break **or** regime loss **or** DD stop (we used 25%).
- **Positioning:** Same cost treatment as above.

---

## Results (full dataset, net of swap costs)
- **BEST v4 (dynamic-DD):** **1674.72%** (final equity ≈ **17747.24 DAI**), round trips: **34**.
- Prior static-DD best (20%): ~1532.88% for reference.
- Tight Trend-Follower (12/2, 25% DD): ~274.78% on full history.

### Windowed results
| window    |     BH % |   BEST v4 % |      TF % |   BEST trades |   TF trades |
|:----------|---------:|------------:|----------:|--------------:|------------:|
| Last 30d  | -23.2709 |    -10.8638 |  -4.02956 |             1 |           1 |
| Last 90d  | -32.4444 |    -38.5279 | -17.5168  |             5 |           6 |
| Last 365d | -26.4514 |    829.512  | 190.956   |            15 |          32 |
| Full      | 191.333  |   1674.72   | 274.777   |            34 |          60 |

---

## Reproduce exactly — single-file script

```python
import pandas as pd, numpy as np, json, math, matplotlib.pyplot as plt

DATA = "/mnt/data/pdai_ohlcv_dai_730day_5m.csv"
SWAP = "/mnt/data/swap_cost_cache.json"
INITIAL = 1000.0
BARS_PER_DAY = int(24*60/5)

# 1) Load
df = pd.read_csv(DATA, parse_dates=["timestamp"]).sort_values("timestamp").set_index("timestamp")
closes, highs, lows = df["close"], df["high"], df["low"]

# 2) Swap-cost model
sc = json.load(open(SWAP))
STEP = int(sc["metadata"]["step_notional"])
ENTRIES = {int(k): v for k,v in sc["entries"].items()}
MIN_STEP, MAX_STEP = min(ENTRIES), max(ENTRIES)

def ceil_to_step(n): return min(int(math.ceil(max(n, MIN_STEP)/STEP)*STEP), MAX_STEP)

def per_side_cost(n, side):
    e = ENTRIES[ceil_to_step(n)]
    loss_rate = float(e["derived"]["loss_rate"])
    gas = float(e[side]["gas_use_estimate_usd"])
    return n*(loss_rate/2.0) + gas

# 3) Backtester
def simulate(closes, signals, initial=INITIAL):
    cap=initial; tokens=0.0; in_pos=False; trades=[]
    for t,p,s in zip(closes.index, closes.values, signals.values):
        if not in_pos and s==1:
            cost = per_side_cost(cap,"buy")
            if cap>cost:
                cap_after=cap-cost; tokens=cap_after/p; cap=0.0; in_pos=True
                trades.append(dict(time=t, type="BUY", price=float(p), notional=float(cap_after+cost), cost=float(cost), step=ceil_to_step(cap_after+cost)))
        elif in_pos and s==0:
            notional=tokens*p; cost=per_side_cost(notional,"sell"); cap=notional-cost; tokens=0.0; in_pos=False
            trades.append(dict(time=t, type="SELL", price=float(p), notional=float(notional), cost=float(cost), step=ceil_to_step(notional)))
    if in_pos:
        p=closes.iloc[-1]; notional=tokens*p; cost=per_side_cost(notional,"sell"); cap=notional-cost
        trades.append(dict(time=closes.index[-1], type="SELL", price=float(p), notional=float(notional), cost=float(cost), step=ceil_to_step(notional)))
    return dict(final_equity=float(cap), total_return_pct=(cap/initial-1.0)*100.0, trades=trades)

def simulate_with_equity(closes, signals, initial=INITIAL):
    cap=initial; tokens=0.0; in_pos=False; eq=[]; trades=[]
    for t,p,s in zip(closes.index, closes.values, signals.values):
        if not in_pos and s==1:
            cost=per_side_cost(cap,"buy")
            if cap>cost:
                cap_after=cap-cost; tokens=cap_after/p; cap=0.0; in_pos=True
                trades.append(dict(time=t, type="BUY", price=float(p), notional=float(cap_after+cost), cost=float(cost), step=ceil_to_step(cap_after+cost)))
        elif in_pos and s==0:
            notional=tokens*p; cost=per_side_cost(notional,"sell"); cap=notional-cost; tokens=0.0; in_pos=False
            trades.append(dict(time=t, type="SELL", price=float(p), notional=float(notional), cost=float(cost), step=ceil_to_step(notional)))
        eq.append(cap if not in_pos else tokens*p)
    if in_pos:
        p=closes.iloc[-1]; notional=tokens*p; cost=per_side_cost(notional,"sell"); cap=notional-cost; eq[-1]=cap
        trades.append(dict(time=closes.index[-1], type="SELL", price=float(p), notional=float(notional), cost=float(cost), step=ceil_to_step(notional)))
    return dict(equity=pd.Series(eq, index=closes.index), trades=trades, final_equity=float(eq[-1]), return_pct=(eq[-1]/initial-1.0)*100.0)

def ema(series, span): return series.ewm(span=span, adjust=False).mean()

# ATR(1d) and ATR ratio
def atr_series(closes, highs, lows, window=288):
    prev=closes.shift(1)
    tr = pd.concat([(highs-lows).abs(), (highs-prev).abs(), (lows-prev).abs()], axis=1).max(axis=1)
    return tr.rolling(window, min_periods=window).mean()
atr_1d = atr_series(closes, highs, lows, 288); atr_ratio = (atr_1d / closes).fillna(0)

# 4) Signals — BEST v4
def signals_breakout_exitEMA_dynDD(closes, e_days=11, x_days=2, ema_days=3, dd_base=0.18, dd_k=0.5, dd_min=0.12, dd_max=0.30):
    e, x, ema_bars = e_days*BARS_PER_DAY, x_days*BARS_PER_DAY, ema_days*BARS_PER_DAY
    hi = closes.rolling(e, min_periods=e).max().shift(1)
    lo = closes.rolling(x, min_periods=x).min().shift(1)
    ema_exit = ema(closes, ema_bars)
    sig = pd.Series(0, index=closes.index, dtype=int); long=False; peak=None
    for i in range(len(closes)):
        c = closes.iloc[i]
        if not long:
            if i>=e and c >= hi.iloc[i]: sig.iloc[i]=1; long=True; peak=c
        else:
            sig.iloc[i]=1; peak = max(peak, c) if peak is not None else c
            dd_t = dd_base + dd_k * float(atr_ratio.iloc[i])
            dd_t = min(max(dd_t, dd_min), dd_max)
            if (i>=x and c <= lo.iloc[i] and c < ema_exit.iloc[i]) or (peak is not None and c <= peak*(1-dd_t)):
                sig.iloc[i]=0; long=False; peak=None
    return sig

if __name__ == "__main__":
    sig = signals_breakout_exitEMA_dynDD(closes)
    res = simulate_with_equity(closes, sig, initial=INITIAL)
    print(res["return_pct"], res["final_equity"], len(res["trades"])//2)
```

---

## Artifacts in this run
- **Blotter:** `/mnt/data/best_v4_blotter_dynDD.csv`
- **Plots:**  
  - `/mnt/data/equity_best_v4_dynDD_vs_bh_full.png`  
  - `/mnt/data/equity_best_v4_dynDD_vs_bh_365d.png`  
  - `/mnt/data/equity_best_v4_dynDD_vs_bh_90d.png`  
  - `/mnt/data/equity_best_v4_dynDD_vs_bh_30d.png`

*Generated: 2025-10-08T11:06:45.161865Z*
