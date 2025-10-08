# Swap-Cost-Aware Breakout — v5 (Dynamic DD + Gain-Loosening) — Full Reproduction Guide
This is a **standalone** guide. It contains all rules, the cost model, and a single-file script to reproduce the exact results on your files. No previous history needed.
## Data
- OHLCV (5m): `/mnt/data/pdai_ohlcv_dai_730day_5m.csv`
- Swap costs: `/mnt/data/swap_cost_cache.json`
## Cost model (step-rounded)
| Step (DAI) | Roundtrip loss% | Eff. buy-side % | Eff. sell-side % |
|---:|---:|---:|---:|
| 5,000 | 2.914% | 1.461% | 1.463% |
| 10,000 | 4.650% | 2.328% | 2.328% |
| 15,000 | 6.170% | 3.087% | 3.087% |
| 20,000 | 7.764% | 3.883% | 3.883% |
| 25,000 | 9.195% | 4.598% | 4.598% |

---

## Strategy logic
**BEST v5 (Dynamic-DD with gain loosening)**
- **Entry:** close > previous **11-day high** (no look-ahead; rolling high shifted by 1 bar).
- **Exit:** earliest of
  1) close < previous **2-day low** **and** close < **EMA(3-day)**; or
  2) **Dynamic drawdown** from **peak since entry** greater than `DD_t` where
     `DD_t = clip( dd_base + k * ATRratio_1d + w * gain , dd_min , dd_max )`.
     - `gain = (peak/entry_price) - 1`, `ATRratio_1d = ATR(1 day)/close`.
     - Parameters we used: `dd_base=0.16`, `k=0.5`, `w=0.10`, `dd_min=0.10`, `dd_max=0.45`.
- **Positioning:** long/flat, full-capital entries; swap costs charged per side using the **rounded-up notional step**.

Rationale: v4’s dynamic-DD crushed performance. v5 further **loosens** the trailing stop as an uptrend’s unrealized gain grows, letting us **ride longer** while still cutting losers via the (2d-low & EMA) exit.

---

## Results (full dataset, net of swap costs)
- **BEST v5:** **1846.00%** (final equity ≈ **19459.98 DAI**), round trips: **32**.
- Prior best (v4, dynamic-DD base0.18 k0.5): ~1674.72%.

### Windowed results
| window    |     BH % |   BEST v5 % |   trades |
|:----------|---------:|------------:|---------:|
| Last 30d  | -23.2709 |    -10.8638 |        1 |
| Last 90d  | -32.4444 |    -38.9921 |        5 |
| Last 365d | -26.4514 |    949.817  |       14 |
| Full      | 191.333  |   1846      |       32 |

---

## Single-file reproduction script

```python
import pandas as pd, numpy as np, json, math, matplotlib.pyplot as plt

DATA = "/mnt/data/pdai_ohlcv_dai_730day_5m.csv"
SWAP = "/mnt/data/swap_cost_cache.json"
INITIAL = 1000.0
BARS_PER_DAY = int(24*60/5)

# Load
df = pd.read_csv(DATA, parse_dates=["timestamp"]).sort_values("timestamp").set_index("timestamp")
closes, highs, lows = df["close"], df["high"], df["low"]

# Swap-cost model
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

# Backtest
def simulate(closes, signals, initial=INITIAL):
    cap=initial; tokens=0.0; in_pos=False; trades=[]
    for t,p,s in zip(closes.index, closes.values, signals.values):
        if not in_pos and s==1:
            cost=per_side_cost(cap,"buy")
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

# Indicators
def ema(series, span): return series.ewm(span=span, adjust=False).mean()
def atr_series(closes, highs, lows, window=288):
    prev=closes.shift(1)
    tr = pd.concat([(highs-lows).abs(), (highs-prev).abs(), (lows-prev).abs()], axis=1).max(axis=1)
    return tr.rolling(window, min_periods=window).mean()

atr_1d = atr_series(closes, highs, lows, 288); atr_ratio = (atr_1d / closes).fillna(0)

# BEST v5 signals
def signals_dynDD_gain_loosen(closes, e_days=11, x_days=2, ema_days=3, dd_base=0.16, k=0.5, dd_min=0.10, dd_max=0.45, entry_buffer_frac=0.0, loosen_gain_w=0.10):
    BARS = BARS_PER_DAY
    e, x, ema_bars = e_days*BARS, x_days*BARS, ema_days*BARS
    hi = closes.rolling(e, min_periods=e).max().shift(1)
    lo = closes.rolling(x, min_periods=x).min().shift(1)
    ema_exit = ema(closes, ema_bars)
    sig = pd.Series(0, index=closes.index, dtype=int); long=False; peak=None; entry_price=None
    for i in range(len(closes)):
        c = closes.iloc[i]
        if not long:
            if i>=e and c >= hi.iloc[i]*(1.0+entry_buffer_frac):
                sig.iloc[i]=1; long=True; peak=c; entry_price=c
        else:
            sig.iloc[i]=1
            if c>peak: peak=c
            dd_t = dd_base + k * float(atr_ratio.iloc[i])
            if entry_price is not None and peak>entry_price:
                gain = (peak/entry_price)-1.0
                dd_t += loosen_gain_w * gain
            dd_t = min(max(dd_t, dd_min), dd_max)
            if (i>=x and c <= lo.iloc[i] and c < ema_exit.iloc[i]) or (peak is not None and c <= peak*(1-dd_t)):
                sig.iloc[i]=0; long=False; peak=None; entry_price=None
    return sig

if __name__ == "__main__":
    sig = signals_dynDD_gain_loosen(closes)
    res = simulate_with_equity(closes, sig, initial=INITIAL)
    print(res["return_pct"], res["final_equity"], len(res["trades"])//2)
```

---

## Artifacts
- Blotter: `/mnt/data/best_v5_blotter_dynDD_loosen.csv`
- Plots:  
  - `/mnt/data/equity_best_v5_vs_bh_full.png`  
  - `/mnt/data/equity_best_v5_vs_bh_365d.png`  
  - `/mnt/data/equity_best_v5_vs_bh_90d.png`  
  - `/mnt/data/equity_best_v5_vs_bh_30d.png`

*Generated: 2025-10-08T11:19:08.564613Z*
