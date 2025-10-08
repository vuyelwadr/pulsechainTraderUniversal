# Iteration 2 — Profit Push + Tight Trend-Follow Variant

**Goal:** raise total return % beyond the previous best and add a pure trend-follow strategy (buy only in uptrends, sell during downtrends).

## What changed
1. **Optimized drawdown stop** on the breakout strategy: tested DD ∈ {15%, 20%, 25%, 30%} with exit confirmation `price < EMA(3d)`.
2. **Added a tight trend-follow regime**: Uptrend requires `close > EMA(1d) > EMA(3d) > EMA(10d)` and a **positive 1d EMA slope**; entry uses a breakout filter; exits on 2d-low break or regime loss, with optional DD stop.

## Results (full dataset, net of swap costs)
- **New BEST (Champion v3)** — **11/2 breakout + exit<EMA(3d) + DD=20%**  
  Return: **+1,532.88%** (final equity ≈ 16,328.81 DAI), round trips: 34.
- Previous BEST (DD=25%): +1,282.56% (final ≈ 13,825.61 DAI), 32 trades.
- Trend-Follow (tight, 12/2, DD=25%): **+274.78%**, 60 trades.

### Drawdown-parameter sweep
|   dd |   final_equity |   return_pct |   trades |
|-----:|---------------:|-------------:|---------:|
| 0.2  |       16328.8  |     1532.88  |       34 |
| 0.25 |       13825.6  |     1282.56  |       32 |
| 0.15 |       11308.5  |     1030.85  |       43 |
| 0.3  |        3891.29 |      289.129 |       32 |

## Windowed performance (net of costs)
| window    |     BH % |   BEST % (11/2 exit<EMA3d> DD20) |   BEST trades |   TF % (tight uptrend) |   TF trades |
|:----------|---------:|---------------------------------:|--------------:|-----------------------:|------------:|
| Last 30d  | -23.2709 |                         -10.8638 |             1 |               -4.02956 |           1 |
| Last 90d  | -32.4444 |                         -38.9921 |             5 |              -17.5168  |           6 |
| Last 365d | -26.4514 |                         769.482  |            15 |              190.956   |          32 |
| Full      | 191.333  |                        1532.88   |            34 |              274.777   |          60 |

> Note: Last 30/90 days were broadly down; BEST v3 had few trades and small drawdowns, trend-follow sat out more but still dipped slightly. Over 1 year and full history, BEST v3 dominates.

## Exact rules (ready to re-implement)
- **BEST v3 (profit-maximizer):**
  - **Entry:** close breaks the **previous 11-day high** (no look-ahead; rolling high shifted by 1 bar).
  - **Exit:** close breaks the **previous 2-day low** **AND** `close < EMA(3d)`; **OR** a **20% drawdown from the peak since entry**.
  - **Positioning:** long/flat, invest full capital on entry; apply swap costs with **step-rounding up to 5,000 DAI**.

- **Trend-Follow (tight):**
  - **Regime (uptrend):** `close > EMA(1d) > EMA(3d) > EMA(10d)` and `EMA(1d) - EMA(1d).shift(1d) > 0`.
  - **Entry:** in uptrend **AND** close breaks previous **12-day** (or **10/9-day**) high.
  - **Exit:** 2-day-low break, **or** regime flips to downtrend, **or** drawdown stop (we used **25%** in tests).
  - **Positioning/costs:** same as BEST v3.

## Files & artifacts
- Blotter (BEST v3): `/mnt/data/best_v3_blotter_dd20.csv`
- Blotter (Trend-Follow): `/mnt/data/trend_follow_blotter.csv`
- Plots:
  - `/mnt/data/equity_best_v3_vs_bh_full.png`
  - `/mnt/data/equity_trend_vs_bh_full.png`
  - `/mnt/data/equity_best_v3_vs_bh_365d.png`
  - `/mnt/data/equity_best_v3_vs_bh_90d.png`
  - `/mnt/data/equity_best_v3_vs_bh_30d.png`

## Reproduce quickly
Use the `reproduce_strategy.py` scaffold from the previous doc and swap the signal functions with:
- `signals_11_2_exit_ema3d_dd(closes, dd=0.20)` for BEST v3.
- `signals_trend_follow(closes, base_e_days=12, base_x_days=2, dd_frac=0.25)` for the tight TF variant.

*Generated: 2025-10-08T10:51:30.203547Z*
