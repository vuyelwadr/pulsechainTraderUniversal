
# Iteration Report — V2 (Cost-Aware)

**Dataset:** `/mnt/data/pdai_ohlcv_dai_730day_5m.csv`  
**Swap-cost cache:** `/mnt/data/swap_cost_cache.json` (ceil to 5k buckets; per-side rate = loss_rate/2; side gas; USD≈DAI).  
**Start capital:** 1,000 DAI

## Prior Best (baseline)
- **SMA(576) reversion + RSI≤30 + exit at SMA+5%** (all-in/out, cost-aware).  
- Net results: **90d +212.5%**, **365d +1,575%**, **730d +3,973%** (final ≈ 40,733 DAI).

## New Experiments (V2 assets)
All outputs are saved with `_V2` postfix.

### 1) **Hybrid_V2** (Regime-switch: mean‑revert in ranges, trend‑follow in uptrends)
- Files:  
  - 30d: [trades](sandbox:/mnt/data/hybrid_V2_30d_trades_V2.csv), [equity](sandbox:/mnt/data/hybrid_V2_30d_equity_V2.png)  
  - 90d: [trades](sandbox:/mnt/data/hybrid_V2_90d_trades_V2.csv), [equity](sandbox:/mnt/data/hybrid_V2_90d_equity_V2.png)  
  - 365d: [trades](sandbox:/mnt/data/hybrid_V2_365d_trades_V2.csv), [equity](sandbox:/mnt/data/hybrid_V2_365d_equity_V2.png)  
  - 730d: [trades](sandbox:/mnt/data/hybrid_V2_730d_trades_V2.csv), [equity](sandbox:/mnt/data/hybrid_V2_730d_equity_V2.png)
- Quick summary (net):  
  - 30d **+5.5%**, 90d **+67.9%**, 365d **+412.9%**, **730d −73.5%** (underperforms long-span)

### 2) **v3V2** (SMA reversion with dynamic exit_up; stronger z-mult; longer max-hold)
- Files:  
  - 30d: [trades](sandbox:/mnt/data/v3V2_30d_trades_V2.csv), [equity](sandbox:/mnt/data/v3V2_30d_equity_V2.png)  
  - 90d: [trades](sandbox:/mnt/data/v3V2_90d_trades_V2.csv), [equity](sandbox:/mnt/data/v3V2_90d_equity_V2.png)  
  - 365d: [trades](sandbox:/mnt/data/v3V2_365d_trades_V2.csv), [equity](sandbox:/mnt/data/v3V2_365d_equity_V2.png)  
  - 730d: [trades](sandbox:/mnt/data/v3V2_730d_trades_V2.csv), [equity](sandbox:/mnt/data/v3V2_730d_equity_V2.png)
- Quick summary (net):  
  - 30d **0%**, 90d **+62.2%**, 365d **+390.1%**, 730d **+1,325.6%** (final ≈ 14,256 DAI)

### 3) **MR Partial V2** (80/20 TP with runner & trailing)
- Files: 730d only for now: [trades](sandbox:/mnt/data/mr_partial_V2_730d_trades_V2.csv), [equity](sandbox:/mnt/data/mr_partial_V2_730d_equity_V2.png)
- Quick summary (net): **730d +2,539%** (final ≈ 27,397 DAI). Improves robustness vs v3V2 but still below the prior best 40,733 DAI.

### 4) **Tight Trend Follower V2** (EMA/Donchian/slope/strength)
- Files:  
  - 30d: [trades](sandbox:/mnt/data/ttf_v2_30d_trades.csv), [equity](sandbox:/mnt/data/ttf_v2_30d_equity.png)  
  - 90d: [trades](sandbox:/mnt/data/ttf_v2_90d_trades.csv), [equity](sandbox:/mnt/data/ttf_v2_90d_equity.png)  
  - 365d: [trades](sandbox:/mnt/data/ttf_v2_365d_trades.csv), [equity](sandbox:/mnt/data/ttf_v2_365d_equity.png)  
  - 730d: [trades](sandbox:/mnt/data/ttf_v2_730d_trades.csv), [equity](sandbox:/mnt/data/ttf_v2_730d_equity.png)
- Quick summary: **Underperformed** (trend signals whipsawed under real costs). Keep as a module for future regime filters; not enabled as default.

## New Modules (drop-in)
- **Hybrid strategy:** [strategy_hybrid_v2.py](sandbox:/mnt/data/strategy_hybrid_v2.py)  
- **Trend follower:** [strategy_trend_ttf_v2.py](sandbox:/mnt/data/strategy_trend_ttf_v2.py)

## Conclusion (so far)
- Your **original v2 SMA‑reversion** remains the champion on this pair and full 730d sample (final ~**40,733 DAI**).  
- The V2 hybrids and trend follow attempts did **not** surpass it yet. The partial‑take‑profit variant shows promise but still trails the best.

## Next to try (immediately runnable ideas)
- **Regime‑adaptive exit_up** with a *longer* target during confirmed uptrends (e.g., hold until `EMAfast>EMAslow` turns down, then TP).  
- **ATR‑clock max‑hold**: exit if no touch of SMA+target after X ATRs of time to boost capital velocity.  
- **Two‑tier TP**: TP1 at SMA+5%, TP2 at *SMA+5%+k·z⁺*, then trail.  
- **Time‑of‑day/session filter** to avoid low-liquidity chop.

