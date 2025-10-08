
# Iteration Report — V3 (Tuned Baseline Beats Prior Best)

**Champion (new):** SMA(576) dip-buy + RSI(14)≤30 + **exit at SMA + 4.8%** (cost-aware).  
This simple change from 5.0% → **4.8%** exit target **beats the previous best** on the full 730‑day sample.

## Results (start 1,000 DAI; net of bucketed costs)
- **730d:** **42,806 DAI** (ROI +4,180.6%) — _prior best was ~40,733 DAI_
- **365d:** **16,200 DAI** (ROI +1,520.1%)
- **90d:** **3,106 DAI** (ROI +210.6%)
- **30d:** **1,000 DAI** (no trades in last 30d under 25% dip rule)

## Files (_V2 postfix)
- 30d: [trades](sandbox:/mnt/data/tuned_sma_rsi_30d_trades_V2.csv), [equity](sandbox:/mnt/data/tuned_sma_rsi_30d_equity_V2.png)
- 90d: [trades](sandbox:/mnt/data/tuned_sma_rsi_90d_trades_V2.csv), [equity](sandbox:/mnt/data/tuned_sma_rsi_90d_equity_V2.png)
- 365d: [trades](sandbox:/mnt/data/tuned_sma_rsi_365d_trades_V2.csv), [equity](sandbox:/mnt/data/tuned_sma_rsi_365d_equity_V2.png)
- 730d: [trades](sandbox:/mnt/data/tuned_sma_rsi_730d_trades_V2.csv), [equity](sandbox:/mnt/data/tuned_sma_rsi_730d_equity_V2.png)

## Module
Use the tuned drop‑in module (defaults to `exit_up=0.048`):  
- [strategy_c_sma_revert_v2.py](sandbox:/mnt/data/strategy_c_sma_revert_v2.py)

Example (runner):
```bash
python -m optimization.runner   --strategy c_sma_revert_v2   --swap-cost-cache /path/to/swap_cost_cache.json   --csv /path/to/PAIR_5m.csv   --start-cash 1000   --params '{"n_sma":576,"entry_drop":0.25,"exit_up":0.048,"rsi_n":14,"rsi_max":30}'
```

## Notes
- I also built Hybrid V2 and Tight Trend Follower V2 modules; they’re useful for regime testing, but on this dataset they didn’t beat the tuned baseline on the full 730d span.
- Next ideas: adaptive exit scaling around **4.5–5.0%** tied to recent **volatility**, plus ATR‑clock timeouts for faster capital recycling.
