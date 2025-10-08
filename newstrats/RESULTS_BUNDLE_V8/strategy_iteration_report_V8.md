
# Iteration Report — V8 (Aggressive Runner, Cost-Aware)

**Change vs V7:** Increase constant runner to **15%** with trailing **30%** (uptrend-gated at TP).  
Entry/TP unchanged: SMA(576) dip-buy + RSI≤30, TP at **SMA + 4.8%**.

## Results (net of costs)
- 30d: 1000.00  | ROI 0.00%  — [equity](sandbox:/mnt/data/v8_runner15_30d_equity_V2.png), [trades](sandbox:/mnt/data/v8_runner15_30d_trades_V2.csv)
- 90d: 3106.44  | ROI 210.64%  — [equity](sandbox:/mnt/data/v8_runner15_90d_equity_V2.png), [trades](sandbox:/mnt/data/v8_runner15_90d_trades_V2.csv)
- 365d: 15356.36 | ROI 1435.64% — [equity](sandbox:/mnt/data/v8_runner15_365d_equity_V2.png), [trades](sandbox:/mnt/data/v8_runner15_365d_trades_V2.csv)
- 730d: **43521.92** | **ROI 4252.19%** — [equity](sandbox:/mnt/data/v8_runner15_730d_equity_V2.png), [trades](sandbox:/mnt/data/v8_runner15_730d_trades_V2.csv)

> New high watermark on 730d: **43521.92 DAI** with runner_frac=0.15, trail=0.30.

**Note:** These exports use a fast path (equity-only). For detailed trade logs, use the V7 module (identical logic with logging).
