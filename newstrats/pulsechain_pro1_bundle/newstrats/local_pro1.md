
# Local Notes — pro1 additions

This document summarizes **local-only** considerations for the `pro1` strategies.

## What you can change safely

- **Trade size used for cost-gating** (`trade_amount_quote` in each strategy).
  This *does not* change your real trade size — it only tells the gate how
  much cost to assume when deciding whether a signal clears the edge threshold.
- **Route key** (`route_key`), if your `swap_cost_cache.json` stores different
  pools/paths under specific keys (e.g., `"HEX/USDC"` vs `"HEX/DAI"`).

## What the strategies expect from data

Columns required by all three:
- `open`, `high`, `low`, `close` (5‑minute candles recommended)

Optional:
- `volume` (used by the liquidity-aware breakout filter)

## Quick smoke tests

```bash
# From repo root
python pulsechain_trading_bot.py --backtest --days 30 --strategy RegimeSwitchingPro1
python pulsechain_trading_bot.py --backtest --days 30 --strategy MeanReversionZScorePro1
python pulsechain_trading_bot.py --backtest --days 30 --strategy LiquidityAwareBreakoutPro1
```

If your local harness supports `signal_strength` for sizing, you should see
scaled position sizes; otherwise the engine will still use the boolean
`buy_signal` / `sell_signal` columns.
