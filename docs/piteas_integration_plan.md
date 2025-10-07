# Piteas Integration Blueprint

_Last updated: 2025-10-07_

The bot will eventually route **all swap decisions** through Piteas. This document captures the full requirements so the implementation is consistent once API credentials are available.

---

## 1. Scope

- Applies to **both live execution** and the **optimizer/backtester**.
- Replaces the current PulseX-only routing logic.
- Maintains pessimistic assumptions in backtests so live trading always matches or exceeds simulated performance.

---

## 2. Piteas API Baseline

| Item | Notes |
|------|-------|
| Endpoint | `GET https://sdk.piteas.io/quote` |
| Params | `chainId`, `tokenInAddress`, `tokenOutAddress`, `amount` (integer), `allowedSlippage` |
| Response | JSON containing `destAmount` (net of fees), `priceImpact`, `gasUseEstimate`, `methodParameters` (calldata + value), and route metadata |
| Rate limit (beta) | **10 requests per minute** unless higher quota is granted. Exceeding the cap can block requests for ~1 hour. |
| Authentication | None required for beta; production uses API key (obtain via `discord.gg/piteas`). |
| Fees | Already included in `destAmount`; no additional adjustment needed. |

Implementation must include: request throttling, retries with exponential backoff, and logging of each quote (input, output, price impact, route). All credentials must be stored via environment variables or secrets manager, never hard-coded.

---

## 3. Live Trading Workflow

1. **Quote Step**
   - Determine trade direction and notional size (USD).
   - Submit a single Piteas quote for the full amount.
   - Inspect `priceImpact`.

2. **Threshold check**
   - If `priceImpact ≤ 0.5%` (configurable) or notional ≤ $500, execute immediately using returned `methodParameters`.
   - Otherwise proceed to chunking.

3. **Chunked execution loop**
   ```pseudo
   remaining = total_notional_usd
   chunk_usd = min(initial_chunk_usd, remaining)
   while remaining > 0:
       ensure quote_budget has capacity (≤ 10 requests/min)
       quote chunk_usd
       if priceImpact ≤ threshold or chunk_usd ≤ min_chunk_usd:
           execute calldata
           remaining -= chunk_usd
       else:
           chunk_usd = max(chunk_usd / 2, min_chunk_usd)
   ```
   - Always re-quote before each chunk; Piteas rebalances routes after previous fills.
   - For each execution, record `destAmount`, `priceImpact`, route breakdown, gas estimate.

4. **Rate-limit management**
   - Use a token bucket per minute (capacity 10) to track outstanding quotes.
   - If the bucket is empty, sleep until at least one token refills (≈6 seconds per token) before requesting the next quote.

5. **Resilience**
   - Retry quotes on transient failures (HTTP errors, timeouts) with exponential backoff.
   - Abort the trade if retries exceed threshold or if Piteas returns a hard error.
   - Fallback path: log and notify; do not attempt blind execution without a valid quote.

---

## 4. Optimizer / Backtester

1. **Slippage Ladder (once per run)**
   - Build USD ladder: `[500, 1k, 2.5k, 5k, 7.5k, 10k, 15k, 20k, 30k, 40k, 50k]`.
   - For each rung:
     - Request buy quote (DAI ➜ PDAI).
     - Request sell quote (PDAI ➜ DAI).
   - Store `{side, notionalUSD, destAmount, priceImpact, gas, timestamp}`.
   - Persist to `cache/slippage_piteas_<start_ts>.json` and reference it in resume metadata.

2. **During simulation**
   - Convert trade size to USD.
   - Use the cached rung with `notionalUSD ≥ trade_size_usd` (ceiling). Optionally interpolate.
   - Apply the **worst of buy or sell** if direction is ambiguous (pessimistic default).
   - No periodic refresh; always use the run’s original cache to ensure deterministic comparisons.

3. **Fallbacks**
   - If ladder fetch fails (API down or rate limit hit), revert to constant-product estimate with +1% buffer and flag the run as “slippage fallback used”.

---

## 5. Testing Checklist

- **Quote client smoke test**: fetch entire ladder with the current rate limit, logging status codes and latency.
- **Chunking simulator**: dry-run the chunk loop (without sending tx) to ensure rate-limiter and retries function correctly.
- **Live dry-run**: broadcast a single real swap with test funds to confirm the calldata executes as expected.
- **Optimizer spot check**: run a slim optimization job (e.g., 2 strategies × 2 timeframes × 3 calls) to confirm cached slippage is applied and no live API calls occur mid-run.

---

## 6. Security & Ops Notes

- Store API keys/tokens (when issued) in `.env` / key vault; load via `Config`.
- Every live trade must log: timestamp, chunk size, dest amount, price impact, route composition, gas used.
- Consider alerting when cumulative price impact across chunks exceeds configuration or when the rate limit forces a delay mid-trade.
- Monitor API health; if sustained failures occur, disable live trading until resolved.

---

Once higher-throughput credentials arrive, this blueprint can be implemented across the bot and the optimizer without revisiting the requirements.
