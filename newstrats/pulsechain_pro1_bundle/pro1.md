
# PulseChain HEX Trading — pro1 Strategy Pack

**Goal:** Improve absolute return percentage *while* respecting real swap
costs from `swap_cost_cache.json`. Strategies are designed to be robust across
multiple time periods by switching behavior based on market regime and gating
all trades on cost-adjusted edge.

> ⚠️ This pack does **not** alter your existing pipeline. It only adds new,
> cost-aware strategies and a lightweight batch runner.

---

## What’s included

| File | Purpose |
|---|---|
| `utils/cost_model_pro1.py` | Flexible reader for `swap_cost_cache.json` + cost gate utilities |
| `strategies/regime_switching_pro1.py` | Regime‑switching (trend ↔︎ mean‑reversion) with Donchian + EMAs |
| `strategies/mean_reversion_zscore_pro1.py` | Conservative EMA‑anchored z‑score mean‑reversion |
| `strategies/liquidity_aware_breakout_pro1.py` | Donchian breakout filtered by on‑chain volume/liquidity |
| `newstrats/local_pro1.md` | Local notes on parameters, running quick tests |
| `scripts/run_backtests_pro1.sh` | Batch runner across common lookbacks |

Every new file uses the **`pro1`** suffix so it’s easy to attribute the origin.

---

## Design principles

1. **Cost awareness first.** Every entry/exit is gated by an estimate of
   round‑trip cost from `swap_cost_cache.json`. You only trade when the
   expected move is comfortably larger than fees + slippage.
2. **Regime adaptability.** Markets alternate between trending and sideways.
   We use a rolling **R² of log‑price** and a **normalized ATR** to pick a
   behavior suited to the present regime.
3. **Signal strength for sizing.** We emit `signal_strength` in `[0,1]`
   proportional to *edge/cost*. If your engine sizes from strength, these
   strategies naturally trade larger when the edge is big and sit out when
   it’s marginal.
4. **No fragile indicators.** Only EMAs, Donchian channels, ATR, and simple
   regressions — minimal overfitting, easy to reason about.

---

## Respecting `swap_cost_cache.json`

We do **not** assume a specific schema. The loader looks for keys such as
`fee_bps`, `slippage_bps`, `avg_bps`, `total_bps`, or nested `median.bps`.
If a route‑specific key is present (e.g., `"HEX/DAI"`), pass it via
`route_key` to the strategy. Otherwise we fall back to `"default"`.

> If your JSON looks different, tweak `_extract_cost_bps_from_cache_entry(...)`
> in `utils/cost_model_pro1.py` — it’s centralized for this reason.

---

## How to install (drop‑in)

From the repository root:

```
# 1) Copy the files into place (preserving folders)
#    utils/cost_model_pro1.py
#    strategies/regime_switching_pro1.py
#    strategies/mean_reversion_zscore_pro1.py
#    strategies/liquidity_aware_breakout_pro1.py
#    newstrats/local_pro1.md
#    scripts/run_backtests_pro1.sh

# 2) (Optional) Add to your strategy registry, e.g. in validate_strategies.py
# AVAILABLE_STRATEGIES.update({
#     "RegimeSwitchingPro1": "strategies.regime_switching_pro1:RegimeSwitchingPro1",
#     "MeanReversionZScorePro1": "strategies.mean_reversion_zscore_pro1:MeanReversionZScorePro1",
#     "LiquidityAwareBreakoutPro1": "strategies.liquidity_aware_breakout_pro1:LiquidityAwareBreakoutPro1",
# })

# 3) Run a few quick tests
bash scripts/run_backtests_pro1.sh
```

If your CLI is different, run something akin to:

```
python pulsechain_trading_bot.py --backtest --days 90  --strategy RegimeSwitchingPro1
python pulsechain_trading_bot.py --backtest --days 180 --strategy MeanReversionZScorePro1
python pulsechain_trading_bot.py --backtest --days 365 --strategy LiquidityAwareBreakoutPro1
```

---

## Strategy details

### 1) RegimeSwitchingPro1

- **Regime detection**: rolling R² on log‑price (window 50) and ATR normalized
  by its 200‑bar mean. If `R² ≥ 0.30` and `ATR_norm ≥ 0.90` → *trend*.
- **Trend behavior**: require EMA alignment (`21 > 55 > 200`) **and** a
  Donchian(20) breakout → long. Exit on close < EMA55.
- **Sideways behavior**: z‑score of `(close − EMA55)` with entry at `−1.25`,
  exit at `−0.20`.
- **Gating**: compute an **expected edge** in bps (channel height/ATR proxy),
  trade only when `edge ≥ 1.15 × (round‑trip cost)`.
- **Strength**: proportional to `(edge / round‑trip‑cost)` in `[0,1]`.

Why it generalizes: the engine only acts when there’s room beyond costs; it can
capture both persistent trends and range oscillations without whipsawing in
noisy chop.

### 2) MeanReversionZScorePro1

- EMA(60) anchor with dynamic residual volatility estimate.
- Enter when `z ≤ −1.6`, exit when `z ≥ −0.25`. Strong bias to cut risk quickly.
- Same cost gate; same strength mapping.

### 3) LiquidityAwareBreakoutPro1

- Donchian(55) breakout filtered by recent **on‑chain volume**: require
  `volume ≥ 0.8 × SMA(volume, 288)` (≈ one day at 5‑minute bars).
- Uses channel height as expected edge; gates on costs; sizes by strength.

---

## Suggested defaults (5‑minute data)

| Parameter | RegimeSwitchingPro1 | MeanReversionZScorePro1 | LiquidityAwareBreakoutPro1 |
|---|---:|---:|---:|
| `trade_amount_quote` | 500 | 500 | 500 |
| `route_key` | `"HEX/DAI"` | `"HEX/DAI"` | `"HEX/DAI"` |
| `min_edge_multiple` | 1.15 | 1.20 | 1.20 |

These values only affect the **gate**; your real position sizing remains up to
the engine (or `signal_strength` if used).

---

## Measuring performance (return %)

To compare apples-to-apples across periods:

1. Use the **same dataset** (real on‑chain candles) and the **same cost cache**.
2. Test multiple lookbacks (`7, 30, 90, 180, 365, 730` days).
3. Report at least: Total Return %, Max Drawdown, Profit Factor, and Trades.

The `scripts/run_backtests_pro1.sh` helper runs a balanced grid. Capture the
generated HTML/JSON reports from your harness as usual.

> I didn’t execute live backtests from here (no chain access in this environment),
> but the strategy logic is deterministic. After you run your local harness, paste
> the key metrics below for a permanent record.

---

## Results (fill after running)

| Strategy | 30d Return | 90d Return | 180d Return | 365d Return | 730d Return | Max DD | PF | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| RegimeSwitchingPro1 | | | | | | | | |
| MeanReversionZScorePro1 | | | | | | | | |
| LiquidityAwareBreakoutPro1 | | | | | | | | |

---

## Next steps

1. **Walk‑Forward Optimization**: tune `r2_trend_threshold`, `mr_band_z`,
   and `donchian_n` on expanding IS/OOS windows (keep cost gate on).
2. **Route‑specific costs**: if your cache has per‑AMM routes, pass the
   `route_key` correctly and/or split strategies per route.
3. **Strength → sizing**: if your engine supports it, map `signal_strength`
   to max trade % with a concave function (e.g., `size = strength^0.6`).
4. **Event filters**: optionally skip bars where `gas` spikes if you maintain
   a gas/latency cache (avoids stale quotes and slippage cliffs).
5. **Multi‑timeframe confirm**: for 15m/1h data, scale EMA/Donchian windows
   ~linearly with bar duration; regime detection remains unchanged.

---

## Safety & disclaimers

- This is **not financial advice**. You bear the risk of using automated
  strategies on-chain. Always demo/backtest first with **real** on-chain data
  and fees.
- The cost gate prevents many unprofitable micro‑trades but it cannot
  eliminate all adverse selection during regime shifts.

---

### Credits & attribution

Produced by the **pro1** agent. All new files include the `pro1` suffix.
