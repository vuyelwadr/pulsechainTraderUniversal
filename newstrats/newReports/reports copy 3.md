Got it — I dug through the repo structure and the artifacts you referenced, then put together concrete, code‑level changes you can drop in to address each ask. I can’t execute the runs from here, so I’m giving you surgical diffs, sensible defaults, and a one‑shot checklist to rerun + document the results.

Below you’ll find:

1. **CSMARevertStrategy** — drawdown control (< −40%) without killing WFO returns
2. **GridTradingStrategyV2Aggressive** — realistic execution (slippage, partial fills, gas) so PF isn’t ∞
3. **CompositeMomentumIndexStrategy** — regime/volatility gating to avoid collapses
4. **Validation workflow** — make the WFO more statistically robust and cheaper to run

Where I reference fees/AMM math or walk‑forward/CV best practices, I cite the primary sources (Uniswap V2 constant‑product & price impact; PulseX 0.29% fee; gas fee ranges; ATR/Chandelier stops; ADX/ER trend gating; purged/embargo CV). Your repo README also confirms the dataset has *real on‑chain prices* and that reserve columns are only present when a Sync occurs — we’ll lean on that for AMM slippage when available. ([GitHub][1])

---

## 0) What I verified in the repo

* The project layout and files you referenced are present (`strategies/`, `optimization/` with runner, scoring, orchestrator, evaluator; datasets & cost cache; strategy lists), and README describes 5‑minute real OHLCV from PulseX swaps & Sync events (reserves present only if a Sync occurred). That’s enough to implement an AMM execution model using reserve snapshots when available, otherwise fall back to a calibrated proxy. ([GitHub][1])

---

## 1) CSMARevertStrategy — cut staged DD below −40% (keep WFO gains)

**Diagnosis (based on your runs):** the revert logic is getting steam‑rolled in trend extensions. Your first ATR trail helped but left high staged drawdowns (~−90%). We need to:

* **Gate entries** when the market is strongly trending (mean‑reversion dies in trends).
* **Normalize risk to volatility** so position sizing shrinks in high‑ATR regimes.
* **Lock profits early** (scaled exit + breakeven) to tame tail risks.
* **Use a better trailing stop** (Chandelier‑style high‑minus‑k×ATR, only tightens) to ride winners while capping reversals. Chandelier/ATR trailing is a standard, well‑documented approach. ([chartschool.stockcharts.com][2])

**Regime filters that are simple and effective**

* **ADX cap** (e.g., only take reversion if ADX ≤ 20–22). Values <20–25 are widely used as “non‑trend” filters. ([Investopedia][3])
* **Kaufman Efficiency Ratio (ER)** cap (e.g., ER ≤ 0.30–0.35). ER measures how “trend‑like” price action is; lower = choppy/mean‑revertable. ([help.tc2000.com][4])
* **Vol‑of‑Vol (VoV) z‑score** cap to avoid chaotic spikes that blow through stops (e.g., z(|ΔATR%|) ≤ 2).

**Scaled exits & break‑even shift**

* **TP1 at ~1.4–1.8×ATR** on 40–60% of size; **move stop to entry+fees** after TP1 (or to entry+fees+half spread).
* Chandelier trail thereafter: `stop = max(prev_stop, highestSinceEntry – atr_mult_trail * ATR)` for longs (mirror for shorts). ([chartschool.stockcharts.com][2])

**Vol‑normalized sizing**

* Size so **risk_per_trade ≈ target** (e.g., 1% of equity per trade) using ATR%: `size = min(max_leverage, risk_target / ATR%)`.

### Patch (drop‑in diff)

> **File:** `strategies/CSMARevertStrategy.py`
> (Names may differ slightly; adapt to your base strategy API.)

```diff
 class CSMARevertStrategy(BaseStrategy):
-    # existing params ...
+    params = dict(
+        # existing CSMA params...
+        atr_len=48,
+        atr_mult_stop=2.2,          # initial catastrophic stop
+        atr_mult_trail=2.8,         # chandelier trail multiple
+        tp1_mult=1.6,               # partial take-profit at 1.6*ATR
+        tp1_size=0.5,               # scale-out 50% at TP1
+        breakeven_after_tp1=True,
+        adx_len=14,                 # regime filter: avoid strong trends
+        adx_max_for_entry=22,
+        er_len=20,                  # Kaufman Efficiency Ratio
+        er_max_for_entry=0.35,
+        vov_lookback=20,            # VoV filter
+        vov_z_max=2.0,
+        risk_per_trade=0.01,        # 1% of equity
+        max_leverage=1.0
+    )

+    def _efficiency_ratio(self, close, n):
+        # ER in [0,1]; 0=noisy chop; 1=straight line
+        if len(close) < n+1: return 1.0
+        change = abs(close[-1] - close[-1-n])
+        volatility = np.sum(np.abs(np.diff(close[-1-n:])))
+        return 0.0 if volatility == 0 else change / volatility

+    def _atr_percent(self, high, low, close, n):
+        tr = np.maximum(high[1:], close[:-1]) - np.minimum(low[1:], close[:-1])
+        atr = pd.Series(tr).ewm(span=n, adjust=False).mean().iloc[-1]
+        return atr / close[-1]

+    def _zscore(self, series, n):
+        s = pd.Series(series[-n:])
+        return 0.0 if s.std() == 0 else (s.iloc[-1] - s.mean()) / s.std()

+    def _vol_filters(self, i):
+        adx = ta.trend.adx(self.high[:i+1], self.low[:i+1], self.close[:i+1], window=self.params['adx_len']).iloc[-1]
+        er  = self._efficiency_ratio(self.close[:i+1], self.params['er_len'])
+        atrp_series = 100 * pd.Series([self._atr_percent(self.high[:j+1], self.low[:j+1], self.close[:j+1], self.params['atr_len'])
+                                       for j in range(max(self.params['atr_len']+1, 2), i+1)])
+        vov_z = self._zscore(atrp_series.values, self.params['vov_lookback']) if len(atrp_series) >= self.params['vov_lookback'] else 0.0
+        return (adx <= self.params['adx_max_for_entry']) and (er <= self.params['er_max_for_entry']) and (abs(vov_z) <= self.params['vov_z_max'])

+    def _risk_normalized_size(self, i):
+        atrp = self._atr_percent(self.high[:i+1], self.low[:i+1], self.close[:i+1], self.params['atr_len'])
+        target = self.params['risk_per_trade']
+        return float(np.clip(target / max(1e-6, atrp), 0.0, self.params['max_leverage']))

+    def _place_entry_and_init_stops(self, side, i):
+        size = self._risk_normalized_size(i)
+        price = self.close[i]
+        self.enter(side=side, size=size, price=price)  # use your framework’s enter()
+        atr = self._atr_percent(self.high[:i+1], self.low[:i+1], self.close[:i+1], self.params['atr_len']) * price
+        stop0 = price - self.params['atr_mult_stop'] * atr if side=='long' else price + self.params['atr_mult_stop'] * atr
+        self.set_stop(stop0)  # catastrophic stop

+    def _manage_trade(self, i):
+        if not self.position: return
+        price = self.close[i]; entry = self.position.entry_price
+        atr_abs = self._atr_percent(self.high[:i+1], self.low[:i+1], self.close[:i+1], self.params['atr_len']) * price
+        # TP1
+        if not self.position.meta.get('tp1_done'):
+            tp1 = entry + self.params['tp1_mult']*atr_abs if self.position.is_long else entry - self.params['tp1_mult']*atr_abs
+            if (self.position.is_long and price >= tp1) or (self.position.is_short and price <= tp1):
+                self.scale_out(self.params['tp1_size'])
+                self.position.meta['tp1_done'] = True
+                if self.params['breakeven_after_tp1']:
+                    be = entry + self.estimated_fees_slippage(entry) * (1 if self.position.is_long else -1)
+                    self.set_stop(max(self.stop, be) if self.position.is_long else min(self.stop, be))
+        # Chandelier trail
+        hh = float(np.max(self.high[self.position.entry_index:i+1]))
+        ll = float(np.min(self.low[self.position.entry_index:i+1]))
+        trail = (hh - self.params['atr_mult_trail']*atr_abs) if self.position.is_long else (ll + self.params['atr_mult_trail']*atr_abs)
+        self.set_stop(max(self.stop, trail) if self.position.is_long else min(self.stop, trail))

     def on_bar(self, i):
-        # existing entry/exit logic
+        self._manage_trade(i)
+        if not self.position and self._vol_filters(i):
+            # mean-revert toward SMA only in non-trend regime
+            side = 'long' if self.close[i] < self.sma[i] else 'short'
+            self._place_entry_and_init_stops(side, i)
```

**How to sweep quickly**

* Keep your **current ATR lengths**; sweep `tp1_mult ∈ {1.4, 1.6, 1.8}`, `tp1_size ∈ {0.4, 0.5, 0.6}`, `atr_mult_trail ∈ {2.4, 2.8, 3.2}`, and `adx_max_for_entry ∈ {20, 22, 25}` with **~120 calls** for CSMA only.
* **Success criteria:** staged **max DD ≥ −40%**; WFO hold‑out ≥ +10–15% median; trades **≥ 30** per walk‑forward sweep so it’s not under‑trading.

---

## 2) GridTradingStrategyV2Aggressive — realistic execution (no ∞ PF)

Your current perfect fills are causing zero recorded losses → ∞ PF. Replace the fill model with an **AMM‑aware execution model** that adds:

* **AMM price impact** using the **constant‑product** formula with fee on input (Uniswap V2 mechanics), which PulseX (a UniV2‑style AMM) follows. ([Uniswap Docs][5])
* **DEX fee** at **0.29% per swap** (PulseX published fee). ([pulsex.com][6])
* **Gas randomness** sampled from a log‑normal or triangular distribution around current swap costs (e.g., ~$0.008 median, heavy‑tailed when the chain gets busy). ([gopulse.com][7])
* **Partial fills (order slicing)** — execute a grid order in N slices within the bar; only a fraction fills if price barely tags the level.

### A small, pluggable execution engine

> **New file:** `optimization/engine/execution_models.py`

```python
# execution_models.py
from dataclasses import dataclass
import numpy as np

@dataclass
class AMMParams:
    dex_fee: float = 0.0029           # PulseX 0.29%
    slices: int = 4                    # per order
    gas_usd_logn_mean: float = 0.007   # median gas cost in USD (approx)
    gas_usd_logn_sigma: float = 0.6
    min_fill_touch_ratio: float = 0.05 # if bar barely breaks, fill fraction is small

class AMMExecutionModel:
    def __init__(self, params: AMMParams):
        self.p = params

    def _sample_gas(self, rng):
        # lognormal around mean (approx) with fat tail
        mu = np.log(self.p.gas_usd_logn_mean) - 0.5*self.p.gas_usd_logn_sigma**2
        return float(np.exp(rng.normal(mu, self.p.gas_usd_logn_sigma)))

    def _slice_weights(self, rng, n):
        w = rng.dirichlet(alpha=np.ones(n))
        return w

    def _price_impact_v2(self, x_reserve, y_reserve, dx_after_fee):
        # Uniswap V2: x*y=k; output dy = y - k/(x+dx_after_fee)
        k = x_reserve * y_reserve
        return y_reserve - (k / (x_reserve + dx_after_fee))

    def simulate_fill(self, side, qty_base, bar, reserves, rng):
        """
        side: 'buy' (spend quote to get base) or 'sell' (sell base for quote)
        qty_base: desired base amount
        bar: dict with open/high/low/close mid or best estimate
        reserves: dict like {'base': x_res, 'quote': y_res} at or near trade time
        rng: np.random.Generator
        Returns: avg_fill_price, filled_qty_base, total_costs_usd
        """
        if self.p.slices <= 1:
            slices = [1.0]
        else:
            slices = self._slice_weights(rng, self.p.slices)

        filled_base = 0.0
        notional_usd = 0.0

        # Fill fraction heuristic: if touched level barely, reduce slice size
        touch_ratio = max(0.0, (bar['high'] - bar['low'])) / max(1e-9, bar['close'])
        touch_scale = max(self.p.min_fill_touch_ratio, min(1.0, touch_ratio))

        x = reserves['quote'] if side == 'buy' else reserves['base']
        y = reserves['base']  if side == 'buy' else reserves['quote']

        for w in slices:
            target_base = qty_base * w * touch_scale
            if target_base <= 0: continue
            # invert to input amount with price impact
            # start with small Newton iteration: find dx s.t. dy == target_base
            # apply fee on input
            dx = 0.0
            for _ in range(12):
                dx_eff = dx * (1.0 - self.p.dex_fee)
                dy = self._price_impact_v2(x, y, dx_eff)
                err = (dy if side=='buy' else dx_eff) - (target_base if side=='buy' else target_base) # symmetric
                if abs(err) < 1e-9: break
                dx += err * 0.5
                dx = max(dx, 0.0)
            dx_eff = dx * (1.0 - self.p.dex_fee)
            dy = self._price_impact_v2(x, y, dx_eff)

            if side == 'buy':
                got_base = dy
                paid_quote = dx
                avg_px = paid_quote / max(1e-9, got_base)
            else:
                # selling base → receive quote; inverse leg:
                # reuse same function by swapping roles; approximate avg price:
                sold_base = target_base
                # compute received quote using constant product in the other direction
                # (for brevity, approximate with mid minus same impact)
                received_quote = self._price_impact_v2(y, x, sold_base * (1.0 - self.p.dex_fee))
                got_base = sold_base
                paid_quote = -received_quote
                avg_px = received_quote / max(1e-9, sold_base)

            filled_base += got_base
            notional_usd += avg_px * got_base

            # update reserves after slice (impact accumulates)
            if side == 'buy':
                x += dx_eff
                y -= got_base
            else:
                y += paid_quote * (1.0 - self.p.dex_fee)
                x -= got_base

        gas_cost = self._sample_gas(rng)
        total_costs = gas_cost
        avg_fill = notional_usd / max(1e-9, filled_base)
        return avg_fill, filled_base, total_costs
```

> **File:** `optimization/engine/evaluator.py` (inject the model)

```diff
-class Evaluator:
-    def __init__(..., slippage_model=None, ...):
+class Evaluator:
+    def __init__(..., execution_model=None, ...):
         ...
-        self.slippage_model = slippage_model
+        self.execution_model = execution_model

     def _execute(self, order, bar, ctx):
-        # OLD: mid/close fill ± bucketed costs
-        price = bar['close']
-        fee  = ctx.fee_lookup(price, order.size)
-        return price, order.size, fee
+        # NEW: AMM-aware fills
+        reserves = ctx.get_reserves_near(bar['ts'])  # uses Sync-based snapshot if present
+        avg_price, filled, extra_costs = self.execution_model.simulate_fill(
+            'buy' if order.is_buy else 'sell',
+            qty_base=abs(order.size),
+            bar=bar,
+            reserves=reserves or ctx.proxy_reserves(bar),  # fallback proxy if no Sync
+            rng=ctx.rng
+        )
+        fee = 0.0  # AMM fee priced inside avg_price (input side)
+        return avg_price, filled, (fee + extra_costs)
```

> **File:** `optimization/runner_cli.py` (CLI switch)

```diff
 parser.add_argument('--exec-model', default='amm', choices=['amm','naive'])
 parser.add_argument('--slices', type=int, default=4)
 parser.add_argument('--gas-lognorm-mean-usd', type=float, default=0.007)
 parser.add_argument('--gas-lognorm-sigma', type=float, default=0.6)
```

> **File:** `optimization/orchestrator.py` (wire params)

```diff
 if args.exec_model == 'amm':
     from optimization.engine.execution_models import AMMExecutionModel, AMMParams
     exec_model = AMMExecutionModel(AMMParams(
         slices=args.slices,
         gas_usd_logn_mean=args.gas_lognorm_mean_usd,
         gas_usd_logn_sigma=args.gas_lognorm_sigma
     ))
 else:
     exec_model = None
 evaluator = Evaluator(..., execution_model=exec_model, ...)
```

**Notes**

* The **0.29% fee** is taken on the input in UniV2‑style AMMs (PulseX). We fold it into the price impact computation (as above). ([Uniswap Docs][5])
* For **reserves**, call a helper that returns the latest Sync (if your collector saved them); README says reserves only exist if a Sync occurred inside the candle — when missing, use a **proxy** (e.g., liquidity inferred from your slippage study or a rolling median reserve). ([GitHub][1])
* **Gas**: current public trackers put PulseChain swap costs in the **fractions of a cent to ~1 cent** range; we seed the log‑normal around **$0.007** for swaps, adjust later with your `pdai_slippage_analysis.md`. ([gopulse.com][7])

**Result you should see after re‑run**

* PF becomes **finite** (expect 1.4–3.0 depending on grid spacing), win‑rate dips from 100% to something realistic, and expectancy per trade compresses. The equity curve should still be smooth, but not “perfect.”

---

## 3) CompositeMomentumIndexStrategy — regime/vol gating

**Goal:** keep the momentum edge but avoid the OOS windows where it collapses.

**Two simple gates (cheap to compute)**

* **Trend‑quality minimum**: only take signals when **ER ≥ 0.35–0.45** (i.e., market is moving directionally, not chopping). ([help.tc2000.com][4])
* **Volatility band**: realized **ATR% in [p25, p85]** percentile of last 180–252 bars — too low → no follow‑through; too high → whipsaws/exhaustion.

**Optional: ADX floor** — require **ADX ≥ 20–25** while ER is high to confirm trending conditions. ([Investopedia][3])

> **File:** `strategies/CompositeMomentumIndexStrategy.py`

```diff
 params.update(dict(
     er_len=20,
-    # no gates
+    er_min_for_entry=0.40,
+    atrp_window=252,
+    atrp_min_pct=0.25,
+    atrp_max_pct=0.85,
+    adx_len=14,
+    adx_min_for_entry=20
 ))

 def _regime_ok(self, i):
     window = slice(max(0, i- self.params['atrp_window']), i+1)
     er  = self._efficiency_ratio(self.close[:i+1], self.params['er_len'])
     atrp = self._atr_percent(self.high[window], self.low[window], self.close[window], n=14_series) # use your ATR
     p = np.nanpercentile(atrp, [100*self.params['atrp_min_pct'], 100*self.params['atrp_max_pct']])
     adx = ta.trend.adx(self.high[:i+1], self.low[:i+1], self.close[:i+1], window=self.params['adx_len']).iloc[-1]
-    return True
+    return (er >= self.params['er_min_for_entry']) and (p[0] <= atrp[-1] <= p[1]) and (adx >= self.params['adx_min_for_entry'])

 def on_bar(self, i):
-    if self.signal_is_long(i): self.buy()
+    if self._regime_ok(i) and self.signal_is_long(i): self.buy()
```

**Expectations:** slightly fewer trades, higher PF & Sharpe on OOS chunks that previously failed.

---

## 4) Validation workflow — make WFO tougher and faster

**What you have:** WFO with window **90d**, step **30d**, **top N=5**, **200 calls** each step.

**Recommended changes**

1. **Train/OOS proportions:**  **120d train / 30d OOS**, step **30d** (still overlapping). This gives more data per fit while keeping OOS fresh.
2. **Purged / embargo cross‑validation inside each training window** (for parameter selection), then one OOS pass — this *materially* reduces leakage vs naïve K‑Fold or single IS fit. Keep it small (e.g., **PurgedKFold=3**, **purge=2 days**, **embargo=1 day**) to control runtime. These techniques are the standard way to avoid information leakage in financial time series. ([Wikipedia][8])
3. **Cheaper search**: drop **calls from 200 → 140** for shortlist runs (CSMA‑only sweeps can stay at 120).
4. **Score for robustness** in `optimization/scoring_engine.py`:

   * Prefer **OOS Calmar** (CAGR / |maxDD|), **OOS Sharpe**, and **Winsorized PF** (PF with small ε in the denominator to avoid ∞); add a **penalty if trades<OOS_min**.
   * Example: `score = 0.45*Sharpe_OOS + 0.35*Calmar_OOS + 0.20*PF_oos_winsor - 0.10*penalty_low_trades`.

> **File:** `optimization/orchestrator.py`

```diff
 parser.add_argument('--train-days', type=int, default=120)
 parser.add_argument('--oos-days', type=int, default=30)
 parser.add_argument('--wfo-step-days', type=int, default=30)
 parser.add_argument('--cv-folds', type=int, default=3)
 parser.add_argument('--cv-purge-days', type=int, default=2)
 parser.add_argument('--cv-embargo-days', type=int, default=1)
```

> **File:** `optimization/engine/evaluator.py` (inside a training window)

```diff
-params = sample_params(n=args.calls)
-best = max((score(train(params_i)), params_i) for params_i in params)
+splits = build_purged_splits(train_index, folds=args.cv_folds,
+                             purge_days=args.cv_purge_days, embargo_days=args.cv_embargo_days)
+def cv_score(p):
+    scores = []
+    for tr_idx, va_idx in splits:
+        scores.append(score(train_on(tr_idx, p), validate_on(va_idx, p)))
+    return np.median(scores)
+params = sample_params(n=args.calls)
+best = max((cv_score(p), p) for p in params)
 # then evaluate best on OOS window
```

**Why this matters:** walk‑forward is good, but selecting params with **purged/embargo CV** *within* each IS window is meaningfully more robust than a single fit and keeps OOS honest. ([Wikipedia][8])

---

## 5) Rerun checklist (one pass)

**Install / run (as you already do):**

```bash
pip install -r requirements.txt

# shortlist with AMM execution enabled
.venv/bin/python -m optimization.runner_cli \
  --strategies-file strats_top_walkforward.json \
  --calls 140 \
  --trade-size 1000 \
  --cpu-fraction 0.8 \
  --exec-model amm \
  --slices 4 \
  --gas-lognorm-mean-usd 0.007 \
  --gas-lognorm-sigma 0.6 \
  --train-days 120 --oos-days 30 --wfo-step-days 30 \
  --cv-folds 3 --cv-purge-days 2 --cv-embargo-days 1
```

**Single‑strategy CSMA run (faster param sweep):**

```bash
.venv/bin/python -m optimization.runner_cli \
  --strategies-file strats_csma.json \
  --calls 120 \
  --trade-size 1000 \
  --cpu-fraction 0.8 \
  --exec-model amm --slices 3 --train-days 120 --oos-days 30 \
  --cv-folds 3 --cv-purge-days 2 --cv-embargo-days 1
```

**Spot checks / exports (unchanged):**

```bash
scripts/evaluate_vost_strategies.py --trade-size 1000
scripts/export_strategy_performance_csv.py
```

---

## 6) What to write into `newstrats/local.md`

Append under:

> **11. 2025‑10‑08 walk‑forward focus run** and **12. 2025‑10‑08 CSMA trailing‑stop iteration**:

* **13. 2025‑10‑08 CSMA drawdown control**

  * Filters: ADX≤22, ER≤0.35, VoV z≤2
  * Exits: TP1 at 1.6×ATR (50%), breakeven after TP1; Chandelier trail 2.8×ATR
  * Sizing: risk 1% via ATR%
  * *Target:* staged DD < −40%, WFO median hold‑out ≥ +10–15%, ~40–60 trades

* **14. 2025‑10‑08 Grid V2 realistic execution**

  * AMM price impact (constant product w/ 0.29% fee on input), partial fills (4 slices), log‑normal gas
  * *Target:* PF finite; win‑rate & expectancy realistic; smoother (but not perfect) equity

* **15. 2025‑10‑08 Composite Momentum gating**

  * Entry only if ER≥0.40, ADX≥20 and ATR% within [p25, p85] of last 252 bars
  * *Target:* Avoid collapse windows; keep 60%+ of prior WFO edge

* **16. 2025‑10‑08 WFO & scoring changes**

  * 120/30d WFO; PurgedKFold(3, purge=2d, embargo=1d) for IS selection
  * Score: 0.45*Sharpe_OOS + 0.35*Calmar_OOS + 0.20*PF_winsor – low‑trade penalty

(Where you cite background: Uniswap V2 AMM pricing and price impact; PulseX 0.29% fee; gas magnitudes; ATR/Chandelier exits; ADX & ER gating; purged‑embargo CV.) ([Uniswap Docs][5])

---

## 7) Suggested commit messages (include key metrics)

> **feat(exec): AMM-aware execution (UniV2/ PulseX), partial fills, stochastic gas**
> - Replace naive fills with constant‑product impact + 0.29% input fee; 4‑slice partial fills; log‑normal gas (μ≈$0.007, σ≈0.6).
> - Runner: `--exec-model amm` + knobs.
> **Result (WFO 120/30 step 30)**: Grid V2 Aggressive PF 2.1 (from ∞), Win% 78%, MaxDD −17%, Trades 84, OOS CAGR +42%.

> **feat(csma): regime filters (ADX/ER/VoV), scaled TP1, chandelier trail, vol‑normalized size**
> **Result (staged)**: MaxDD −36% (from −90%); **WFO** median hold‑out +14%, Trades 47, Calmar 0.35.

> **feat(cmi): trend & vol gating (ER/ADX + ATR% band)**
> **Result (WFO)**: OOS PF 1.6→2.0, fewer collapses on weak windows; trades −18%.

> **feat(validation): Purged/Embargo CV inside each IS; 120/30 WFO; robust scoring**
> Reduced compute (calls 200→140), tighter dispersion of OOS metrics.

(Replace metrics with your actual rerun numbers from `reports/optimizer_run_*/`.)

---

## 8) Why these choices (supporting references)

* **AMM impact model** — UniV2 price impact is governed by the **constant product** formula; **price impact** is the difference between mid and execution after accounting for the curve and fee. PulseX publishes a **0.29%** swap fee; we include it on the input leg (UniV2 mechanics). ([Uniswap Docs][5])
* **Gas** — PulseChain swap gas is typically **sub‑cent** (~$0.005–$0.01) with variability; treating it as log‑normal captures the heavy tail. ([gopulse.com][7])
* **ATR/Chandelier exits** — standard volatility‑adaptive trailing that locks in trend gains while capping DD. ([chartschool.stockcharts.com][2])
* **ADX (≤ ~20 for non‑trend / ≥ ~25 for trend)** and **Efficiency Ratio** gates are common, lightweight regime filters. ([Investopedia][3])
* **Purged/Embargo CV** reduces leakage when labels or features bleed over time; it’s a best practice for time‑series model selection and plays well with WFO. ([Wikipedia][8])

---

## 9) Final “how to test” (quick)

1. **Install & run shortlist** with `--exec-model amm` as above.
2. **Re‑inspect** `reports/optimizer_run_*` (CSVs + HTML): confirm **CSMA staged DD < −40%**, **Grid PF finite**, **CMI fewer OOS blowups**.
3. **Export** `strategy_performance_summary.csv` and paste the top‑line changes into commit messages.

---

### Small caveats

* If your dataset lacks Sync‑based reserves on a bar, the AMM model falls back to a proxy (e.g., rolling liquidity or your `swap_cost_cache`). You can refine the proxy with the analysis in `pdai_slippage_analysis.md`. ([GitHub][1])
* Names may differ a little across your base classes — the diffs show intent and minimal APIs to wire.

If you want, I can also add a tiny helper (`ctx.get_reserves_near(ts)`) and a one‑pager in `docs/` that explains the execution model math for future contributors.

[1]: https://github.com/vuyelwadr/pulsechainTraderUniversal "GitHub - vuyelwadr/pulsechainTraderUniversal"
[2]: https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-overlays/chandelier-exit?utm_source=chatgpt.com "Chandelier Exit - ChartSchool - StockCharts.com"
[3]: https://www.investopedia.com/articles/trading/07/adx-trend-indicator.asp?utm_source=chatgpt.com "ADX: The Trend Strength Indicator"
[4]: https://help.tc2000.com/m/69445/l/755884-kaufman-efficiency-ratio?utm_source=chatgpt.com "Kaufman Efficiency Ratio | Personal Criteria Formulas (PCF)"
[5]: https://docs.uniswap.org/contracts/v2/concepts/protocol-overview/glossary?utm_source=chatgpt.com "Glossary"
[6]: https://pulsex.com/?utm_source=chatgpt.com "PulseX.com"
[7]: https://gopulse.com/gas?utm_source=chatgpt.com "gas fees"
[8]: https://en.wikipedia.org/wiki/Purged_cross-validation?utm_source=chatgpt.com "Purged cross-validation"
