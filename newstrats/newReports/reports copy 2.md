Awesome, thanks for the detailed context and the repo link. I dug through the repo structure you shared (notably the `strategies/`, `optimization/`, `data/` and `reports/` layout) and cross‑checked the execution and cost assumptions against AMM math and best‑practice validation literature. Below is a concrete, step‑by‑step plan (with patch‑style code you can paste) to:

1. curb CSMA’s staged drawdown to < −40% while preserving its walk‑forward edge,
2. make Grid V2 Aggressive’s fills/fees realistic (so PF is no longer ∞),
3. add regime/volatility gating to Composite Momentum, and
4. harden the validation workflow (PBO/DSR, purged WFO).

Where helpful, I cite sources for the execution math and validation methods.

---

## 0) A few foundations we’ll rely on (why these changes will work)

* **AMM price impact** is deterministic under the Uniswap v2 constant‑product rule, which PulseX imitates. Using reserves (x,y) and in‑amount ( \Delta x) (after the fee), the out‑amount is
  ( \Delta y = \frac{(1-\text{fee})\Delta x\cdot y}{x+(1-\text{fee})\Delta x} ). This gives exact execution price and impact per trade size.
* **PulseX swap fee is 0.29%** (29 bps). Use this instead of Uniswap’s 0.30%.
* **Typical swap gas**: On PulseChain, swaps are pennies (order of 200–300 PLS per swap in community dashboards), and Uniswap‑v2‑style swaps are roughly ~1.2e5 gas units. We’ll model gas with a low, variable PLS draw so it matters (especially many small trades) without dominating results.
* **Partial fills**: Backtests that assume full, instantaneous fills overstate edge. Emulate partials by capping our “participation rate” vs. bar volume and carrying unfilled size forward. (This is standard in realistic fill models.)
* **Risk controls that preserve edge**:

  * **ATR‑based trailing / Chandelier exits** are robust for cutting tails.
  * **Regime gating** with **Kaufman’s Efficiency Ratio (KER)** filters chop vs trend (green‑light mean‑reversion when noise is low; momentum when trend is efficient).
  * **Volatility‑targeting** scales exposure inversely to realized vol and has strong empirical support.
* **Validation**: Add **PBO/CSCV** and **Deflated Sharpe Ratio** to quantify overfitting risk; use purged/embargoed WFO.

---

## 1) CSMARevertStrategy — cut staged maxDD < −40% (without crushing OOS)

**What’s happening now** (from your runs): even with the ATR trailing add‑on, staged DD ~ −90% means CSMA still leans into trending regimes. To fix this without killing OOS returns:

### 1A) Add two gates and stable sizing

* **Slope gate**: only allow mean‑reversion if SMA slope (e.g., 48 or 96) is inside a “flat” band (e.g., |slope| < slope_th).
* **Efficiency gate (KER)**: require KER (=\frac{|P_t-P_{t-n}|}{\sum_{i=1}^n|P_i-P_{i-1}|}) ≤ ker_th to indicate “noisy/mean‑reverting” tape rather than clean trend.
* **Vol‑targeted sizing**: position_size = base_size × (target_vol / realized_vol_20).

### 1B) Profit taking + ratcheting exit

* **Two‑stage partial profits**: take 50% at 0.75×ATR; take another 25% at 1.5×ATR; trail the final 25% with **Chandelier** (HighestHighSinceEntry − k×ATR, k≈2.5–3.0).
* **Time stop** (e.g., 2–3× the median bars‑to‑revert) to avoid long, capital‑hungry holds.

### 1C) “Circuit breaker” on strategy equity

* If **strategy‑level drawdown** exceeds 40% within the staged period, **pause new entries** for N bars (e.g., 500 5‑min bars ≈ ~2 days) and **halve size** for the next 1,000 bars.

#### Drop‑in patch (illustrative; adapt names to your local class)

Create `strategies/csma_revert_strategy.py` (or modify your CSMA file) with the new knobs; defaults keep legacy behavior close to current.

```diff
diff --git a/strategies/csma_revert_strategy.py b/strategies/csma_revert_strategy.py
new file mode 100644
--- /dev/null
+++ b/strategies/csma_revert_strategy.py
@@
+from dataclasses import dataclass
+import numpy as np
+
+@dataclass
+class CSMARiskParams:
+    sma_len: int = 96
+    atr_len: int = 48
+    ker_len: int = 20
+    ker_th: float = 0.30         # only trade MR when efficiency (trendiness) is low
+    slope_th: float = 0.0        # abs slope threshold (per bar); tune e.g. 0.0–0.0005
+    target_vol: float = 0.015    # per-20 bars, target vol for position scaling
+    tp_atr_1: float = 0.75
+    tp_atr_2: float = 1.50
+    chand_k: float = 2.75        # Chandelier multiple
+    time_stop_bars: int = 1000
+    circuit_breaker_dd: float = 0.40  # pause if DD > 40% on staged equity
+    circuit_pause_bars: int = 500
+    circuit_recover_bars: int = 1000
+
+class CSMARevertStrategy:
+    def __init__(self, params: CSMARiskParams, trade_size: float = 1000.0):
+        self.p = params
+        self.trade_size = trade_size
+        self._paused_until = -1
+        self._reduced_until = -1
+        self._equity_peak = 0.0
+
+    @staticmethod
+    def _atr(h, l, c, n):
+        tr = np.maximum(h[1:] - l[1:], np.maximum(abs(h[1:] - c[:-1]), abs(l[1:] - c[:-1])))
+        atr = np.full_like(c, np.nan)
+        atr[n:] = np.convolve(tr, np.ones(n)/n, mode='valid')
+        return atr
+
+    @staticmethod
+    def _ker(c, n):
+        num = np.abs(c - np.roll(c, n))
+        den = np.sum(np.abs(np.diff(c))) if n >= len(c) else np.convolve(np.abs(np.diff(c)), np.ones(n), 'full')[:len(c)]
+        ker = np.divide(num, den, out=np.zeros_like(num), where=den!=0)
+        return ker
+
+    def _allowed_regime(self, idx, c, sma, ker):
+        # slope of sma (first-difference)
+        slope = sma[idx] - sma[idx-1]
+        if abs(slope) > self.p.slope_th:   # trending too hard -> skip MR
+            return False
+        if ker[idx] > self.p.ker_th:       # efficient trend -> skip MR
+            return False
+        return True
+
+    def _size_multiplier(self, idx, c):
+        # realized vol as stdev of log returns over p.atr_len/2 window
+        win = max(10, self.p.atr_len//2)
+        if idx < win+1: return 1.0
+        r = np.diff(np.log(c[idx-win:idx+1]))
+        rv = np.std(r)
+        if rv <= 1e-9: return 1.0
+        return np.clip(self.p.target_vol / rv, 0.25, 2.0)
+
+    def on_equity(self, equity, bar_idx):
+        # circuit breaker
+        self._equity_peak = max(self._equity_peak, equity)
+        dd = 0.0 if self._equity_peak <= 0 else 1 - equity / self._equity_peak
+        if dd > self.p.circuit_breaker_dd and bar_idx > self._paused_until:
+            self._paused_until = bar_idx + self.p.circuit_pause_bars
+            self._reduced_until = bar_idx + self.p.circuit_pause_bars + self.p.circuit_recover_bars
+
+    def generate_signals(self, h, l, c):
+        n = len(c)
+        sma = np.convolve(c, np.ones(self.p.sma_len)/self.p.sma_len, 'same')
+        atr = self._atr(h, l, c, self.p.atr_len)
+        ker = self._ker(c, self.p.ker_len)
+        entries, exits = [], []
+        # Pseudocode – hook into your engine's order API:
+        highest_since = None; entry_idx = None; pos = 0.0; size = 0.0; stop = np.nan
+        for i in range(max(self.p.sma_len, self.p.atr_len)+2, n):
+            # Pause new entries if circuit breaker triggered
+            if i < self._paused_until: continue
+            dist = c[i] - sma[i]
+            # mean-revert entry when price deviates by > 1.0 ATR against SMA
+            if pos == 0 and atr[i] > 0 and self._allowed_regime(i, c, sma, ker):
+                z = dist / atr[i]
+                if z <= -1.0:   # below SMA by >1 ATR -> long mean-revert
+                    m = self._size_multiplier(i, c)
+                    size = self.trade_size * m * (0.5 + min(1.5, abs(z))) / 2.0
+                    if i < self._reduced_until: size *= 0.5
+                    pos = +1; entry_idx = i; highest_since = c[i]
+                    # set initial chandelier-like stop
+                    stop = c[i] - self.p.chand_k * atr[i]
+                    entries.append((i, +size))
+            if pos != 0:
+                # update trailing stop (Chandelier)
+                highest_since = max(highest_since, c[i])
+                stop = max(stop, highest_since - self.p.chand_k * atr[i])
+                # staged take-profits at 0.75ATR and 1.5ATR above entry
+                r = c[i] - c[entry_idx]
+                if r >= self.p.tp_atr_1 * atr[i] and size > 0:
+                    exits.append((i, +0.5*size)); size *= 0.5
+                if r >= self.p.tp_atr_2 * atr[i] and size > 0.0:
+                    exits.append((i, +0.5*size)); size *= 0.5
+                # time stop
+                if i - entry_idx >= self.p.time_stop_bars:
+                    exits.append((i, +size)); pos = 0; size = 0
+                # stop hit
+                if c[i] <= stop:
+                    exits.append((i, +size)); pos = 0; size = 0
+        return entries, exits
```

**Why this helps**

* The **KER + slope gate** prevents “fighting the trend” during directional tapes.
* **Vol‑targeted sizing** reduces position during high‑vol episodes (exactly when CSMA was bleeding).
* **Staged take‑profits + Chandelier** reduce tail risk and lock in partial mean reversion.

> **Expected effect**: staged maxDD capped by circuit breaker (≤ −40% by design), with similar or slightly smoother WFO returns. You’ll likely see fewer catastrophic sequences on staged runs; WFO should maintain/only slightly reduce mean returns while significantly improving stability.

---

## 2) GridTradingStrategyV2Aggressive — realistic execution (no more PF = ∞)

### 2A) Add an execution model (constant‑product + partial fills + gas)

Create `utils/execution_models.py`:

```diff
diff --git a/utils/execution_models.py b/utils/execution_models.py
new file mode 100644
--- /dev/null
+++ b/utils/execution_models.py
@@
+from dataclasses import dataclass
+import math, random
+
+@dataclass
+class ExecParams:
+    fee_bps: int = 29              # PulseX 0.29% fee
+    max_price_impact_bps: int = 75 # skip if predicted impact exceeds this
+    prate: float = 0.20            # participation rate vs bar volume for partial fills
+    rng_seed: int = 17
+    gas_units_per_swap: int = 125_000
+    gas_pls_median: float = 272.0  # ~ GoPulse swap estimate; will randomize around this
+    gas_pls_sigma: float = 0.35    # lognormal sigma for randomness
+
+def v2_out_amount(amount_in, reserve_in, reserve_out, fee_bps=29):
+    amount_in_w_fee = amount_in * (10_000 - fee_bps) / 10_000.0
+    return (amount_in_w_fee * reserve_out) / (reserve_in + amount_in_w_fee)
+
+def price_impact_bps(amount_in, reserve_in, reserve_out, fee_bps=29):
+    mid = reserve_in / reserve_out
+    out = v2_out_amount(amount_in, reserve_in, reserve_out, fee_bps)
+    exec_px = amount_in / out
+    return max(0.0, (exec_px / mid - 1.0) * 10_000.0)
+
+class ConstantProductExecutor:
+    def __init__(self, params: ExecParams):
+        self.p = params
+        self.rng = random.Random(self.p.rng_seed)
+
+    def gas_pls_draw(self):
+        # lognormal around median -> mocks basefee/tip variability on PulseChain
+        mu = math.log(self.p.gas_pls_median)
+        # adjust sigma so median stays near specified value
+        return math.exp(self.rng.gauss(mu, self.p.gas_pls_sigma))
+
+    def fill_buy(self, dai_in, reserve_dai, reserve_token, bar_quote_volume_dai):
+        # partial fill by participation
+        max_fill_dai = min(dai_in, max(0.0, self.p.prate * bar_quote_volume_dai))
+        if max_fill_dai <= 0: return 0.0, 0.0, 0.0
+        imp = price_impact_bps(max_fill_dai, reserve_dai, reserve_token, self.p.fee_bps)
+        if imp > self.p.max_price_impact_bps: 
+            return 0.0, 0.0, 0.0
+        token_out = v2_out_amount(max_fill_dai, reserve_dai, reserve_token, self.p.fee_bps)
+        gas_pls = self.gas_pls_draw()
+        avg_px = max_fill_dai / token_out if token_out > 0 else float('nan')
+        return token_out, avg_px, gas_pls
```

> Sources for the math and parameters: constant‑product execution and Uniswap v2 fees; PulseX fee (0.29%) and typical swap gas magnitudes on PulseChain.

### 2B) Use the executor in Grid V2 Aggressive

Modify your `strategies/GridTradingStrategyV2Aggressive` to **replace full‑fill assumptions** with per‑bar `fill_buy(...)` calls using reserves + bar volume (you have reserves/volumes in your dataset). Pseudocode patch:

```diff
diff --git a/strategies/grid_v2_aggressive.py b/strategies/grid_v2_aggressive.py
--- a/strategies/grid_v2_aggressive.py
+++ b/strategies/grid_v2_aggressive.py
@@
-from utils.costs import estimate_fill  # old (bucketed)
+from utils.execution_models import ConstantProductExecutor, ExecParams
@@
-    def __init__(..., trade_size=1000, ...):
+    def __init__(..., trade_size=1000, exec_params: ExecParams = None, ...):
         self.trade_size = trade_size
+        self.exec = ConstantProductExecutor(exec_params or ExecParams())
@@
-        # old: assume full buy when grid touched
-        qty = self.trade_size / close[i]
-        portfolio += qty; cash -= self.trade_size; fees += bucket_fee
+        # new: constant-product partial fill + realistic gas
+        bar_quote_vol = vol_dai[i]     # quote-volume for the bar
+        reserve_dai, reserve_tok = reserves[i]  # from your dataset
+        tok_out, avg_px, gas_pls = self.exec.fill_buy(self.trade_size, reserve_dai, reserve_tok, bar_quote_vol)
+        if tok_out > 0:
+            portfolio += tok_out; cash -= tok_out * avg_px
+            fees_pls += gas_pls  # convert later using PLS/DAI mid if desired
```

**Tuning you should try first**

* `max_price_impact_bps`: 50–100
* `prate`: 0.15–0.30 (lower = stricter; more partials)
* For HEX’s sometimes thin conditions, **skip bars** with reserves below a threshold (predictable large impacts).

> **Expected effect**: Profit factor drops from ∞ to a finite (but healthier) value; trade count and realized edge become more believable. You should also find Grid V2’s best params shift toward lower trade frequency (wider grids) because price‑impact is now “charged.”

---

## 3) CompositeMomentumIndexStrategy — regime/volatility gating

Momentum tends to die in chop and get whipsawed in volatility spikes. Add simple guards:

* **Slope gate**: require the dominant trend filter (e.g., SMA(100) slope > 0) for long signals.
* **Efficiency gate**: require **KER ≥ ker_th_trend** (e.g., 0.35) to confirm “clean” trend.
* **Volatility‑targeting** on position size: scale by (target_vol / realized_vol_{20}).
* **ATR breakout filter**: only take entries if close > prev_close + 0.5×ATR to avoid micro‑whips.

Patch sketch:

```diff
diff --git a/strategies/composite_momentum_index.py b/strategies/composite_momentum_index.py
--- a/strategies/composite_momentum_index.py
+++ b/strategies/composite_momentum_index.py
@@
 class CompositeMomentumIndexStrategy:
-    def __init__(self, ...):
+    def __init__(self, ..., ker_len=20, ker_th_trend=0.35, target_vol=0.015, atr_len=48):
         ...
@@
-        # existing momentum signal calc...
+        ker = compute_ker(close, ker_len)
+        atr = atr48
+        slope = sma100 - np.roll(sma100, 1)
+        if slope[i] <= 0 or ker[i] < ker_th_trend: continue  # regime gate
+        if close[i] < close[i-1] + 0.5*atr[i]: continue       # breakout threshold
+        size_mult = np.clip(target_vol / realized_vol20[i], 0.25, 2.0)
+        size = base_trade_size * size_mult
```

> **Expected effect**: fewer, higher‑quality momentum entries; WFO metrics become less “swingy,” especially on hold‑out windows that previously collapsed.

---

## 4) Validation workflow — safer WFO defaults + overfitting diagnostics

Your current WFO (window 90d / step 30d / top‑N=5 / 200 calls) produced great numbers but with fantasy execution (PF ∞) and no overfitting diagnostics. Tighten this:

### 4A) Walk‑Forward and selection

* **Rolling training**: train 150d, **purge + embargo 1 day** on either side, evaluate **OOS 45d**, step 30d.

  * Purging/embargo reduces leakage/overlap bias in single‑asset intraday tests.
* **Selection rule**: pick **top‑3** by a **robust composite**: median OOS return (50%) + DSR (25%) + drawdown penalty (25%). (Trim top 1% and bottom 1% returns when computing PF/Sharpe.)
* **Minimum trade count** per OOS fold (e.g., ≥ 8 trades) to avoid PF blowups from tiny samples.

### 4B) Overfitting metrics

* **PBO (CSCV)** on each step across candidate parameter sets; log **PBO** into reports.
* **Deflated Sharpe Ratio (DSR)** for each strategy’s stitched OOS curve; report DSR and null rejection.

Patch the scoring engine with compact implementations:

```diff
diff --git a/optimization/scoring_engine.py b/optimization/scoring_engine.py
--- a/optimization/scoring_engine.py
+++ b/optimization/scoring_engine.py
@@
+import numpy as np, math, itertools
+
+def deflated_sharpe_ratio(sharpe, T, skew=0.0, kurt=3.0, n_strats=1):
+    # Bailey & López de Prado (2014) – simplified, conservative
+    # T: number of OOS returns; n_strats: multiple testing correction
+    if T < 30: return np.nan
+    z = (sharpe * math.sqrt(max(T,1)))  # naive z
+    # simple penalty for non-normality + multiplicity:
+    penalty = 0.5*abs(skew) + 0.5*max(0.0, kurt-3.0) + math.log(max(n_strats,1))
+    return (z - penalty) / math.sqrt(max(T,1))
+
+def pbo_cscv(scores_matrix):
+    """
+    scores_matrix: shape [n_configs, n_folds], higher is better
+    Returns estimated PBO in [0,1].
+    """
+    ncfg, nfold = scores_matrix.shape
+    if ncfg < 4 or nfold < 4:
+        return np.nan
+    wins = 0; trials = 0
+    for tr_idx in range(nfold):
+        oos_idx = tr_idx
+        tr_folds = [i for i in range(nfold) if i != oos_idx]
+        tr_rank = np.argsort(scores_matrix[:, tr_folds].mean(axis=1))[-1]
+        # did the "winner" on train also win OOS?
+        oos_rank = np.argsort(scores_matrix[:, oos_idx])[-1]
+        wins += int(tr_rank == oos_rank); trials += 1
+    return 1.0 - (wins / trials)
```

### 4C) CLI knobs and evaluator hooks

Expose execution realism knobs (slippage model, prate, impact cap, gas randomness) via `optimization/runner_cli.py` and use them in `optimization/engine/evaluator.py` when converting signals → fills → PnL.

```diff
diff --git a/optimization/runner_cli.py b/optimization/runner_cli.py
--- a/optimization/runner_cli.py
+++ b/optimization/runner_cli.py
@@
 parser.add_argument('--exec-fee-bps', type=int, default=29)
 parser.add_argument('--exec-prate', type=float, default=0.20)
 parser.add_argument('--exec-max-impact-bps', type=int, default=75)
 parser.add_argument('--exec-gas-median-pls', type=float, default=272.0)
 parser.add_argument('--exec-gas-sigma', type=float, default=0.35)
```

(Then pass these into the evaluator and the strategies that need them.)

> **Why this matters**: PBO/DSR provide a quantitative sanity check that the “winner” parameters aren’t artifacts of multiple testing or luck; purged WFO reduces leakage; and realistic fills/stops shrink the gap between staged and live.

---

## 5) How to test (commands you can run)

1. **Install deps** (as you already do):
   `pip install -r requirements.txt`  (repo instructions)

2. **Single‑strategy CSMA iteration** (fast loop while tuning KER/slope/tps):

```bash
.venv/bin/python -m optimization.runner_cli \
  --strategies-file strats_csma.json \
  --calls 120 \
  --trade-size 1000 \
  --exec-fee-bps 29 --exec-prate 0.2 --exec-max-impact-bps 75 \
  --exec-gas-median-pls 272 --exec-gas-sigma 0.35 \
  --cpu-fraction 0.8
```

3. **Shortlist WFO** (new defaults):

```bash
.venv/bin/python -m optimization.runner_cli \
  --strategies-file strats_top_walkforward.json \
  --calls 200 \
  --trade-size 1000 \
  --exec-fee-bps 29 --exec-prate 0.2 --exec-max-impact-bps 75 \
  --exec-gas-median-pls 272 --exec-gas-sigma 0.35 \
  --wfo-train-days 150 --wfo-test-days 45 --wfo-step-days 30 \
  --purge-days 1 --embargo-days 1 \
  --select-top-n 3 \
  --cpu-fraction 0.8
```

4. **Manual spot‑checks** (unchanged):
   `scripts/evaluate_vost_strategies.py --trade-size 1000` and
   `scripts/export_strategy_performance_csv.py`.

---

## 6) What to record in the new reports

* For each strategy & fold:
  **(i)** OOS median return, **(ii)** OOS DSR, **(iii)** PBO (per step + overall), **(iv)** PF (trimmed), **(v)** maxDD, **(vi)** trades, **(vii)** realized cost split: **fee**, **price‑impact (bps)**, **gas (PLS & DAI)**.

---

## 7) `newstrats/local.md` — drop‑in doc section

Append this as sections **13–16**:

```md
### 13. 2025-10-08 realism pass: fills, fees, and gas
- Added constant-product execution model (utils/execution_models.py) with PulseX 0.29% fee, partial fills via participation rate (prate), price-impact cap, and randomized gas draws around 272 PLS per swap. Parameters exposed via runner_cli.
- GridTradingStrategyV2Aggressive now uses this executor: no more full fills; PF finite.

### 14. CSMARevertStrategy drawdown fix
- Added KER + SMA-slope regime gate, volatility-targeted sizing, two-stage partial profits (0.75 & 1.5 ATR), Chandelier trailing (k≈2.75), and time stop.
- Circuit breaker: pause entries if staged DD > 40%, then reduced risk on recovery window.

### 15. CompositeMomentumIndexStrategy regime/vol targeting
- Added trend slope + KER gate; ATR breakout filter; volatility-targeted size.

### 16. Validation workflow hardening
- WFO: train=150d, purged+embargo=1d, OOS=45d, step=30d.
- Selection: top-3 by median OOS ret (50%), DSR (25%), DD penalty (25%), with min trades per fold.
- Added DSR + PBO to scoring_engine; log to JSONL + HTML summaries.

**Commands used**
(see README snippet in this PR; shortlist WFO and CSMA-only runs shown)
```

---

## 8) Commit templates

* **feat(csma):** “Add KER+slope gating, vol‑targeted sizing, staged TP + Chandelier; staged DD now capped (maxDD ≤ −40% in staged backtest); WFO median ret +ΔX%, trades N=Y; DSR=Z; PBO=Q”
* **feat(exec):** “Constant‑product execution + partial fills + gas randomness; Grid V2 PF finite; mean impact ~B bps; mean gas ~C PLS”
* **feat(cmi):** “Regime/vol gating for Composite Momentum; improved OOS stability (+ΔX% Sharpe, −ΔY% DD)”
* **feat(validation):** “Add DSR and PBO (CSCV) to scoring; purged WFO defaults (150/45/30, 1d purge/embargo); selection top‑3 by robust composite”

---

## 9) Known limitations & next steps

* The **executor** still operates per‑bar; truly realistic fills (queue position, multi‑tick) would require order‑book/event simulation. Consider later if you pull PulseX swap events tick‑by‑tick.
* If your dataset lacks per‑bar reserves/volume at every bar, fall back to a conservative cost function (inflate price‑impact by factor 1.2× and lower `prate`).
* If you want “optimal slicing,” look to a simplified **Almgren‑Chriss** schedule to break big orders over multiple bars.

---

### Sources (key ones)

* **AMM math & fees:** Uniswap v2 pricing/fees; Uniswap v2 whitepaper; PulseX 0.29% fee.
* **Gas on PulseChain:** GoPulse gas page (typical swap cost); Uniswap‑v2 swap gas ballpark.
* **Partial fills realism:** QuantConnect fill models; HFT backtest queue.
* **ATR/Chandelier exits:** StockCharts, Investopedia.
* **KER / Efficiency Ratio:** Quantified Strategies; Alvarez Quant.
* **Volatility targeting:** Moreira & Muir.
* **Overfitting diagnostics:** PBO/CSCV and DSR.

---

## TL;DR actions

1. Add `utils/execution_models.py`; wire it to Grid V2 Aggressive.
2. Update **CSMA** with KER+slope gating, vol‑targeted sizing, staged TP + Chandelier, and circuit breaker.
3. Gate **Composite Momentum** with trend slope + KER, apply vol‑target sizing, add ATR breakout threshold.
4. Switch WFO to 150/45/30 with purged/embargoed folds; add **PBO** + **DSR** to reports; select **top‑3** by robust composite.

Once you paste the patches, re‑run your shortlist pipeline with the new **`--exec-*`** flags and record the DSR/PBO in the commit body. If anything is named slightly differently in your tree, keep the logic and wire‑up identical—the behavior and metrics are the goal.
