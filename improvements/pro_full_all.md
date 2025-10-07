Absolutely—here’s a **ready‑to‑paste replacement** for `optimization/optimizer_bayes.py` that merges the strongest ideas from your attached variants **without** multi‑fidelity screening and **without** environment guardrails. It keeps your 30→90→1y staged design intact, and folds in:

* **Walk‑forward** (expanding‑window) CV with optional warm‑up
* **Adaptive acquisition** switching (EI/PI/LCB) with mild **xi/kappa annealing**
* **Dynamic GP noise** estimated from fold variance (robust to noisy backtests)
* **Mini‑batch** optimization via iterative `gp_minimize` with `x0/y0`
* **Parallel** fold evaluation (threads) for speed, avoiding risky process nesting

> Provenance: this file integrates the best mechanics from your drafts/notes (the four optimizer diffs + design notes) while honoring your constraints to **leave out multi‑fidelity** and **skip OMP/MKL/OPENBLAS env guards**.
> The choices below also line up with standard practice: walk‑forward CV for time‑series, and skopt’s EI/PI/LCB with `xi`/`kappa` (and a **noise** parameter you can set to a variance estimate). ([scikit-optimize.github.io][1])

---

## `optimization/optimizer_bayes.py` (drop‑in replacement)

```python
#!/usr/bin/env python3
"""
Bayesian Parameter Optimizer — best-of (no multi-fidelity)

What you get:
- Walk-forward (expanding window) cross-validation for time series, with optional warm-up tail
- Adaptive acquisition (EI/PI/LCB) + gentle xi/kappa annealing when progress stalls
- Dynamic GP noise estimated from fold variance (robust to noisy backtests)
- Batched optimization: iterative gp_minimize calls, growing x0/y0
- Optional parallelization across validation folds (threads), no env guardrails

Expected backtest signature:
    backtest_func(strategy_instance, data_df, params_dict) -> float (higher is better)

Notes:
- gp_minimize minimizes; we pass negative score back from our objective.
- The `noise` parameter we hand to gp_minimize acts as the GP's observation noise level.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer, Categorical
except ImportError:
    raise ImportError("scikit-optimize is required. Install with: pip install scikit-optimize")


# ---------------------------------------------------------------------------

@dataclass
class OptimizationResult:
    """Standard result container for your pipeline."""
    strategy_name: str
    timeframe: str
    best_params: Dict[str, Any]
    best_cps_score: float
    improvement_percent: float
    convergence_iteration: Optional[int]
    total_iterations: int
    optimization_time: float
    param_history: List[Dict[str, Any]]
    score_history: List[float]


# ---------------------------------------------------------------------------

class BayesianOptimizer:
    """
    Consolidated optimizer (multi-fidelity intentionally omitted to match user’s request).

    Key arguments:
        n_calls: total objective evaluations
        n_initial_points: initial random samples before BO
        acq_func: 'EI' | 'PI' | 'LCB' (used unless acq_funcs schedule is provided)
        xi, kappa: exploration parameters (annealed adaptively)
        noise_floor: minimum GP noise; dynamic noise from fold variance >= this
        walk_forward_folds/min_size/warmup: time-series CV config
        parallel_folds: run validation folds in parallel (threads)
        batch_size: grow gp_minimize in small batches via x0/y0

    Everything else is kept simple and explicit.
    """

    def __init__(self,
                 n_calls: int = 100,
                 n_initial_points: int = 20,
                 acq_func: str = "EI",
                 xi: float = 0.01,
                 kappa: float = 1.96,
                 noise_floor: float = 1e-10,
                 n_jobs: int = 1,
                 random_state: int = 42,
                 convergence_threshold: float = 1e-3,
                 convergence_patience: int = 10,
                 acq_funcs: Optional[List[str]] = None,
                 # Walk-forward CV:
                 walk_forward_folds: int = 4,
                 walk_forward_min_size: int = 120,
                 walk_forward_warmup: int = 200,
                 parallel_folds: bool = True,
                 fold_workers: Optional[int] = None,
                 # Mini-batch loop:
                 batch_size: int = 5,
                 # Noise history:
                 noise_window: int = 7,
                 max_noise: float = 0.5):
        # General BO setup
        self.n_calls = int(n_calls)
        self.n_initial_points = int(n_initial_points)
        self.acq_func = (acq_func or "EI").upper()
        self.xi = float(xi)
        self.kappa = float(kappa)

        # GP noise control
        self.noise_floor = float(max(noise_floor, 1e-12))
        self.max_noise = float(max(max_noise, self.noise_floor))

        self.n_jobs = int(max(1, n_jobs))
        self.random_state = int(random_state)
        self.convergence_threshold = float(max(1e-8, convergence_threshold))
        self.convergence_patience = int(max(1, convergence_patience))
        self.acq_sequence = self._normalize_acq_sequence(acq_funcs)

        # Time-series CV
        self.walk_forward_folds = int(max(1, walk_forward_folds))
        self.walk_forward_min_size = int(max(10, walk_forward_min_size))
        self.walk_forward_warmup = int(max(0, walk_forward_warmup))
        self.parallel_folds = bool(parallel_folds)
        self.fold_workers = fold_workers  # None -> auto threads

        # Loop batching
        self.batch_size = int(max(1, batch_size))

        # Noise estimation memory
        self.noise_window = int(max(1, noise_window))
        self.observation_variances: List[float] = []

        # Tracking
        self.param_history: List[Dict[str, Any]] = []
        self.score_history: List[float] = []
        self.best_score: float = -np.inf
        self.no_improvement_count: int = 0
        self.convergence_iteration: Optional[int] = None
        self.total_evaluations: int = 0

        # Acquisition parameters (annealed dynamically)
        self.current_xi = self.xi
        self.current_kappa = self.kappa
        self.min_xi, self.max_xi = 1e-3, 0.25
        self.min_kappa, self.max_kappa = 0.5, 6.0

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _normalize_acq_sequence(acq_funcs: Optional[List[str]]) -> Optional[List[str]]:
        if not acq_funcs:
            return None
        seq = []
        for name in acq_funcs:
            label = (name or "").strip().upper()
            if label in {"EI", "PI", "LCB"}:
                seq.append(label)
        return seq or None

    @staticmethod
    def _clone_strategy(strategy):
        """Try deep copy; fall back to .clone() if the class supplies one."""
        try:
            return copy.deepcopy(strategy)
        except Exception:
            clone = getattr(strategy, "clone", None)
            if callable(clone):
                try:
                    return clone()
                except Exception:
                    return None
        return None

    def _generate_walk_forward_splits(self, data: pd.DataFrame) -> List[Tuple[slice, slice]]:
        """
        Build expanding-window train/validation splits:
            [0:step] -> [step:2*step], then [0:2*step] -> [2*step:3*step], ...
        """
        n = len(data)
        if n < max(2 * self.walk_forward_min_size, 20):
            return []
        folds = min(self.walk_forward_folds, max(1, n // self.walk_forward_min_size) - 1)
        if folds <= 0:
            return []
        step = max(self.walk_forward_min_size, n // (folds + 1))
        splits: List[Tuple[slice, slice]] = []
        start = step
        for _ in range(folds):
            end = min(n, start + step)
            if end <= start:
                break
            splits.append((slice(0, start), slice(start, end)))
            start = end
            if end >= n:
                break
        return splits

    def _prepare_fold_data(self, data: pd.DataFrame, train_slice: slice, val_slice: slice) -> pd.DataFrame:
        """
        Optionally prepend a warm-up tail from the training window to the validation chunk.
        This helps indicators that need initial state (MAs, ATR, etc).
        """
        train_df = data.iloc[train_slice]
        val_df = data.iloc[val_slice]
        if self.walk_forward_warmup > 0 and len(train_df) > 0:
            warm = train_df.tail(min(len(train_df), self.walk_forward_warmup))
            out = pd.concat([warm, val_df])
            return out[~out.index.duplicated(keep="last")]
        return val_df

    def _record_noise_from_scores(self, scores: List[float]) -> None:
        """Track a robust variance summary from fold scores to set GP noise."""
        if not scores:
            return
        xs = [float(s) for s in scores if np.isfinite(s)]
        if not xs:
            return
        var = float(np.var(xs, ddof=1)) if len(xs) > 1 else 0.0
        var = float(np.clip(var, self.noise_floor, self.max_noise))
        self.observation_variances.append(var)
        # bound history
        if len(self.observation_variances) > self.noise_window * 5:
            del self.observation_variances[:-self.noise_window * 5]

    def _current_noise(self) -> float:
        """Median of recent fold variances with a minimum floor."""
        if not self.observation_variances:
            return self.noise_floor
        recent = self.observation_variances[-self.noise_window:]
        return float(np.clip(np.median(recent), self.noise_floor, self.max_noise))

    def _recent_improvement(self, window: int = 6) -> float:
        if len(self.score_history) < 2:
            return float("inf")
        w = max(1, min(window, len(self.score_history) - 1))
        recent = self.score_history[-(w + 1):]
        return float(np.max(recent) - np.min(recent))

    def _select_acquisition(self) -> Tuple[str, float, float]:
        """
        Choose EI/PI/LCB and anneal xi/kappa a bit based on progress.
        - On stagnation: push exploration (LCB) & inflate xi/kappa.
        - On recent gains: exploit (PI) & deflate xi/kappa.
        - Else: drift back to user-provided defaults.
        """
        # Respect a user-specified schedule if present
        if self.acq_sequence:
            idx = (len(self.score_history) // max(1, self.batch_size)) % len(self.acq_sequence)
            base = self.acq_sequence[idx]
        else:
            base = self.acq_func

        improv = self._recent_improvement(window=6)
        stagnating = improv < self.convergence_threshold * 6

        if stagnating:
            self.current_xi = min(self.max_xi, self.current_xi * 1.5)
            self.current_kappa = min(self.max_kappa, self.current_kappa * 1.3)
            return "LCB", self.current_xi, self.current_kappa

        if self.score_history and (self.score_history[-1] >= self.best_score - self.convergence_threshold):
            self.current_xi = max(self.min_xi, self.current_xi * 0.7)
            self.current_kappa = max(self.min_kappa, self.current_kappa * 0.9)
            return "PI", self.current_xi, self.current_kappa

        # drift toward defaults
        self.current_xi += (self.xi - self.current_xi) * 0.25
        self.current_kappa += (self.kappa - self.current_kappa) * 0.25
        self.current_xi = float(np.clip(self.current_xi, self.min_xi, self.max_xi))
        self.current_kappa = float(np.clip(self.current_kappa, self.min_kappa, self.max_kappa))
        return (base if base in {"EI", "PI", "LCB"} else "EI"), self.current_xi, self.current_kappa

    # -----------------------------------------------------------------------
    # Parameter space helper (adjust ranges as needed)
    # -----------------------------------------------------------------------

    def get_parameter_space(self, strategy_type: str) -> List:
        name = (strategy_type or "").lower()
        if "momentum" in name:
            return [
                Integer(5, 50, name="fast_period"),
                Integer(10, 200, name="slow_period"),
                Real(0.001, 0.1, name="signal_threshold"),
                Real(0.01, 0.2, name="stop_loss"),
                Real(0.02, 0.5, name="take_profit"),
            ]
        if "trend" in name:
            return [
                Integer(10, 100, name="trend_period"),
                Real(0.5, 3.0, name="atr_multiplier"),
                Real(0.001, 0.05, name="min_trend_strength"),
                Integer(5, 30, name="confirmation_period"),
                Real(0.01, 0.15, name="trailing_stop"),
            ]
        if "volatility" in name:
            return [
                Integer(10, 40, name="lookback"),
                Real(0.5, 4.0, name="bb_width"),
                Integer(10, 80, name="oversold_level"),
                Integer(20, 80, name="overbought_level"),
                Integer(3, 20, name="smoothing_period"),
                Real(0.001, 0.05, name="divergence_threshold"),
            ]
        if "volume" in name:
            return [
                Integer(10, 50, name="volume_period"),
                Real(1.5, 5.0, name="volume_multiplier"),
                Integer(5, 30, name="price_period"),
                Real(0.001, 0.05, name="confirmation_threshold"),
                Real(0.01, 0.1, name="risk_per_trade"),
            ]
        # generic fallback
        return [
            Integer(5, 50, name="period1"),
            Integer(10, 100, name="period2"),
            Real(0.001, 0.1, name="threshold1"),
            Real(0.01, 0.2, name="threshold2"),
            Real(0.01, 0.3, name="risk_limit"),
        ]

    # -----------------------------------------------------------------------
    # Objective
    # -----------------------------------------------------------------------

    def create_objective(self,
                         strategy,
                         data: pd.DataFrame,
                         backtest_func,
                         cross_validate: bool = True,
                         dimensions: Optional[List] = None):
        """
        Wrap the user's backtest into a skopt-compatible objective.
        """
        usable_cv = cross_validate and len(data) >= (self.walk_forward_min_size * 2)

        def objective(params_list: List[Any]) -> float:
            try:
                # list -> dict mapping aligned with the given dimensions
                space = dimensions if dimensions is not None else self.get_parameter_space(strategy.__class__.__name__)
                params: Dict[str, Any] = {dim.name: val for dim, val in zip(space, params_list)}
                self.param_history.append(params)

                # Evaluate (WFV if available)
                fold_scores: List[float] = []
                if usable_cv:
                    splits = self._generate_walk_forward_splits(data)
                    if splits:
                        if self.parallel_folds and len(splits) > 1:
                            workers = self.fold_workers or min(len(splits), max(1, self.n_jobs))
                            with ThreadPoolExecutor(max_workers=workers) as ex:
                                futs = []
                                for tr, va in splits:
                                    fold_df = self._prepare_fold_data(data, tr, va)
                                    inst = self._clone_strategy(strategy) or strategy
                                    futs.append(ex.submit(backtest_func, inst, fold_df, params))
                                for f in futs:
                                    fold_scores.append(float(f.result()))
                        else:
                            for tr, va in splits:
                                fold_df = self._prepare_fold_data(data, tr, va)
                                inst = self._clone_strategy(strategy) or strategy
                                fold_scores.append(float(backtest_func(inst, fold_df, params)))
                # If no folds or not enough data for WFV, do a single full evaluation
                if not fold_scores:
                    inst = self._clone_strategy(strategy) or strategy
                    final_score = float(backtest_func(inst, data, params))
                else:
                    final_score = float(np.mean(fold_scores))

                # Update noise stats from folds (or from the single score)
                self._record_noise_from_scores(fold_scores if fold_scores else [final_score])

                # Bookkeeping for convergence
                self.score_history.append(final_score)
                self.total_evaluations += 1
                if final_score > self.best_score:
                    gain = final_score - (self.best_score if np.isfinite(self.best_score) else -np.inf)
                    if not np.isfinite(self.best_score) or gain >= self.convergence_threshold:
                        self.no_improvement_count = 0
                    else:
                        self.no_improvement_count += 1
                    self.best_score = final_score
                else:
                    self.no_improvement_count += 1

                if self.no_improvement_count >= self.convergence_patience and self.convergence_iteration is None:
                    self.convergence_iteration = len(self.score_history)

                # skopt minimizes -> return negative
                return -final_score

            except Exception as e:
                print(f"[objective] error: {e}")
                # worst possible (we minimize), but finite to keep GP stable
                return 0.0

        return objective

    # -----------------------------------------------------------------------
    # Main entry
    # -----------------------------------------------------------------------

    def optimize_strategy(self,
                          strategy,
                          strategy_type: str,
                          data: pd.DataFrame,
                          backtest_func,
                          initial_score: Optional[float] = None,
                          dimensions: Optional[List] = None,
                          initial_params: Optional[List[Dict[str, Any]]] = None,
                          timeframe: str = "") -> OptimizationResult:
        """Run the BO loop with batching and adaptive acquisition."""
        t0 = time.time()

        # Reset trackers
        self.param_history.clear()
        self.score_history.clear()
        self.observation_variances.clear()
        self.best_score = -np.inf
        self.no_improvement_count = 0
        self.convergence_iteration = None
        self.total_evaluations = 0
        self.current_xi, self.current_kappa = self.xi, self.kappa

        # Space + objective
        space = dimensions if dimensions is not None else self.get_parameter_space(strategy_type)
        objective = self.create_objective(strategy, data, backtest_func, cross_validate=True, dimensions=space)

        # Helper: dict -> point according to space ordering
        def _dict_to_point(p: Dict[str, Any]) -> Optional[List[Any]]:
            out: List[Any] = []
            for dim in space:
                if dim.name not in p:
                    return None
                raw = p[dim.name]
                try:
                    if isinstance(dim, Integer):
                        v = int(round(float(raw)))
                        v = max(dim.low, min(dim.high, v))
                    elif isinstance(dim, Real):
                        v = float(raw)
                        v = min(dim.high, max(dim.low, v))
                    else:
                        v = raw
                except Exception:
                    return None
                out.append(v)
            return out

        # Seed with optional initial points
        evaluated_x: List[List[Any]] = []
        evaluated_y: List[float] = []
        if initial_params:
            for p in initial_params:
                pt = _dict_to_point(p)
                if pt is None:
                    continue
                y = float(objective(pt))
                evaluated_x.append(pt)
                evaluated_y.append(y)

        target_calls = max(0, int(self.n_calls))
        remaining = max(0, target_calls - len(evaluated_x))
        if not evaluated_x and remaining <= 0:
            raise ValueError("Expected at least one optimization call or a valid initial sample")

        # Iterative partial gp_minimize: append batch_size calls per loop
        while remaining > 0:
            acq, xi, kappa = self._select_acquisition()
            step = min(self.batch_size, remaining)
            total_calls = len(evaluated_x) + step
            if total_calls <= len(evaluated_x):
                break

            n_init = max(0, min(self.n_initial_points, total_calls) - len(evaluated_x))
            noise_level = self._current_noise()

            res = gp_minimize(
                func=objective,
                dimensions=space,
                n_calls=total_calls,
                n_initial_points=n_init,
                acq_func=acq,
                xi=xi,
                kappa=kappa,
                noise=noise_level,    # dynamic GP noise (see skopt gp_minimize docs)
                n_jobs=self.n_jobs,   # relevant when acq_optimizer='lbfgs'
                random_state=self.random_state,
                x0=evaluated_x if evaluated_x else None,
                y0=evaluated_y if evaluated_y else None,
            )

            evaluated_x = list(res.x_iters)
            evaluated_y = list(map(float, res.func_vals))
            remaining = max(0, target_calls - len(evaluated_x))

            if self.convergence_iteration is not None and len(self.score_history) >= self.convergence_iteration:
                break

        if not evaluated_x:
            raise RuntimeError("Optimization produced no evaluations")

        # Extract best
        best_idx = int(np.argmin(evaluated_y))
        best_point = evaluated_x[best_idx]
        best_params: Dict[str, Any] = {}
        for dim, val in zip(space, best_point):
            if isinstance(dim, Integer):
                best_params[dim.name] = int(round(val))
            elif isinstance(dim, Real):
                best_params[dim.name] = float(val)
            else:
                best_params[dim.name] = val

        best_cps_score = float(-evaluated_y[best_idx])
        improvement_pct = 0.0
        if initial_score is not None:
            try:
                improvement_pct = ((best_cps_score - float(initial_score)) / float(initial_score)) * 100.0
            except ZeroDivisionError:
                improvement_pct = 0.0

        runtime = float(time.time() - t0)
        return OptimizationResult(
            strategy_name=strategy.__class__.__name__,
            timeframe=str(timeframe or ""),
            best_params=best_params,
            best_cps_score=best_cps_score,
            improvement_percent=improvement_pct,
            convergence_iteration=self.convergence_iteration,
            total_iterations=len(self.score_history),
            optimization_time=runtime,
            param_history=list(self.param_history),
            score_history=list(self.score_history),
        )
```

### Why these exact changes (and where they came from)

* **Walk‑forward CV** replaces the older fixed thirds split to avoid leakage and match how you’ll trade—training always precedes validation in time. This directly follows your “use walk‑forward” note and the more robust WFV variants in your diffs.   This practice is also the standard recommendation for time‑series validation. ([Scikit-learn][2])
* **Adaptive EI/PI/LCB** with **xi/kappa** tweaks: if progress stalls, push exploration (LCB / higher `kappa`, larger `xi`); if gains are recent, tighten to exploit (PI / smaller `xi`/`kappa`). This captures the best logic sprinkled across your variants.   It aligns with skopt’s acquisition functions and their trade‑offs. ([scikit-optimize.github.io][3])
* **Dynamic noise (fold variance → GP noise)**: noisy backtests (slippage/fills/timing) shouldn’t derail the surrogate. We roll a variance over recent folds and hand it to `gp_minimize(noise=...)`. This is the exact “model observation noise better” point you highlighted.   Skopt allows per‑run `noise` to reflect observation uncertainty. ([scikit-optimize.github.io][1])
* **Batched loop using `x0/y0`**: iterative `gp_minimize` calls let you add a few evaluations at a time while keeping control over acquisition and noise each step. This mirrors the batched/partial approach from your drafts.   The approach is fully compatible with skopt’s API (see parameters `x0`, `y0`, `xi`, `kappa`, `noise`, `n_jobs`). ([scikit-optimize.github.io][1])
* **Parallel fold eval (threads)**: fold validations can run concurrently to speed up the objective, but we intentionally avoid deeper nesting/guards per your request to skip environment thread caps. 

---

### Drop‑in instructions

1. **Replace** your file at `optimization/optimizer_bayes.py` with the code above.
2. Keep using your existing staged 30 → 90 → 1y pipeline; this optimizer deliberately **does not** add multi‑fidelity screening and **does not** add OMP/MKL/OPENBLAS env limits, as you asked.

If you want me to produce a **unified git patch** against your current tree, say the word and I’ll generate a `git apply` block that mirrors this file exactly.

**References (for your reviewers):** skopt `gp_minimize` parameters and the EI/PI/LCB behavior; time‑series walk‑forward validation. ([scikit-optimize.github.io][1])

**Pulled ideas (internal provenance)**: the combined improvements reflect the best bits from your four optimizer drafts and your planning notes.

---

*Model note:* This implementation keeps to scikit‑optimize conventions (EI/PI/LCB, `xi`, `kappa`, `noise`, `x0/y0`, `n_jobs`) to stay stable across environments. See the skopt docs for those exact parameters and semantics. ([scikit-optimize.github.io][1])

If you want me to also **wire a tiny refinement pass** (local ±jitter around the best set) as a separate module, I can provide it—but I left it out here to keep the replacement strictly 1:1 with your file footprint and your “no multi‑fidelity / no env guards” request.

[1]: https://scikit-optimize.github.io/stable/modules/generated/skopt.gp_minimize.html "skopt.gp_minimize — scikit-optimize 0.8.1 documentation"
[2]: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html?utm_source=chatgpt.com "TimeSeriesSplit — scikit-learn 1.7.2 documentation"
[3]: https://scikit-optimize.github.io/stable/modules/acquisition.html?utm_source=chatgpt.com "1. Acquisition — scikit-optimize 0.8.1 documentation"
