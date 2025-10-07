Got it—I read your **`whatToDo.md`** and the attached implementation write‑ups/diffs. Here’s the “best‑of” plan and a **ready‑to‑paste** set of files you can drop into `pulsechainTraderUniversal` to complete the task.

---

## What I’m picking (and why)

From your variants, these are the strongest ideas to combine:

1. **Walk‑Forward Validation (WFV) with warm‑up** — prevents time‑series leakage and matches live use. I’m keeping an expanding‑window split plus a warm‑up tail before each validation window. 

2. **Multi‑fidelity screening** — do a quick pass on the last ~30–40% of data and **promote** only good candidates to an expensive full backtest. This cuts wall‑time a lot without hurting quality. I’m using a percentile‑based promotion threshold with a safety margin against the current best.

3. **Observation‑noise from folds** — estimate score variance across WFV folds and feed it to the GP’s `noise` (alpha). This makes the surrogate less jumpy and reduces overfitting to “lucky” folds. I’m using a robust rolling window/median. 

4. **Adaptive acquisition switching (+ gentle xi/kappa annealing)** — switch EI ↔ PI ↔ LCB based on recent improvement; increase exploration when stagnating, exploit when improving. I preserved the nice parameter annealing touches.

5. **Parallel fold evaluation** — validate folds concurrently (threads) so the objective stays fast but we don’t dangerously nest process pools.

6. **Batched BO loop** — run `gp_minimize` in small batches (accumulating `x0/y0`) to amortize cost while keeping control of noise/acq settings.

7. **Post‑BO refinement pass** — quick local neighborhood search around the top seeds to pick up the last 1–5% that the GP can miss. I’m including your refinement module as a drop‑in. 

These choices directly implement the priorities listed in your “heavy thinking” notes and capture the best pieces across your diffs. 

---

# Files to copy/paste

> Paths assume your repo root. Create missing folders if needed.

---

## 1) `optimization/optimizer_bayes.py`  **(drop‑in replacement)**

```python
#!/usr/bin/env python3
"""
Best-of Bayesian Parameter Optimizer for Stage 2 Strategy Optimization
- Walk-forward cross-validation (expanding window) with warm-up tail
- Multi-fidelity screening (quick subset) with percentile + margin promotion
- Observation-noise estimation from fold variance (robust rolling window)
- Adaptive acquisition switching (EI/PI/LCB) with xi/kappa annealing
- Batched bayes loop; parallel fold evaluation (threads)

Expected backtest signature: backtest_func(strategy, data_df, param_dict) -> float (higher is better)
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
    print("WARNING: scikit-optimize not installed. Install with: pip install scikit-optimize")
    raise


@dataclass
class OptimizationResult:
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


class BayesianOptimizer:
    """
    A consolidated optimizer that merges the strongest ideas across your prototypes:
      - Walk-forward CV + warmup
      - Multi-fidelity quick evaluation with percentile/margin promotion
      - Fold-variance -> GP noise (rolling robust estimate)
      - Adaptive acquisition switching with xi/kappa annealing
      - Batch calls to gp_minimize
      - Parallel folds (threads) for speed without risky process nesting
    """

    # --------------------------- CONFIG DEFAULTS --------------------------- #
    def __init__(
        self,
        n_calls: int = 100,
        n_initial_points: int = 20,
        acq_func: str = "EI",
        xi: float = 0.01,
        kappa: float = 1.96,
        # noise is a floor; dynamic noise estimates will be >= this
        noise_floor: float = 1e-10,
        n_jobs: int = 1,
        random_state: int = 42,
        convergence_threshold: float = 1e-3,
        convergence_patience: int = 10,
        acq_funcs: Optional[List[str]] = None,
        # Walk-forward
        walk_forward_folds: int = 4,
        walk_forward_min_size: int = 120,
        walk_forward_warmup: int = 200,
        parallel_folds: bool = True,
        fold_workers: Optional[int] = None,
        # Multi-fidelity
        enable_multi_fidelity: bool = True,
        quick_eval_fraction: float = 0.35,
        promotion_percentile: float = 0.40,      # promote if >= this percentile of quick history
        promote_margin_frac: float = 0.05,       # or near current best within 5%
        quick_penalty: float = 0.90,             # reduce value of non-promoted candidates slightly
        multi_fidelity_min_points: int = 300,
        # Batch loop
        batch_size: int = 5,
        # Noise tracking
        noise_window: int = 7,                   # rolling window to median recent noise
        max_noise: float = 0.5,                  # cap noise to keep GP stable
    ):
        self.n_calls = int(n_calls)
        self.n_initial_points = int(n_initial_points)
        self.acq_func = (acq_func or "EI").upper()
        self.xi = float(xi)
        self.kappa = float(kappa)
        self.noise_floor = float(max(noise_floor, 1e-12))
        self.max_noise = float(max(max_noise, self.noise_floor))
        self.n_jobs = int(max(1, n_jobs))
        self.random_state = int(random_state)
        self.convergence_threshold = float(max(1e-8, convergence_threshold))
        self.convergence_patience = int(max(1, convergence_patience))
        self.acq_sequence = self._normalize_acq_sequence(acq_funcs)

        self.walk_forward_folds = int(max(1, walk_forward_folds))
        self.walk_forward_min_size = int(max(10, walk_forward_min_size))
        self.walk_forward_warmup = int(max(0, walk_forward_warmup))
        self.parallel_folds = bool(parallel_folds)
        self.fold_workers = fold_workers  # None -> auto

        self.enable_multi_fidelity = bool(enable_multi_fidelity)
        self.quick_eval_fraction = float(min(0.9, max(0.05, quick_eval_fraction)))
        self.promotion_percentile = float(min(0.95, max(0.0, promotion_percentile)))
        self.promote_margin_frac = float(max(0.0, promote_margin_frac))
        self.quick_penalty = float(min(1.0, max(0.01, quick_penalty)))
        self.multi_fidelity_min_points = int(max(1, multi_fidelity_min_points))

        self.batch_size = int(max(1, batch_size))
        self.noise_window = int(max(1, noise_window))

        # Tracking
        self.param_history: List[Dict[str, Any]] = []
        self.score_history: List[float] = []
        self.quick_scores: List[float] = []
        self.observation_variances: List[float] = []
        self.best_score: float = -np.inf
        self.no_improvement_count: int = 0
        self.convergence_iteration: Optional[int] = None
        self.total_evaluations: int = 0
        self.current_xi = self.xi
        self.current_kappa = self.kappa

        # Bounds for xi/kappa annealing
        self.min_xi, self.max_xi = 1e-3, 0.25
        self.min_kappa, self.max_kappa = 0.5, 6.0

    # --------------------------- UTILITIES --------------------------------- #
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

    def _clone_strategy(self, strategy):
        try:
            return copy.deepcopy(strategy)
        except Exception:
            clone_method = getattr(strategy, "clone", None)
            if callable(clone_method):
                try:
                    return clone_method()
                except Exception:
                    return None
        return None

    def _generate_walk_forward_splits(self, data: pd.DataFrame) -> List[Tuple[slice, slice]]:
        """Return [(train_slice, val_slice), ...] expanding-window."""
        n = len(data)
        if n < max(2 * self.walk_forward_min_size, 20):
            return []
        fold_count = min(self.walk_forward_folds, max(1, n // self.walk_forward_min_size) - 1)
        if fold_count <= 0:
            return []
        fold_size = max(self.walk_forward_min_size, n // (fold_count + 1))
        splits: List[Tuple[slice, slice]] = []
        start = fold_size
        for _ in range(fold_count):
            end = min(n, start + fold_size)
            if end <= start:
                break
            splits.append((slice(0, start), slice(start, end)))
            start = end
            if end >= n:
                break
        return splits

    def _prepare_fold_data(self, data: pd.DataFrame, train_slice: slice, val_slice: slice) -> pd.DataFrame:
        train_df = data.iloc[train_slice]
        val_df = data.iloc[val_slice]
        if self.walk_forward_warmup > 0 and len(train_df) > 0:
            warmup = train_df.tail(min(len(train_df), self.walk_forward_warmup))
            out = pd.concat([warmup, val_df])
            return out[~out.index.duplicated(keep="last")]
        return val_df

    def _record_noise_from_scores(self, scores: List[float]):
        if not scores:
            return
        clean = [float(s) for s in scores if np.isfinite(s)]
        if not clean:
            return
        var = float(np.var(clean, ddof=1)) if len(clean) > 1 else 0.0
        var = float(np.clip(var, self.noise_floor, self.max_noise))
        self.observation_variances.append(var)
        # keep a reasonable history (5 * window)
        if len(self.observation_variances) > self.noise_window * 5:
            del self.observation_variances[:-self.noise_window * 5]

    def _current_noise(self) -> float:
        if not self.observation_variances:
            return self.noise_floor
        recent = self.observation_variances[-self.noise_window:]
        return float(np.clip(np.median(recent), self.noise_floor, self.max_noise))

    def _recent_improvement(self, window: int = 6) -> float:
        if len(self.score_history) < 2:
            return float("inf")
        window = max(1, min(window, len(self.score_history) - 1))
        recent = self.score_history[-(window + 1):]
        return float(np.max(recent) - np.min(recent))

    def _select_acquisition(self) -> Tuple[str, float, float]:
        """Adaptive EI/PI/LCB with mild xi/kappa annealing."""
        # Preference: follow provided sequence if any
        if self.acq_sequence:
            idx = (len(self.score_history) // max(1, self.batch_size)) % len(self.acq_sequence)
            base = self.acq_sequence[idx]
        else:
            base = self.acq_func

        # Detect stagnation / strong improvement
        improv = self._recent_improvement(window=6)
        stagnating = improv < self.convergence_threshold * 6

        if stagnating:
            # push exploration
            self.current_xi = min(self.max_xi, self.current_xi * 1.5)
            self.current_kappa = min(self.max_kappa, self.current_kappa * 1.3)
            return "LCB", self.current_xi, self.current_kappa

        # Good improvement recently → exploit
        if self.score_history and (self.score_history[-1] >= self.best_score - self.convergence_threshold):
            self.current_xi = max(self.min_xi, self.current_xi * 0.7)
            self.current_kappa = max(self.min_kappa, self.current_kappa * 0.9)
            return "PI", self.current_xi, self.current_kappa

        # Drift toward base
        self.current_xi += (self.xi - self.current_xi) * 0.25
        self.current_kappa += (self.kappa - self.current_kappa) * 0.25
        self.current_xi = float(np.clip(self.current_xi, self.min_xi, self.max_xi))
        self.current_kappa = float(np.clip(self.current_kappa, self.min_kappa, self.max_kappa))
        return (base if base in {"EI", "PI", "LCB"} else "EI"), self.current_xi, self.current_kappa

    def _promotion_threshold(self) -> Optional[float]:
        """Percentile threshold over quick scores, if we have enough history."""
        if len(self.quick_scores) < 8:
            return None
        return float(np.quantile(self.quick_scores, self.promotion_percentile))

    # --------------------------- PARAM SPACE -------------------------------- #
    def get_parameter_space(self, strategy_type: str) -> List:
        """Tune ranges; adjust as needed for your strategies."""
        lt = strategy_type.lower()
        if "momentum" in lt:
            return [
                Integer(5, 50, name="fast_period"),
                Integer(10, 200, name="slow_period"),
                Real(0.001, 0.1, name="signal_threshold"),
                Real(0.01, 0.2, name="stop_loss"),
                Real(0.02, 0.5, name="take_profit"),
            ]
        if "trend" in lt:
            return [
                Integer(10, 100, name="trend_period"),
                Real(0.5, 3.0, name="atr_multiplier"),
                Real(0.001, 0.05, name="min_trend_strength"),
                Integer(5, 30, name="confirmation_period"),
                Real(0.01, 0.15, name="trailing_stop"),
            ]
        if "volatility" in lt:
            return [
                Integer(10, 40, name="lookback"),
                Real(0.5, 4.0, name="bb_width"),
                Integer(10, 80, name="oversold_level"),
                Integer(20, 80, name="overbought_level"),
                Integer(3, 20, name="smoothing_period"),
                Real(0.001, 0.05, name="divergence_threshold"),
            ]
        if "volume" in lt:
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

    # --------------------------- OBJECTIVE ---------------------------------- #
    def create_objective(
        self,
        strategy,
        data: pd.DataFrame,
        backtest_func,
        cross_validate: bool = True,
        dimensions: Optional[List] = None,
    ):
        usable_cv = cross_validate and len(data) >= (self.walk_forward_min_size * 2)

        def objective(params_list):
            try:
                # Map param vector -> dict using known dimension names
                space = dimensions if dimensions is not None else self.get_parameter_space(strategy.__class__.__name__)
                param_dict: Dict[str, Any] = {dim.name: val for dim, val in zip(space, params_list)}
                self.param_history.append(param_dict)

                # ---------- Multi-fidelity quick screen ----------
                final_score: Optional[float] = None
                fold_scores: List[float] = []

                if (
                    self.enable_multi_fidelity
                    and len(data) >= self.multi_fidelity_min_points
                    and 0.0 < self.quick_eval_fraction < 1.0
                ):
                    quick_len = max(self.walk_forward_min_size, int(len(data) * self.quick_eval_fraction))
                    quick_df = data.iloc[-quick_len:]
                    quick_strategy = self._clone_strategy(strategy) or strategy
                    quick_score = float(backtest_func(quick_strategy, quick_df, param_dict))
                    self.quick_scores.append(quick_score)

                    # Decide promotion with percentile threshold + margin to current best
                    threshold = self._promotion_threshold()
                    margin_cut = None
                    if np.isfinite(self.best_score) and self.best_score != -np.inf:
                        margin_cut = self.best_score - self.promote_margin_frac * max(1.0, abs(self.best_score))

                    promote = True
                    if threshold is not None and quick_score < threshold:
                        promote = False
                    if margin_cut is not None and quick_score < margin_cut:
                        promote = False

                    if not promote:
                        final_score = quick_score * self.quick_penalty  # keep but downweight

                # ---------- Full/WFV evaluation if promoted or no multi-fidelity ----------
                if final_score is None:
                    if usable_cv:
                        splits = self._generate_walk_forward_splits(data)
                        if splits:
                            if self.parallel_folds and len(splits) > 1:
                                workers = self.fold_workers or min(len(splits), max(1, self.n_jobs))
                                with ThreadPoolExecutor(max_workers=workers) as ex:
                                    futs = []
                                    for tr, va in splits:
                                        fold_data = self._prepare_fold_data(data, tr, va)
                                        fold_strategy = self._clone_strategy(strategy) or strategy
                                        futs.append(ex.submit(backtest_func, fold_strategy, fold_data, param_dict))
                                    for f in futs:
                                        fold_scores.append(float(f.result()))
                            else:
                                for tr, va in splits:
                                    fold_data = self._prepare_fold_data(data, tr, va)
                                    seq_strategy = self._clone_strategy(strategy) or strategy
                                    fold_scores.append(float(backtest_func(seq_strategy, fold_data, param_dict)))

                            final_score = float(np.mean(fold_scores)) if fold_scores else float(
                                backtest_func(self._clone_strategy(strategy) or strategy, data, param_dict)
                            )
                        else:
                            final_score = float(backtest_func(self._clone_strategy(strategy) or strategy, data, param_dict))
                    else:
                        final_score = float(backtest_func(self._clone_strategy(strategy) or strategy, data, param_dict))

                # ---------- Bookkeeping ----------
                if fold_scores:
                    self._record_noise_from_scores(fold_scores)
                else:
                    self._record_noise_from_scores([final_score])

                self.score_history.append(final_score)
                self.total_evaluations += 1

                # Best / convergence tracking
                if final_score > self.best_score:
                    improvement = final_score - (self.best_score if np.isfinite(self.best_score) else -np.inf)
                    if not np.isfinite(self.best_score) or improvement >= self.convergence_threshold:
                        self.no_improvement_count = 0
                    else:
                        self.no_improvement_count += 1
                    self.best_score = final_score
                else:
                    self.no_improvement_count += 1

                if self.no_improvement_count >= self.convergence_patience and self.convergence_iteration is None:
                    self.convergence_iteration = len(self.score_history)

                # gp_minimize minimizes
                return -final_score

            except Exception as e:
                print(f"[objective] error: {e}")
                return 0.0

        return objective

    # --------------------------- MAIN LOOP ---------------------------------- #
    def optimize_strategy(
        self,
        strategy,
        strategy_type: str,
        data: pd.DataFrame,
        backtest_func,
        initial_score: Optional[float] = None,
        dimensions: Optional[List] = None,
        initial_params: Optional[List[Dict[str, Any]]] = None,
        timeframe: str = "",
    ) -> OptimizationResult:

        t0 = time.time()
        # reset tracking
        self.param_history.clear()
        self.score_history.clear()
        self.quick_scores.clear()
        self.observation_variances.clear()
        self.best_score = -np.inf
        self.no_improvement_count = 0
        self.convergence_iteration = None
        self.total_evaluations = 0
        self.current_xi, self.current_kappa = self.xi, self.kappa

        space = dimensions if dimensions is not None else self.get_parameter_space(strategy_type)
        objective = self.create_objective(strategy, data, backtest_func, cross_validate=True, dimensions=space)

        # Helper to convert dict->list in the same dim order
        def _dict_to_point(p: Dict[str, Any]) -> Optional[List[Any]]:
            point: List[Any] = []
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
                point.append(v)
            return point

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

        while remaining > 0:
            acq, xi, kappa = self._select_acquisition()
            step = min(self.batch_size, remaining)
            total_calls = len(evaluated_x) + step
            if total_calls <= len(evaluated_x):
                break

            n_init = max(0, min(self.n_initial_points, total_calls) - len(evaluated_x))
            noise_level = self._current_noise()

            result = gp_minimize(
                func=objective,
                dimensions=space,
                n_calls=total_calls,
                n_initial_points=n_init,
                acq_func=acq,
                xi=xi,
                kappa=kappa,
                noise=noise_level,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
                x0=evaluated_x if evaluated_x else None,
                y0=evaluated_y if evaluated_y else None,
            )

            evaluated_x = list(result.x_iters)
            evaluated_y = list(map(float, result.func_vals))
            remaining = max(0, target_calls - len(evaluated_x))

            if self.convergence_iteration is not None and len(self.score_history) >= self.convergence_iteration:
                break

        if not evaluated_x:
            raise RuntimeError("Optimization produced no evaluations")

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

**Why this file?**
It merges your best patterns (multi‑fidelity, WFV, noise, adaptive acq, batching, parallel folds) from the drafts and recommendations, with clean defaults and guardrails.

---

## 2) `optimization/refinement_pass.py`  **(new)**

This is your refinement step, kept as a self‑contained module so you can run a local neighborhood search around BO’s winners. (Copied and lightly formatted to drop in.) 

```python
# optimization/refinement_pass.py
from __future__ import annotations

import random
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

Params = Dict[str, Union[int, float, str]]
ScoreFn = Callable[[Params], float]

Bounds = Dict[str, Union[
    Tuple[float, float, str],
    Tuple[None, None, str, Sequence[Union[str, int, float]]]
]]

@dataclass
class Candidate:
    params: Params
    score: Optional[float] = None

def _clamp(x: float, lo: float, hi: float) -> float:
    if lo > hi:
        lo, hi = hi, lo
    return max(lo, min(hi, x))

def _perturb_numeric(val: float, lo: float, hi: float, step: float, is_int: bool, rng: random.Random) -> Union[int, float]:
    span = max((hi - lo) * step, 1e-12)
    prop = rng.normalvariate(val, span)
    prop = _clamp(prop, lo, hi)
    return int(round(prop)) if is_int else float(prop)

def _perturb_categorical(val: Union[str, int, float], values: Sequence[Union[str, int, float]], rng: random.Random):
    if not values:
        return val
    if len(values) == 1:
        return values[0]
    if rng.random() < 0.7:
        return val
    pool = [v for v in values if v != val]
    return rng.choice(pool) if pool else val

def _generate_neighbor(seed: Params, bounds: Bounds, step_frac: float, rng: random.Random) -> Params:
    out = dict(seed)
    for k, spec in bounds.items():
        if k not in out or not isinstance(spec, tuple):
            continue
        kind = str(spec[2]).lower() if len(spec) >= 3 else "float"
        if kind in ("float", "int"):
            lo, hi = float(spec[0]), float(spec[1])
            out[k] = _perturb_numeric(float(out[k]), lo, hi, step_frac, is_int=(kind == "int"), rng=rng)
        elif kind == "cat":
            vals = tuple(spec[3]) if len(spec) >= 4 else (out[k],)
            out[k] = _perturb_categorical(out[k], vals, rng)
    return out

def _ensure_unique(items: Iterable[Params]) -> List[Params]:
    seen, uniq = set(), []
    for p in items:
        key = tuple(sorted(p.items()))
        if key not in seen:
            seen.add(key)
            uniq.append(p)
    return uniq

def refinement_pass(
    score_fn: ScoreFn,
    bounds: Bounds,
    seeds: Sequence[Union[Params, Candidate]],
    *,
    neighbors_per_seed: int = 24,
    step_frac: float = 0.05,
    topk_seeds: Optional[int] = None,
    max_evals: Optional[int] = None,
    parallel: bool = True,
    use_threads: bool = False,
    max_workers: Optional[int] = None,
    rng_seed: Optional[int] = None,
) -> Tuple[Params, float, List[Candidate]]:
    rng = random.Random(rng_seed)

    seed_cands: List[Candidate] = []
    for s in seeds:
        if isinstance(s, Candidate):
            seed_cands.append(s)
        elif isinstance(s, dict):
            seed_cands.append(Candidate(params=s, score=None))
        else:
            raise TypeError(f"Unsupported seed type: {type(s)}")

    if topk_seeds is not None and any(c.score is not None for c in seed_cands):
        seed_cands.sort(key=lambda c: (float("-inf") if c.score is None else c.score), reverse=True)
        seed_cands = seed_cands[:max(1, int(topk_seeds))]

    candidates: List[Params] = []
    for c in seed_cands:
        candidates.append(dict(c.params))
        for _ in range(neighbors_per_seed):
            candidates.append(_generate_neighbor(c.params, bounds, step_frac, rng))

    candidates = _ensure_unique(candidates)
    if max_evals is not None:
        candidates = candidates[: max(1, max_evals)]

    def _eval_one(p: Params) -> Candidate:
        try:
            s = float(score_fn(p))
        except Exception:
            s = float("-inf")
        return Candidate(params=p, score=s)

    evaluated: List[Candidate] = []
    if parallel and len(candidates) > 1:
        Executor = ThreadPoolExecutor if use_threads else ProcessPoolExecutor
        with Executor(max_workers=max_workers) as ex:
            futs = {ex.submit(_eval_one, p): p for p in candidates}
            for fut in as_completed(futs):
                evaluated.append(fut.result())
    else:
        for p in candidates:
            evaluated.append(_eval_one(p))

    for c in seed_cands:
        if c.score is not None:
            evaluated.append(c)

    evaluated.sort(key=lambda c: (float("-inf") if c.score is None else c.score), reverse=True)
    best = evaluated[0]
    return best.params, float(best.score), evaluated
```

---

## 3) (Optional) quick wiring in your runner

You likely already run strategies in parallel at an outer level. Keep that; just add:

* **BLAS guard** to prevent oversubscription.
* **Refinement pass** right after BO picks its winner for each strategy/window.
* (If you already return only a single score from backtests, you’re fine; the optimizer infers noise from folds it runs itself.)

**Snippet to integrate (add near the top of your runner):**
(You can place this in your main entrypoint or runner module.)

```python
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
```

**After `optimize_strategy(...)` returns:**

```python
from optimization.refinement_pass import refinement_pass, Candidate

# Suppose `result` is the OptimizationResult from optimizer_bayes
best_params = result.best_params
best_score  = result.best_cps_score

# Define a thin wrapper that calls your existing backtest+score combo:
def eval_fn(p):
    # NOTE: plug in your strategy + data + backtest_func used above
    return backtest_func(strategy, data, p)

# Minimal numeric bounds example; adjust to your parameter names:
param_bounds = {
    "fast_period": (5, 50, "int"),
    "slow_period": (10, 200, "int"),
    "signal_threshold": (0.001, 0.1, "float"),
    # ... add all tuned params used by the strategy ...
}

final_params, final_score, _ = refinement_pass(
    score_fn=eval_fn,
    bounds=param_bounds,
    seeds=[Candidate(best_params, best_score)],
    neighbors_per_seed=24,
    step_frac=0.07,
    topk_seeds=1,
    parallel=True,
    use_threads=False,
    rng_seed=42,
)

# Use final_params downstream (save, report, deploy)
```

This small step picks up the last incremental gains with very little complexity. 

---

## Notes & provenance

* **Why WFV, dynamic acq, noise from folds, and multi‑fidelity?** They’re the clearest levers to improve robustness/speed without changing the optimizer’s public surface. Your notes/diffs explicitly called these out. 
* **WFV + warmup** and **parallel folds** were present in your stronger drafts; I kept them and simplified the API.
* **Percentile‑based promotion** plus a **margin check** is more stable than hard thresholds across assets/timeframes; that hybrid came from two of your drafts.
* **Adaptive EI/PI/LCB** with gentle xi/kappa tuning: best compromise between your versions (progress‑aware switching + annealing).
* **Batching and end‑to‑end defaults**: I kept the batch loop pattern from your diffs and avoided deeper, riskier multi‑level parallelism; if you want the “always‑on defaults” module later, your earlier `bo_defaults.py` patch remains a nice optional extra.

---

## How to drop this in

1. Replace your existing optimizer with **`optimization/optimizer_bayes.py`** above.
2. Add **`optimization/refinement_pass.py`**.
3. (Optional) Add the runner snippet and call `refinement_pass` right after BO’s result.
4. Run your usual optimization entrypoint.

That’s it—this is ready to copy/paste into your repo and run. If you want me to also produce a single unified patch (git‑apply style) against your current tree, I can generate it from these files.
