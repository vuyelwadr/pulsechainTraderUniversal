#!/usr/bin/env python3
"""Bayesian parameter optimizer tuned for PulseChain HEX strategies."""

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
except ImportError as exc:  # pragma: no cover - explicit error for missing dependency
    raise ImportError("scikit-optimize is required: pip install scikit-optimize") from exc

from .refinement_pass import Candidate, refinement_pass


@dataclass
class OptimizationResult:
    """Container for optimization artefacts."""

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
    """Integrated Bayesian optimizer with walk-forward CV and adaptive acquisition."""

    def __init__(
        self,
        *,
        n_calls: int = 100,
        n_initial_points: int = 20,
        acq_func: str = "EI",
        xi: float = 0.01,
        kappa: float = 1.96,
        noise: float = 1e-10,
        n_jobs: int = 1,
        random_state: int = 42,
        convergence_threshold: float = 1e-3,
        convergence_patience: int = 10,
        acq_funcs: Optional[List[str]] = None,
        walk_forward_folds: int = 4,
        walk_forward_min_size: int = 120,
        walk_forward_warmup: int = 200,
        batch_size: int = 5,
        parallel_folds: bool = True,
        fold_workers: Optional[int] = None,
        enable_refinement: bool = True,
        refinement_neighbors: int = 12,
        refinement_step: float = 0.07,
        refinement_topk: Optional[int] = 5,
        refinement_max_evals: Optional[int] = None,
        refinement_workers: Optional[int] = None,
    ) -> None:
        self.n_calls = int(max(0, n_calls))
        self.n_initial_points = int(max(0, n_initial_points))
        self.acq_func = (acq_func or "EI").upper()
        self.xi = float(xi)
        self.kappa = float(kappa)
        self.noise_floor = float(max(noise, 1e-12))
        self.dynamic_noise = self.noise_floor
        self.n_jobs = int(max(1, n_jobs))
        self.random_state = int(random_state)
        self.convergence_threshold = float(max(0.0, convergence_threshold))
        self.convergence_patience = int(max(1, convergence_patience))
        self.acq_sequence = self._normalize_acq_sequence(acq_funcs)

        self.walk_forward_folds = int(max(1, walk_forward_folds))
        self.walk_forward_min_size = int(max(10, walk_forward_min_size))
        self.walk_forward_warmup = int(max(0, walk_forward_warmup))

        self.batch_size = int(max(1, batch_size))
        self.parallel_folds = bool(parallel_folds)
        self.fold_workers = fold_workers

        self.enable_refinement = bool(enable_refinement)
        self.refinement_neighbors = int(max(1, refinement_neighbors))
        self.refinement_step = float(max(1e-4, refinement_step))
        self.refinement_topk = refinement_topk if refinement_topk is None else int(max(1, refinement_topk))
        self.refinement_max_evals = refinement_max_evals if refinement_max_evals is None else int(max(1, refinement_max_evals))
        self.refinement_workers = refinement_workers

        # tracking state
        self.param_history: List[Dict[str, Any]] = []
        self.score_history: List[float] = []
        self.best_score: float = -np.inf
        self.best_params_snapshot: Optional[Dict[str, Any]] = None
        self.no_improvement_count: int = 0
        self.convergence_iteration: Optional[int] = None
        self.total_evaluations: int = 0
        self.observation_variances: List[float] = []

    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_acq_sequence(acq_funcs: Optional[List[str]]) -> Optional[List[str]]:
        if not acq_funcs:
            return None
        normalized: List[str] = []
        for name in acq_funcs:
            if not name:
                continue
            label = name.strip().upper()
            if label not in {"EI", "PI", "LCB"}:
                raise ValueError(f"Unsupported acquisition function '{name}'")
            normalized.append(label)
        return normalized or None

    @staticmethod
    def _clone_strategy(strategy) -> Optional[Any]:
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

    def _prepare_fold_data(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> pd.DataFrame:
        if self.walk_forward_warmup <= 0 or train_df.empty:
            return val_df.copy()
        warm = train_df.iloc[-min(len(train_df), self.walk_forward_warmup) :]
        combined = pd.concat([warm, val_df])
        if hasattr(combined, "index"):
            combined = combined[~combined.index.duplicated(keep="last")]
        return combined.copy()

    def _generate_walk_forward_splits(self, data: pd.DataFrame) -> List[Tuple[slice, slice]]:
        n = len(data)
        if n < (self.walk_forward_min_size * 2):
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
            if start >= n:
                break
        return splits

    def _record_noise_from_scores(self, scores: List[float]) -> None:
        clean = [float(s) for s in scores if np.isfinite(s)]
        if not clean:
            return
        variance = float(np.var(clean, ddof=1)) if len(clean) > 1 else 0.0
        if not np.isfinite(variance):
            variance = 0.0
        self.observation_variances.append(max(variance, self.noise_floor))
        if len(self.observation_variances) > 100:
            self.observation_variances = self.observation_variances[-100:]

    def _estimate_noise_level(self) -> float:
        if not self.observation_variances:
            return self.noise_floor
        recent = self.observation_variances[-min(20, len(self.observation_variances)) :]
        dyn = float(np.mean(recent) + 1e-6)
        return float(max(self.noise_floor, dyn))

    def _recent_improvement(self) -> float:
        if len(self.score_history) < 2:
            return float("inf")
        return float(np.max(self.score_history) - np.min(self.score_history))

    def _select_acquisition(self) -> Tuple[str, float, float]:
        return self.acq_func, self.xi, self.kappa

    # ------------------------------------------------------------------
    def get_parameter_space(self, strategy_type: str) -> List[Any]:
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
                Integer(5, 60, name="lookback"),
                Real(0.5, 5.0, name="atr_mult"),
                Real(0.001, 0.1, name="entry_z"),
                Real(0.01, 0.3, name="risk_limit"),
            ]
        return [
            Integer(5, 50, name="period1"),
            Integer(10, 100, name="period2"),
            Real(0.001, 0.1, name="threshold1"),
            Real(0.01, 0.2, name="threshold2"),
            Real(0.01, 0.3, name="risk_limit"),
        ]

    # ------------------------------------------------------------------
    def create_objective(
        self,
        strategy,
        data: pd.DataFrame,
        backtest_func,
        *,
        cross_validate: bool = True,
        dimensions: Optional[List[Any]] = None,
    ):
        space = dimensions if dimensions is not None else self.get_parameter_space(strategy.__class__.__name__)
        usable_cv = bool(cross_validate and len(data) >= (self.walk_forward_min_size * 2))
        splits: List[Tuple[slice, slice]] = self._generate_walk_forward_splits(data) if usable_cv else []

        def objective(params_list: List[Any]) -> float:
            param_dict: Dict[str, Any] = {}
            for dim, value in zip(space, params_list):
                if isinstance(dim, Integer):
                    param_dict[dim.name] = int(round(value))
                elif isinstance(dim, Real):
                    param_dict[dim.name] = float(value)
                else:
                    param_dict[dim.name] = value

            self.param_history.append(param_dict.copy())

            final_score: float
            fold_scores: List[float] = []

            if usable_cv:
                splits = self._generate_walk_forward_splits(data)
                if splits:
                    if self.parallel_folds and len(splits) > 1:
                        workers = self.fold_workers or min(len(splits), max(1, self.n_jobs))
                        with ThreadPoolExecutor(max_workers=workers) as executor:
                            futures = [
                                executor.submit(
                                    backtest_func,
                                    self._clone_strategy(strategy) or strategy,
                                    self._prepare_fold_data(data[s_train], data[s_val]),
                                    param_dict,
                                )
                                for s_train, s_val in splits
                            ]
                            for future in futures:
                                fold_scores.append(float(future.result()))
                    else:
                        for s_train, s_val in splits:
                            fold_df = self._prepare_fold_data(data[s_train], data[s_val])
                            score_val = backtest_func(self._clone_strategy(strategy) or strategy, fold_df, param_dict)
                            fold_scores.append(float(score_val))
                    final_score = float(np.mean(fold_scores)) if fold_scores else float(backtest_func(strategy, data, param_dict))
                else:
                    final_score = float(backtest_func(strategy, data, param_dict))
            else:
                final_score = float(backtest_func(strategy, data, param_dict))

            if fold_scores:
                self._record_noise_from_scores(fold_scores)
            else:
                self._record_noise_from_scores([final_score])

            self.score_history.append(final_score)
            improved = final_score > self.best_score
            if improved:
                if self.best_score != -np.inf and (final_score - self.best_score) < self.convergence_threshold:
                    self.no_improvement_count += 1
                else:
                    self.no_improvement_count = 0
                self.best_score = final_score
                self.best_params_snapshot = param_dict.copy()
            else:
                self.no_improvement_count += 1

            if self.no_improvement_count >= self.convergence_patience and self.convergence_iteration is None:
                self.convergence_iteration = len(self.score_history)

            return -final_score

        return objective

    # ------------------------------------------------------------------
    def _space_bounds(self, space: List[Any]) -> Dict[str, Tuple[Any, ...]]:
        bounds: Dict[str, Tuple[Any, ...]] = {}
        for dim in space:
            if isinstance(dim, Integer):
                bounds[dim.name] = (float(dim.low), float(dim.high), "int")
            elif isinstance(dim, Real):
                bounds[dim.name] = (float(dim.low), float(dim.high), "float")
            elif isinstance(dim, Categorical):
                bounds[dim.name] = (None, None, "cat", tuple(dim.categories))
        return bounds

    def _dict_to_list(self, params: Dict[str, Any], space: List[Any]) -> List[Any]:
        ordered: List[Any] = []
        for dim in space:
            value = params[dim.name]
            if isinstance(dim, Integer):
                ordered.append(int(round(value)))
            elif isinstance(dim, Real):
                ordered.append(float(value))
            else:
                ordered.append(value)
        return ordered

    # ------------------------------------------------------------------
    def optimize_strategy(
        self,
        *,
        strategy,
        strategy_type: str,
        data: pd.DataFrame,
        backtest_func,
        initial_score: Optional[float] = None,
        dimensions: Optional[List[Any]] = None,
        initial_params: Optional[List[Dict[str, Any]]] = None,
    ) -> OptimizationResult:
        start = time.time()

        self.param_history.clear()
        self.score_history.clear()
        self.observation_variances.clear()
        self.best_score = -np.inf
        self.best_params_snapshot = None
        self.no_improvement_count = 0
        self.convergence_iteration = None
        self.total_evaluations = 0

        space = dimensions if dimensions is not None else self.get_parameter_space(strategy_type)
        objective = self.create_objective(strategy, data, backtest_func, dimensions=space)

        evaluated_x: List[List[Any]] = []
        evaluated_y: List[float] = []

        if initial_params:
            for param_dict in initial_params:
                try:
                    point = self._dict_to_list(param_dict, space)
                except KeyError:
                    continue
                y_val = float(objective(point))
                evaluated_x.append(point)
                evaluated_y.append(y_val)

        target_calls = max(0, int(self.n_calls))
        if target_calls == 0 and not evaluated_x:
            raise ValueError("Expected at least one optimization call or a valid initial sample")

        remaining = max(0, target_calls - len(evaluated_x))
        if not evaluated_x and remaining <= 0:
            raise ValueError("Expected at least one optimization call or a valid initial sample")

        while remaining > 0:
            acq, xi, kappa = self._select_acquisition()
            calls_this_iter = min(self.batch_size, remaining)
            total_calls = len(evaluated_x) + calls_this_iter
            if total_calls <= len(evaluated_x):
                break
            n_init = max(0, min(self.n_initial_points, total_calls) - len(evaluated_x))
            self.dynamic_noise = self._estimate_noise_level()

            result = gp_minimize(
                func=objective,
                dimensions=space,
                n_calls=total_calls,
                n_initial_points=n_init,
                acq_func=acq,
                xi=xi,
                kappa=kappa,
                noise=self.dynamic_noise,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
                x0=evaluated_x if evaluated_x else None,
                y0=evaluated_y if evaluated_y else None,
            )

            evaluated_x = list(result.x_iters)
            evaluated_y = [float(v) for v in result.func_vals]

            remaining = max(0, target_calls - len(evaluated_x))
            self.total_evaluations = len(self.score_history)

            if self.convergence_iteration is not None and len(self.score_history) >= self.convergence_iteration:
                break

        if not evaluated_x:
            raise RuntimeError("Optimization produced no evaluations")

        best_idx = int(np.argmin(evaluated_y))
        best_point = evaluated_x[best_idx]
        default_best_params = {
            dim.name: int(round(val)) if isinstance(dim, Integer) else float(val) if isinstance(dim, Real) else val
            for dim, val in zip(space, best_point)
        }
        best_params = self.best_params_snapshot.copy() if self.best_params_snapshot else default_best_params.copy()

        if self.enable_refinement and best_params:
            bounds = self._space_bounds(space)
            seeds: List[Candidate] = []
            for point, score in zip(evaluated_x, evaluated_y):
                param_dict = {
                    dim.name: int(round(val)) if isinstance(dim, Integer) else float(val) if isinstance(dim, Real) else val
                    for dim, val in zip(space, point)
                }
                seeds.append(Candidate(params=param_dict, score=float(-score)))
            seeds.sort(key=lambda cand: float(cand.score or float("-inf")), reverse=True)
            max_workers = self.refinement_workers or (self.n_jobs if self.n_jobs > 1 else None)

            def _score_fn(p: Dict[str, Any]) -> float:
                ordered = self._dict_to_list(p, space)
                return float(-objective(ordered))

            refined_params, refined_score, _ = refinement_pass(
                score_fn=_score_fn,
                bounds=bounds,
                seeds=seeds,
                neighbors_per_seed=self.refinement_neighbors,
                step_frac=self.refinement_step,
                topk_seeds=self.refinement_topk,
                max_evals=self.refinement_max_evals,
                parallel=max_workers is not None,
                use_threads=True,
                max_workers=max_workers,
                rng_seed=self.random_state,
            )
            if refined_score > float(self.best_score):
                best_params = refined_params
                self.best_score = float(refined_score)

        best_cps_score = float(self.best_score if np.isfinite(self.best_score) else -evaluated_y[best_idx])
        improvement_percent = 0.0
        if initial_score:
            try:
                improvement_percent = ((best_cps_score - float(initial_score)) / float(initial_score)) * 100.0
            except ZeroDivisionError:
                improvement_percent = 0.0

        elapsed = time.time() - start
        return OptimizationResult(
            strategy_name=strategy.__class__.__name__,
            timeframe="auto",
            best_params=best_params,
            best_cps_score=best_cps_score,
            improvement_percent=improvement_percent,
            convergence_iteration=self.convergence_iteration,
            total_iterations=len(self.score_history),
            optimization_time=elapsed,
            param_history=self.param_history.copy(),
            score_history=self.score_history.copy(),
        )
