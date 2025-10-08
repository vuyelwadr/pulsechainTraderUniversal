"""Orchestrator for the fast optimizer runner."""

from __future__ import annotations

import math
import multiprocessing as mp
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from optimization.engine.evaluator import (
    TrialConfig,
    evaluate_trial,
    init_worker,
    resolve_strategy_class,
)
from optimization.persistence import TrialStore


@dataclass
class StrategySpec:
    name: str
    cls: Any
    base_parameters: Dict[str, Any]
    param_space: Dict[str, Tuple[float, float]]


def load_strategy_specs(strategy_names: Sequence[str]) -> List[StrategySpec]:
    specs: List[StrategySpec] = []
    for name in strategy_names:
        cls = resolve_strategy_class(name)
        if cls is None:
            raise ImportError(f"Strategy class '{name}' could not be loaded.")
        instance = cls()
        base_params = dict(getattr(instance, 'parameters', {}) or {})
        try:
            param_space = dict(cls.parameter_space())
        except Exception:
            param_space = {}
        specs.append(StrategySpec(name=name, cls=cls, base_parameters=base_params, param_space=param_space))
    return specs


def _coerce_param(default: Any, lo: float, hi: float, rng: random.Random) -> Any:
    if isinstance(default, bool):
        return bool(rng.randint(0, 1))
    if isinstance(default, int):
        lo_i, hi_i = int(math.floor(lo)), int(math.ceil(hi))
        if lo_i > hi_i:
            lo_i, hi_i = hi_i, lo_i
        return int(rng.randint(lo_i, hi_i))
    return float(rng.uniform(lo, hi))


def sample_parameters(spec: StrategySpec, rng: random.Random) -> Dict[str, Any]:
    params = dict(spec.base_parameters)
    if not spec.param_space:
        return params
    for key, bounds in spec.param_space.items():
        lo, hi = bounds
        default = params.get(key)
        if default is None:
            default = 0.0
        # Occasionally stay with default to keep anchor
        if rng.random() < 0.2:
            continue
        sampled = _coerce_param(default, float(lo), float(hi), rng)
        params[key] = sampled
    return params


def _max_workers(cpu_fraction: float) -> int:
    total = mp.cpu_count()
    target = max(1, int(total * cpu_fraction))
    return max(1, min(total, target))


def run_optimization(
    *,
    run_dir: Path,
    store: TrialStore,
    strategy_specs: Sequence[StrategySpec],
    objectives: Sequence[str],
    total_calls: int,
    trade_size: float,
    data_path: Path,
    swap_cost_path: Path,
    stage_label: str,
    stage_days: Optional[int],
    extra_windows: Sequence[Tuple[str, Optional[int]]],
    cpu_fraction: float,
    seed: Optional[int] = None,
    save_trades: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Execute optimisation run, returning completed trials and best score summary."""
    existing_trials = store.load_trials()
    completed_ids = {trial['trial_id'] for trial in existing_trials}
    start_id = (max(completed_ids) + 1) if completed_ids else 1
    remaining_calls = max(0, total_calls - len(existing_trials))
    rng = random.Random(seed if seed is not None else random.SystemRandom().randint(1, 1_000_000))

    best_scores: Dict[str, Dict[str, Any]] = {}
    for trial in existing_trials:
        for objective in objectives:
            obj = (trial.get('objective_scores') or {}).get(objective)
            if not obj:
                continue
            score = float(obj.get('score', 0.0))
            current = best_scores.get(objective)
            if current is None or score > current['score']:
                best_scores[objective] = {
                    'score': score,
                    'trial_id': trial['trial_id'],
                    'strategy': trial['strategy'],
                    'parameters': trial.get('parameters', {}),
                }

    if remaining_calls == 0:
        return existing_trials, best_scores

    trades_dir: Optional[Path] = run_dir / 'trades' if save_trades else None
    if trades_dir is not None:
        trades_dir.mkdir(parents=True, exist_ok=True)

    tasks: List[TrialConfig] = []
    for idx in range(remaining_calls):
        spec = strategy_specs[idx % len(strategy_specs)]
        params = sample_parameters(spec, rng)
        trial_id = start_id + idx
        tasks.append(
            TrialConfig(
                trial_id=trial_id,
                strategy_name=spec.name,
                parameters=params,
                trade_size=trade_size,
                objectives=objectives,
                stage_label=stage_label,
                stage_days=stage_days,
                extra_windows=extra_windows,
                save_trades=save_trades,
                trades_dir=trades_dir,
            )
        )

    max_workers = _max_workers(cpu_fraction)
    results: List[Dict[str, Any]] = []

    with mp.get_context("spawn").Pool(
        processes=max_workers,
        initializer=init_worker,
        initargs=(data_path, swap_cost_path, stage_label, stage_days, extra_windows),
    ) as pool:
        try:
            for result in pool.imap_unordered(evaluate_trial, tasks):
                result['completed_at'] = datetime.utcnow().isoformat()
                store.append_trial(result)
                results.append(result)
                if result.get('status') == 'ok':
                    for objective in objectives:
                        obj = (result.get('objective_scores') or {}).get(objective)
                        if not obj:
                            continue
                        score = float(obj.get('score', 0.0))
                        current = best_scores.get(objective)
                        if current is None or score > current['score']:
                            best_scores[objective] = {
                                'score': score,
                                'trial_id': result['trial_id'],
                                'strategy': result['strategy'],
                                'parameters': result.get('parameters', {}),
                            }
                checkpoint = {
                    'next_trial_id': start_id + len(results),
                    'completed': len(existing_trials) + len(results),
                    'total_requests': total_calls,
                    'remaining_calls': max(0, total_calls - (len(existing_trials) + len(results))),
                    'best_scores': best_scores,
                }
                store.write_checkpoint(checkpoint)
        finally:
            pool.close()
            pool.join()

    all_trials = existing_trials + results
    final_checkpoint = {
        'next_trial_id': start_id + len(results),
        'completed': len(all_trials),
        'total_requests': total_calls,
        'remaining_calls': max(0, total_calls - len(all_trials)),
        'best_scores': best_scores,
    }
    store.write_checkpoint(final_checkpoint)
    return all_trials, best_scores
