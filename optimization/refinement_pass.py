"""Local refinement pass for Bayesian optimizer outputs."""

from __future__ import annotations

import random
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

Params = Dict[str, Union[int, float, str]]
ScoreFn = Callable[[Params], float]
Bounds = Dict[str, Union[
    Tuple[float, float, str],
    Tuple[None, None, str, Sequence[Union[str, int, float]]],
]]


@dataclass
class Candidate:
    params: Params
    score: Optional[float] = None


def _clamp(x: float, lo: float, hi: float) -> float:
    if lo > hi:
        lo, hi = hi, lo
    return max(lo, min(hi, x))


def _perturb_numeric(val: float, lo: float, hi: float, step: float, as_int: bool, rng: random.Random):
    sd = max((hi - lo) * step, 1e-12)
    prop = rng.normalvariate(val, sd)
    prop = _clamp(prop, lo, hi)
    return int(round(prop)) if as_int else float(prop)


def _perturb_categorical(val, values: Sequence, rng: random.Random):
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
    for key, spec in bounds.items():
        if key not in out or not isinstance(spec, tuple):
            continue
        kind = (spec[2] if len(spec) >= 3 else "float").lower()
        if kind in ("float", "int"):
            lo, hi = float(spec[0]), float(spec[1])
            out[key] = _perturb_numeric(float(out[key]), lo, hi, step_frac, as_int=(kind == "int"), rng=rng)
        elif kind == "cat":
            values = tuple(spec[3]) if len(spec) >= 4 else (out[key],)
            out[key] = _perturb_categorical(out[key], values, rng)
    return out


def _ensure_unique(items: Iterable[Params]) -> List[Params]:
    seen, uniq = set(), []
    for params in items:
        key = tuple(sorted(params.items()))
        if key not in seen:
            seen.add(key)
            uniq.append(params)
    return uniq


def refinement_pass(
    *,
    score_fn: ScoreFn,
    bounds: Bounds,
    seeds: Sequence[Union[Params, Candidate]],
    neighbors_per_seed: int = 24,
    step_frac: float = 0.07,
    topk_seeds: Optional[int] = None,
    max_evals: Optional[int] = None,
    parallel: bool = True,
    use_threads: bool = False,
    max_workers: Optional[int] = None,
    rng_seed: Optional[int] = None,
) -> Tuple[Params, float, List[Candidate]]:
    rng = random.Random(rng_seed)

    seed_candidates: List[Candidate] = []
    for seed in seeds:
        if isinstance(seed, Candidate):
            seed_candidates.append(seed)
        elif isinstance(seed, dict):
            seed_candidates.append(Candidate(params=seed, score=None))
        else:
            raise TypeError(f"Unsupported seed type: {type(seed)}")

    if topk_seeds is not None and any(c.score is not None for c in seed_candidates):
        seed_candidates.sort(key=lambda c: float("-inf") if c.score is None else float(c.score), reverse=True)
        seed_candidates = seed_candidates[: max(1, int(topk_seeds))]

    candidates: List[Params] = []
    for cand in seed_candidates:
        candidates.append(dict(cand.params))
        for _ in range(max(1, neighbors_per_seed)):
            candidates.append(_generate_neighbor(cand.params, bounds, step_frac, rng))
    candidates = _ensure_unique(candidates)
    if max_evals is not None:
        candidates = candidates[: max(1, int(max_evals))]

    def _evaluate(params: Params) -> Candidate:
        try:
            score = float(score_fn(params))
        except Exception:
            score = float("-inf")
        return Candidate(params=params, score=score)

    evaluated: List[Candidate] = []
    if parallel and len(candidates) > 1:
        Executor = ThreadPoolExecutor if use_threads else ProcessPoolExecutor
        with Executor(max_workers=max_workers) as executor:
            futures = {executor.submit(_evaluate, params): params for params in candidates}
            for future in as_completed(futures):
                evaluated.append(future.result())
    else:
        for params in candidates:
            evaluated.append(_evaluate(params))

    for cand in seed_candidates:
        if cand.score is not None:
            evaluated.append(cand)

    evaluated.sort(key=lambda c: float("-inf") if c.score is None else float(c.score), reverse=True)
    best = evaluated[0]
    return best.params, float(best.score), evaluated
