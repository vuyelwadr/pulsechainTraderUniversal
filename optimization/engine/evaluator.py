"""High-performance evaluation utilities for the optimizer runner."""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_vost_strategies import (
    load_dataset,
    load_swap_costs,
    run_strategy,
)
from functools import lru_cache
import importlib

import logging

from optimization import runner as runner_mod  # type: ignore
from optimization.runner import score_detail_from_results  # type: ignore
from optimization.scoring_engine import StrategyMetrics  # type: ignore

# ---------------------------------------------------------------------------
# Worker-scoped caches
# ---------------------------------------------------------------------------

BASE_DATA: Optional[pd.DataFrame] = None
SWAP_COSTS: Optional[Dict[str, Dict[int, Dict[str, float]]]] = None
DATA_PATH: Optional[Path] = None
COST_PATH: Optional[Path] = None
STAGE_LABEL: str = "stage"
STAGE_DAYS: Optional[int] = None
EXTRA_WINDOWS: List[Tuple[str, Optional[int]]] = []
WINDOW_CACHE: Dict[Tuple[Optional[int], str], pd.DataFrame] = {}
STRATEGY_MODULES: Optional[List[str]] = None


@lru_cache(maxsize=None)
def _fallback_strategy_modules() -> List[str]:
    modules: List[str] = []
    strategies_dir = REPO_ROOT / 'strategies'
    if not strategies_dir.exists():
        return modules
    for path in strategies_dir.rglob('*.py'):
        if path.name.startswith('_'):
            continue
        try:
            rel = path.resolve().relative_to(REPO_ROOT)
            modules.append('.'.join(rel.with_suffix('').parts))
        except Exception:
            continue
    return modules


def resolve_strategy_class(strategy_name: str):
    logger = logging.getLogger(runner_mod.__name__)
    previous_level = logger.level
    logger.setLevel(logging.ERROR)
    try:
        cls = runner_mod.load_strategy_class(strategy_name)
    finally:
        logger.setLevel(previous_level)
    if cls is not None:
        return cls
    for module_name in _fallback_strategy_modules():
        try:
            mod = importlib.import_module(module_name)
        except Exception:
            continue
        if hasattr(mod, strategy_name):
            return getattr(mod, strategy_name)
    raise ImportError(f"Strategy class '{strategy_name}' not found in strategies directory.")


def _prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure timestamps are datetime and sorted."""
    out = df.copy()
    if 'timestamp' in out.columns:
        out['timestamp'] = pd.to_datetime(out['timestamp'])
        out = out.sort_values('timestamp').reset_index(drop=True)
    else:
        if not isinstance(out.index, pd.DatetimeIndex):
            out.index = pd.to_datetime(out.index)
        out = out.sort_index()
    return out


def _slice_last_days(df: pd.DataFrame, days: Optional[int]) -> pd.DataFrame:
    if days is None:
        return df
    if 'timestamp' in df.columns:
        cutoff = df['timestamp'].max() - pd.Timedelta(days=days)
        return df[df['timestamp'] >= cutoff].reset_index(drop=True)
    idx = df.index
    if isinstance(idx, pd.DatetimeIndex):
        cutoff = idx.max() - pd.Timedelta(days=days)
        return df.loc[idx >= cutoff]
    return df


def init_worker(
    data_path: Union[str, Path],
    swap_cost_path: Union[str, Path],
    stage_label: str,
    stage_days: Optional[int],
    extra_windows: Sequence[Tuple[str, Optional[int]]],
) -> None:
    """Initialise per-worker dataset and cost caches."""
    global BASE_DATA, DATA_PATH, SWAP_COSTS, COST_PATH, STAGE_LABEL, STAGE_DAYS, EXTRA_WINDOWS, WINDOW_CACHE

    data_path = Path(data_path).resolve()
    cost_path = Path(swap_cost_path).resolve()

    if BASE_DATA is None or DATA_PATH != data_path:
        BASE_DATA = _prepare_dataframe(load_dataset(data_path))
        DATA_PATH = data_path
        WINDOW_CACHE.clear()

    if SWAP_COSTS is None or COST_PATH != cost_path:
        SWAP_COSTS = load_swap_costs(cost_path)
        COST_PATH = cost_path

    STAGE_LABEL = stage_label
    STAGE_DAYS = stage_days
    EXTRA_WINDOWS = list(extra_windows)
    WINDOW_CACHE.clear()


def _get_window(label: str, days: Optional[int]) -> pd.DataFrame:
    global WINDOW_CACHE
    key = (days, label)
    if key in WINDOW_CACHE:
        return WINDOW_CACHE[key]
    if BASE_DATA is None:
        raise RuntimeError("Worker dataset not initialised.")
    df = _slice_last_days(BASE_DATA, days)
    WINDOW_CACHE[key] = df
    return df


def _sanitize_stats(stats: Dict[str, Any]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    for key, value in stats.items():
        if key in ('equity_curve', 'strategy_returns', 'trade_returns'):
            continue
        if isinstance(value, (np.floating, np.integer)):
            summary[key] = float(value)
        elif isinstance(value, np.ndarray):
            summary[key] = value.astype(float).tolist()
        else:
            summary[key] = value
    return summary


def _instantiate_strategy(strategy_cls, parameters: Dict[str, Any]):
    if hasattr(strategy_cls, '__call__'):
        return strategy_cls(parameters=parameters or None)
    return strategy_cls(parameters=parameters or None)


@dataclass
class TrialConfig:
    trial_id: int
    strategy_name: str
    parameters: Dict[str, Any]
    trade_size: float
    objectives: Sequence[str]
    stage_label: str
    stage_days: Optional[int]
    extra_windows: Sequence[Tuple[str, Optional[int]]]


def evaluate_trial(config: TrialConfig) -> Dict[str, Any]:
    """Evaluate a strategy configuration and compute objective scores."""
    start_time = time.time()
    try:
        if BASE_DATA is None or SWAP_COSTS is None:
            raise RuntimeError("Worker not initialised with dataset/cost cache.")

        strategy_cls = resolve_strategy_class(config.strategy_name)
        stage_df = _get_window(config.stage_label, config.stage_days)
        stage_strategy = _instantiate_strategy(strategy_cls, config.parameters)
        stage_stats = run_strategy(
            stage_strategy,
            stage_df,
            swap_costs=SWAP_COSTS,
            trade_notional=config.trade_size,
        )

        objective_scores: Dict[str, Dict[str, Any]] = {}
        for objective in config.objectives:
            compounds = score_detail_from_results(
                config.strategy_name,
                stage_stats,
                stage_df,
                objective=objective,
            )
            metrics_obj = compounds.get('metrics')
            if isinstance(metrics_obj, StrategyMetrics):
                compounds['metrics'] = metrics_obj.__dict__
            compounds['score'] = float(compounds.get('score', 0.0))
            objective_scores[objective] = compounds

        timeframe_metrics: Dict[str, Dict[str, Any]] = {
            config.stage_label: _sanitize_stats(stage_stats)
        }

        for label, days in config.extra_windows:
            if label == config.stage_label and days == config.stage_days:
                continue
            df = _get_window(label, days)
            if df.empty:
                timeframe_metrics[label] = {}
                continue
            window_strategy = _instantiate_strategy(strategy_cls, config.parameters)
            window_stats = run_strategy(
                window_strategy,
                df,
                swap_costs=SWAP_COSTS,
                trade_notional=config.trade_size,
            )
            timeframe_metrics[label] = _sanitize_stats(window_stats)

        elapsed = time.time() - start_time
        return {
            'trial_id': config.trial_id,
            'strategy': config.strategy_name,
            'parameters': config.parameters,
            'trade_size': config.trade_size,
            'objective_scores': objective_scores,
            'timeframe_metrics': timeframe_metrics,
            'elapsed_seconds': elapsed,
            'status': 'ok',
        }
    except Exception as exc:
        return {
            'trial_id': config.trial_id,
            'strategy': config.strategy_name,
            'parameters': config.parameters,
            'trade_size': config.trade_size,
            'status': 'error',
            'error': f"{exc}",
            'traceback': traceback.format_exc(),
        }
