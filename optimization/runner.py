#!/usr/bin/env python3
"""
Unified Optimizer Runner

Runs a multi-stage, parallel optimization pipeline using the full
Gaussian-Process Bayesian optimizer (Stage-2). Stages:

  1) 30 days OHLCV  -> select top N1
  2) 90 days OHLCV  -> select top N2 from previous top
  3) 1 year OHLCV   -> select top N3 (typically 1) => final

Key characteristics:
- Uses real OHLCV resampled from recorded swaps (no synthetic data)
- Optimizes strategy-owned parameters only (reads each strategy's default
  parameters and/or explicit parameter_space())
- CPS scoring (CompositePerformanceScorer) to favor high return and low drawdown
- Fully parallel across strategies and timeframes

Usage:
  # Full 3-stage run (30d -> 90d -> 1y)
  python -m src.pipelines.optimizer_runner \
      --top-n1 60 --top-n2 20 --top-n3 1 \
      --timeframes 5min,15min,30min,1h,2h,4h,8h,16h,1d

  # Single-stage runs
  # Stage 1 (30d) only
  python -m src.pipelines.optimizer_runner --stage 30d --top-n1 60

  # Stage 2 (90d) only using previous Stage 1 summary
  python -m src.pipelines.optimizer_runner --stage 90d --top-n2 20 \
      --from-summary reports/optimizer_pipeline_<ts>/stage1_30d/summary.json

  # Stage 3 (1y) only using previous Stage 2 summary
  python -m src.pipelines.optimizer_runner --stage 1y --top-n3 1 \
      --from-summary reports/optimizer_pipeline_<ts>/stage2_90d/summary.json

Outputs timestamped results under reports/optimizer_pipeline_YYYYmmdd_HHMMSS/
"""

import os
import sys
import json
import gzip
import logging
import math
from dataclasses import dataclass
import subprocess
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional, Union
import hashlib
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from pathlib import Path
import importlib
from decimal import Decimal

import pandas as pd
import numpy as np
import math

# Ensure repo root on path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# External components
from bot.config import Config
from bot.backtest_engine import BacktestEngine
from optimization.scoring_engine import (
    CompositePerformanceScorer,
    StrategyMetrics,
)
try:
    # Dynamic utility scorer (profit-led, DD-aware)
    from optimization.scoring_engine import DynamicUtilityScorer
except Exception:
    DynamicUtilityScorer = None  # Fallback if not available
from optimization.optimizer_bayes import BayesianOptimizer
from importlib.machinery import SourceFileLoader
import re
from utils.swap_cost_cache import (
    initialize_swap_cost_cache,
    ensure_worker_cache_initialized,
    get_swap_cost_cache,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
    """Atomically write JSON data to prevent corruption from concurrent writes."""
    import tempfile
    import os
    
    # Write to temporary file first
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, suffix='.tmp', delete=False) as tmp:
        try:
            json.dump(data, tmp, indent=2, default=str)
            tmp.flush()
            os.fsync(tmp.fileno())  # Force write to disk
            tmp_path = Path(tmp.name)
        except Exception:
            os.unlink(tmp.name)
            raise
    
    # Atomic move
    try:
        tmp_path.replace(path)
    except Exception:
        os.unlink(tmp_path)
        raise


TIMEFRAME_LIST = ['5min', '15min', '30min', '1h', '4h', '8h', '16h', '1d']
# Unified 2-year source file used for all stages (walk-forward)
WFO_DATA_PATH = str((REPO_ROOT / Config.data_path('{asset}_ohlcv_{quote}_730day_5m.csv')).resolve())

# Per-run cache directory name (under repo tmp/). Removed at end of run.
RUN_CACHE_ROOT_NAME = 'optimizer_cache'


def load_strategy_class(strategy_name: str, path_map: Optional[Dict[str, str]] = None):
    """Dynamically load a strategy class by name (resilient to layout differences)."""
    try:
        # Try multiple module path forms (lowercase and snake_case)
        def _to_snake(name: str) -> str:
            import re
            s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
            return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

        lower = strategy_name.lower()
        snake = _to_snake(strategy_name)
        paths_to_try = [
            f"strategies.lazybear.technical_indicators.technical_indicators.{lower}",
            f"strategies.lazybear.technical_indicators.technical_indicators.{snake}",
            f"strategies.lazybear.technical_indicators.{lower}",
            f"strategies.lazybear.technical_indicators.{snake}",
            f"strategies.lazybear.{lower}",
            f"strategies.lazybear.{snake}",
            f"strategies.custom.{lower}",
            f"strategies.{lower}",
            f"strategies.custom.{snake}",
            f"strategies.{snake}",
            "strategies.tradingview_core_strategies",
        ]
        for module_path in paths_to_try:
            try:
                module = __import__(module_path, fromlist=[strategy_name])
                if hasattr(module, strategy_name):
                    return getattr(module, strategy_name)
            except ImportError:
                continue
        # manifest-driven path
        manifest_path = REPO_ROOT / 'task' / 'strategy_manifest.json'
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            for _, info in manifest.get('strategies', {}).items():
                if info.get('implementation_class') == strategy_name:
                    fp = info.get('file_path')
                    if fp:
                        abs_path = Path(fp)
                        if not abs_path.exists():
                            abs_path = REPO_ROOT / fp
                        if abs_path.exists():
                            mod = SourceFileLoader(f"dyn_{strategy_name}", str(abs_path)).load_module()
                            if hasattr(mod, strategy_name):
                                return getattr(mod, strategy_name)
        # discovered path map
        if not path_map:
            path_map = discover_strategy_classes([
                'strategies/lazybear/technical_indicators/technical_indicators',
                'strategies/lazybear/technical_indicators',
                'strategies',
            ])
        if strategy_name in path_map:
            module_name = path_map[strategy_name]
            try:
                mod = importlib.import_module(module_name)
                if hasattr(mod, strategy_name):
                    return getattr(mod, strategy_name)
            except Exception as imp_err:
                logger.error(f"Error importing {strategy_name} from {module_name}: {imp_err}")
    except Exception as e:
        logger.error(f"Error loading {strategy_name}: {e}")
    logger.warning(f"Strategy {strategy_name} not found")
    return None


def discover_strategy_classes(paths: List[str]) -> Dict[str, str]:
    """Scan given directories for classes inheriting BaseStrategy (direct declaration)."""
    class_map: Dict[str, str] = {}
    pattern = re.compile(r"class\s+([A-Za-z0-9_]+)\(BaseStrategy\):")
    for root in paths:
        root_path = REPO_ROOT / root
        if not root_path.exists():
            continue
        for p in root_path.rglob('*.py'):
            try:
                text = p.read_text(errors='ignore')
                for m in pattern.finditer(text):
                    cls_name = m.group(1)
                    try:
                        rel = p.resolve().relative_to(REPO_ROOT)
                        module_name = '.'.join(rel.with_suffix('').parts)
                        class_map[cls_name] = module_name
                    except Exception:
                        continue
            except Exception:
                continue
    return class_map


def build_skopt_dimensions(param_space: Dict[str, Tuple[float, float]]):
    """Convert a simple {name:(lo,hi)} dict into skopt dimensions (Real/Integer)."""
    from skopt.space import Real, Integer
    dims = []
    for k, (lo, hi) in (param_space or {}).items():
        # Integer if bounds look integer
        if float(lo).is_integer() and float(hi).is_integer():
            dims.append(Integer(int(lo), int(hi), name=k))
        else:
            dims.append(Real(float(lo), float(hi), name=k))
    return dims


def _coerce_param_types(raw: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not raw:
        return {}
    coerced: Dict[str, Any] = {}
    for key, value in raw.items():
        if isinstance(value, str):
            text = value.strip()
            if text.lower() in {'true', 'false'}:
                coerced[key] = 1 if text.lower() == 'true' else 0
                continue
            try:
                if text.startswith(('0x', '0X')):
                    coerced[key] = int(text, 16)
                    continue
                int_val = int(text)
                coerced[key] = int_val
                continue
            except ValueError:
                try:
                    coerced[key] = float(text)
                    continue
                except ValueError:
                    pass
            coerced[key] = value
        else:
            coerced[key] = value
    return coerced


def _sanitize_initial_params(initial_params: Optional[List[Dict[str, Any]]], dims: List[Any]) -> List[Dict[str, Any]]:
    if not initial_params or not dims:
        return []
    try:
        from skopt.space import Integer, Real
    except ImportError:
        Integer = Real = None  # type: ignore

    sanitized: List[Dict[str, Any]] = []
    for raw in initial_params:
        if not isinstance(raw, dict):
            continue
        entry: Dict[str, Any] = {}
        valid = True
        for dim in dims:
            name = getattr(dim, 'name', None)
            if name is None or name not in raw:
                valid = False
                break
            val = raw[name]
            try:
                if Integer is not None and isinstance(dim, Integer):
                    coerced = int(round(float(val)))
                    coerced = max(dim.low, min(dim.high, coerced))
                elif Real is not None and isinstance(dim, Real):
                    coerced = float(val)
                    coerced = min(dim.high, max(dim.low, coerced))
                else:
                    coerced = val
            except Exception:
                valid = False
                break
            entry[name] = coerced
        if valid:
            sanitized.append(entry)
    return sanitized


def _apply_bounds_override(pspace: Dict[str, Tuple[Any, Any]],
                           override: Optional[Dict[str, Any]]) -> Dict[str, Tuple[Any, Any]]:
    if not override:
        return dict(pspace)
    adjusted: Dict[str, Tuple[Any, Any]] = dict(pspace)
    for name, spec in override.items():
        if name not in adjusted:
            continue
        try:
            if isinstance(spec, dict):
                lo = spec.get('min', spec.get('low'))
                hi = spec.get('max', spec.get('high'))
            elif isinstance(spec, (list, tuple)) and len(spec) >= 2:
                lo, hi = spec[0], spec[1]
            else:
                continue
        except Exception:
            continue
        orig = adjusted[name]
        if not isinstance(orig, (list, tuple)) or len(orig) < 2:
            continue
        orig_lo, orig_hi = orig[0], orig[1]
        lo = orig_lo if lo is None else lo
        hi = orig_hi if hi is None else hi
        try:
            lo_f = float(lo)
            hi_f = float(hi)
        except Exception:
            continue
        if lo_f > hi_f:
            continue
        if isinstance(orig_lo, int) and isinstance(orig_hi, int):
            lo_cast = int(round(lo_f))
            hi_cast = int(round(hi_f))
        else:
            lo_cast = lo_f
            hi_cast = hi_f
        adjusted[name] = (lo_cast, hi_cast)
    return adjusted


def prepare_data_for_timeframes(base_data: pd.DataFrame, timeframes: List[str]) -> Dict[str, pd.DataFrame]:
    data_dict = {}
    if 'timestamp' in base_data.columns:
        base_data = base_data.copy()
        base_data['timestamp'] = pd.to_datetime(base_data['timestamp'])
        base_data = base_data.set_index('timestamp')
    if not isinstance(base_data.index, pd.DatetimeIndex):
        base_data.index = pd.to_datetime(base_data.index)
    # ensure OHLCV
    if 'close' not in base_data.columns and 'price' in base_data.columns:
        base_data['close'] = base_data['price']
    if 'open' not in base_data.columns:
        base_data['open'] = base_data['close'].shift(1).fillna(base_data['close'])
    if 'high' not in base_data.columns:
        base_data['high'] = base_data['close']
    if 'low' not in base_data.columns:
        base_data['low'] = base_data['close']
    if 'volume' not in base_data.columns:
        base_data['volume'] = 0

    rule_map = {
        '5min': '5min', '15min': '15min', '30min': '30min',
        '1h': '1h', '2h': '2h', '4h': '4h', '8h': '8h', '16h': '16h', '1d': '1d'
    }
    for tf in timeframes:
        rule = rule_map.get(tf)
        if not rule:
            continue
        res = base_data.resample(rule).agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
        }).dropna()
        res['price'] = res['close']
        data_dict[tf] = res.reset_index()
    return data_dict


def timeframe_to_minutes(tf: str) -> int:
    mapping = {
        '5min': 5,
        '15min': 15,
        '30min': 30,
        '1h': 60,
        '2h': 120,
        '4h': 240,
        '8h': 480,
        '16h': 960,
        '1d': 1440,
    }
    return int(mapping.get(tf, 60))


def _save_parquet_or_csv(df: pd.DataFrame, out_path_parquet: Path, out_path_csv: Path) -> str:
    """Save DataFrame as Parquet when possible; fall back to CSV.

    Returns the path actually written.
    """
    # Ensure parent dirs
    out_path_parquet.parent.mkdir(parents=True, exist_ok=True)
    out_path_csv.parent.mkdir(parents=True, exist_ok=True)
    # Normalize timestamp dtype if present
    try:
        if 'timestamp' in df.columns:
            df = df.copy()
            df['timestamp'] = pd.to_datetime(df['timestamp'])
    except Exception:
        pass
    # Prefer Parquet
    try:
        # Prefer pyarrow engine with snappy compression for speed/size balance
        df.to_parquet(str(out_path_parquet), index=False, compression='snappy', engine='pyarrow')
        try:
            # warm read
            pd.read_parquet(str(out_path_parquet), columns=df.columns[:1]).head(1)
        except Exception:
            pass
        return str(out_path_parquet)
    except Exception:
        # Fallback to CSV
        df.to_csv(str(out_path_csv), index=False)
        try:
            pd.read_csv(str(out_path_csv), nrows=1)
        except Exception:
            pass
        return str(out_path_csv)


def _build_run_cache(unified_csv_path: Union[str, Path], timeframes: List[str], cache_root: Path) -> Dict[str, str]:
    """Build a per-run cache: base file and resampled per-timeframe files.

    Returns mapping {tf: file_path} (Parquet when possible, CSV fallback).
    """
    src_path = str(unified_csv_path)
    cache_root.mkdir(parents=True, exist_ok=True)
    # If a meta file exists, and matches fingerprint, reuse existing cache
    meta_path = cache_root / 'meta.json'
    try:
        src = Path(src_path)
        fp = f"{src.resolve()}|{src.stat().st_mtime_ns}|{src.stat().st_size}|{','.join(sorted(timeframes))}"
        h = hashlib.sha1(fp.encode('utf-8')).hexdigest()
    except Exception:
        h = ''
    cache_root.mkdir(parents=True, exist_ok=True)
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            if meta.get('hash') == h:
                # Build mapping from existing files
                per_tf: Dict[str, str] = {}
                for tf in timeframes:
                    pq = cache_root / f'resampled_{tf}.parquet'
                    cs = cache_root / f'resampled_{tf}.csv'
                    if pq.exists():
                        per_tf[tf] = str(pq)
                    elif cs.exists():
                        per_tf[tf] = str(cs)
                if per_tf:
                    return per_tf
        except Exception:
            pass

    # Load unified CSV once (fast parsing options)
    try:
        base = pd.read_csv(
            src_path,
            parse_dates=['timestamp'] if 'timestamp' in pd.read_csv(src_path, nrows=0).columns else None,
            na_filter=False,
            memory_map=True,
        )
    except Exception:
        base = pd.read_csv(src_path)
    # Normalize to OHLCV + datetime index
    if 'timestamp' in base.columns:
        base['timestamp'] = pd.to_datetime(base['timestamp'])
        base = base.set_index('timestamp')
    if not isinstance(base.index, pd.DatetimeIndex):
        base.index = pd.to_datetime(base.index)
    if 'close' not in base.columns and 'price' in base.columns:
        base['close'] = base['price']
    if 'open' not in base.columns:
        base['open'] = base['close'].shift(1).fillna(base['close'])
    if 'high' not in base.columns:
        base['high'] = base['close']
    if 'low' not in base.columns:
        base['low'] = base['close']
    if 'volume' not in base.columns:
        base['volume'] = 0
    # Save base
    try:
        _save_parquet_or_csv(base.reset_index(), cache_root / 'base.parquet', cache_root / 'base.csv')
    except Exception:
        pass
    # Resample per timeframe one time
    rule_map = {
        '5min': '5min', '15min': '15min', '30min': '30min',
        '1h': '1h', '2h': '2h', '4h': '4h', '8h': '8h', '16h': '16h', '1d': '1d'
    }
    per_tf: Dict[str, str] = {}
    for tf in timeframes:
        rule = rule_map.get(tf)
        if not rule:
            continue
        try:
            res = base.resample(rule).agg({
                'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
            }).dropna()
            res['price'] = res['close']
            res = res.reset_index().rename(columns={'index': 'timestamp'})
            outp = _save_parquet_or_csv(res, cache_root / f'resampled_{tf}.parquet', cache_root / f'resampled_{tf}.csv')
            per_tf[tf] = outp
        except Exception:
            continue
    # Write metadata for reuse
    try:
        meta_path.write_text(json.dumps({'hash': h, 'source': src_path, 'timeframes': timeframes}, indent=2))
    except Exception:
        pass
    return per_tf


def _purge_stale_caches(tmp_dir: Path, prefix: str, ttl_hours: int = 24) -> None:
    """Remove cache directories older than TTL to avoid buildup after hard kills."""
    try:
        if not tmp_dir.exists():
            return
        import time
        now = time.time()
        ttl = ttl_hours * 3600
        for p in tmp_dir.iterdir():
            try:
                if not p.is_dir():
                    continue
                if not p.name.startswith(prefix):
                    continue
                age = now - p.stat().st_mtime
                if age > ttl:
                    shutil.rmtree(p, ignore_errors=True)
            except Exception:
                continue
    except Exception:
        pass


def _read_df_from_path(sp: str) -> pd.DataFrame:
    """Read a dataframe from Parquet/Feather/CSV with consistent schema."""
    try:
        if sp.endswith('.parquet'):
            return pd.read_parquet(sp)
        if sp.endswith('.feather'):
            return pd.read_feather(sp)
        # CSV fallback with faster settings
        try:
            return pd.read_csv(sp, parse_dates=['timestamp'], na_filter=False, memory_map=True)
        except Exception:
            return pd.read_csv(sp)
    except Exception as e:
        raise


def _extract_error_message(payload: Dict[str, Any]) -> str:
    """Attempt to extract a human-friendly error message from a payload."""
    if not isinstance(payload, dict):
        return ''
    if isinstance(payload.get('error'), str):
        return payload.get('error')  # type: ignore[return-value]
    try:
        folds = payload.get('folds') or []
        for fold in folds:
            if not isinstance(fold, dict):
                continue
            metrics = fold.get('oos_metrics') or {}
            if isinstance(metrics, dict) and isinstance(metrics.get('error'), str):
                return str(metrics.get('error'))
    except Exception:
        return ''
    return ''


def _classify_stage_result(payload: Dict[str, Any]) -> str:
    """Classify an existing stage result as success, zero-trades, or error."""
    if not isinstance(payload, dict):
        return 'error'
    if payload.get('error'):
        return 'error'
    try:
        folds = payload.get('folds') or []
        for fold in folds:
            if not isinstance(fold, dict):
                continue
            metrics = fold.get('oos_metrics') or {}
            if isinstance(metrics, dict) and metrics.get('error'):
                return 'error'
    except Exception:
        return 'error'
    agg = payload.get('results_agg') or {}
    total_trades = agg.get('total_trades')
    if isinstance(total_trades, (int, float)) and total_trades <= 0:
        return 'zero_trades'
    results_inner = payload.get('results') or {}
    if isinstance(results_inner, dict):
        if results_inner.get('error'):
            return 'error'
        inner_trades = results_inner.get('total_trades')
        if isinstance(inner_trades, (int, float)) and inner_trades <= 0:
            return 'zero_trades'
    return 'success'


def _iter_stage_result_files(stage_dir: Path):
    """Yield per-strategy result files (supports .json and .json.gz)."""
    seen_bases = set()
    for pattern in ('*.json', '*.json.gz'):
        for path in stage_dir.glob(pattern):
            name = path.name
            if name.startswith('_'):
                continue
            if name.startswith('summary.json'):
                continue
            base = name.split('.json')[0]
            if base in seen_bases and path.suffix == '.gz':
                # Prefer uncompressed copy if both exist
                continue
            seen_bases.add(base)
            yield path


def _read_json_file(path: Path) -> Optional[Dict[str, Any]]:
    """Load JSON from plain or gzipped file; returns None on failure."""
    try:
        if path.suffix == '.gz' or path.name.endswith('.json.gz'):
            with gzip.open(path, 'rt', encoding='utf-8') as fh:
                return json.load(fh)
        with path.open('r', encoding='utf-8') as fh:
            return json.load(fh)
    except Exception:
        return None


def _load_stage_results(stage_dir: Path) -> Dict[str, Any]:
    """Load all per-strategy JSON results present under a stage directory."""
    entries: Dict[Tuple[str, str], Dict[str, Any]] = {}
    statuses: Dict[Tuple[str, str], str] = {}
    status_counts: Dict[str, int] = {'success': 0, 'zero_trades': 0, 'error': 0}
    has_failures = False
    error_items: List[Dict[str, Any]] = []
    zero_trade_items: List[Dict[str, Any]] = []
    if not stage_dir.exists() or not stage_dir.is_dir():
        return {
            'entries': entries,
            'statuses': statuses,
            'status_counts': status_counts,
            'has_failures': False,
            'count': 0,
            'path': stage_dir,
            'error_items': error_items,
            'zero_trade_items': zero_trade_items,
        }
    for path in _iter_stage_result_files(stage_dir):
        payload = _read_json_file(path)
        if not isinstance(payload, dict):
            continue
        strat = payload.get('strategy')
        tf = payload.get('timeframe')
        if not strat or not tf:
            continue
        key = (str(strat), str(tf))
        status = _classify_stage_result(payload)
        entries[key] = payload
        statuses[key] = status
        if status not in status_counts:
            status_counts[status] = 0
        status_counts[status] += 1
        if status != 'success':
            has_failures = True
        if status == 'error':
            message = _extract_error_message(payload) or 'error'
            error_items.append({
                'strategy': key[0],
                'timeframe': key[1],
                'message': message,
                'path': str(path),
            })
        elif status == 'zero_trades':
            message = 'total trades = 0'
            agg = payload.get('results_agg') or {}
            if isinstance(agg, dict) and agg.get('total_trades') is not None:
                message = f"total_trades={agg.get('total_trades')}"
            zero_trade_items.append({
                'strategy': key[0],
                'timeframe': key[1],
                'message': message,
                'path': str(path),
            })
    return {
        'entries': entries,
        'statuses': statuses,
        'status_counts': status_counts,
        'has_failures': has_failures,
        'count': len(entries),
        'path': stage_dir,
        'error_items': error_items,
        'zero_trade_items': zero_trade_items,
    }


def _analyze_existing_run(out_root: Path,
                          objectives: List[str],
                          stages: List[str]) -> Dict[str, Any]:
    """Inspect the output directory for existing per-stage results."""
    info: Dict[str, Any] = {'has_data': False, 'has_failures': False, 'objectives': {}, 'error_items': [], 'zero_trade_items': []}
    for obj in objectives:
        stage_map: Dict[str, Any] = {}
        for stage_label in stages:
            stage_dir = out_root / f"{obj}_{stage_label}"
            stage_info = _load_stage_results(stage_dir)
            if stage_info['count'] > 0:
                info['has_data'] = True
                if stage_info['has_failures']:
                    info['has_failures'] = True
                stage_map[stage_label] = stage_info
                for item in stage_info.get('error_items', []):
                    info['error_items'].append((obj, stage_label, item))
                for item in stage_info.get('zero_trade_items', []):
                    info['zero_trade_items'].append((obj, stage_label, item))
        if stage_map:
            info['objectives'][obj] = stage_map
    progress_path = out_root / '_progress.json'
    if progress_path.exists():
        try:
            info['progress'] = json.loads(progress_path.read_text())
        except Exception:
            info['progress'] = None
    else:
        info['progress'] = None
    return info


def run_backtest(strategy_class, params: Dict[str, Any], data: pd.DataFrame, pool_reserves: Dict[str, Tuple[int, int]] = None) -> Dict[str, Any]:
    strat = strategy_class(parameters=params)
    engine = BacktestEngine(pool_reserves=pool_reserves)
    trade_pct = None
    try:
        if isinstance(params, dict):
            trade_pct = params.get('trade_amount_pct')
    except Exception:
        trade_pct = None
    return engine.run_backtest(strat, data, trade_amount_pct=trade_pct)


def _worker_init():
    """Initializer for worker processes to clamp BLAS threads and quiet logging."""
    # Clamp BLAS/NumPy threading to avoid oversubscription
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    try:
        # Quiet noisy loggers in workers (only show errors)
        logging.getLogger().setLevel(logging.ERROR)
        logging.getLogger('bot.backtest_engine').setLevel(logging.ERROR)
        # Suppress common warnings to keep console clean
        import warnings
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        try:
            ensure_worker_cache_initialized()
        except Exception:
            pass
    except Exception:
        pass

def optimize_wfo(strategy_name: str,
                 timeframe: str,
                 ohlcv: Any,
                 top_calls: int = 80,
                 path_map: Optional[Dict[str, str]] = None,
                 objective: str = 'utility',
                 utility_kwargs: Optional[Dict[str, Any]] = None,
                 window_days: int = 90,
                 train_days: int = 45,
                 test_days: int = 15,
                 step_days: int = 15,
                 random_state: Optional[int] = None,
                 acq_funcs: Optional[List[str]] = None,
                 initial_params: Optional[List[Dict[str, Any]]] = None,
                 bounds_override: Optional[Dict[str, Any]] = None,
                 thread_workers: Optional[int] = None,
                 pool_reserves: Optional[Dict[str, Tuple[int, int]]] = None) -> Dict[str, Any]:
    strategy_class = load_strategy_class(strategy_name, path_map=path_map)
    if not strategy_class:
        return {'strategy': strategy_name, 'timeframe': timeframe, 'error': 'not found'}
    # Param space
    try:
        if hasattr(strategy_class, 'parameter_space'):
            pspace = strategy_class.parameter_space() or {}
        else:
            pspace = {}
    except Exception as e:
        logger.warning(f"parameter_space() error for {strategy_name}: {e}")
        pspace = {}
    if not pspace:
        try:
            from strategies.base_strategy import BaseStrategy as _BS
            inst = strategy_class()
            pspace = _BS._derive_param_space_from_defaults(getattr(inst, 'parameters', {}))
        except Exception:
            pspace = {}
    pspace = _apply_bounds_override(pspace, bounds_override)
    dims = build_skopt_dimensions(pspace)
    # Load timeframe data (prefer pre-resampled cache if provided)
    try:
        data_tf = None
        if isinstance(ohlcv, dict):
            src = ohlcv.get(timeframe)
            if src:
                try:
                    data_tf = _read_df_from_path(str(src))
                except Exception as e:
                    return {'strategy': strategy_name, 'timeframe': timeframe, 'error': f'data_load_failed: {e}'}
        if data_tf is None:
            if isinstance(ohlcv, (str, Path)):
                df_src = _read_df_from_path(str(ohlcv))
            else:
                df_src = ohlcv
            tf_data_map = prepare_data_for_timeframes(df_src, [timeframe])
            data_tf = tf_data_map.get(timeframe)
    except Exception as e:
        return {'strategy': strategy_name, 'timeframe': timeframe, 'error': f'data_load_failed: {e}'}
    if data_tf is None or len(data_tf) == 0:
        return {'strategy': strategy_name, 'timeframe': timeframe, 'error': 'empty timeframe data'}

    # Fetch trading reserves once for accurate slippage calculation unless supplied
    if pool_reserves is None:
        try:
            from collectors.reserve_fetcher import fetch_trading_reserves
            pool_reserves = fetch_trading_reserves()
            logger.info(f"Fetched {len(pool_reserves)} pool reserves for slippage calculation")
        except Exception as e:
            raise RuntimeError(f"Failed to fetch pool reserves for slippage calculation: {e}")
    
    # Generate folds
    folds = generate_wfo_folds(data_tf, window_days=window_days, train_days=train_days, test_days=test_days, step_days=step_days)
    if not folds:
        return {'strategy': strategy_name, 'timeframe': timeframe, 'error': 'no folds generated'}

    seed = 42 if random_state is None else int(random_state)
    acq_sequence = acq_funcs if acq_funcs else None
    acq_primary = acq_sequence[0] if acq_sequence else 'EI'
    if top_calls <= 0:
        init_points = 1
    else:
        init_points = max(1, min(top_calls, max(5, top_calls // 4)))
    thread_workers = max(1, int(thread_workers)) if thread_workers else 1
    optimizer = BayesianOptimizer(
        n_calls=top_calls,
        # Respect small --calls values; ensure 1 <= n_initial_points <= n_calls when positive
        n_initial_points=init_points,
        acq_func=acq_primary,
        random_state=seed,
        acq_funcs=acq_sequence,
        n_jobs=thread_workers,
        fold_workers=thread_workers if thread_workers > 1 else None,
        refinement_workers=thread_workers if thread_workers > 1 else None,
        refinement_neighbors=12,
    )

    warm_start_params = _sanitize_initial_params(initial_params, dims) if initial_params else []

    def backtest_func(strategy_obj, data_fold: pd.DataFrame, params: Dict[str, Any]) -> float:
        try:
            res = run_backtest(strategy_class, params, data_fold, pool_reserves)
        except Exception as e:
            # Swallow strategy errors and treat as zero score to keep pool healthy
            return 0.0
        if 'error' in res:
            return 0.0
        try:
            val, _ = score_from_results(strategy_name, res, data_fold, objective, utility_kwargs)
            return float(val)
        except Exception:
            return 0.0

    strat = strategy_class()
    fold_results = []
    oos_scores: List[float] = []
    last_params = getattr(strategy_class(), 'parameters', {})
    # Collect fields for aggregate OOS summary
    agg_factor = 1.0  # chained return factor across folds
    agg_init_balance = None
    agg_total_trades = 0
    agg_buy_trades = 0
    agg_sell_trades = 0
    agg_profitable_trades = 0
    agg_losing_trades = 0
    agg_max_dd = 0.0
    sharpe_weighted_sum = 0.0
    sharpe_weights = 0.0
    vol_weighted_sum = 0.0
    vol_weights = 0.0
    earliest_start = None
    latest_end = None
    pos_folds = 0
    num_folds = 0
    # Buy & Hold aggregation across OOS folds (chain factors)
    bh_factor = 1.0
    for (is_start, is_end, oos_start, oos_end) in folds:
        is_slice = data_tf[(data_tf['timestamp'] >= is_start) & (data_tf['timestamp'] < is_end)]
        oos_slice = data_tf[(data_tf['timestamp'] >= oos_start) & (data_tf['timestamp'] < oos_end)]
        if len(is_slice) < 10 or len(oos_slice) < 5:
            continue
        opt_res = optimizer.optimize_strategy(
            strategy=strat,
            strategy_type='generic',
            data=is_slice,
            backtest_func=backtest_func,
            initial_score=None,
            dimensions=dims,
            initial_params=warm_start_params,
        )
        best_params = opt_res.best_params
        last_params = best_params
        # Evaluate OOS
        try:
            final_oos = run_backtest(strategy_class, best_params, oos_slice, pool_reserves)
            oos_val, _ = score_from_results(strategy_name, final_oos, oos_slice, objective, utility_kwargs)
        except Exception:
            final_oos = {}
            oos_val = 0.0
        oos_scores.append(float(oos_val))
        # Aggregate OOS metrics
        try:
            m = final_oos or {}
            num_folds += 1
            # chained return via final_balance/initial_balance if present; else total_return_pct
            fi = float(m.get('final_balance')) if m.get('final_balance') is not None else None
            ii = float(m.get('initial_balance')) if m.get('initial_balance') is not None else None
            if agg_init_balance is None and ii is not None:
                agg_init_balance = ii
            if fi is not None and ii is not None and ii > 0:
                fold_factor = fi / ii
            else:
                r_pct = float(m.get('total_return_pct', 0.0) or 0.0)
                fold_factor = 1.0 + (r_pct / 100.0)
            agg_factor *= max(0.0, fold_factor)
            tr = int(m.get('total_trades', 0) or 0)
            bt = int(m.get('buy_trades', 0) or 0)
            st = int(m.get('sell_trades', 0) or 0)
            pt = int(m.get('profitable_trades', 0) or 0)
            lt = int(m.get('losing_trades', 0) or 0)
            agg_total_trades += tr
            agg_buy_trades += bt
            agg_sell_trades += st
            agg_profitable_trades += pt
            agg_losing_trades += lt
            try:
                dd = float(m.get('max_drawdown_pct', 0.0) or 0.0)
                if dd > agg_max_dd:
                    agg_max_dd = dd
            except Exception:
                pass
            try:
                sh = float(m.get('sharpe_ratio', 0.0) or 0.0)
                w = float(m.get('data_points', 0) or 0.0)
                sharpe_weighted_sum += sh * (w if w > 0 else 1.0)
                sharpe_weights += (w if w > 0 else 1.0)
            except Exception:
                pass
            try:
                vol = float(m.get('volatility_pct', 0.0) or 0.0)
                wv = float(m.get('data_points', 0) or 0.0)
                vol_weighted_sum += vol * (wv if wv > 0 else 1.0)
                vol_weights += (wv if wv > 0 else 1.0)
            except Exception:
                pass
            try:
                # fold positive if total_return_pct > 0
                rpct = float(m.get('total_return_pct', 0.0) or 0.0)
                if rpct > 0:
                    pos_folds += 1
            except Exception:
                pass
            try:
                # derive OOS window bounds
                st_ts = pd.to_datetime(m.get('start_time')) if m.get('start_time') else None
                en_ts = pd.to_datetime(m.get('end_time')) if m.get('end_time') else None
                if st_ts is not None:
                    earliest_start = st_ts if earliest_start is None else min(earliest_start, st_ts)
                if en_ts is not None:
                    latest_end = en_ts if latest_end is None else max(latest_end, en_ts)
            except Exception:
                pass
            try:
                # Compute buy & hold for this OOS fold from slice
                # Use 'close' when available; else 'price'
                if len(oos_slice) > 1:
                    if 'close' in oos_slice.columns:
                        p0 = float(oos_slice.iloc[0]['close'])
                        p1 = float(oos_slice.iloc[-1]['close'])
                    else:
                        p0 = float(oos_slice.iloc[0]['price'])
                        p1 = float(oos_slice.iloc[-1]['price'])
                    if p0 > 0:
                        fold_bh_factor = p1 / p0
                        bh_factor *= max(0.0, fold_bh_factor)
            except Exception:
                pass
        except Exception:
            pass
        fold_results.append({
            'is_start': str(is_start), 'is_end': str(is_end),
            'oos_start': str(oos_start), 'oos_end': str(oos_end),
            'best_params': best_params,
            'oos_score': float(oos_val),
            'oos_metrics': final_oos,
        })
    final_score = float(np.mean(oos_scores)) if oos_scores else 0.0
    # Build aggregated OOS summary in a BacktestEngine-like shape
    try:
        final_balance = float((agg_init_balance if agg_init_balance is not None else 1000.0) * agg_factor)
        init_balance = float(agg_init_balance if agg_init_balance is not None else 1000.0)
        total_return_pct = ((final_balance / init_balance) - 1.0) * 100.0 if init_balance > 0 else 0.0
        sharpe_avg = (sharpe_weighted_sum / sharpe_weights) if sharpe_weights > 0 else 0.0
        vol_avg = (vol_weighted_sum / vol_weights) if vol_weights > 0 else 0.0
        # overall win rate from sells
        win_rate_pct = (agg_profitable_trades / agg_sell_trades * 100.0) if agg_sell_trades > 0 else 0.0
        oos_return_pct = float(((agg_factor) - 1.0) * 100.0)
        oos_bh_return_pct = float(((bh_factor) - 1.0) * 100.0) if bh_factor > 0 else 0.0
        results_agg = {
            'strategy_name': 'WFO OOS Aggregate',
            'start_time': str(earliest_start) if earliest_start is not None else '',
            'end_time': str(latest_end) if latest_end is not None else '',
            'duration_days': None,  # not reliable across folds
            'data_points': None,
            'initial_balance': init_balance,
            'final_balance': final_balance,
            'total_return_pct': float(total_return_pct),
            'total_fees': None,
            'total_trades': int(agg_total_trades),
            'buy_trades': int(agg_buy_trades),
            'sell_trades': int(agg_sell_trades),
            'win_rate_pct': float(win_rate_pct),
            'profitable_trades': int(agg_profitable_trades),
            'losing_trades': int(agg_losing_trades),
            'avg_win_pct': None,
            'avg_loss_pct': None,
            'profit_factor': None,
            'max_drawdown_pct': float(agg_max_dd),
            'volatility_pct': float(vol_avg),
            'sharpe_ratio': float(sharpe_avg),
            'pos_folds': int(pos_folds),
            'num_folds': int(num_folds),
            'bh_return_pct': float(oos_bh_return_pct),
            'oos_return_pct': float(oos_return_pct),
        }
    except Exception:
        results_agg = {
            'strategy_name': 'WFO OOS Aggregate',
            'initial_balance': 1000.0,
            'final_balance': 1000.0,
            'total_return_pct': 0.0,
            'total_trades': 0,
            'max_drawdown_pct': 0.0,
            'sharpe_ratio': 0.0,
            'volatility_pct': 0.0,
            'pos_folds': 0,
            'num_folds': 0,
            'bh_return_pct': 0.0,
            'oos_return_pct': 0.0,
        }
    return {
        'strategy': strategy_name,
        'timeframe': timeframe,
        'selected_params': last_params,
        'score': final_score,
        'score_cv': '',
        'components': {'objective': objective},
        'objective': objective,
        'folds': fold_results,
        'results_agg': results_agg,
    }


def _build_metrics(strategy_name: str, results: Dict[str, Any], data: pd.DataFrame) -> Tuple[StrategyMetrics, float]:
    """Build StrategyMetrics and compute buy&hold for the window."""
    # Dynamic buy & hold from data window
    buy_hold_return = 0.0
    if data is not None and len(data) > 0:
        try:
            if 'close' in data.columns:
                p0 = float(data.iloc[0]['close'])
                p1 = float(data.iloc[-1]['close'])
            elif 'price' in data.columns:
                p0 = float(data.iloc[0]['price'])
                p1 = float(data.iloc[-1]['price'])
            else:
                p0 = p1 = 0.0
            if p0 > 0:
                buy_hold_return = (p1 - p0) / p0
        except Exception:
            buy_hold_return = 0.0

    # Duration and trade frequency (trades per month)
    duration_days = int(results.get('duration_days', 30) or 30)
    if duration_days == 0 and data is not None:
        try:
            if 'timestamp' in data.columns:
                duration_days = max(1, int((pd.to_datetime(data['timestamp'].iloc[-1]) - pd.to_datetime(data['timestamp'].iloc[0])).days))
            elif isinstance(data.index, pd.DatetimeIndex) and len(data.index) > 0:
                duration_days = max(1, int((data.index.max() - data.index.min()).days))
            else:
                duration_days = 30
        except Exception:
            duration_days = 30
    total_trades = int(results.get('total_trades', 0) or 0)
    trade_frequency = (total_trades / max(1, duration_days)) * 30.0

    # Map fields with proper percent-to-decimal conversions
    metrics = StrategyMetrics(
        strategy_name=strategy_name,
        timeframe='window',
        total_return=float(results.get('total_return_pct', 0.0) or 0.0) / 100.0,
        max_drawdown=float(results.get('max_drawdown_pct', 0.0) or 0.0) / 100.0,
        sharpe_ratio=float(results.get('sharpe_ratio', 0.0) or 0.0),
        sortino_ratio=float(results.get('sortino_ratio', 0.0) or 0.0),
        total_trades=total_trades,
        winning_trades=int(results.get('profitable_trades', 0) or 0),
        losing_trades=int(results.get('losing_trades', 0) or 0),
        win_rate=float(results.get('win_rate_pct', 0.0) or 0.0) / 100.0,
        avg_win=float(results.get('avg_win_pct', 0.0) or 0.0) / 100.0,
        avg_loss=float(results.get('avg_loss_pct', 0.0) or 0.0) / 100.0,
        profit_factor=float(results.get('profit_factor', 0.0) or 0.0),
        recovery_factor=float(results.get('recovery_factor', 0.0) or 0.0),
        trade_frequency=float(trade_frequency),
        reversal_catches=int(results.get('reversal_catches', 0) or 0),
        reversal_opportunities=int(results.get('reversal_opportunities', 0) or 0),
    )
    return metrics, buy_hold_return


def score_from_results(
    strategy_name: str,
    results: Dict[str, Any],
    data: pd.DataFrame,
    objective: str = 'utility',
    utility_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[float, Dict[str, Any]]:
    """Compute objective score and component breakdown for a single window.

    Returns (score_value, components_dict).
    """
    metrics, bh = _build_metrics(strategy_name, results, data)
    # Profit/Drawdown ratio (MAR-like) objective
    if objective == 'mar':
        try:
            dd = max(1e-9, float(metrics.max_drawdown))
            mar = float(metrics.total_return) / dd
        except Exception:
            mar = 0.0
        return float(mar), {'mar': float(mar), 'buy_hold_return': bh}
    if objective == 'final_balance':
        final_balance = float(results.get('final_balance', 0.0) or 0.0)
        return final_balance, {'final_balance': final_balance}
    if objective == 'return_dd_ratio':
        total_return_pct = float(results.get('total_return_pct', 0.0) or 0.0)
        max_drawdown_pct = abs(float(results.get('max_drawdown_pct', 0.0) or 0.0))
        total_trades = int(results.get('total_trades', 0) or 0)
        min_return_pct = 5.0
        ratio_denominator = max(1.0, max_drawdown_pct)
        raw_ratio = total_return_pct / ratio_denominator if ratio_denominator > 0 else 0.0
        if total_trades > 1:
            trade_count_scaler = min(1.0, math.log(total_trades) / math.log(50.0))
        else:
            trade_count_scaler = 0.0
        score = raw_ratio * trade_count_scaler
        if total_return_pct < min_return_pct:
            score = 0.0
        comps = {
            'return_dd_ratio': float(score),
            'raw_ratio': float(raw_ratio),
            'trade_count_scaler': float(trade_count_scaler),
            'total_return_pct': total_return_pct,
            'max_drawdown_pct': max_drawdown_pct,
            'max_drawdown_duration_days': float(results.get('max_drawdown_duration_days', 0.0) or 0.0),
            'total_trades': total_trades,
        }
        return float(score), comps
    if objective == 'cps':
        scorer = CompositePerformanceScorer(buy_hold_return=bh)
        out = scorer.calculate_cps(metrics)
        val = float(out['cps'] if isinstance(out, dict) else out)
        comps = {
            'cps': val,
            'buy_hold_return': bh,
            'profit_score': out.get('profit_score', 0.0),
            'preservation_score': out.get('preservation_score', 0.0),
            'risk_adjusted_score': out.get('risk_adjusted_score', 0.0),
            'activity_score': out.get('activity_score', 0.0),
            'trend_detection_score': out.get('trend_detection_score', 0.0),
        }
        return val, comps
    # Profit-biased CPS v2 (no external deps)
    if objective in ('profit_biased','cps_v2','cps_v2_profit_biased'):
        # Components in percent space
        R = float(results.get('total_return_pct', 0.0) or 0.0)
        DD = abs(float(results.get('max_drawdown_pct', 0.0) or 0.0))
        S = float(results.get('sharpe_ratio', 0.0) or 0.0)
        N = float(results.get('total_trades', 0) or 0)
        RBH = float(bh * 100.0)
        def sigmoid(x):
            try:
                return 1.0 / (1.0 + np.exp(-float(x)))
            except Exception:
                return 0.0
        # Profit vs B&H (strong bias on alpha)
        P = sigmoid(0.06 * (R - RBH))
        # Preservation: lower drawdown is better; centered around 10%
        Pr = 1.0 - sigmoid(0.20 * (DD - 10.0))
        # Risk-adjusted via Sharpe
        Q = sigmoid(0.5 * S)
        # Activity: bell around ~25 trades
        Activity = float(np.exp(-((N - 25.0) / 25.0) ** 2)) if N > 0 else 0.0
        cps2 = 0.60 * P + 0.25 * Pr + 0.12 * Q + 0.03 * Activity
        # Hard cap penalty for extreme drawdown
        if DD > 60.0:
            cps2 *= 0.5
        comps = {
            'cps_v2': float(cps2),
            'profit_bias_P': float(P),
            'preservation_Pr': float(Pr),
            'risk_Q': float(Q),
            'activity': float(Activity),
            'buy_hold_return': bh,
        }
        return float(cps2), comps
    # default utility
    if DynamicUtilityScorer is None:
        # Fallback to CPS if utility scorer is not available
        scorer = CompositePerformanceScorer(buy_hold_return=bh)
        out = scorer.calculate_cps(metrics)
        val = float(out['cps'] if isinstance(out, dict) else out)
        return val, {'cps': val, 'fallback': True}
    kwargs = utility_kwargs or {}
    util = DynamicUtilityScorer(
        lambda_dd=kwargs.get('lambda_dd', 2.0),
        dd_power=kwargs.get('dd_power', 1.25),
        epsilon=kwargs.get('epsilon', 0.01),
        reliability_k=kwargs.get('reliability_k', 5.0),
        alpha_oos=kwargs.get('alpha_oos', 0.25),
    )
    out = util.score(metrics)
    return float(out['score']), out


def score_detail_from_results(
    strategy_name: str,
    results: Dict[str, Any],
    data: pd.DataFrame,
    objective: str,
    utility_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Detailed output for the chosen objective."""
    score, comps = score_from_results(strategy_name, results, data, objective, utility_kwargs)
    comps = dict(comps or {})
    comps['score'] = score
    comps['objective'] = objective
    return comps


@dataclass
class StageConfig:
    label: str     # '30d'|'90d'|'1y'
    csv_path: str  # OHLCV file path
    top_n: int


def load_ohlcv_for_label(label: str) -> str:
    # Always prefer unified 2-year source; fallback to legacy mapping if missing
    try:
        p = Path(WFO_DATA_PATH)
        if p.exists():
            return str(p)
    except Exception:
        pass
        file_map = {
            '30d': Config.data_path('{asset}_ohlcv_{quote}_30day_5m.csv'),
            '90d': Config.data_path('{asset}_ohlcv_{quote}_90day_5m.csv'),
            '1y': Config.data_path('{asset}_ohlcv_{quote}_365day_5m.csv'),
        }
    path = REPO_ROOT / file_map.get(label, '')
    if path.exists():
        return str(path)
    alt = Config.resolve_ohlcv_path()
    if not alt:
        raise FileNotFoundError(f"OHLCV not found for {label}")
    return alt

def _ensure_ts(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if 'timestamp' in d.columns:
        d['timestamp'] = pd.to_datetime(d['timestamp'])
    else:
        if not isinstance(d.index, pd.DatetimeIndex):
            d.index = pd.to_datetime(d.index)
        d = d.reset_index().rename(columns={'index': 'timestamp'})
    return d

def generate_wfo_folds(df_tf: pd.DataFrame, *, window_days: int, train_days: int, test_days: int, step_days: int) -> List[Tuple[pd.Timestamp,pd.Timestamp,pd.Timestamp,pd.Timestamp]]:
    """Create time-ordered walk-forward folds from the tail of the TF dataset."""
    d = _ensure_ts(df_tf)
    d = d.sort_values('timestamp')
    end_ts = d['timestamp'].max()
    start_window = end_ts - pd.Timedelta(days=window_days)
    d = d.loc[d['timestamp'] >= start_window]
    folds: List[Tuple[pd.Timestamp,pd.Timestamp,pd.Timestamp,pd.Timestamp]] = []
    cur_is_start = d['timestamp'].min()
    while True:
        is_start = cur_is_start
        is_end = is_start + pd.Timedelta(days=train_days)
        oos_start = is_end
        oos_end = oos_start + pd.Timedelta(days=test_days)
        if oos_end > end_ts:
            break
        folds.append((is_start, is_end, oos_start, oos_end))
        cur_is_start = is_start + pd.Timedelta(days=step_days)
        if cur_is_start + pd.Timedelta(days=train_days + test_days) > end_ts:
            break
    return folds


def collect_strategies() -> List[str]:
    # 1) Manifest
    manifest_path = REPO_ROOT / 'task' / 'strategy_manifest.json'
    names: List[str] = []
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        for _, info in manifest.get('strategies', {}).items():
            if info.get('status') == 'implemented' and info.get('implementation_class'):
                names.append(info['implementation_class'])
    # 2) Scan
    discovered = discover_strategy_classes([
        'strategies/lazybear/technical_indicators/technical_indicators',
        'strategies/lazybear/technical_indicators',
        'strategies'
    ])
    names.extend(discovered.keys())
    # dedupe + sort, drop placeholder strategies that should not be optimized
    excluded = {'StubStrategy'}
    return sorted(n for n in set(names) if n not in excluded)


def aggregate_stage(stage_dir: Path) -> None:
    """Generate CSV and Markdown summaries for a completed stage directory."""
    try:
        agg_path = REPO_ROOT / 'optimization' / 'aggregate.py'
        mod = SourceFileLoader('agg_opt_results', str(agg_path)).load_module()
        # Reuse its functions directly to avoid argv coupling
        rows = mod.load_results(stage_dir)
        # Always sort by score (highest first) for consistency
        import math as _math
        def _score_key(r: dict) -> float:
            try:
                v = float(r.get('score') if r.get('score') not in (None, '') else 'nan')
                return (-1e30 if _math.isnan(v) else v)
            except Exception:
                return -1e30
        rows_sorted = sorted(rows, key=_score_key, reverse=True)
        mod.write_csv(rows_sorted, stage_dir / 'stage_aggregate.csv')
        best = mod.best_per_strategy(rows_sorted)
        best_rows_sorted = sorted(list(best.values()), key=_score_key, reverse=True)
        mod.write_csv(best_rows_sorted, stage_dir / 'stage_best.csv')
        mod.write_report(stage_dir, best)
        # Also generate the interactive HTML dashboard to mirror earlier runs
        try:
            mod.generate_html_dashboard(stage_dir, rows_sorted, top_n=100)
        except Exception as _e:
            logger.warning("Dashboard generation skipped for %s: %s", stage_dir, _e)
        logger.info("Aggregation complete for %s", stage_dir)
    except Exception as e:
        logger.error("Aggregation failed for %s: %s", stage_dir, e)


def build_run_overall_aggregate(root_dir: Path) -> Optional[Path]:
    """Combine every stage_aggregate.csv in a run into a single CSV sorted by return."""
    try:
        rows = []
        for child in sorted(root_dir.iterdir()):
            if not child.is_dir():
                continue
            csv_path = child / 'stage_aggregate.csv'
            if not csv_path.exists():
                continue
            df = pd.read_csv(csv_path).copy()
            if 'stage_dir' not in df.columns:
                df.insert(0, 'stage_dir', child.name)
            parts = child.name.split('_', 1)
            obj = parts[0]
            window = parts[1] if len(parts) == 2 else ''
            if 'objective' in df.columns:
                df['objective'] = df['objective'].fillna(obj)
            else:
                df.insert(1, 'objective', obj)
            if 'window' in df.columns:
                df['window'] = df['window'].fillna(window)
            else:
                df.insert(2, 'window', window)
            rows.append(df)
        if not rows:
            return None
        combo = pd.concat(rows, ignore_index=True)
        if 'total_return_pct' in combo.columns:
            combo = combo.sort_values(by='total_return_pct', ascending=False, na_position='last')
        out_path = root_dir / 'overall_stage_aggregate.csv'
        combo.to_csv(out_path, index=False)
        return out_path
    except Exception as exc:  # pragma: no cover - aggregation should not break the run
        logger.warning("Failed to build run-level aggregate for %s: %s", root_dir, exc)
        return None


def update_global_reports_aggregate() -> None:
    """Rebuild reports/report_aggregate.csv from every run-level aggregate."""
    reports_root = REPO_ROOT / 'reports'
    try:
        rows = []
        for run_dir in sorted(reports_root.glob('optimizer_pipeline_*')):
            agg_path = run_dir / 'overall_stage_aggregate.csv'
            if not agg_path.exists():
                continue
            df = pd.read_csv(agg_path).copy()
            df.insert(0, 'run_dir', run_dir.name)
            rows.append(df)
        if not rows:
            return
        combo = pd.concat(rows, ignore_index=True)
        if 'total_return_pct' in combo.columns:
            combo = combo.sort_values(by='total_return_pct', ascending=False, na_position='last')
        out_path = reports_root / 'report_aggregate.csv'
        combo.to_csv(out_path, index=False)
    except Exception as exc:  # pragma: no cover
        logger.warning("Failed to update reports aggregate: %s", exc)


def optimize_one(strategy_name: str,
                 timeframe: str,
                 ohlcv: pd.DataFrame,
                 top_calls: int = 80,
                 path_map: Optional[Dict[str, str]] = None,
                 objective: str = 'utility',
                 utility_kwargs: Optional[Dict[str, Any]] = None,
                 random_state: Optional[int] = None,
                 acq_funcs: Optional[List[str]] = None,
                 initial_params: Optional[List[Dict[str, Any]]] = None,
                 bounds_override: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    # Load class
    strategy_class = load_strategy_class(strategy_name, path_map=path_map)
    if not strategy_class:
        return {'strategy': strategy_name, 'timeframe': timeframe, 'error': 'not found'}

    # Build param space from strategy-owned definition or default params
    try:
        if hasattr(strategy_class, 'parameter_space'):
            pspace = strategy_class.parameter_space() or {}
        else:
            pspace = {}
    except Exception as e:
        logger.warning(f"parameter_space() error for {strategy_name}: {e}")
        pspace = {}
    if not pspace:
        try:
            from strategies.base_strategy import BaseStrategy as _BS
            inst = strategy_class()
            pspace = _BS._derive_param_space_from_defaults(getattr(inst, 'parameters', {}))
        except Exception:
            pspace = {}
    if not pspace:
        # No optimizable params -> run defaults once
        params = getattr(strategy_class(), 'parameters', {})
        res = run_backtest(strategy_class, params, ohlcv)
        score_val, comps = score_from_results(strategy_name, res, ohlcv, objective, utility_kwargs) if 'error' not in res else (0.0, {})
        return {
            'strategy': strategy_name,
            'timeframe': timeframe,
            'best_params': params,
            'score': score_val,
            'results': res,
            'results_agg': res,
            'components': comps,
            'objective': objective,
        }

    # Fetch trading reserves once for accurate slippage calculation
    try:
        from collectors.reserve_fetcher import fetch_trading_reserves
        pool_reserves = fetch_trading_reserves()
        logger.info(f"Fetched {len(pool_reserves)} pool reserves for slippage calculation")
    except Exception as e:
        raise RuntimeError(f"Failed to fetch pool reserves for slippage calculation: {e}")

    pspace = _apply_bounds_override(pspace, bounds_override)
    dims = build_skopt_dimensions(pspace)

    # Stage-2 optimizer with explicit dimensions and CV/early-stopping logic
    if random_state is None:
        seed = 42
    else:
        seed = int(random_state)
    acq_sequence = acq_funcs if acq_funcs else None
    acq_primary = acq_sequence[0] if acq_sequence else 'EI'
    if top_calls <= 0:
        init_points = 1
    else:
        init_points = max(1, min(top_calls, max(5, top_calls // 4)))
    optimizer = BayesianOptimizer(
        n_calls=top_calls,
        # Respect small --calls values; ensure 1 <= n_initial_points <= n_calls
        n_initial_points=init_points,
        acq_func=acq_primary,
        random_state=seed,
        acq_funcs=acq_sequence,
    )

    def backtest_func_stage2(strategy_obj, data_fold: pd.DataFrame, params: Dict[str, Any]) -> float:
        # Always construct a fresh instance using provided params
        try:
            res = run_backtest(strategy_class, params, data_fold, pool_reserves)
        except Exception:
            return 0.0
        if 'error' in res:
            return 0.0
        try:
            val, _ = score_from_results(strategy_name, res, data_fold, objective, utility_kwargs)
            return float(val)
        except Exception:
            return 0.0

    strat = strategy_class()  # baseline instance for optimizer bookkeeping
    warm_start_params = _sanitize_initial_params(initial_params, dims) if initial_params else []
    opt_res = optimizer.optimize_strategy(
        strategy=strat,
        strategy_type='generic',
        data=ohlcv,
        backtest_func=backtest_func_stage2,
        initial_score=None,
        dimensions=dims,
        initial_params=warm_start_params,
    )

    best_params = opt_res.best_params
    best_cv = float(opt_res.best_cps_score)
    final = run_backtest(strategy_class, best_params, ohlcv, pool_reserves)
    # Compute objective score on the full window
    try:
        detail = score_detail_from_results(strategy_name, final, ohlcv, objective, utility_kwargs) if 'error' not in final else {'score': 0.0}
        score_full = float(detail.get('score', 0.0))
    except Exception:
        detail = {'score': 0.0}
        score_full = 0.0
    return {
        'strategy': strategy_name,
        'timeframe': timeframe,
        'best_params': best_params,
        'score': score_full,           # Full-window objective score
        'score_cv': best_cv,           # Cross-validated score from optimizer
        'components': detail,
        'objective': objective,
        'results': final,
        'results_agg': final,
        'tuning': {
            'n_calls': int(optimizer.n_calls),
            'n_initial_points': int(optimizer.n_initial_points),
            'param_history': getattr(opt_res, 'param_history', []),
            'score_history': getattr(opt_res, 'score_history', []),
            'convergence_iteration': getattr(opt_res, 'convergence_iteration', None),
            'improvement_percent': getattr(opt_res, 'improvement_percent', None),
            'total_iterations': getattr(opt_res, 'total_iterations', None),
        }
    }


def stage_run(label: str,
              ohlcv_csv: Union[str, Dict[str, str]],
              strategies: List[str],
              timeframes: List[str],
              workers: int = 12,
              calls: int = 80,
              out_dir: Path = None,
              objective: str = 'utility',
              utility_kwargs: Optional[Dict[str, Any]] = None,
              random_state: Optional[int] = None,
              progress_prefix: str = '',
              root_progress_path: Optional[Path] = None,
              acq_funcs: Optional[List[str]] = None,
              initial_params: Optional[List[Dict[str, Any]]] = None,
              bounds_override: Optional[Dict[str, Any]] = None,
              resume_mode: bool = False,
              resume_state: Optional[Dict[str, Any]] = None,
              rerun_errors: bool = False,
              rerun_zero_trades: bool = False,
              strategy_timeframes: Optional[Dict[str, List[str]]] = None) -> List[Dict[str, Any]]:
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
    resume_state = resume_state or {}
    existing_entries: Dict[Tuple[str, str], Dict[str, Any]] = {}
    status_map: Dict[Tuple[str, str], str] = {}
    if resume_mode:
        existing_entries = {k: v for k, v in (resume_state.get('entries') or {}).items()}
        status_map = {k: v for k, v in (resume_state.get('statuses') or {}).items()}

    # Build tasks as lightweight tuples including original index so random seeds remain stable.
    tasks_all: List[Tuple[int, str, str]] = []
    for s in strategies:
        tf_list = (strategy_timeframes or {}).get(s, timeframes)
        if not tf_list:
            tf_list = timeframes
        for tf in tf_list:
            tasks_all.append((len(tasks_all), s, tf))

    results: List[Dict[str, Any]] = []
    tasks_to_run: List[Tuple[int, str, str]] = []
    for idx, s, tf in tasks_all:
        key = (str(s), str(tf))
        entry = existing_entries.get(key)
        status = status_map.get(key)
        if resume_mode and entry is not None:
            rerun = False
            if status == 'error' and rerun_errors:
                rerun = True
            elif status == 'zero_trades' and rerun_zero_trades:
                rerun = True
            if not rerun:
                results.append(entry)
                continue
        tasks_to_run.append((idx, s, tf))

    skipped_count = len(results)
    total_tasks = len(tasks_all)
    # Build a shared path map once so workers can load any strategy file
    path_map = discover_strategy_classes([
        'strategies/lazybear/technical_indicators',
        'strategies',
    ])
    start_ts = datetime.now()

    stage_key = f"{objective}_{label}"

    shared_pool_reserves: Optional[Dict[str, Tuple[int, int]]] = None
    if tasks_to_run:
        try:
            from collectors.reserve_fetcher import fetch_trading_reserves
            shared_pool_reserves = fetch_trading_reserves() or {}
            logger.info(f"Pre-fetched {len(shared_pool_reserves)} pool reserves for stage {stage_key}")
        except Exception as e:
            logger.error(f"Failed to fetch pool reserves for stage {stage_key}: {e}")
            raise

    def emit_progress(current_completed: int, stage_eta_seconds: int) -> None:
        pct = (current_completed / total_tasks * 100.0) if total_tasks else 100.0
        if out_dir:
            try:
                stage_progress = {
                    'objective': objective,
                    'stage': label,
                    'completed': current_completed,
                    'total': total_tasks,
                    'pct': pct,
                    'eta_seconds': stage_eta_seconds,
                }
                atomic_write_json(out_dir / '_progress.json', stage_progress)
            except Exception:
                pass
        if root_progress_path is None:
            return
        try:
            if root_progress_path.exists():
                g = json.loads(root_progress_path.read_text())
            else:
                g = {}
            g.setdefault('objectives', [])
            if objective not in g['objectives']:
                g['objectives'].append(objective)
            g.setdefault('stages', {})
            g.setdefault('stage_units', {'completed': 0, 'total': 0})
            if 'started_at' not in g or not g['started_at']:
                g['started_at'] = datetime.now().isoformat()
            stage_entry = g['stages'].get(stage_key, {'completed': 0, 'total': total_tasks})
            prev_completed = int(stage_entry.get('completed', 0))
            stage_entry['total'] = total_tasks
            stage_entry['completed'] = min(total_tasks, max(current_completed, 0))
            g['stages'][stage_key] = stage_entry
            # Recompute overall totals to avoid double counting
            total_all = sum(int(st.get('total', 0)) for st in g['stages'].values())
            completed_all = sum(int(st.get('completed', 0)) for st in g['stages'].values())
            g['total'] = total_all
            g['completed'] = min(total_all, completed_all)
            g['current'] = {'objective': objective, 'stage': label}
            if stage_eta_seconds is not None:
                g['stage_eta_seconds'] = stage_eta_seconds
            elapsed_all = (datetime.now() - pd.to_datetime(g['started_at'])).total_seconds() if g.get('started_at') else 0
            g['pct'] = (g['completed'] / g['total'] * 100.0) if g['total'] else 100.0
            rate_all = (g['completed'] / elapsed_all) if elapsed_all > 0 else 0
            eta_all = int((g['total'] - g['completed']) / rate_all) if rate_all > 0 else 0
            g['eta_seconds'] = eta_all
            atomic_write_json(root_progress_path, g)
            # Emit concise overall summaries mirroring the original output
            eta_h_all = eta_all // 3600
            eta_m_all = (eta_all % 3600) // 60
            eta_s_all = eta_all % 60
            elapsed_h = int(elapsed_all // 3600)
            elapsed_m = int((elapsed_all % 3600) // 60)
            elapsed_s = int(elapsed_all % 60)
            obj_summ = []
            for obj_name in g.get('objectives', []):
                tot = comp = 0
                for st_key, st_rec in g.get('stages', {}).items():
                    if st_key.startswith(f"{obj_name}_"):
                        tot += int(st_rec.get('total', 0) or 0)
                        comp += int(st_rec.get('completed', 0) or 0)
                if tot > 0:
                    obj_summ.append(f"{obj_name} {comp}/{tot}")
            obj_part = "; ".join(obj_summ) if obj_summ else ""
            if obj_part:
                print(f"[overall] {g['completed']}/{g['total']} ({g['pct']:.1f}%) elapsed {elapsed_h:02d}:{elapsed_m:02d}:{elapsed_s:02d} ETA {eta_h_all:02d}:{eta_m_all:02d}:{eta_s_all:02d} | {obj_part}", flush=True)
            else:
                print(f"[overall] {g['completed']}/{g['total']} ({g['pct']:.1f}%) elapsed {elapsed_h:02d}:{elapsed_m:02d}:{elapsed_s:02d} ETA {eta_h_all:02d}:{eta_m_all:02d}:{eta_s_all:02d}", flush=True)
            su = g.get('stage_units', {'completed': 0, 'total': 0})
            su_c = int(su.get('completed', 0))
            su_t = int(su.get('total', 0))
            stage_pct = (su_c / su_t * 100.0) if su_t else 0.0
            stage_eta = int(g.get('stage_eta_seconds', 0)) if isinstance(g.get('stage_eta_seconds'), (int, float)) else 0
            sh = stage_eta // 3600
            sm = (stage_eta % 3600) // 60
            ss = stage_eta % 60
            print(f"[overall-stages] {su_c}/{su_t} ({stage_pct:.1f}%) elapsed {elapsed_h:02d}:{elapsed_m:02d}:{elapsed_s:02d} ETA {sh:02d}:{sm:02d}:{ss:02d}", flush=True)
        except Exception:
            pass

    if resume_mode and skipped_count:
        pct_skipped = (skipped_count / total_tasks * 100.0) if total_tasks else 100.0
        print(f"{progress_prefix}[{objective}][{label}] resume skipping {skipped_count}/{total_tasks} ({pct_skipped:.1f}%) previously completed results", flush=True)

    if total_tasks == 0:
        emit_progress(0, 0)
        return results

    completed = skipped_count
    last_reported = skipped_count

    # Align overall progress state with the current completion count.
    if resume_mode and skipped_count:
        emit_progress(skipped_count, 0)

    # macOS: enforce spawn and set worker initializer
    if tasks_to_run:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx, initializer=_worker_init) as ex:
            # Baseline WFO settings per stage (will be scaled per timeframe to ensure min bars)
            if label == '30d':
                base = dict(window_days=90, train_days=45, test_days=15, step_days=15)
            elif label == '90d':
                base = dict(window_days=270, train_days=120, test_days=30, step_days=30)
            else:
                base = dict(window_days=365, train_days=180, test_days=30, step_days=30)
            futs = []
            # Minimum bars needed per fold to avoid short-window errors
            MIN_IS_BARS = 180
            MIN_OOS_BARS = 60
            total_cores = os.cpu_count() or 8
            threads_per_worker = max(1, total_cores // max(1, workers))
            for orig_idx, s, tf in tasks_to_run:
                tf_min = timeframe_to_minutes(tf)
                bars_to_days = lambda bars: int(np.ceil(bars * tf_min / (24 * 60)))
                train_days_tf = max(base['train_days'], bars_to_days(MIN_IS_BARS))
                test_days_tf = max(base['test_days'], bars_to_days(MIN_OOS_BARS))
                window_days_tf = max(base['window_days'], train_days_tf + test_days_tf + base['step_days'])

                tf_source = ohlcv_csv.get(tf) if isinstance(ohlcv_csv, dict) else ohlcv_csv
                seed = None if random_state is None else int(random_state) + orig_idx
                futs.append(ex.submit(
                    optimize_wfo,
                    s, tf, tf_source,
                    calls, path_map,
                    objective,
                    utility_kwargs or {},
                    window_days_tf,
                    train_days_tf,
                    test_days_tf,
                    base['step_days'],
                    random_state=seed,
                    acq_funcs=acq_funcs,
                    initial_params=initial_params,
                    bounds_override=bounds_override,
                    thread_workers=threads_per_worker,
                    pool_reserves=shared_pool_reserves,
                ))

            import traceback
            for fut in as_completed(futs):
                try:
                    r = fut.result()
                    results.append(r)
                    if out_dir:
                        result_path = out_dir / f"{r.get('strategy','unknown')}_{r.get('timeframe','tf')}.json.gz"
                        with gzip.open(result_path, 'wt', encoding='utf-8') as fh:
                            json.dump(r, fh, indent=2, default=str)
                except Exception as e:
                    logger.error(f"Stage {label} task failed: {e!r}")
                    logger.debug("%s", traceback.format_exc())
                finally:
                    completed += 1
                    if completed > total_tasks:
                        completed = total_tasks
                    # Light progress with ETA every 5 completions or on last
                    if completed == total_tasks or (completed - last_reported) >= 5:
                        elapsed = (datetime.now() - start_ts).total_seconds()
                        rate = completed / elapsed if elapsed > 0 else 0.0
                        remaining = max(0, total_tasks - completed)
                        eta_s = int(remaining / rate) if rate > 0 else 0
                        eta_h = eta_s // 3600
                        eta_m = (eta_s % 3600) // 60
                        eta_sec = eta_s % 60
                        pct = (completed / total_tasks * 100.0) if total_tasks else 100.0
                        print(f"{progress_prefix}[{objective}][{label}] {completed}/{total_tasks} ({pct:.1f}%) ETA {eta_h:02d}:{eta_m:02d}:{eta_sec:02d}", flush=True)
                        emit_progress(completed, eta_s)
                        last_reported = completed

    if completed < total_tasks:
        # Finalize progress so downstream aggregation treats the stage as complete.
        emit_progress(completed, 0)
    else:
        emit_progress(total_tasks, 0)

    if resume_mode and not tasks_to_run and skipped_count:
        # Ensure we report a completed line if nothing had to re-run.
        pct = (skipped_count / total_tasks * 100.0) if total_tasks else 100.0
        print(f"{progress_prefix}[{objective}][{label}] {skipped_count}/{total_tasks} ({pct:.1f}%) already complete", flush=True)

    return results


def select_top(results: List[Dict[str, Any]], top_n: int) -> List[Dict[str, Any]]:
    """Select top strategies ranked by aggregated OOS returns.

    Returns a list of dict entries containing strategy and best timeframe.
    """
    best_map: Dict[str, Dict[str, Any]] = {}
    fallback_map: Dict[str, Dict[str, Any]] = {}
    for r in results:
        strategy = r.get('strategy')
        timeframe = r.get('timeframe')
        if not strategy or not timeframe:
            continue
        agg = r.get('results_agg') or {}
        raw_return = agg.get('oos_return_pct', agg.get('total_return_pct'))
        try:
            ret_val = float(raw_return)
        except (TypeError, ValueError):
            continue
        if math.isnan(ret_val):
            continue
        trades = agg.get('total_trades', 0)
        try:
            trades_val = int(trades)
        except (TypeError, ValueError):
            trades_val = 0
        bh = agg.get('bh_return_pct', 0.0)
        try:
            bh_val = float(bh)
        except (TypeError, ValueError):
            bh_val = 0.0
        entry = {
            'strategy': strategy,
            'timeframe': timeframe,
            'return_pct': ret_val,
            'bh_return_pct': bh_val,
            'total_trades': trades_val,
        }
        # Zero-trade guard: only drop if market was flat/up (bh >= 0)
        if trades_val == 0 and bh_val >= 0.0:
            if strategy not in fallback_map or ret_val > fallback_map[strategy]['return_pct']:
                fallback_map[strategy] = entry
            continue
        current = best_map.get(strategy)
        if current is None or ret_val > current['return_pct']:
            best_map[strategy] = entry
    if not best_map and fallback_map:
        best_map = fallback_map
    ranked_entries = sorted(best_map.values(), key=lambda e: e['return_pct'], reverse=True)
    top_entries = ranked_entries[:max(1, top_n)]
    return top_entries


def main():
    import argparse
    ap = argparse.ArgumentParser(description='Unified Optimizer Runner (multi-stage)')
    ap.add_argument('--top-n1', type=int, default=20)
    ap.add_argument('--top-n2', type=int, default=10)
    ap.add_argument('--top-n3', type=int, default=5)
    ap.add_argument('--timeframes', type=str, default=','.join(TIMEFRAME_LIST))
    # Default workers determined dynamically to target ~90% CPU utilization
    ap.add_argument('--workers', type=int, default=0)
    ap.add_argument('--calls', type=int, default=80)
    ap.add_argument('--stage', type=str, choices=['all','30d','90d','1y'], default='all',
                    help='Run all stages or a single stage')
    ap.add_argument('--from-summary', type=str, default='',
                    help='Path to a prior stage summary.json containing a "top" array to seed strategies')
    ap.add_argument('--strategies-file', type=str, default='',
                    help='Path to a JSON list file of strategy class names to seed a single-stage run')
    # Objective/scoring controls (defaults set here; no env usage)
    ap.add_argument('--objective', type=str, choices=['mar','utility','cps','profit_biased','cps_v2','cps_v2_profit_biased','final_balance'], default='mar', help='Scoring objective')
    ap.add_argument('--objectives', type=str, default='', help='Comma-separated list of objectives to run sequentially (overrides --objective). Allowed: mar,utility,cps,profit_biased,cps_v2,cps_v2_profit_biased,final_balance')
    ap.add_argument('--lambda-dd', type=float, default=2.0, help='Utility lambda for DD penalty')
    ap.add_argument('--dd-power', type=float, default=1.25, help='Utility power for DD penalty')
    ap.add_argument('--epsilon', type=float, default=0.01, help='Small epsilon for diagnostic ratios')
    ap.add_argument('--reliability-k', type=float, default=5.0, help='Soft reliability discount k (0 disables)')
    ap.add_argument('--alpha-oos', type=float, default=0.25, help='OOS blend weight (0..1), 0.25 favors OOS')
    ap.add_argument('--out-dir', type=str, default='', help='Custom output directory root')
    ap.add_argument('--random-state', type=int, default=None, help='Random seed for Bayesian optimizer (per run); default reproducible (42 per task)')
    ap.add_argument('--acq-funcs', type=str, default='auto', help='Acquisition functions to use (comma-separated ei,pi,lcb, "cycle", or "auto")')
    ap.add_argument('--initial-sample', type=str, default='', help='Path to JSON with parameter dict(s) for warm start ("best_current" shortcut)')
    ap.add_argument('--bounds-file', type=str, default='', help='JSON file specifying parameter bounds overrides (relative paths resolve under optimization/bounds/)')
    args = ap.parse_args()

    acq_arg = (args.acq_funcs or '').strip()
    acq_funcs_list: Optional[List[str]] = None
    if acq_arg:
        lowered = acq_arg.lower()
        if lowered not in ('auto', ''):
            if lowered == 'cycle':
                acq_funcs_list = ['EI', 'PI', 'LCB']
            else:
                parts = [p.strip() for p in acq_arg.split(',') if p.strip()]
                seq: List[str] = []
                for part in parts:
                    label = part.upper()
                    if label not in {'EI', 'PI', 'LCB'}:
                        raise SystemExit(f"Unsupported acquisition function '{part}' (use ei, pi, lcb, cycle, or auto)")
                    seq.append(label)
                acq_funcs_list = seq or None

    initial_param_samples: Optional[List[Dict[str, Any]]] = None
    sample_spec = (args.initial_sample or '').strip()
    if sample_spec:
        candidates: List[Path] = []
        if sample_spec.lower() == 'best_current':
            candidates.append(REPO_ROOT / 'reports' / 'gridTradeAggressive' / 'best_current' / 'best_params.json')
        else:
            candidates.append(Path(sample_spec))
        sample_path = next((p for p in candidates if p.exists()), None)
        if sample_path is None:
            raise SystemExit(f"Initial sample file not found: {candidates[0]}")
        try:
            payload = json.loads(sample_path.read_text())
        except Exception as exc:
            raise SystemExit(f"Failed to parse initial sample '{sample_path}': {exc}")
        entries = payload if isinstance(payload, list) else [payload]
        parsed: List[Dict[str, Any]] = []
        for entry in entries:
            if isinstance(entry, dict):
                parsed.append(_coerce_param_types(entry))
        if parsed:
            initial_param_samples = parsed

    bounds_override: Optional[Dict[str, Tuple[Any, Any]]] = None
    bounds_spec = (args.bounds_file or '').strip()
    if bounds_spec:
        cand_paths: List[Path] = []
        base_path = Path(bounds_spec)
        if base_path.is_absolute():
            cand_paths.append(base_path)
        else:
            cand_paths.extend([
                Path(bounds_spec),
                REPO_ROOT / bounds_spec,
                REPO_ROOT / 'optimization' / 'bounds' / bounds_spec,
            ])
        bounds_path = next((p for p in cand_paths if p.exists()), None)
        if bounds_path is None:
            raise SystemExit(f"Bounds file not found (checked: {[str(p) for p in cand_paths]})")
        try:
            raw_bounds = json.loads(bounds_path.read_text())
        except Exception as exc:
            raise SystemExit(f"Failed to parse bounds file '{bounds_path}': {exc}")
        if isinstance(raw_bounds, dict):
            if 'bounds' in raw_bounds and isinstance(raw_bounds['bounds'], dict):
                raw_bounds = raw_bounds['bounds']
            overrides: Dict[str, Tuple[Any, Any]] = {}
            for key, val in raw_bounds.items():
                lo = hi = None
                if isinstance(val, dict):
                    lo = val.get('min', val.get('low'))
                    hi = val.get('max', val.get('high'))
                elif isinstance(val, (list, tuple)) and len(val) >= 2:
                    lo, hi = val[0], val[1]
                else:
                    continue
                if lo is None or hi is None:
                    continue
                try:
                    lo_f = float(lo)
                    hi_f = float(hi)
                except Exception:
                    continue
                overrides[str(key)] = (lo_f, hi_f)
            if overrides:
                bounds_override = overrides

    # Multi-objective orchestration (run sequentially via subprocess for isolation)
    tf_list = [tf.strip() for tf in (args.timeframes or '').split(',') if tf.strip()]
    if not tf_list:
        tf_list = TIMEFRAME_LIST

    multi = (args.objectives or '').strip()
    if multi:
        allowed = {'mar','utility','cps','profit_biased','cps_v2','cps_v2_profit_biased','final_balance'}
        obj_list = [o.strip() for o in multi.split(',') if o.strip()]
        bad = [o for o in obj_list if o not in allowed]
        if bad:
            raise SystemExit(f"Unsupported objectives: {bad}. Allowed: mar, utility, cps")
        base_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_dirs = []
        # Use a single root folder for this multi-objective run
        root_dir = args.out_dir if args.out_dir else str(REPO_ROOT / 'reports' / f'optimizer_pipeline_{base_ts}')
        Path(root_dir).mkdir(parents=True, exist_ok=True)
        os.environ['SWAP_COST_CACHE_DIR'] = str(root_dir)
        os.environ['SWAP_COST_CACHE_PRODUCER'] = '1'
        try:
            initialize_swap_cost_cache(root_dir, producer=True, initial_target=Decimal('100000'))
        except Exception as exc:
            print(f"[runner] Failed to initialize swap cost cache producer: {exc}")
            raise
        # Initialize a shared progress file upfront so children don't assume a single-objective total
        try:
            stage_labels = ['30d','90d','1y'] if args.stage in ('all',) else [args.stage]
            stage_units_total = len(obj_list) * len(stage_labels)
            progress_path = Path(root_dir) / '_progress.json'
            if not progress_path.exists():
                progress_path.write_text(json.dumps({
                    'objectives': obj_list,
                    'stages': {},
                    'completed': 0,
                    'total': 0,
                    'started_at': datetime.now().isoformat(),
                    'pct': 0.0,
                    'eta_seconds': 0,
                    'stage_units': {'completed': 0, 'total': stage_units_total},
                    'stage_eta_seconds': 0,
                    'stage_pct': 0.0,
                    'stage_time_starts': {},
                    'stage_durations_seconds': []
                }, indent=2))
                # Show initial overall-stages line for the whole multi-objective run
                print(f"[overall-stages] 0/{stage_units_total} (0.0%) elapsed 00:00:00 ETA --:--:--", flush=True)
            else:
                # Progress file exists (resume case), ensure stage_units total is correct for current objectives
                try:
                    g = json.loads(progress_path.read_text())
                    exp_total = len(obj_list) * len(stage_labels)
                    su = g.get('stage_units', {'completed': 0, 'total': exp_total})
                    if int(su.get('total', 0) or 0) != exp_total:
                        su['total'] = exp_total
                        g['stage_units'] = su
                        # Reset per-run counters for resume - completed counter will be re-incremented by children
                        g['stage_units']['completed'] = 0
                        g['stage_eta_seconds'] = 0
                        g['stage_pct'] = 0.0
                        g['stage_time_starts'] = {}
                        g['stage_durations_seconds'] = []
                        atomic_write_json(progress_path, g)
                        print(f"[overall-stages] Resuming with corrected total: 0/{exp_total} (0.0%) elapsed 00:00:00 ETA --:--:--", flush=True)
                except Exception as exc:
                    print(f"[runner] Failed to update existing progress file for resume: {exc}")
        except Exception:
            # Non-fatal: children will still run; they also attempt to correct totals if possible
            pass
        # Build a shared cache path derived from source CSV fingerprint so it can be reused across objectives
        unified_csv_path = load_ohlcv_for_label('1y')
        try:
            sp = Path(unified_csv_path)
            fp = f"{sp.resolve()}|{sp.stat().st_mtime_ns}|{sp.stat().st_size}|{','.join(sorted(tf_list))}"
            h = hashlib.sha1(fp.encode('utf-8')).hexdigest()
        except Exception:
            h = base_ts
        shared_cache_root = REPO_ROOT / 'tmp' / f"{RUN_CACHE_ROOT_NAME}_{h}"
        # Purge stale caches once at parent startup
        _purge_stale_caches(REPO_ROOT / 'tmp', RUN_CACHE_ROOT_NAME, ttl_hours=24)
        for obj in obj_list:
            # Child runners will create per-stage folders named '<objective>_<stage>' under this root
            out_dir = root_dir
            cmd = [
                sys.executable, '-m', 'optimization.runner',
                '--stage', args.stage,
                '--workers', str(args.workers),
                '--calls', str(args.calls),
                '--top-n1', str(args.top_n1), '--top-n2', str(args.top_n2), '--top-n3', str(args.top_n3),
                '--objective', obj,
                '--lambda-dd', str(args.lambda_dd), '--dd-power', str(args.dd_power), '--epsilon', str(args.epsilon),
                '--reliability-k', str(args.reliability_k), '--alpha-oos', str(args.alpha_oos),
                '--out-dir', out_dir,
                '--timeframes', ','.join(tf_list),
            ]
            if args.random_state is not None:
                cmd += ['--random-state', str(args.random_state)]
            if args.strategies_file:
                cmd += ['--strategies-file', args.strategies_file]
            if args.from_summary:
                cmd += ['--from-summary', args.from_summary]
            print(f"[runner] Launching objective={obj} out_dir={out_dir}")
            env = os.environ.copy()
            env['OPTIMIZER_SHARED_CACHE'] = str(shared_cache_root)
            env['SWAP_COST_CACHE_DIR'] = str(root_dir)
            env['SWAP_COST_CACHE_PRODUCER'] = '0'
            subprocess.run(cmd, check=True, env=env)
            out_dirs.append((obj, root_dir))
        # After all objectives finished, build per-stage combined aggregates
        try:
            stage_labels = ['30d','90d','1y'] if args.stage in ('all',) else [args.stage]
            allowed = {'mar','utility','cps'}
            for lbl in stage_labels:
                frames = []
                for (obj, od) in out_dirs:
                    p = Path(od) / f"{obj}_{lbl}" / 'stage_aggregate.csv'
                    if p.exists():
                        try:
                            df = pd.read_csv(p)
                            df.insert(0, 'objective', obj)
                            frames.append(df)
                        except Exception:
                            continue
                if not frames:
                    continue
                combo = pd.concat(frames, ignore_index=True)
                # Robust score sort
                def _key(x):
                    try:
                        return float(x)
                    except Exception:
                        return -1e30
                try:
                    combo = combo.sort_values(by='score', key=lambda s: s.map(_key), ascending=False)
                except Exception:
                    pass
                # Write combined at the root once per stage label
                outp = Path(root_dir) / f"all_objectives_{lbl}.csv"
                try:
                    combo.to_csv(outp, index=False)
                except Exception:
                    pass
            # Also copy per-objective aggregates to the root for convenience
            for lbl in stage_labels:
                for (obj, od) in out_dirs:
                    st_dir = Path(od) / f"{obj}_{lbl}"
                    agg = st_dir / 'stage_aggregate.csv'
                    best = st_dir / 'stage_best.csv'
                    if agg.exists():
                        try:
                            df = pd.read_csv(agg)
                            df = df.sort_values(by='score', ascending=False, key=lambda s: s.map(lambda x: float(x) if str(x) not in ('nan','') else -1e30) if df.shape[0] else s)
                            df.to_csv(Path(root_dir) / f"{obj}_{lbl}_aggregate.csv", index=False)
                        except Exception:
                            pass
                    if best.exists():
                        try:
                            dfb = pd.read_csv(best)
                            if 'score' in dfb.columns:
                                dfb = dfb.sort_values(by='score', ascending=False, key=lambda s: s.map(lambda x: float(x) if str(x) not in ('nan','') else -1e30) if dfb.shape[0] else s)
                            dfb.to_csv(Path(root_dir) / f"{obj}_{lbl}_best.csv", index=False)
                        except Exception:
                            pass

            # Build run-level and global aggregates for convenience
            run_agg_path = build_run_overall_aggregate(Path(root_dir))
            if run_agg_path:
                update_global_reports_aggregate()
        except Exception as e:
            print(f"[runner] Failed to build combined per-stage aggregates: {e}")
        # Cleanup shared cache after multi-objective run
        try:
            shutil.rmtree(shared_cache_root, ignore_errors=True)
        except Exception:
            pass
        try:
            get_swap_cost_cache().stop()
        except Exception:
            pass
        return

    timeframes = tf_list

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    # Compute default workers if not provided
    if not args.workers or args.workers <= 0:
        try:
            cpu_total = os.cpu_count() or 8
        except Exception:
            cpu_total = 8
        # Aim for ~90% utilization, at least 2
        default_workers = max(2, int(round(cpu_total * 0.9)))
        args.workers = default_workers
        print(f"[config] auto workers set to {args.workers} based on {cpu_total} CPUs (~90%)", flush=True)

    if args.out_dir:
        out_root = Path(args.out_dir)
    else:
        suffix = '' if args.stage == 'all' else f'_{args.stage}'
        out_root = REPO_ROOT / 'reports' / f'optimizer_pipeline_{timestamp}{suffix}'
    out_root.mkdir(parents=True, exist_ok=True)

    os.environ['SWAP_COST_CACHE_DIR'] = str(out_root)
    producer_env = os.environ.get('SWAP_COST_CACHE_PRODUCER')
    should_produce = producer_env is None or producer_env != '0'
    os.environ['SWAP_COST_CACHE_PRODUCER'] = '1' if should_produce else '0'
    initialize_swap_cost_cache(
        out_root,
        producer=should_produce,
        initial_target=Decimal('100000'),
    )

    # Build base utility kwargs
    base_opts = {
        'utility_kwargs': {
            'lambda_dd': args.lambda_dd,
            'dd_power': args.dd_power,
            'epsilon': args.epsilon,
            'reliability_k': args.reliability_k,
            'alpha_oos': args.alpha_oos,
        },
        'acq_funcs': acq_funcs_list,
        'initial_params': initial_param_samples,
        'bounds_override': bounds_override,
    }

    # Stage configs (all use unified data path)
    unified_csv = load_ohlcv_for_label('1y')
    stage1 = StageConfig('30d', unified_csv, args.top_n1)
    stage2 = StageConfig('90d', unified_csv, args.top_n2)
    stage3 = StageConfig('1y', unified_csv, args.top_n3)

    # Purge stale caches at startup (older than 24h), then build per-run cache (Parquet preferred)
    tmp_dir = REPO_ROOT / 'tmp'
    _purge_stale_caches(tmp_dir, RUN_CACHE_ROOT_NAME, ttl_hours=24)

    # If a shared cache path is provided via env (multi-objective parent), reuse it
    shared_cache_env = os.environ.get('OPTIMIZER_SHARED_CACHE', '').strip()
    cache_root = Path(shared_cache_env) if shared_cache_env else (REPO_ROOT / 'tmp' / f"{RUN_CACHE_ROOT_NAME}_{timestamp}")
    try:
        cache_map = _build_run_cache(unified_csv, TIMEFRAME_LIST, cache_root)
    except Exception as e:
        print(f"[runner] Cache build failed ({e}); using direct CSV reads.")
        cache_map = {}

    # Strategy set helpers
    def _parse_seed_entries(entries: List[Any]) -> Tuple[List[str], Dict[str, List[str]]]:
        names: List[str] = []
        tf_map: Dict[str, List[str]] = {}
        for entry in entries:
            if isinstance(entry, dict):
                name = entry.get('strategy') or entry.get('name')
                tf = entry.get('timeframe')
            else:
                name = str(entry)
                tf = None
            if not name:
                continue
            if name not in names:
                names.append(name)
            if tf:
                tf_map.setdefault(name, [])
                if tf not in tf_map[name]:
                    tf_map[name].append(tf)
        return names, tf_map

    def load_seed_strategies() -> Tuple[List[str], Dict[str, List[str]]]:
        # --strategies-file overrides
        if args.strategies_file:
            p = Path(args.strategies_file)
            if not p.exists():
                raise FileNotFoundError(f"strategies-file not found: {p}")
            arr = json.loads(p.read_text())
            return _parse_seed_entries(arr if isinstance(arr, list) else [arr])
        # --from-summary supplies {top: [...]}
        if args.from_summary:
            p = Path(args.from_summary)
            if not p.exists():
                raise FileNotFoundError(f"from-summary not found: {p}")
            data = json.loads(p.read_text())
            if isinstance(data, dict):
                if 'top' in data and isinstance(data['top'], list) and data['top']:
                    return _parse_seed_entries(data['top'])
                tf_map_data = data.get('timeframe_map')
                if isinstance(tf_map_data, dict) and tf_map_data:
                    names = list(tf_map_data.keys())
                    tf_map_normalized = {str(k): list(v) for k, v in tf_map_data.items()}
                    return names, tf_map_normalized
                if 'selected' in data and isinstance(data['selected'], list):
                    return _parse_seed_entries(data['selected'])
                # legacy format fallbacks
                flat = data.get('strategies') or data.get('top_strategies') or []
                return _parse_seed_entries(flat if isinstance(flat, list) else [flat])
            return _parse_seed_entries([])
        # Fallback to full discovery
        return collect_strategies(), {}

    # Full discovery only needed for stage all or stage 30d without seeds
    strategies_all = collect_strategies()
    logger.info(f"Total strategies discovered: {len(strategies_all)}")

    # Determine objectives sequence
    objectives: List[str] = []
    if args.objectives:
        objectives = [o.strip().lower() for o in args.objectives.split(',') if o.strip()]
    if not objectives:
        objectives = [args.objective.lower()]

    overall_summary = {}
    # Ensure spawn context is set once (macOS)
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    # Root overall progress file (initialize overall stage plan as well)
    root_progress_path = out_root / '_progress.json'
    stages_to_run = ['30d','90d','1y'] if args.stage == 'all' else [args.stage]

    resume_mode = False
    rerun_errors = False
    rerun_zero_trades = False
    resume_plan: Dict[str, Dict[str, Any]] = {}
    existing_run_info: Dict[str, Any] = {}
    if out_root.exists():
        try:
            existing_run_info = _analyze_existing_run(out_root, objectives, stages_to_run)
        except Exception as exc:
            print(f"[resume] Failed to inspect existing run data: {exc}")
            existing_run_info = {'has_data': False, 'has_failures': False, 'objectives': {}, 'error_items': [], 'zero_trade_items': []}
        if existing_run_info.get('has_data'):
            print('[resume] Existing results detected in output directory:')
            progress_state = existing_run_info.get('progress') or {}
            if isinstance(progress_state, dict) and progress_state.get('total'):
                total = int(progress_state.get('total') or 0)
                completed = int(progress_state.get('completed') or 0)
                pct = float(progress_state.get('pct') or 0.0)
                eta_sec = int(progress_state.get('eta_seconds') or 0)
                started_at = progress_state.get('started_at')
                if started_at:
                    try:
                        elapsed_sec = int((datetime.now() - pd.to_datetime(started_at)).total_seconds())
                    except Exception:
                        elapsed_sec = 0
                else:
                    elapsed_sec = 0
                def _fmt_hms(seconds: int) -> str:
                    seconds = max(0, int(seconds))
                    h = seconds // 3600
                    m = (seconds % 3600) // 60
                    s = seconds % 60
                    return f"{h:02d}:{m:02d}:{s:02d}"
                print(f"           overall progress: {completed}/{total} ({pct:.1f}%) elapsed {_fmt_hms(elapsed_sec)} ETA {_fmt_hms(eta_sec)}")
            for obj, stage_map in existing_run_info.get('objectives', {}).items():
                for stage_label, stage_info in stage_map.items():
                    counts = stage_info.get('status_counts', {})
                    succ = counts.get('success', 0)
                    zero = counts.get('zero_trades', 0)
                    err = counts.get('error', 0)
                    total = stage_info.get('count', 0)
                    print(f"           objective={obj} stage={stage_label}: total={total} success={succ} zero_trades={zero} errors={err}")
            error_entries = existing_run_info.get('error_items', []) or []
            zero_entries = existing_run_info.get('zero_trade_items', []) or []
            if error_entries:
                print(f"           error results ({len(error_entries)}):")
                for obj, stage_label, item in error_entries:
                    message = item.get('message', '')
                    msg_suffix = f" -> {message}" if message else ''
                    print(f"             - {obj}/{stage_label}: {item.get('strategy')} [{item.get('timeframe')}] {msg_suffix}")
            if zero_entries:
                print(f"           zero-trade results ({len(zero_entries)}):")
                for obj, stage_label, item in zero_entries:
                    message = item.get('message', '')
                    msg_suffix = f" -> {message}" if message else ''
                    print(f"             - {obj}/{stage_label}: {item.get('strategy')} [{item.get('timeframe')}] {msg_suffix}")
            if sys.stdin.isatty():
                answer = input('Resume from existing results? [y/N]: ').strip().lower()
                if answer == 'y':
                    resume_mode = True
                    resume_plan = existing_run_info.get('objectives', {})
                    if error_entries:
                        rerun_answer = input('Re-run strategies with prior errors (e.g., "not found")? [y/N]: ').strip().lower()
                        rerun_errors = (rerun_answer == 'y')
                    if zero_entries:
                        rerun_zero_answer = input('Re-run strategies that produced zero trades? [y/N]: ').strip().lower()
                        rerun_zero_trades = (rerun_zero_answer == 'y')
                    # Clean up progress file for resume: preserve existing data, add new objectives
                    if root_progress_path.exists():
                        try:
                            g = json.loads(root_progress_path.read_text())
                            # Preserve all existing objectives, add any new ones
                            existing_objs = set(g.get('objectives', []))
                            for obj in objectives:
                                if obj not in existing_objs:
                                    g.setdefault('objectives', []).append(obj)
                            # Preserve all existing stages, they'll be used for historical context
                            # Handle stage_units carefully for multi-objective vs single-objective runs
                            current_expected_total = len(objectives) * len(stages_to_run)
                            existing_total = int(g.get('stage_units', {}).get('total', 0))
                            # If existing total is much larger than current expected, this is likely a child in multi-objective run
                            # Preserve the existing total, just reset counters for current run tracking
                            if existing_total > current_expected_total * 2:  # Heuristic for multi-objective
                                # Multi-objective run: preserve existing stage_units total, reset per-run counters
                                g['stage_eta_seconds'] = 0
                                g['stage_pct'] = 0.0
                                g['stage_time_starts'] = {}
                                g['stage_durations_seconds'] = []
                            else:
                                # Single-objective or new run: reset stage_units for current scope
                                g['stage_units'] = {'completed': 0, 'total': current_expected_total}
                                g['stage_eta_seconds'] = 0
                                g['stage_pct'] = 0.0
                                g['stage_time_starts'] = {}
                                g['stage_durations_seconds'] = []
                            # Don't modify the overall totals - they should reflect all historical + current work
                            atomic_write_json(root_progress_path, g)
                        except Exception as exc:
                            print(f"[resume] Failed to update progress file for resume: {exc}")
                else:
                    print('[resume] Starting a fresh run in this directory (existing files may be overwritten).')
            else:
                print('[resume] Non-interactive session detected; continuing without automatic resume.')

    stage_units_total = len(objectives) * len(stages_to_run)
    if not root_progress_path.exists():
        atomic_write_json(root_progress_path, {
            'objectives': objectives,
            'stages': {},
            'completed': 0,
            'total': 0,
            'started_at': datetime.now().isoformat(),
            'pct': 0.0,
            'eta_seconds': 0,
            # Overall stage-unit tracker (counts completed stages across all objectives)
            'stage_units': {'completed': 0, 'total': stage_units_total},
            'stage_eta_seconds': 0,
            'stage_pct': 0.0,
            'stage_time_starts': {},
            'stage_durations_seconds': []
        })
        # Print initial overall-stages line (0 completed)
        print(f"[overall-stages] 0/{stage_units_total} (0.0%) elapsed 00:00:00 ETA --:--:--", flush=True)
    else:
        # If the progress file exists (e.g., parent initialized for multi-objective), ensure the total is correct
        try:
            g = json.loads(root_progress_path.read_text())
            exp_total = max(1, len(g.get('objectives', objectives)) * len(stages_to_run))
            su = g.get('stage_units', {'completed': 0, 'total': exp_total})
            if int(su.get('total', 0) or 0) != exp_total:
                su['total'] = exp_total
                g['stage_units'] = su
                atomic_write_json(root_progress_path, g)
        except Exception:
            pass

    try:
        for obj in objectives:
            # Build per-objective options
            objective = obj
            utility_kwargs = base_opts['utility_kwargs']
            obj_tag = objective

            base_seed_strategies, base_seed_map = (load_seed_strategies() if (args.strategies_file or args.from_summary) else (strategies_all, {}))

            def _count_combos(strats: List[str], tf_map: Dict[str, List[str]]) -> int:
                total = 0
                for name in strats:
                    tf_list = tf_map.get(name)
                    if tf_list:
                        total += len(tf_list)
                    else:
                        total += len(timeframes)
                return total

            current_strategies = list(base_seed_strategies)
            current_tf_map = {k: list(v) for k, v in base_seed_map.items()}
            top1_entries: List[Dict[str, Any]] = []
            top2_entries: List[Dict[str, Any]] = []
            top3_entries: List[Dict[str, Any]] = []
            stage1_input_count = len(current_strategies) if current_strategies else len(strategies_all)
            stage2_input_count = 0
            stage3_input_count = 0

            # Stage 1
            if args.stage in ('all', '30d'):
                s1_dir = out_root / f"{obj_tag}_30d"
                s1_dir.mkdir(exist_ok=True)
                input_strategies = current_strategies or list(strategies_all)
                input_tf_map = current_tf_map
                stage1_input_count = len(input_strategies)
                combos = _count_combos(input_strategies, input_tf_map)
                if combos <= 0:
                    combos = len(input_strategies) * len(timeframes)
                print(f"[progress] Starting Stage 1 (30d) for objective={objective} with {len(input_strategies)} strategies × {combos} combos", flush=True)
                try:
                    g = json.loads(root_progress_path.read_text()) if root_progress_path.exists() else {}
                    g.setdefault('stage_time_starts', {})
                    g['stage_time_starts'][f"{objective}_30d"] = datetime.now().isoformat()
                    atomic_write_json(root_progress_path, g)
                except Exception:
                    pass
                stage1_resume_state = resume_plan.get(objective, {}).get(stage1.label) if resume_mode else None
                res1 = stage_run(
                    stage1.label,
                    cache_map if cache_map else stage1.csv_path,
                    input_strategies,
                    timeframes,
                    args.workers,
                    args.calls,
                    s1_dir,
                    objective,
                    utility_kwargs,
                    args.random_state,
                    progress_prefix='',
                    root_progress_path=root_progress_path,
                    acq_funcs=base_opts.get('acq_funcs'),
                    initial_params=base_opts.get('initial_params'),
                    bounds_override=base_opts.get('bounds_override'),
                    resume_mode=resume_mode,
                    resume_state=stage1_resume_state,
                    rerun_errors=rerun_errors,
                    rerun_zero_trades=rerun_zero_trades,
                    strategy_timeframes=input_tf_map if input_tf_map else None,
                )
                top1_entries = select_top(res1, stage1.top_n)
                if top1_entries:
                    current_strategies = [entry['strategy'] for entry in top1_entries]
                    current_tf_map = {entry['strategy']: [entry['timeframe']] for entry in top1_entries if entry.get('timeframe')}
                else:
                    current_strategies = input_strategies[:stage1.top_n]
                    if input_tf_map:
                        current_tf_map = {s: input_tf_map.get(s, []) for s in current_strategies if input_tf_map.get(s)}
                stage1_summary = {
                    'top': top1_entries,
                    'top_strategies': current_strategies,
                    'timeframe_map': current_tf_map,
                    'total': len(input_strategies),
                }
                (s1_dir / 'summary.json').write_text(json.dumps(stage1_summary, indent=2))
                aggregate_stage(s1_dir)
                try:
                    g = json.loads(root_progress_path.read_text()) if root_progress_path.exists() else {}
                    g.setdefault('stage_units', {'completed': 0, 'total': stage_units_total})
                    try:
                        exp_total = max(1, len(g.get('objectives', objectives)) * len(stages_to_run))
                        if int(g['stage_units'].get('total', 0) or 0) != exp_total:
                            g['stage_units']['total'] = exp_total
                    except Exception:
                        pass
                    g.setdefault('stage_time_starts', {})
                    g.setdefault('stage_durations_seconds', [])
                    g.setdefault('started_at', datetime.now().isoformat())
                    st_key = f"{objective}_30d"
                    st_start = pd.to_datetime(g['stage_time_starts'].get(st_key)) if g['stage_time_starts'].get(st_key) else pd.to_datetime(datetime.now())
                    dur = int(max(0, (datetime.now() - st_start.to_pydatetime()).total_seconds()))
                    g['stage_durations_seconds'].append(dur)
                    g['stage_units']['completed'] = int(g['stage_units'].get('completed', 0)) + 1
                    comp_u = int(g['stage_units']['completed'])
                    tot_u = int(g['stage_units']['total'])
                    avg = (sum(g['stage_durations_seconds']) / max(1, len(g['stage_durations_seconds']))) if g['stage_durations_seconds'] else 0
                    rem = max(0, tot_u - comp_u)
                    g['stage_eta_seconds'] = int(rem * avg)
                    g['stage_pct'] = (comp_u / tot_u * 100.0) if tot_u else 100.0
                    atomic_write_json(root_progress_path, g)
                    eta = g['stage_eta_seconds']; eh=int(((pd.to_datetime(datetime.now())-pd.to_datetime(g['started_at'])).total_seconds())//3600); em=int((((pd.to_datetime(datetime.now())-pd.to_datetime(g['started_at'])).total_seconds())%3600)//60); es=int(((pd.to_datetime(datetime.now())-pd.to_datetime(g['started_at'])).total_seconds())%60)
                    th=eta//3600; tm=(eta%3600)//60; ts=eta%60
                    print(f"[overall-stages] {comp_u}/{tot_u} ({g['stage_pct']:.1f}%) elapsed {eh:02d}:{em:02d}:{es:02d} ETA {th:02d}:{tm:02d}:{ts:02d}", flush=True)
                except Exception:
                    pass
                if args.stage == '30d':
                    print(json.dumps({'stage': '30d', 'objective': objective, 'top': top1_entries, 'total': len(input_strategies)}, indent=2))
                    continue
            else:
                if not current_strategies:
                    current_strategies = list(base_seed_strategies)
                if not current_tf_map:
                    current_tf_map = {k: list(v) for k, v in base_seed_map.items()}

            # Stage 2
            if args.stage in ('all', '90d'):
                s2_dir = out_root / f"{obj_tag}_90d"
                s2_dir.mkdir(exist_ok=True)
                input_strategies = current_strategies or list(base_seed_strategies)
                input_tf_map = current_tf_map
                stage2_input_count = len(input_strategies)
                combos = _count_combos(input_strategies, input_tf_map)
                if combos <= 0:
                    combos = len(input_strategies) * len(timeframes)
                print(f"[progress] Starting Stage 2 (90d) for objective={objective} with {len(input_strategies)} strategies × {combos} combos", flush=True)
                try:
                    g = json.loads(root_progress_path.read_text()) if root_progress_path.exists() else {}
                    g.setdefault('stage_time_starts', {})
                    g['stage_time_starts'][f"{objective}_90d"] = datetime.now().isoformat()
                    atomic_write_json(root_progress_path, g)
                except Exception:
                    pass
                stage2_resume_state = resume_plan.get(objective, {}).get(stage2.label) if resume_mode else None
                res2 = stage_run(
                    stage2.label,
                    cache_map if cache_map else stage2.csv_path,
                    input_strategies,
                    timeframes,
                    args.workers,
                    args.calls,
                    s2_dir,
                    objective,
                    utility_kwargs,
                    args.random_state,
                    progress_prefix='',
                    root_progress_path=root_progress_path,
                    acq_funcs=base_opts.get('acq_funcs'),
                    initial_params=base_opts.get('initial_params'),
                    bounds_override=base_opts.get('bounds_override'),
                    resume_mode=resume_mode,
                    resume_state=stage2_resume_state,
                    rerun_errors=rerun_errors,
                    rerun_zero_trades=rerun_zero_trades,
                    strategy_timeframes=input_tf_map if input_tf_map else None,
                )
                top2_entries = select_top(res2, stage2.top_n)
                if top2_entries:
                    current_strategies = [entry['strategy'] for entry in top2_entries]
                    current_tf_map = {entry['strategy']: [entry['timeframe']] for entry in top2_entries if entry.get('timeframe')}
                else:
                    current_strategies = input_strategies[:stage2.top_n]
                    if input_tf_map:
                        current_tf_map = {s: input_tf_map.get(s, []) for s in current_strategies if input_tf_map.get(s)}
                stage2_summary = {
                    'top': top2_entries,
                    'from': len(input_strategies),
                    'top_strategies': current_strategies,
                    'timeframe_map': current_tf_map,
                }
                (s2_dir / 'summary.json').write_text(json.dumps(stage2_summary, indent=2))
                aggregate_stage(s2_dir)
                try:
                    g = json.loads(root_progress_path.read_text()) if root_progress_path.exists() else {}
                    g.setdefault('stage_units', {'completed': 0, 'total': stage_units_total})
                    try:
                        exp_total = max(1, len(g.get('objectives', objectives)) * len(stages_to_run))
                        if int(g['stage_units'].get('total', 0) or 0) != exp_total:
                            g['stage_units']['total'] = exp_total
                    except Exception:
                        pass
                    g.setdefault('stage_time_starts', {})
                    g.setdefault('stage_durations_seconds', [])
                    g.setdefault('started_at', datetime.now().isoformat())
                    st_key = f"{objective}_90d"
                    st_start = pd.to_datetime(g['stage_time_starts'].get(st_key)) if g['stage_time_starts'].get(st_key) else pd.to_datetime(datetime.now())
                    dur = int(max(0, (datetime.now() - st_start.to_pydatetime()).total_seconds()))
                    g['stage_durations_seconds'].append(dur)
                    g['stage_units']['completed'] = int(g['stage_units'].get('completed', 0)) + 1
                    comp_u = int(g['stage_units']['completed'])
                    tot_u = int(g['stage_units']['total'])
                    avg = (sum(g['stage_durations_seconds']) / max(1, len(g['stage_durations_seconds']))) if g['stage_durations_seconds'] else 0
                    rem = max(0, tot_u - comp_u)
                    g['stage_eta_seconds'] = int(rem * avg)
                    g['stage_pct'] = (comp_u / tot_u * 100.0) if tot_u else 100.0
                    atomic_write_json(root_progress_path, g)
                    eta = g['stage_eta_seconds']; eh=int(((pd.to_datetime(datetime.now())-pd.to_datetime(g['started_at'])).total_seconds())//3600); em=int((((pd.to_datetime(datetime.now())-pd.to_datetime(g['started_at'])).total_seconds())%3600)//60); es=int(((pd.to_datetime(datetime.now())-pd.to_datetime(g['started_at'])).total_seconds())%60)
                    th=eta//3600; tm=(eta%3600)//60; ts=eta%60
                    print(f"[overall-stages] {comp_u}/{tot_u} ({g['stage_pct']:.1f}%) elapsed {eh:02d}:{em:02d}:{es:02d} ETA {th:02d}:{tm:02d}:{ts:02d}", flush=True)
                except Exception:
                    pass
                if args.stage == '90d':
                    print(json.dumps({'stage': '90d', 'objective': objective, 'top': top2_entries, 'from': len(input_strategies)}, indent=2))
                    continue
            else:
                if not current_strategies:
                    current_strategies = list(base_seed_strategies)
                if not current_tf_map:
                    current_tf_map = {k: list(v) for k, v in base_seed_map.items()}

            # Stage 3
            s3_dir = out_root / f"{obj_tag}_1y"
            s3_dir.mkdir(exist_ok=True)
            input_strategies = current_strategies or list(base_seed_strategies)
            input_tf_map = current_tf_map
            stage3_input_count = len(input_strategies)
            combos = _count_combos(input_strategies, input_tf_map)
            if combos <= 0:
                combos = len(input_strategies) * len(timeframes)
            print(f"[progress] Starting Stage 3 (1y) for objective={objective} with {len(input_strategies)} strategies × {combos} combos", flush=True)
            try:
                g = json.loads(root_progress_path.read_text()) if root_progress_path.exists() else {}
                g.setdefault('stage_time_starts', {})
                g['stage_time_starts'][f"{objective}_1y"] = datetime.now().isoformat()
                atomic_write_json(root_progress_path, g)
            except Exception:
                pass
            stage3_resume_state = resume_plan.get(objective, {}).get(stage3.label) if resume_mode else None
            res3 = stage_run(
                stage3.label,
                cache_map if cache_map else stage3.csv_path,
                input_strategies,
                timeframes,
                args.workers,
                args.calls,
                s3_dir,
                objective,
                utility_kwargs,
                args.random_state,
                progress_prefix='',
                root_progress_path=root_progress_path,
                acq_funcs=base_opts.get('acq_funcs'),
                initial_params=base_opts.get('initial_params'),
                bounds_override=base_opts.get('bounds_override'),
                resume_mode=resume_mode,
                resume_state=stage3_resume_state,
                rerun_errors=rerun_errors,
                rerun_zero_trades=rerun_zero_trades,
                strategy_timeframes=input_tf_map if input_tf_map else None,
            )
            top3_entries = select_top(res3, stage3.top_n)
            if top3_entries:
                current_strategies = [entry['strategy'] for entry in top3_entries]
                current_tf_map = {entry['strategy']: [entry['timeframe']] for entry in top3_entries if entry.get('timeframe')}
                tf_summary_map = current_tf_map
            else:
                if input_tf_map:
                    current_tf_map = {s: input_tf_map.get(s, []) for s in input_strategies if input_tf_map.get(s)}
                    tf_summary_map = {s: input_tf_map.get(s, []) for s in input_strategies if input_tf_map.get(s)}
                else:
                    tf_summary_map = {}
            stage3_summary = {
                'top': top3_entries,
                'from': len(input_strategies),
                'top_strategies': [entry['strategy'] for entry in top3_entries] if top3_entries else input_strategies,
                'timeframe_map': tf_summary_map,
            }
            (s3_dir / 'summary.json').write_text(json.dumps(stage3_summary, indent=2))
            aggregate_stage(s3_dir)
            try:
                g = json.loads(root_progress_path.read_text()) if root_progress_path.exists() else {}
                g.setdefault('stage_units', {'completed': 0, 'total': stage_units_total})
                # Correct total if parent initialized a different set of objectives
                try:
                    exp_total = max(1, len(g.get('objectives', objectives)) * len(stages_to_run))
                    if int(g['stage_units'].get('total', 0) or 0) != exp_total:
                        g['stage_units']['total'] = exp_total
                except Exception:
                    pass
                g.setdefault('stage_time_starts', {})
                g.setdefault('stage_durations_seconds', [])
                g.setdefault('started_at', datetime.now().isoformat())
                st_key = f"{objective}_1y"
                st_start = pd.to_datetime(g['stage_time_starts'].get(st_key)) if g['stage_time_starts'].get(st_key) else pd.to_datetime(datetime.now())
                dur = int(max(0, (datetime.now() - st_start.to_pydatetime()).total_seconds()))
                g['stage_durations_seconds'].append(dur)
                g['stage_units']['completed'] = int(g['stage_units'].get('completed', 0)) + 1
                comp_u = int(g['stage_units']['completed'])
                tot_u = int(g['stage_units']['total'])
                avg = (sum(g['stage_durations_seconds']) / max(1, len(g['stage_durations_seconds']))) if g['stage_durations_seconds'] else 0
                rem = max(0, tot_u - comp_u)
                g['stage_eta_seconds'] = int(rem * avg)
                g['stage_pct'] = (comp_u / tot_u * 100.0) if tot_u else 100.0
                atomic_write_json(root_progress_path, g)
                eta = g['stage_eta_seconds']; eh=int(((pd.to_datetime(datetime.now())-pd.to_datetime(g['started_at'])).total_seconds())//3600); em=int((((pd.to_datetime(datetime.now())-pd.to_datetime(g['started_at'])).total_seconds())%3600)//60); es=int(((pd.to_datetime(datetime.now())-pd.to_datetime(g['started_at'])).total_seconds())%60)
                th=eta//3600; tm=(eta%3600)//60; ts=eta%60
                print(f"[overall-stages] {comp_u}/{tot_u} ({g['stage_pct']:.1f}%) elapsed {eh:02d}:{em:02d}:{es:02d} ETA {th:02d}:{tm:02d}:{ts:02d}", flush=True)
            except Exception:
                pass

            # Collect final objective summary when running full
            overall_summary[obj] = {
                'stage1_total': stage1_input_count,
                'stage1_top': top1_entries,
                'stage2_from': stage2_input_count,
                'stage2_top': top2_entries,
                'stage3_from': stage3_input_count,
                'stage3_top': top3_entries,
            }
    finally:
        # Cleanup cache directory unless using a shared cache provided by parent
        try:
            if not shared_cache_env:
                shutil.rmtree(cache_root, ignore_errors=True)
        except Exception:
            pass

    # Final print only for full run with multiple or single objectives and stage=all
    if args.stage == 'all' and overall_summary:
        (out_root / 'final_summary.json').write_text(json.dumps(overall_summary if len(objectives) > 1 else list(overall_summary.values())[0], indent=2))
        print(json.dumps(overall_summary if len(objectives) > 1 else list(overall_summary.values())[0], indent=2))

    try:
        get_swap_cost_cache().stop()
    except Exception:
        pass


if __name__ == '__main__':
    main()
