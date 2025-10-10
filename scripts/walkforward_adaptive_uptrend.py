#!/usr/bin/env python3

import argparse
import json
import logging
import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import List

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from bot.walkforward_uptrend_engine import (
    AnalysisType,
    SimResult,
    SimulationConfig,
    run_simulation,
)
from strategies.adaptive_trend_hold_strategy import AdaptiveTrendHoldStrategy

# Configuration
DEFAULT_ANALYSIS_DIR = Path("analysis_v2")
DEFAULT_OUTPUT_CSV = Path("reports/wf_adaptive_uptrend_summary.csv")
DEFAULT_COST_RATE = 0.015  # 1.5% round-trip
TIMEFRAMES = ("5min", "15min", "30min", "1h", "2h", "4h", "8h", "16h", "1d", "2d")

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def _load_adaptive_config(file_path: str) -> List[dict]:
    """Load adaptive strategy configurations from JSON file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Adaptive config file not found: {path}")
    
    with open(path) as fp:
        data = json.load(fp)
    
    if isinstance(data, dict):
        return [data]
    elif isinstance(data, list):
        return data
    else:
        raise ValueError("Adaptive config must be a dict or list of dicts")


def _adaptive_config_to_sim_config(adaptive_config: dict, tag: str) -> SimulationConfig:
    """Convert an adaptive configuration dict to a SimulationConfig."""
    params = {
        "adaptive_config_file": Path("tmp/adaptive_configs_v1.json").absolute(),
    }
    # Add any adaptive-specific parameters
    for key, value in adaptive_config.items():
        if key not in ["name", "adaptive_mode", "base_config", "adaptive_parameters", 
                     "volatility_thresholds", "trend_strength_thresholds",
                     "volatility_adaptation", "trend_strength_adaptation"]:
            params[key] = value
    
    # Use some aggressive baseline parameters to test against the static maximum
    base_config = {
        "threshold_mode": "quantile_grid",
        "threshold_quantile": 0.1,
        "trailing_mode": "atr",
        "atr_mult": 20.0,
        "atr_floor": 0.000000001,
        "cooldown_bars": 0,
        "require_strength_positive": False,
    }
    
    return SimulationConfig(
        name=f"{tag}_{adaptive_config['name']}",
        **base_config,
        cost_rate=DEFAULT_COST_RATE,
        analysis_dir=DEFAULT_ANALYSIS_DIR,
        output_csv=Path(f"reports/wf_adaptive-{adaptive_config['name']}.csv"),
        timeframes=TIMEFRAMES,
        strategy_params=params
    )


def build_configs(args) -> List[SimulationConfig]:
    """Build adaptive configurations from command line args."""
    configs = []
    
    # Load adaptive configurations
    adaptive_configs = _load_adaptive_config(args.adaptive_config)
    
    for adaptive_config in adaptive_configs:
        sim_config = _adaptive_config_to_sim_config(adaptive_config, args.tag)
        configs.append(sim_config)
    
    # Add baseline static strategy for comparison
    baseline_config = SimulationConfig(
        name=f"{args.tag}_baseline_static",
        threshold_mode="quantile_grid",
        threshold_grid=[0.0, 0.1, 0.3, 0.92, 0.99],
        trailing_mode="atr",
        atr_mult=20.0,
        atr_floor=0.000000001,
        cooldown_bars=0,
        require_strength_positive=False,
        cost_rate=DEFAULT_COST_RATE,
        analysis_dir=DEFAULT_ANALYSIS_DIR,
        output_csv=Path(f"reports/wf_adaptive_baseline_static.csv"),
        timeframes=TIMEFRAMES,
    )
    configs.append(baseline_config)
    
    return configs


def run_adaptive_walkforward(config: SimulationConfig) -> SimResult:
    """Run walk-forward simulation with adaptive strategy."""
    strategy = AdaptiveTrendHoldStrategy(config.strategy_params)
    
    results = []
    for tf in config.timeframes:
        result = run_simulation(strategy, tf, AnalysisType.WALKFORWARD, config)
        if result:
            results.append(result)
    
    if not results:
        return None
    
    # Aggregate results across timeframes
    total_return_pct = sum(r.total_return_pct for r in results) / len(results)
    buy_hold_total_return_pct = sum(r.buy_hold_total_return_pct for r in results) / len(results)
    trades = sum(r.trades for r in results)
    
    return SimResult(
        timeframe="all",
        train_start=None,
        train_end=None,
        test_start=None,
        test_end=None,
        final_balance=1000.0 * (1 + total_return_pct),
        total_return_pct=total_return_pct,
        buy_hold_final_balance=1000.0 * (1 + buy_hold_total_return_pct),
        buy_hold_total_return_pct=buy_hold_total_return_pct,
        trades=trades,
        config_name=config.name
    )


def serialize_sim_results(results: List[SimResult]) -> pd.DataFrame:
    """Convert simulation results to a DataFrame."""
    rows = []
    for r in results:
        rows.append(asdict(r))
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parallel walk‑forward testing for adaptive uptrend strategies"
    )
    parser.add_argument(
        "--adaptive-config",
        type=str,
        default="tmp/adaptive_configs_v1.json",
        help="Path to adaptive strategy configuration JSON",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="adaptive_v1",
        help="Tag to append to result filenames",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT_CSV),
        help="Path to the summary CSV file",
    )
    parser.add_argument(
        "--parallelism",
        type=int,
        default=max(1, min(6, int(multiprocessing.cpu_count()))),
        help="Number of processes to run in parallel",
    )
    
    args = parser.parse_args()
    
    # Build adaptive configurations
    configs = build_configs(args)
    
    log.info(f"Running {len(configs)} adaptive configurations in parallel...")
    
    # Run simulations in parallel
    results = []
    with ProcessPoolExecutor(max_workers=args.parallelism) as executor:
        for result in executor.map(run_adaptive_walkforward, configs):
            if result is not None:
                results.append(result)
    
    # Save results
    df = serialize_sim_results(results)
    df.sort_values("total_return_pct", ascending=False, inplace=True)
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    log.info(f"Adaptive results saved to {output_path}")
    log.info("\nAdaptive strategy performance summary:")
    for _, row in df.iterrows():
        excess = row["total_return_pct"] - row["buy_hold_total_return_pct"]
        log.info(f"{row['config_name']}: {row['total_return_pct']:.2f}% (excess: {excess:+.2f}%, trades: {row['trades']})")


if __name__ == "__main__":
    main()
