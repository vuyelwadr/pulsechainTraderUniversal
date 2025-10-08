"""Command-line interface for the fast optimizer runner."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from optimization.orchestrator import (
    StrategySpec,
    load_strategy_specs,
    run_optimization,
)
from optimization.persistence import TrialStore
from optimization.reporting import generate_reports


STAGE_MAP = {
    '30d': 30,
    '90d': 90,
    '1y': 365,
    'all': None,
}


def _load_strategy_names(path: Path) -> List[str]:
    payload = json.loads(path.read_text())
    if isinstance(payload, list):
        return [str(item) for item in payload]
    raise ValueError(f"Strategies file {path} must contain a JSON array of strategy names.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--strategies-file', type=Path, required=True, help='JSON file listing strategy names to evaluate.')
    parser.add_argument('--objectives', type=str, default='', help='Comma-separated list of objectives (default: all supported).')
    parser.add_argument('--calls', type=int, default=50, help='Total number of evaluations to perform.')
    parser.add_argument('--trade-size', type=float, default=1000.0, help='Trade notional in DAI used for cost calculation (default 1000).')
    parser.add_argument('--data', type=Path, default=Path('data/pdai_ohlcv_dai_730day_5m.csv'), help='Path to HEX/PDAI OHLCV dataset.')
    parser.add_argument('--swap-cost-cache', type=Path, default=Path('data/swap_cost_cache.json'), help='Path to swap-cost cache JSON.')
    parser.add_argument('--stage', type=str, default=None, choices=list(STAGE_MAP.keys()), help='Single optimisation window (30d, 90d, 1y, all). Mutually exclusive with --stages.')
    parser.add_argument('--stages', type=str, default='', help='Comma-separated list of stages to run sequentially (e.g. "30d,90d,1y"). Overrides --stage when provided.')
    parser.add_argument('--out-dir', type=Path, default=None, help='Base output directory (default: reports/optimizer_run_<timestamp>/).')
    parser.add_argument('--cpu-fraction', type=float, default=0.9, help='Fraction of CPU cores to use (default 0.9).')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for parameter sampling (optional).')
    parser.add_argument('--resume-from', type=Path, default=None, help='Existing run directory to resume.')
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.objectives:
        objectives = [obj.strip() for obj in args.objectives.split(',') if obj.strip()]
    else:
        objectives = ['final_balance','profit_biased','cps_v2_profit_biased','mar','utility','cps','cps_v2']

    strategies = _load_strategy_names(args.strategies_file)
    strategy_specs = load_strategy_specs(strategies)

    if args.stage and args.stages:
        raise SystemExit('Use either --stage or --stages, not both.')

    stages_arg = [s.strip() for s in args.stages.split(',') if s.strip()]
    if stages_arg:
        stage_labels = stages_arg
    elif args.stage:
        stage_labels = [args.stage]
    else:
        stage_labels = ['30d', '90d', '1y', 'all']

    stages_to_run: List[Tuple[str, Optional[int]]] = []
    for label in stage_labels:
        if label not in STAGE_MAP:
            raise SystemExit(f"Unknown stage '{label}' (choose from {', '.join(STAGE_MAP.keys())})")
        stages_to_run.append((label, STAGE_MAP[label]))

    if args.resume_from:
        run_dir = args.resume_from.resolve()
    else:
        timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
        base_out = args.out_dir.resolve() if args.out_dir else Path('reports') / f"optimizer_run_{timestamp}"
        run_dir = base_out

    data_path = args.data.resolve()
    swap_cost_path = args.swap_cost_cache.resolve()
    if not swap_cost_path.exists():
        raise SystemExit(f"Swap-cost cache not found at {swap_cost_path}. Run scripts/fetch_swap_cost_cache.py first.")

    store = TrialStore(run_dir)
    config_payload = {
        'strategies_file': str(args.strategies_file.resolve()),
        'objectives': objectives,
        'calls': args.calls,
        'trade_size': args.trade_size,
        'data_path': str(data_path),
        'swap_cost_cache': str(swap_cost_path),
        'stages': stage_labels,
        'cpu_fraction': args.cpu_fraction,
        'seed': args.seed,
        'created_at': datetime.now(timezone.utc).isoformat(),
    }
    store.write_config(config_payload)

    aggregated_trials: List[Dict[str, Any]] = []
    aggregated_best: Dict[str, Dict[str, Any]] = {}
    aggregated_summary_frames: List[pd.DataFrame] = []

    for label, days in stages_to_run:
        stage_dir = run_dir / f"stage_{label}"
        stage_store = TrialStore(stage_dir)

        extra_windows: List[Tuple[str, Optional[int]]] = []
        timeframe_titles: List[Tuple[str, str]] = [(label, f"Stage ({label})")]

        all_trials, best_scores = run_optimization(
            run_dir=stage_dir,
            store=stage_store,
            strategy_specs=strategy_specs,
            objectives=objectives,
            total_calls=args.calls,
            trade_size=args.trade_size,
            data_path=data_path,
            swap_cost_path=swap_cost_path,
            stage_label=label,
            stage_days=days,
            extra_windows=extra_windows,
            cpu_fraction=args.cpu_fraction,
            seed=args.seed,
        )

        generate_reports(stage_dir, all_trials, objectives, timeframe_titles)

        # aggregate
        for trial in all_trials:
            record = dict(trial)
            record['stage'] = label
            aggregated_trials.append(record)

        summary_path = stage_dir / 'summary_total_return.csv'
        if summary_path.exists():
            df = pd.read_csv(summary_path)
            df.insert(0, 'stage', label)
            aggregated_summary_frames.append(df)

        for obj, info in best_scores.items():
            existing = aggregated_best.get(obj)
            if existing is None or info['score'] > existing['score']:
                aggregated_best[obj] = dict(info, stage=label)

    # Combined summaries
    if aggregated_summary_frames:
        combined = pd.concat(aggregated_summary_frames, ignore_index=True)
        combined = combined.sort_values('total_return_pct', ascending=False)
        combined.to_csv(run_dir / 'aggregate_summary_total_return.csv', index=False)

    print(f"Run complete. Results saved to {run_dir}")
    if aggregated_best:
        print("Top scores by objective:")
        for objective, info in aggregated_best.items():
            print(f"  {objective:<25} score={info['score']:.4f} trial={info['trial_id']} stage={info['stage']} strategy={info['strategy']}")


if __name__ == '__main__':
    main()
