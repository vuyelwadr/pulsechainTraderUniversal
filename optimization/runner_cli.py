"""Command-line interface for the fast optimizer runner."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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


def _select_top_strategies(summary_df: Optional[pd.DataFrame], top_n: int) -> List[str]:
    if summary_df is None or summary_df.empty or top_n <= 0:
        return []
    working = summary_df.copy()
    if 'stage' in working.columns:
        working['__stage_rank'] = working['stage'].apply(lambda s: 0 if str(s).lower() == 'all' else 1)
        working = working.sort_values(['__stage_rank', 'total_return_pct'], ascending=[True, False])
    else:
        working = working.sort_values('total_return_pct', ascending=False)
    selected: List[str] = []
    for _, row in working.iterrows():
        strategy = str(row.get('strategy'))
        total_return = float(row.get('total_return_pct', 0.0))
        if total_return <= 0:
            continue
        if strategy in selected:
            continue
        selected.append(strategy)
        if len(selected) >= top_n:
            break
    return selected


def _instantiate_strategy(strategy_cls, parameters: Dict[str, Any]):
    # Mirror evaluator.instantiate logic without importing private helper
    if hasattr(strategy_cls, '__call__'):
        return strategy_cls(parameters=parameters or None)
    return strategy_cls(parameters=parameters or None)


def _run_walk_forward(
    *,
    run_dir: Path,
    strategy_specs: Sequence[StrategySpec],
    selected_strategies: Sequence[str],
    data_path: Path,
    swap_cost_path: Path,
    trade_size: float,
    window_days: int,
    step_days: int,
    objectives: Sequence[str],
    cpu_fraction: float,
    total_calls: int,
    seed: Optional[int],
    save_trades: bool,
) -> Optional[pd.DataFrame]:
    if window_days <= 0 or step_days <= 0:
        return None
    selected = [name for name in selected_strategies if name]
    if not selected:
        return None

    spec_map: Dict[str, StrategySpec] = {spec.name: spec for spec in strategy_specs}
    subset_specs = [spec_map[name] for name in selected if name in spec_map]
    if not subset_specs:
        return None

    from scripts.evaluate_vost_strategies import load_dataset, load_swap_costs, run_strategy

    data_df = load_dataset(data_path)
    if 'timestamp' not in data_df.columns:
        raise SystemExit("Dataset must include a 'timestamp' column for walk-forward runs.")
    data_df = data_df.sort_values('timestamp').reset_index(drop=True)
    timestamps = pd.to_datetime(data_df['timestamp'])
    start_ts = timestamps.min()
    end_ts = timestamps.max()
    window_delta = pd.Timedelta(days=window_days)
    step_delta = pd.Timedelta(days=step_days)

    if start_ts >= end_ts:
        return None

    swap_costs = load_swap_costs(swap_cost_path)

    walk_dir = run_dir / 'walkforward'
    datasets_dir = walk_dir / 'datasets'
    datasets_dir.mkdir(parents=True, exist_ok=True)

    summary_records: List[Dict[str, Any]] = []

    train_start = start_ts
    step_index = 0

    while True:
        train_end = train_start + window_delta
        holdout_start = train_end
        holdout_end = holdout_start + step_delta

        if holdout_end > end_ts:
            break

        train_mask = (timestamps >= train_start) & (timestamps < train_end)
        holdout_mask = (timestamps >= holdout_start) & (timestamps < holdout_end)

        train_df = data_df.loc[train_mask].reset_index(drop=True)
        holdout_df = data_df.loc[holdout_mask].reset_index(drop=True)

        if len(train_df) < 50 or len(holdout_df) < 10:
            break

        train_csv_path = datasets_dir / f'train_step_{step_index:03d}.csv'
        train_df.to_csv(train_csv_path, index=False)

        step_dir = walk_dir / f'step_{step_index:03d}'
        train_stage_dir = step_dir / 'train'

        store = TrialStore(train_stage_dir)
        train_trials, _ = run_optimization(
            run_dir=train_stage_dir,
            store=store,
            strategy_specs=subset_specs,
            objectives=objectives,
            total_calls=total_calls,
            trade_size=trade_size,
            data_path=train_csv_path,
            swap_cost_path=swap_cost_path,
            stage_label='train',
            stage_days=None,
            extra_windows=[],
            cpu_fraction=cpu_fraction,
            seed=seed,
            save_trades=save_trades,
        )

        generate_reports(train_stage_dir, train_trials, objectives, [('train', f'WF Train Step {step_index:03d}')])

        summary_path = train_stage_dir / 'summary_total_return.csv'
        if not summary_path.exists():
            break

        train_summary = pd.read_csv(summary_path)
        trials_by_id = {trial['trial_id']: trial for trial in train_trials}
        step_records: List[Dict[str, Any]] = []

        for strategy_name in selected:
            strat_rows = train_summary.loc[train_summary['strategy'] == strategy_name]
            if strat_rows.empty:
                continue
            strat_rows = strat_rows.sort_values('total_return_pct', ascending=False)
            best_row = None
            for _, candidate in strat_rows.iterrows():
                if float(candidate.get('total_return_pct', 0.0)) > 0:
                    best_row = candidate
                    break
            if best_row is None:
                continue

            trial_id = int(best_row['trial_id'])
            trial_payload = trials_by_id.get(trial_id)
            if not trial_payload:
                continue

            parameters = dict(trial_payload.get('parameters') or {})
            strategy_spec = spec_map.get(strategy_name)
            if strategy_spec is None:
                continue
            strategy_instance = _instantiate_strategy(strategy_spec.cls, parameters)

            holdout_stage_dir = step_dir / 'holdout'
            holdout_stage_dir.mkdir(parents=True, exist_ok=True)

            stats = run_strategy(
                strategy_instance,
                holdout_df.copy(),
                swap_costs=swap_costs,
                trade_notional=trade_size,
                capture_trades=save_trades,
            )

            trade_path = None
            trades = stats.get('trades') or []
            if save_trades and trades:
                trade_filename = f"{strategy_name}_step_{step_index:03d}.csv.gz"
                trade_path_full = holdout_stage_dir / trade_filename
                pd.DataFrame(trades).to_csv(trade_path_full, index=False, compression='gzip')
                trade_path = str(Path('walkforward') / f'step_{step_index:03d}' / 'holdout' / trade_filename)

            record = {
                'step_index': step_index,
                'strategy': strategy_name,
                'train_start': train_start.isoformat(),
                'train_end': train_end.isoformat(),
                'holdout_start': holdout_start.isoformat(),
                'holdout_end': holdout_end.isoformat(),
                'trial_id': trial_id,
                'train_total_return_pct': float(best_row.get('total_return_pct', 0.0)),
                'train_buy_hold_return_pct': float(best_row.get('buy_hold_return_pct', 0.0)),
                'train_buy_hold_max_drawdown_pct': float(best_row.get('buy_hold_max_drawdown_pct', 0.0)),
                'train_max_drawdown_pct': float(best_row.get('max_drawdown_pct', 0.0)),
                'train_sharpe_ratio': float(best_row.get('sharpe_ratio', 0.0)),
                'train_total_trades': int(best_row.get('total_trades', 0)),
                'parameters': json.dumps(parameters, sort_keys=True),
                'holdout_total_return_pct': float(stats.get('total_return_pct', 0.0)),
                'holdout_buy_hold_return_pct': float(stats.get('buy_hold_return_pct', 0.0)),
                'holdout_buy_hold_max_drawdown_pct': float(stats.get('buy_hold_max_drawdown_pct', 0.0)),
                'holdout_max_drawdown_pct': float(stats.get('max_drawdown_pct', 0.0)),
                'holdout_cagr_pct': float(stats.get('cagr_pct', 0.0)),
                'holdout_num_trades': int(stats.get('num_trades', 0)),
                'holdout_win_rate_pct': float(stats.get('win_rate_pct', 0.0)),
                'holdout_profit_factor': float(stats.get('profit_factor', 0.0)),
                'holdout_total_cost_pct': float(stats.get('total_cost_pct', 0.0)),
                'holdout_avg_cost_per_trade_pct': float(stats.get('avg_cost_per_trade_pct', 0.0)),
                'trade_log_path': trade_path,
            }
            step_records.append(record)
            summary_records.append(record)

        if step_records:
            step_summary_df = pd.DataFrame(step_records)
            step_summary_df.to_csv(step_dir / 'holdout_summary.csv', index=False)

        train_start += step_delta
        step_index += 1

    if summary_records:
        overall_df = pd.DataFrame(summary_records)
        overall_df.to_csv(walk_dir / 'walkforward_summary.csv', index=False)
        return overall_df
    return None


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
    parser.add_argument('--save-trades', action='store_true', help='Persist per-trade logs for each trial (compressed CSV).')
    parser.add_argument('--walk-forward-window', type=int, default=90, help='Training window size in days for walk-forward optimisation (default 90).')
    parser.add_argument('--walk-forward-step', type=int, default=30, help='Forward step size in days for walk-forward validation (default 30).')
    parser.add_argument('--walk-forward-top-n', type=int, default=5, help='Number of top strategies (by profit) to include in walk-forward (default 5).')
    parser.add_argument('--disable-walk-forward', action='store_true', help='Skip walk-forward evaluation.')
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.objectives:
        objectives = [obj.strip() for obj in args.objectives.split(',') if obj.strip()]
    else:
        objectives = ['return_dd_ratio']

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
        'save_trades': bool(args.save_trades),
        'walk_forward': {
            'enabled': not args.disable_walk_forward,
            'window_days': args.walk_forward_window,
            'step_days': args.walk_forward_step,
            'top_n': args.walk_forward_top_n,
        },
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
            save_trades=bool(args.save_trades),
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
    combined_summary: Optional[pd.DataFrame] = None
    if aggregated_summary_frames:
        combined_summary = pd.concat(aggregated_summary_frames, ignore_index=True)
        combined_summary = combined_summary.sort_values('total_return_pct', ascending=False)
        combined_summary.to_csv(run_dir / 'aggregate_summary_total_return.csv', index=False)

    print(f"Run complete. Results saved to {run_dir}")
    if aggregated_best:
        print("Top scores by objective:")
        for objective, info in aggregated_best.items():
            print(f"  {objective:<25} score={info['score']:.4f} trial={info['trial_id']} stage={info['stage']} strategy={info['strategy']}")

    walkforward_df: Optional[pd.DataFrame] = None
    if not args.disable_walk_forward:
        top_strategy_names = _select_top_strategies(combined_summary, args.walk_forward_top_n)
        if top_strategy_names:
            walkforward_df = _run_walk_forward(
                run_dir=run_dir,
                strategy_specs=strategy_specs,
                selected_strategies=top_strategy_names,
                data_path=data_path,
                swap_cost_path=swap_cost_path,
                trade_size=args.trade_size,
                window_days=args.walk_forward_window,
                step_days=args.walk_forward_step,
                objectives=objectives,
                cpu_fraction=args.cpu_fraction,
                total_calls=args.calls,
                seed=args.seed,
                save_trades=bool(args.save_trades),
            )
            if walkforward_df is not None and not walkforward_df.empty:
                best_wf = walkforward_df.sort_values('holdout_total_return_pct', ascending=False).head(5)
                print("Top walk-forward holdout returns:")
                for _, row in best_wf.iterrows():
                    print(
                        f"  step={int(row['step_index'])} strategy={row['strategy']} "
                        f"holdout_return={row['holdout_total_return_pct']:.2f}% "
                        f"trades={int(row['holdout_num_trades'])}"
                    )
        else:
            print("Walk-forward skipped: no strategies with positive returns found in stage summaries.")


if __name__ == '__main__':
    main()
