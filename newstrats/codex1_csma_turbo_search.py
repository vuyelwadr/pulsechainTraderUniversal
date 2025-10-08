"""Grid-search helper for Codex1CSMATurboStrategy.

Usage:
    python newstrats/codex1_csma_turbo_search.py --trade-size 5000 \
        --grid-file newstrats/codex1_turbo_grid.csv
"""

from __future__ import annotations

import argparse
import csv
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from scripts.evaluate_vost_strategies import (
    load_dataset,
    load_swap_costs,
    run_strategy,
)
from strategies.codex1_csma_turbo_strategy import Codex1CSMATurboStrategy


def param_grid(options: Dict[str, Iterable]) -> Iterable[Dict[str, float]]:
    keys = list(options.keys())
    for values in product(*options.values()):
        yield dict(zip(keys, values))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/pdai_ohlcv_dai_730day_5m.csv'))
    parser.add_argument('--swap-cost-cache', type=Path, default=Path('swap_cost_cache.json'))
    parser.add_argument('--trade-size', type=float, default=5000.0)
    parser.add_argument('--grid-file', type=Path, help='Optional CSV to write parameter results to')
    args = parser.parse_args()

    data = load_dataset(args.data)
    swap_costs = load_swap_costs(args.swap_cost_cache)

    grid = {
        'entry_drop': [0.22, 0.26, 0.30],
        'entry_dd_threshold': [-0.75, -0.60, -0.45],
        'rsi_max': [28, 31, 34],
        'profit_target': [0.60, 1.00, 1.60],
        'trailing_pct': [0.18, 0.25, 0.32],
        'hard_stop': [0.25, 0.35, 0.45],
    }

    results: List[Tuple[Dict[str, float], float, float, float]] = []

    for params in param_grid(grid):
        params_payload = {
            'entry_drop': float(params['entry_drop']),
            'entry_dd_threshold': float(params['entry_dd_threshold']),
            'rsi_max': float(params['rsi_max']),
            'profit_target': float(params['profit_target']),
            'trailing_pct': float(params['trailing_pct']),
            'hard_stop': float(params['hard_stop']),
        }
        strategy = Codex1CSMATurboStrategy(parameters=params_payload)
        stats = run_strategy(strategy, data, swap_costs=swap_costs, trade_notional=args.trade_size)
        results.append((params_payload, stats['total_return_pct'], stats['max_drawdown_pct'], stats['num_trades']))
        print(
            f"params={params_payload} -> total_return={stats['total_return_pct']:.2f}%"
            f", maxDD={stats['max_drawdown_pct']:.2f}%, trades={stats['num_trades']}"
        )

    if args.grid_file:
        args.grid_file.parent.mkdir(parents=True, exist_ok=True)
        with args.grid_file.open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['entry_drop', 'entry_dd_threshold', 'rsi_max', 'profit_target', 'trailing_pct', 'hard_stop', 'total_return_pct', 'max_drawdown_pct', 'num_trades'])
            for params, total_ret, max_dd, num_trades in results:
                writer.writerow([
                    params['entry_drop'],
                    params['entry_dd_threshold'],
                    params['rsi_max'],
                    params['profit_target'],
                    params['trailing_pct'],
                    params['hard_stop'],
                    total_ret,
                    max_dd,
                    num_trades,
                ])

    best = max(results, key=lambda item: item[1])
    print('\nBest by total return:')
    print(best)


if __name__ == '__main__':
    main()
