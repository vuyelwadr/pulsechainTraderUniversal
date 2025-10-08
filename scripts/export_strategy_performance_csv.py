#!/usr/bin/env python3
"""Export detailed performance metrics to CSV using real swap costs."""

from __future__ import annotations

import csv
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Type

import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_vost_strategies import (
    load_dataset,
    load_swap_costs,
    per_side_cost_fraction,
)
from strategies.base_strategy import BaseStrategy
from strategies.passive_hold_strategy import PassiveHoldStrategy
from strategies.trailing_hold_strategy import TrailingHoldStrategy
from strategies.multiweek_breakout_strategy import MultiWeekBreakoutStrategy
from strategies.multiweek_breakout_ultra_strategy import MultiWeekBreakoutUltraStrategy
from strategies.c_sma_revert_strategy import CSMARevertStrategy
from strategies.c_sma_revert_pro1_strategy import CSMARevertPro1Strategy
from strategies.donchian_champion_strategy import (
    DonchianChampionStrategy,
    DonchianChampionAggressiveStrategy,
    DonchianChampionDynamicStrategy,
)
from strategies.tight_trend_follow_strategy import TightTrendFollowStrategy
from strategies.hybrid_v2_strategy import HybridV2Strategy


FIVE_MINUTES_PER_YEAR = 365 * 24 * 60 // 5


@dataclass
class StrategyDef:
    name: str
    file_path: str
    description: str
    cls: Type[BaseStrategy]


STRATEGIES: List[StrategyDef] = [
    StrategyDef(
        'PassiveHoldStrategy',
        'strategies/passive_hold_strategy.py',
        'Buy once at dataset start and hold; only one roundtrip so cost is minimal.',
        PassiveHoldStrategy,
    ),
    StrategyDef(
        'TrailingHoldStrategy',
        'strategies/trailing_hold_strategy.py',
        'Single entry with 80% trailing stop to cap catastrophic crashes while keeping turnover tiny.',
        TrailingHoldStrategy,
    ),
    StrategyDef(
        'CSMARevertStrategy',
        'strategies/c_sma_revert_strategy.py',
        'SMA reversion with RSI filter; enters deep dips and exits on mean reversion.',
        CSMARevertStrategy,
    ),
    StrategyDef(
        'CSMARevertPro1Strategy',
        'strategies/c_sma_revert_pro1_strategy.py',
        'Crash-gated SMA reversion with ATR+cooldown for high-cost buckets.',
        CSMARevertPro1Strategy,
    ),
    StrategyDef(
        'DonchianChampionStrategy',
        'strategies/donchian_champion_strategy.py',
        'Donchian 11/2 breakout with EMA-confirmed exits (Champion v1).',
        DonchianChampionStrategy,
    ),
    StrategyDef(
        'DonchianChampionAggressiveStrategy',
        'strategies/donchian_champion_strategy.py',
        'Champion v3 breakout with 20% trailing stop overlay.',
        DonchianChampionAggressiveStrategy,
    ),
    StrategyDef(
        'DonchianChampionDynamicStrategy',
        'strategies/donchian_champion_strategy.py',
        'Champion v4 breakout with ATR-based dynamic drawdown stop.',
        DonchianChampionDynamicStrategy,
    ),
    StrategyDef(
        'TightTrendFollowStrategy',
        'strategies/tight_trend_follow_strategy.py',
        'Tight trend-follow: EMA(1d)>EMA(3d)>EMA(10d) with breakout entry and 25% trail.',
        TightTrendFollowStrategy,
    ),
    StrategyDef(
        'HybridV2Strategy',
        'strategies/hybrid_v2_strategy.py',
        'Hybrid V2: combines deep-dip entries with EMA trend breakouts and adaptive exits.',
        HybridV2Strategy,
    ),
    StrategyDef(
        'MultiWeekBreakoutStrategy',
        'strategies/multiweek_breakout_strategy.py',
        'Multi-week breakout with regime + recovery filters and 26% trail; tuned for high net return and flat recent performance.',
        MultiWeekBreakoutStrategy,
    ),
    StrategyDef(
        'MultiWeekBreakoutUltraStrategy',
        'strategies/multiweek_breakout_ultra_strategy.py',
        'Eight-week breakout variant with stricter filters (~12 trades) to handle larger swap costs.',
        MultiWeekBreakoutUltraStrategy,
    ),
]


def evaluate_strategy(
    strategy: BaseStrategy,
    data: pd.DataFrame,
    *,
    swap_costs: Dict[str, Dict[int, Dict[str, float]]],
    trade_notional: float,
) -> Dict[str, float]:
    from scripts.evaluate_vost_strategies import run_strategy as eval_run_strategy

    stats = eval_run_strategy(
        strategy,
        data,
        swap_costs=swap_costs,
        trade_notional=trade_notional,
    )

    return {
        'total_return_pct': stats['total_return_pct'],
        'buy_hold_return_pct': stats['buy_hold_return_pct'],
        'cagr_pct': stats['cagr_pct'],
        'max_drawdown_pct': stats['max_drawdown_pct'],
        'sharpe_ratio': stats['sharpe_ratio'],
        'sortino_ratio': stats['sortino_ratio'],
        'num_trades': stats['total_trades'],
        'win_rate_pct': stats['win_rate_pct'],
        'total_cost_pct': stats['total_cost_pct'],
        'avg_cost_per_trade_pct': stats['avg_cost_per_trade_pct'],
        'total_cost_dai': stats['total_cost_dai'],
    }


def compute_buy_hold_metrics(close: pd.Series) -> Dict[str, float]:
    total_return = close.iloc[-1] / close.iloc[0] - 1.0
    num_bars = len(close)
    annual_factor = FIVE_MINUTES_PER_YEAR / max(num_bars, 1)
    cagr = (1.0 + total_return) ** annual_factor - 1.0
    drawdowns = close / close.cummax() - 1.0
    max_drawdown = drawdowns.min()
    return {
        'buy_hold_return_pct': total_return * 100,
        'buy_hold_cagr_pct': cagr * 100,
        'buy_hold_max_drawdown_pct': max_drawdown * 100,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='Export cost-aware performance CSV for key strategies.')
    parser.add_argument('--data', type=Path, default=Path('data/pdai_ohlcv_dai_730day_5m.csv'), help='OHLC dataset path.')
    parser.add_argument('--swap-cost-cache', type=Path, default=Path('reports/optimizer_top_top_strats_run/swap_cost_cache.json'), help='Swap-cost cache JSON path.')
    parser.add_argument('--output', type=Path, default=Path('strategy_performance_summary.csv'), help='Destination CSV.')
    args = parser.parse_args()

    output_path = args.output
    data = load_dataset(args.data)
    swap_costs = load_swap_costs(args.swap_cost_cache)
    trade_sizes = [5000, 10000, 25000]

    period_end = data['timestamp'].max()
    timeframes: List[Tuple[str, Optional[pd.Timestamp]]] = [
        ('full', None),
        ('last_3m', period_end - pd.Timedelta(days=90)),
        ('last_1m', period_end - pd.Timedelta(days=30)),
    ]

    rows: List[Dict[str, object]] = []
    for tf_name, start_time in timeframes:
        if start_time is None:
            subset = data.copy()
        else:
            subset = data[data['timestamp'] >= start_time].reset_index(drop=True)
            if subset.empty:
                continue
        period_start = subset['timestamp'].min()
        buy_hold_metrics = compute_buy_hold_metrics(subset['close'])

        for strat_def in STRATEGIES:
            for size in trade_sizes:
                strategy = strat_def.cls()
                metrics = evaluate_strategy(strategy, subset, swap_costs=swap_costs, trade_notional=size)
                row = {
                    'timeframe': tf_name,
                    'strategy': strat_def.name,
                    'file': strat_def.file_path,
                    'description': strat_def.description,
                    'trade_size_dai': size,
                    'period_start': period_start,
                    'period_end': subset['timestamp'].max(),
                    **buy_hold_metrics,
                    **metrics,
                }
                rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f'Wrote {output_path} with {len(df)} rows')


if __name__ == '__main__':
    main()
