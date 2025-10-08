#!/usr/bin/env python3
"""Export detailed performance metrics to CSV using real swap costs."""

from __future__ import annotations

import csv
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
from strategies.codex1_csma_enhanced_strategy import Codex1CSMAEnhancedStrategy
from strategies.codex1_csma_ultra_strategy import Codex1CSMAUltraStrategy
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
        'Codex1CSMAEnhancedStrategy',
        'strategies/codex1_csma_enhanced_strategy.py',
        'Enhanced CSMA variant with RSI<=32 threshold to capture more crash rebounds.',
        Codex1CSMAEnhancedStrategy,
    ),
    StrategyDef(
        'Codex1CSMAUltraStrategy',
        'strategies/codex1_csma_ultra_strategy.py',
        'Ultra-tuned CSMA variant (n_sma=432, entry_drop=24%, exit_up=5.4%) achieving the new 6.8k% net return record.',
        Codex1CSMAUltraStrategy,
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
    """Run strategy and compute detailed metrics, including explicit cost accounting."""

    enriched = strategy.calculate_indicators(data.copy())
    signals = strategy.generate_signals(enriched)

    buy_signal = signals.get('buy_signal', pd.Series(False, index=signals.index)).astype(bool)
    sell_signal = signals.get('sell_signal', pd.Series(False, index=signals.index)).astype(bool)

    # Align execution (execute on bar after signal)
    buy_exec = buy_signal.shift(1, fill_value=False)
    sell_exec = sell_signal.shift(1, fill_value=False)

    position = np.zeros(len(signals), dtype=int)
    pos = 0
    for i in range(len(signals)):
        if pos == 0 and buy_exec.iat[i]:
            pos = 1
        elif pos == 1 and sell_exec.iat[i]:
            pos = 0
        position[i] = pos

    close = data['price'] if 'price' in data.columns else data['close']
    returns = close.pct_change().fillna(0.0).to_numpy()
    strat_returns = position * returns

    position_shift = np.roll(position, 1)
    position_shift[0] = 0
    entries = (position > position_shift)
    exits = (position < position_shift)

    buy_cost_frac = per_side_cost_fraction(swap_costs, trade_notional, 'buy')
    sell_cost_frac = per_side_cost_fraction(swap_costs, trade_notional, 'sell')

    # Apply fees & slippage costs
    strat_returns = strat_returns - buy_cost_frac * entries.astype(float) - sell_cost_frac * exits.astype(float)

    equity_curve = (1.0 + strat_returns).cumprod()

    total_return = float(equity_curve[-1] - 1.0)
    num_bars = len(data)
    annual_factor = FIVE_MINUTES_PER_YEAR / max(num_bars, 1)
    cagr = float(equity_curve[-1] ** annual_factor - 1.0)

    rolling_peak = np.maximum.accumulate(equity_curve)
    drawdowns = equity_curve / rolling_peak - 1.0
    max_drawdown = float(drawdowns.min())

    mean_per_bar = strat_returns.mean()
    std_per_bar = strat_returns.std(ddof=0)
    sharpe = float(mean_per_bar / std_per_bar * np.sqrt(FIVE_MINUTES_PER_YEAR)) if std_per_bar > 0 else 0.0
    downside = strat_returns[strat_returns < 0]
    if len(downside) > 0:
        downside_std = downside.std(ddof=0)
        sortino = float(mean_per_bar / downside_std * np.sqrt(FIVE_MINUTES_PER_YEAR)) if downside_std > 0 else 0.0
    else:
        sortino = 0.0

    total_cost_frac = buy_cost_frac * entries.sum() + sell_cost_frac * exits.sum()
    total_cost_pct = total_cost_frac * 100
    total_cost_dai = total_cost_frac * trade_notional
    avg_cost_per_trade_pct = (total_cost_pct / entries.sum()) if entries.sum() else 0.0

    # Collect trade outcomes for win rate if desired
    trade_returns: List[float] = []
    in_trade = False
    entry_equity = 1.0
    for i in range(len(equity_curve)):
        if entries[i] and not in_trade:
            in_trade = True
            entry_equity = equity_curve[i - 1] if i > 0 else 1.0
        elif exits[i] and in_trade:
            trade_returns.append(equity_curve[i] / entry_equity - 1.0)
            in_trade = False

    if in_trade:
        trade_returns.append(equity_curve[-1] / entry_equity - 1.0)

    win_rate = float(np.mean([r > 0 for r in trade_returns])) if trade_returns else 0.0

    return {
        'total_return_pct': total_return * 100,
        'cagr_pct': cagr * 100,
        'max_drawdown_pct': max_drawdown * 100,
        'sharpe': sharpe,
        'sortino': sortino,
        'num_trades': int(entries.sum()),
        'win_rate_pct': win_rate * 100,
        'total_cost_pct': total_cost_pct,
        'avg_cost_per_trade_pct': avg_cost_per_trade_pct,
        'total_cost_dai': total_cost_dai,
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
    output_path = Path('strategy_performance_summary.csv')
    data = load_dataset(Path('data/pdai_ohlcv_dai_730day_5m.csv'))
    swap_costs = load_swap_costs(Path('reports/optimizer_top_top_strats_run/swap_cost_cache.json'))
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
