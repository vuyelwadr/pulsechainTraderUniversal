#!/usr/bin/env python3
"""Evaluate VOST-based strategies on the PDAI-DAI 5m dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from strategies.vost_trend_rider_strategy import VOSTTrendRiderStrategy
from strategies.vost_breakout_squeeze_strategy import VOSTBreakoutSqueezeStrategy
from strategies.vost_pullback_accumulator_strategy import VOSTPullbackAccumulatorStrategy
from strategies.multiweek_breakout_strategy import MultiWeekBreakoutStrategy
from strategies.long_term_regime_strategy import LongTermRegimeStrategy
from strategies.passive_hold_strategy import PassiveHoldStrategy
from strategies.macro_trend_channel_strategy import MacroTrendChannelStrategy
from strategies.trailing_hold_strategy import TrailingHoldStrategy
from strategies.c_sma_revert_strategy import CSMARevertStrategy
from strategies.codex1_csma_turbo_strategy import Codex1CSMATurboStrategy
from strategies.codex1_csma_enhanced_strategy import Codex1CSMAEnhancedStrategy
from strategies.codex1_csma_ultra_strategy import Codex1CSMAUltraStrategy
from strategies.codex1_phoenix_strategy import Codex1PhoenixStrategy
from strategies.codex1_recovery_trend_strategy import Codex1RecoveryTrendStrategy
from strategies.donchian_champion_strategy import (
    DonchianChampionStrategy,
    DonchianChampionAggressiveStrategy,
    DonchianChampionDynamicStrategy,
)
from strategies.tight_trend_follow_strategy import TightTrendFollowStrategy
from strategies.hybrid_v2_strategy import HybridV2Strategy


FIVE_MINUTES_PER_YEAR = 365 * 24 * 60 // 5


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df


def load_swap_costs(path: Path) -> Dict[str, Dict[int, Dict[str, float]]]:
    with path.open() as f:
        payload = json.load(f)
    step = int(payload['metadata']['step_notional'])
    entries_raw = payload.get('entries', {})
    entries: Dict[int, Dict[str, float]] = {}
    for notional, info in entries_raw.items():
        try:
            key = int(float(notional))
        except ValueError:
            continue
        entries[key] = {
            'loss_rate': float(info['derived']['loss_rate']),
            'buy_gas': float(info['buy']['gas_use_estimate_usd']),
            'sell_gas': float(info['sell']['gas_use_estimate_usd']),
        }
    if not entries:
        raise ValueError('No swap cost entries found in cache.')
    entries = dict(sorted(entries.items()))
    return {'step': step, 'entries': entries}


def _bucket_for_notional(costs: Dict[str, Dict[int, Dict[str, float]]], notional: float) -> int:
    keys = list(costs['entries'].keys())
    for key in keys:
        if notional <= key:
            return key
    return keys[-1]


def per_side_cost(costs: Dict[str, Dict[int, Dict[str, float]]], notional: float, side: str) -> float:
    bucket = _bucket_for_notional(costs, notional)
    info = costs['entries'][bucket]
    loss_rate = info['loss_rate']
    gas = info['buy_gas'] if side.lower().startswith('b') else info['sell_gas']
    return notional * (loss_rate / 2.0) + gas


def per_side_cost_fraction(costs: Dict[str, Dict[int, Dict[str, float]]], notional: float, side: str) -> float:
    if notional <= 0:
        return 0.0
    return per_side_cost(costs, notional, side) / notional


def lookup_roundtrip_cost(costs: Dict[str, Dict[int, Dict[str, float]]], notional: float) -> float:
    return per_side_cost(costs, notional, 'buy') + per_side_cost(costs, notional, 'sell')


def run_strategy(
    strategy,
    data: pd.DataFrame,
    *,
    swap_costs: Dict[int, float],
    trade_notional: float,
) -> Dict:
    enriched = strategy.calculate_indicators(data.copy())
    signals = strategy.generate_signals(enriched)

    buy_signal = signals.get('buy_signal', pd.Series(False, index=signals.index)).astype(bool)
    sell_signal = signals.get('sell_signal', pd.Series(False, index=signals.index)).astype(bool)

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
    strat_returns = strat_returns - buy_cost_frac * entries.astype(float) - sell_cost_frac * exits.astype(float)

    equity_curve = (1.0 + strat_returns).cumprod()

    total_return = float(equity_curve[-1] - 1.0)
    num_bars = len(data)
    annual_factor = FIVE_MINUTES_PER_YEAR / max(num_bars, 1)
    cagr = float((equity_curve[-1] ** annual_factor) - 1.0)

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

    num_trades = int(entries.sum())
    win_rate = float(np.mean([r > 0 for r in trade_returns])) if trade_returns else 0.0

    total_cost_frac = buy_cost_frac * entries.sum() + sell_cost_frac * exits.sum()
    total_cost_pct = total_cost_frac * 100
    total_cost_dai = total_cost_frac * trade_notional
    avg_cost_per_trade_pct = (total_cost_pct / num_trades) if num_trades else 0.0

    buy_hold_return = float(close.iloc[-1] / close.iloc[0] - 1.0)

    return {
        'total_return_pct': total_return * 100,
        'buy_hold_return_pct': buy_hold_return * 100,
        'cagr_pct': cagr * 100,
        'max_drawdown_pct': max_drawdown * 100,
        'sharpe': sharpe,
        'sortino': sortino,
        'num_trades': num_trades,
        'win_rate_pct': win_rate * 100,
        'total_cost_pct': total_cost_pct,
        'total_cost_dai': total_cost_dai,
        'avg_cost_per_trade_pct': avg_cost_per_trade_pct,
        'equity_curve': equity_curve,
        'strategy_returns': strat_returns,
    }


def format_row(name: str, stats: Dict) -> str:
    return (
        f"{name:<32}"
        f" {stats['total_return_pct']:>9.2f}%"
        f" {stats['buy_hold_return_pct']:>9.2f}%"
        f" {stats['cagr_pct']:>8.2f}%"
        f" {stats['max_drawdown_pct']:>8.2f}%"
        f" {stats['sharpe']:>6.2f}"
        f" {stats['sortino']:>7.2f}"
        f" {stats['num_trades']:>6d}"
        f" {stats['win_rate_pct']:>7.2f}%"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--data',
        type=Path,
        default=Path('data/pdai_ohlcv_dai_730day_5m.csv'),
        help='Path to the PDAI-DAI dataset CSV.',
    )
    parser.add_argument(
        '--swap-cost-cache',
        type=Path,
        default=Path('reports/optimizer_top_top_strats_run/swap_cost_cache.json'),
        help='Path to swap cost cache JSON.',
    )
    parser.add_argument(
        '--trade-size',
        type=float,
        default=5000.0,
        help='Notional size per trade in DAI used for cost lookup (default 5k).',
    )
    args = parser.parse_args()

    data = load_dataset(args.data)
    swap_costs = load_swap_costs(args.swap_cost_cache)

    strategies = [
        ('PassiveHold', PassiveHoldStrategy()),
        ('TrailingHold', TrailingHoldStrategy()),
        ('LongTermRegime', LongTermRegimeStrategy()),
        ('MacroTrendChannel', MacroTrendChannelStrategy()),
        ('CSMARevert', CSMARevertStrategy()),
        ('DonchianChampion', DonchianChampionStrategy()),
        ('DonchianChampionAggressive', DonchianChampionAggressiveStrategy()),
        ('DonchianChampionDynamic', DonchianChampionDynamicStrategy()),
        ('Codex1CSMATurbo', Codex1CSMATurboStrategy()),
        ('Codex1CSMAEnhanced', Codex1CSMAEnhancedStrategy()),
        ('Codex1CSMAUltra', Codex1CSMAUltraStrategy()),
        ('Codex1Phoenix', Codex1PhoenixStrategy()),
        ('Codex1RecoveryTrend', Codex1RecoveryTrendStrategy()),
        ('HybridV2', HybridV2Strategy()),
        ('TightTrendFollow', TightTrendFollowStrategy()),
        ('MultiWeekBreakout', MultiWeekBreakoutStrategy()),
        ('VOSTTrendRider', VOSTTrendRiderStrategy()),
        ('VOSTBreakoutSqueeze', VOSTBreakoutSqueezeStrategy()),
        ('VOSTPullbackAccumulator', VOSTPullbackAccumulatorStrategy()),
    ]

    results: List[Tuple[str, Dict]] = []
    for name, strategy in strategies:
        stats = run_strategy(strategy, data, swap_costs=swap_costs, trade_notional=args.trade_size)
        results.append((name, stats))

    header = (
        f"{'Strategy':<32}  TotalRet   Buy&Hold     CAGR   MaxDD  Sharpe Sortino Trades  Win%"
    )
    print(header)
    print('-' * len(header))
    for name, stats in results:
        print(format_row(name, stats))


if __name__ == '__main__':
    main()
