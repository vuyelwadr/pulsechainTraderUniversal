#!/usr/bin/env python3
"""Evaluate VOST-based strategies on the PDAI-DAI 5m dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import logging

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
from strategies.multiweek_breakout_ultra_strategy import MultiWeekBreakoutUltraStrategy
from strategies.long_term_regime_strategy import LongTermRegimeStrategy
from strategies.passive_hold_strategy import PassiveHoldStrategy
from strategies.macro_trend_channel_strategy import MacroTrendChannelStrategy
from strategies.trailing_hold_strategy import TrailingHoldStrategy
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
    logging.warning(
        "Trade notional $%.2f exceeds largest swap-cost bucket ($%.2f); using largest bucket values.",
        notional,
        keys[-1],
    )
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
    swap_costs: Dict[str, Dict[int, Dict[str, float]]],
    trade_notional: float,
    capture_trades: bool = False,
) -> Dict:
    enriched = strategy.calculate_indicators(data.copy())
    signals = strategy.generate_signals(enriched)

    buy_signal = signals.get('buy_signal', pd.Series(False, index=signals.index)).astype(bool)
    sell_signal = signals.get('sell_signal', pd.Series(False, index=signals.index)).astype(bool)

    close_series = data['close'] if 'close' in data.columns else data['price']
    open_series = data['open'] if 'open' in data.columns else close_series

    cash = float(trade_notional)
    tokens = 0.0
    total_cost_dai = 0.0
    total_notional = 0.0
    equity_high = cash
    max_drawdown_observed = 0.0
    drawdown_tripped = False
    max_drawdown_cap: Optional[float] = None
    if hasattr(strategy, 'parameters'):
        cap_val = strategy.parameters.get('max_equity_drawdown_pct')
        if cap_val is not None:
            try:
                max_drawdown_cap = float(cap_val)
            except (TypeError, ValueError):
                max_drawdown_cap = None
    if max_drawdown_cap is None and hasattr(strategy, 'max_equity_drawdown_pct'):
        try:
            max_drawdown_cap = float(getattr(strategy, 'max_equity_drawdown_pct'))
        except (TypeError, ValueError):
            max_drawdown_cap = None
    if max_drawdown_cap is not None:
        if max_drawdown_cap <= 0:
            max_drawdown_cap = None
        else:
            max_drawdown_cap = min(max_drawdown_cap, 0.99)

    equity_history: List[float] = []
    trade_returns: List[float] = []
    trade_log: List[Dict[str, Any]] = []

    profitable_trades = 0
    losing_trades = 0
    positive_sum = 0.0
    negative_sum = 0.0

    in_trade = False
    entry_equity = None
    pending_trade: Optional[Dict[str, Any]] = None

    executed_buys: List[bool] = []
    executed_sells: List[bool] = []

    for i in range(len(data) - 1):
        price_now = float(close_series.iloc[i])
        equity_now = cash + tokens * price_now
        equity_history.append(equity_now)

        if max_drawdown_cap is not None and equity_high > 0:
            drawdown_now = (equity_now - equity_high) / equity_high
            max_drawdown_observed = min(max_drawdown_observed, drawdown_now)
            if drawdown_now <= -max_drawdown_cap:
                drawdown_tripped = True
        if equity_now > equity_high:
            equity_high = equity_now

        # Forced liquidation at current close if circuit breaker tripped
        if drawdown_tripped and tokens > 0:
            notional = tokens * price_now
            cost = per_side_cost(swap_costs, notional, 'sell')
            proceeds = max(notional - cost, 0.0)
            cash += proceeds
            total_cost_dai += cost
            total_notional += notional

            equity_after = cash
            if in_trade and entry_equity and entry_equity > 0.0:
                trade_roi = equity_after / entry_equity - 1.0
                trade_returns.append(trade_roi)
                if trade_roi > 0:
                    profitable_trades += 1
                    positive_sum += trade_roi
                elif trade_roi < 0:
                    losing_trades += 1
                    negative_sum += trade_roi
            in_trade = False
            entry_equity = None
            tokens = 0.0

            if capture_trades and pending_trade is not None:
                pending_trade.update(
                    {
                        'exit_index': i,
                        'exit_price': price_now,
                        'cost_sell_dai': cost,
                        'proceeds_dai': proceeds,
                        'return_pct': trade_roi * 100 if 'trade_roi' in locals() else 0.0,
                    }
                )
                trade_log.append(pending_trade)
                pending_trade = None

            equity_history[-1] = cash
            executed_buys.append(False)
            executed_sells.append(True)
            continue

        exec_price = float(open_series.iloc[i + 1])

        buy_executed = False
        sell_executed = False

        if tokens <= 0 and buy_signal.iat[i] and cash > 0.0 and not drawdown_tripped:
            notional = cash
            cost = per_side_cost(swap_costs, notional, 'buy')
            effective_cash = cash - cost
            if effective_cash > 0.0:
                tokens = effective_cash / exec_price
                cash = 0.0
                total_cost_dai += cost
                total_notional += notional
                in_trade = True
                entry_equity = equity_now
                buy_executed = True
                if capture_trades:
                    pending_trade = {
                        'entry_index': i + 1,
                        'entry_price': exec_price,
                        'cost_buy_dai': cost,
                        'notional_committed_dai': notional,
                    }

        forced_liquidation = drawdown_tripped and tokens > 0
        if tokens > 0 and (sell_signal.iat[i] or forced_liquidation):
            notional = tokens * exec_price
            cost = per_side_cost(swap_costs, notional, 'sell')
            proceeds = max(notional - cost, 0.0)
            cash += proceeds
            total_cost_dai += cost
            total_notional += notional
            sell_executed = True

            equity_after = cash
            if in_trade and entry_equity and entry_equity > 0.0:
                trade_roi = equity_after / entry_equity - 1.0
                trade_returns.append(trade_roi)
                if trade_roi > 0:
                    profitable_trades += 1
                    positive_sum += trade_roi
                elif trade_roi < 0:
                    losing_trades += 1
                    negative_sum += trade_roi
            in_trade = False
            entry_equity = None
            tokens = 0.0

            if capture_trades and pending_trade is not None:
                pending_trade.update(
                    {
                        'exit_index': i + 1,
                        'exit_price': exec_price,
                        'cost_sell_dai': cost,
                        'proceeds_dai': proceeds,
                        'return_pct': trade_roi * 100 if 'trade_roi' in locals() else 0.0,
                    }
                )
                trade_log.append(pending_trade)
                pending_trade = None

        executed_buys.append(buy_executed)
        executed_sells.append(sell_executed)

    final_price = float(close_series.iloc[-1])
    final_equity = cash + tokens * final_price
    equity_history.append(final_equity)

    if tokens > 0:
        notional = tokens * final_price
        cost = per_side_cost(swap_costs, notional, 'sell')
        proceeds = max(notional - cost, 0.0)
        cash += proceeds
        total_cost_dai += cost
        total_notional += notional
        equity_after = cash
        if in_trade and entry_equity and entry_equity > 0.0:
            trade_roi = equity_after / entry_equity - 1.0
            trade_returns.append(trade_roi)
            if trade_roi > 0:
                profitable_trades += 1
                positive_sum += trade_roi
            elif trade_roi < 0:
                losing_trades += 1
                negative_sum += trade_roi
        tokens = 0.0
        in_trade = False
        entry_equity = None
        if capture_trades and pending_trade is not None:
            pending_trade.update(
                {
                    'exit_index': len(data) - 1,
                    'exit_price': final_price,
                    'cost_sell_dai': cost,
                    'proceeds_dai': proceeds,
                    'return_pct': trade_roi * 100 if 'trade_roi' in locals() else 0.0,
                }
            )
            trade_log.append(pending_trade)
            pending_trade = None

    equity_history[-1] = cash

    equity_array = np.array(equity_history)
    strat_returns = np.zeros_like(equity_array)
    strat_returns[1:] = np.diff(equity_array) / np.maximum(equity_array[:-1], 1e-9)

    total_return = float(equity_array[-1] / max(equity_array[0], 1e-9) - 1.0)
    num_bars = len(equity_array)
    annual_factor = FIVE_MINUTES_PER_YEAR / max(num_bars - 1, 1)
    cagr = float((equity_array[-1] / max(equity_array[0], 1e-9)) ** annual_factor - 1.0)

    rolling_peak = np.maximum.accumulate(equity_array)
    drawdowns = equity_array / np.maximum(rolling_peak, 1e-9) - 1.0
    max_drawdown = float(drawdowns.min())
    max_duration_bars = 0
    current_start = None
    for idx, dd in enumerate(drawdowns):
        if dd < 0:
            if current_start is None:
                current_start = idx
            duration = idx - current_start
            if duration > max_duration_bars:
                max_duration_bars = duration
        else:
            current_start = None
    max_drawdown_duration_days = (max_duration_bars * 5.0) / (60.0 * 24.0)

    mean_per_bar = strat_returns.mean()
    std_per_bar = strat_returns.std(ddof=0)
    sharpe = float(mean_per_bar / std_per_bar * np.sqrt(FIVE_MINUTES_PER_YEAR)) if std_per_bar > 0 else 0.0

    downside = strat_returns[strat_returns < 0]
    downside_std = downside.std(ddof=0) if len(downside) > 0 else 0.0
    sortino = float(mean_per_bar / downside_std * np.sqrt(FIVE_MINUTES_PER_YEAR)) if downside_std > 0 else 0.0

    wins = [r for r in trade_returns if r > 0]
    losses = [r for r in trade_returns if r < 0]
    num_trades = len(trade_returns)
    win_rate_pct = len(wins) / num_trades * 100 if num_trades else 0.0
    avg_win_pct = np.mean(wins) * 100 if wins else 0.0
    avg_loss_pct = np.mean(losses) * 100 if losses else 0.0
    profit_factor = (positive_sum / max(1e-9, abs(negative_sum))) if negative_sum != 0 else (float('inf') if positive_sum > 0 else 0.0)
    recovery_factor = total_return / max(1e-9, abs(max_drawdown)) if max_drawdown != 0 else float('inf')

    total_cost_pct = (total_cost_dai / max(1e-9, total_notional)) * 100
    avg_cost_per_trade_pct = total_cost_pct / max(num_trades, 1)

    buy_hold_return = float(close_series.iloc[-1] / max(close_series.iloc[0], 1e-9) - 1.0)
    buy_hold_series = close_series / max(1e-9, float(close_series.iloc[0]))
    buy_hold_peak = np.maximum.accumulate(buy_hold_series.to_numpy())
    buy_hold_drawdowns = buy_hold_series.to_numpy() / buy_hold_peak - 1.0
    buy_hold_max_drawdown = float(buy_hold_drawdowns.min())

    if 'timestamp' in data.columns:
        ts = pd.to_datetime(data['timestamp'])
        duration_days = max(1, int((ts.iloc[-1] - ts.iloc[0]).days))
    elif isinstance(data.index, pd.DatetimeIndex) and len(data.index) > 0:
        duration_days = max(1, int((data.index[-1] - data.index[0]).days))
    else:
        duration_days = max(1, int(len(close_series) * 5 / (60 * 24)))

    avg_buy_step_pct = None
    avg_sell_step_pct = None
    buy_samples = getattr(strategy, '_buy_step_samples', None)
    sell_samples = getattr(strategy, '_sell_step_samples', None)
    if buy_samples:
        avg_buy_step_pct = float(np.mean(buy_samples) * 100.0)
    if sell_samples:
        avg_sell_step_pct = float(np.mean(sell_samples) * 100.0)

    payload = {
        'total_return_pct': total_return * 100,
        'buy_hold_return_pct': buy_hold_return * 100,
        'buy_hold_max_drawdown_pct': buy_hold_max_drawdown * 100,
        'cagr_pct': cagr * 100,
        'max_drawdown_pct': max_drawdown * 100,
        'max_drawdown_duration_days': max_drawdown_duration_days,
        'sharpe': sharpe,
        'sortino': sortino,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'num_trades': num_trades,
        'win_rate_pct': win_rate_pct,
        'total_trades': num_trades,
        'profitable_trades': len(wins),
        'losing_trades': len(losses),
        'avg_win_pct': avg_win_pct,
        'avg_loss_pct': avg_loss_pct,
        'profit_factor': profit_factor,
        'recovery_factor': recovery_factor,
        'duration_days': duration_days,
        'total_cost_pct': total_cost_pct,
        'total_cost_dai': total_cost_dai,
        'avg_cost_per_trade_pct': avg_cost_per_trade_pct,
        'equity_curve': equity_array,
        'strategy_returns': strat_returns,
        'trade_returns': trade_returns,
        'final_balance': float(equity_array[-1]),
        'initial_balance': float(equity_array[0]),
    }
    if avg_buy_step_pct is not None:
        payload['avg_buy_step_pct'] = avg_buy_step_pct
        payload['buy_step_sample_count'] = len(buy_samples)
    if avg_sell_step_pct is not None:
        payload['avg_sell_step_pct'] = avg_sell_step_pct
        payload['sell_step_sample_count'] = len(sell_samples)
    if max_drawdown_cap is not None:
        payload['max_equity_drawdown_pct_observed'] = max_drawdown_observed * 100
        payload['drawdown_circuit_tripped'] = drawdown_tripped

    if capture_trades and trade_log:
        payload['trades'] = trade_log

    return payload


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
        ('CSMARevertPro1', CSMARevertPro1Strategy()),
        ('DonchianChampion', DonchianChampionStrategy()),
        ('DonchianChampionAggressive', DonchianChampionAggressiveStrategy()),
        ('DonchianChampionDynamic', DonchianChampionDynamicStrategy()),
        ('HybridV2', HybridV2Strategy()),
        ('TightTrendFollow', TightTrendFollowStrategy()),
        ('MultiWeekBreakout', MultiWeekBreakoutStrategy()),
        ('MultiWeekBreakoutUltra', MultiWeekBreakoutUltraStrategy()),
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
