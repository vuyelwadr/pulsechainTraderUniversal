#!/usr/bin/env python3
"""
Strategy Validation - Test Best Performing Strategies

Validates the top performing strategies from optimization on multiple timeframes.
"""
import pandas as pd
import numpy as np
import json
import sys
from pathlib import Path
from datetime import datetime

def load_data_period(filepath: str, days: int = 365) -> pd.DataFrame:
    """Load data for specific time period"""
    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    rows_needed = days * 24 * 12
    if len(df) > rows_needed:
        df = df.tail(rows_needed).reset_index(drop=True)

    return df

def load_swap_costs(cache_file: str) -> dict:
    """Load swap cost cache"""
    with open(cache_file, 'r') as f:
        data = json.load(f)

    costs = {}
    for entry in data['entries'].values():
        notional = int(float(entry['notional_dai']))
        costs[notional] = {
            'loss_rate': float(entry['derived']['loss_rate']),
            'roundtrip_loss_dai': float(entry['derived']['roundtrip_loss_dai'])
        }

    return costs

def get_trade_cost(costs: dict, trade_size_dai: float) -> float:
    """Get trading cost"""
    trade_sizes = sorted(costs.keys())
    closest_size = min(trade_sizes, key=lambda x: abs(x - trade_size_dai))

    if trade_size_dai in costs:
        return costs[trade_size_dai]['loss_rate']
    elif trade_size_dai < trade_sizes[0]:
        return costs[trade_sizes[0]]['loss_rate']
    elif trade_size_dai > trade_sizes[-1]:
        return costs[trade_sizes[-1]]['loss_rate']
    else:
        for i in range(len(trade_sizes) - 1):
            if trade_sizes[i] <= trade_size_dai <= trade_sizes[i+1]:
                x0, x1 = trade_sizes[i], trade_sizes[i+1]
                y0 = costs[x0]['loss_rate']
                y1 = costs[x1]['loss_rate']
                return y0 + (y1 - y0) * (trade_size_dai - x0) / (x1 - x0)

    return 0.01

def calculate_buy_hold(data: pd.DataFrame) -> dict:
    """Calculate buy & hold performance"""
    start_price = data['price'].iloc[0]
    end_price = data['price'].iloc[-1]
    bh_return = (end_price - start_price) / start_price

    return {
        'return': bh_return,
        'start_price': start_price,
        'end_price': end_price,
        'period_days': (data['timestamp'].iloc[-1] - data['timestamp'].iloc[0]).days
    }

def test_strategy_with_params(data: pd.DataFrame, costs: dict, params: dict) -> dict:
    """Test strategy with specific parameters"""
    capital = 1000.0
    position = 0
    trades = []

    # Extract parameters
    short_ma = params.get('short_ma', 3)
    long_ma = params.get('long_ma', 15)
    rsi_period = params.get('rsi_period', 8)
    trend_threshold = params.get('trend_threshold', 0.002)
    hold_periods = params.get('hold_periods', 60)
    capital_usage = params.get('capital_usage', 0.995)
    rsi_entry_max = params.get('rsi_entry_max', 75)
    rsi_exit_min = params.get('rsi_exit_min', 85)

    # Calculate indicators
    data = data.copy()
    data['short_ma'] = data['price'].rolling(window=short_ma).mean()
    data['long_ma'] = data['price'].rolling(window=long_ma).mean()
    data['trend_strength'] = abs(data['short_ma'] - data['long_ma']) / data['long_ma']

    delta = data['price'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
    rs = gain / loss
    data['rsi'] = 100 - (100 / (1 + rs))
    data['rsi'] = data['rsi'].fillna(50)

    data['trend_direction'] = np.where(data['short_ma'] > data['long_ma'], 1,
                                     np.where(data['short_ma'] < data['long_ma'], -1, 0))

    signal_count = 0

    for i in range(long_ma, len(data)):
        row = data.iloc[i]
        current_price = row['price']

        if position == 0:
            if (row['trend_direction'] == 1 and
                row['trend_strength'] >= trend_threshold and
                row['rsi'] <= rsi_entry_max):

                position_value = capital * capital_usage
                cost_rate = get_trade_cost(costs, position_value)
                entry_price = current_price * (1 + cost_rate)
                position = position_value / entry_price

                trades.append({
                    'type': 'buy',
                    'price': current_price,
                    'entry_price': entry_price,
                    'position_value': position_value,
                    'cost_rate': cost_rate,
                    'index': i
                })
                signal_count += 1

        elif position > 0:
            since_entry = i - trades[-1]['index']

            if (since_entry >= hold_periods or
                row['trend_direction'] != 1 or
                row['rsi'] >= rsi_exit_min):

                hex_value = position * current_price
                cost_rate = get_trade_cost(costs, hex_value)
                exit_price = current_price * (1 - cost_rate)

                entry_cost = trades[-1]['position_value']
                exit_value = position * exit_price
                trade_pnl = exit_value - entry_cost
                capital += trade_pnl

                trades.append({
                    'type': 'sell',
                    'price': current_price,
                    'exit_price': exit_price,
                    'pnl': trade_pnl,
                    'index': i
                })
                position = 0

    # Calculate metrics
    bh_info = calculate_buy_hold(data)
    strategy_return = (capital - 1000.0) / 1000.0

    return {
        'strategy_return': strategy_return,
        'buy_hold_return': bh_info['return'],
        'final_capital': capital,
        'num_trades': len(trades) // 2,
        'outperformance': strategy_return - bh_info['return'],
        'signals': signal_count,
        'period_days': bh_info['period_days']
    }

def main():
    """Validate best performing strategies"""
    print("=" * 80)
    print("STRATEGY VALIDATION - BEST PERFORMERS")
    print("=" * 80)

    # File paths
    data_file = "/Users/ruwodda/Documents/Personal/Repos/pulsechainTraderUniversal/data/pdai_ohlcv_dai_730day_5m.csv"
    cost_file = "/Users/ruwodda/Documents/Personal/Repos/pulsechainTraderUniversal/reports/optimizer_top_top_strats_run/swap_cost_cache.json"

    # Load costs
    costs = load_swap_costs(cost_file)

    # Best performing strategies from optimization
    best_strategies = {
        "Champion_38.8%": {
            'short_ma': 8, 'long_ma': 23, 'rsi_period': 16, 'trend_threshold': 0.008,
            'hold_periods': 72, 'capital_usage': 0.85, 'rsi_entry_max': 65, 'rsi_exit_min': 75
        },
        "RunnerUp_30.1%": {
            'short_ma': 11, 'long_ma': 25, 'rsi_period': 16, 'trend_threshold': 0.008,
            'hold_periods': 72, 'capital_usage': 0.85, 'rsi_entry_max': 65, 'rsi_exit_min': 75
        },
        "Consistent_18.1%": {
            'short_ma': 14, 'long_ma': 28, 'rsi_period': 18, 'trend_threshold': 0.007,
            'hold_periods': 60, 'capital_usage': 0.80, 'rsi_entry_max': 70, 'rsi_exit_min': 80
        }
    }

    # Test timeframes
    timeframes = [60, 90, 120, 180, 270, 365]  # 2 months to 1 year

    results = []

    for strategy_name, params in best_strategies.items():
        print(f"\n{'='*60}")
        print(f"VALIDATING: {strategy_name}")
        print(f"{'='*60}")

        strategy_results = []

        for days in timeframes:
            # Load data
            period_data = load_data_period(data_file, days)

            # Test strategy
            result = test_strategy_with_params(period_data, costs, params)

            strategy_return_pct = result['strategy_return'] * 100
            bh_return_pct = result['buy_hold_return'] * 100
            outperformance = result['outperformance'] * 100

            print("6d")

            strategy_results.append({
                'days': days,
                'strategy_return': strategy_return_pct,
                'buy_hold_return': bh_return_pct,
                'outperformance': outperformance,
                'trades': result['num_trades'],
                'signals': result['signals']
            })

        results.append((strategy_name, strategy_results))

    # Summary
    print(f"\n{'='*80}")
    print("VALIDATION SUMMARY")
    print(f"{'='*80}")

    for strategy_name, strategy_results in results:
        print(f"\n{strategy_name}:")
        print("-" * 40)

        # Calculate averages
        avg_outperformance = np.mean([r['outperformance'] for r in strategy_results])
        positive_periods = sum(1 for r in strategy_results if r['outperformance'] > 0)
        total_periods = len(strategy_results)

        print(".1f")
        print(f"  Positive periods: {positive_periods}/{total_periods}")

        # Show detailed results
        for result in strategy_results:
            status = "✅" if result['outperformance'] > 0 else "❌"
            print("6d")

    print(f"\n{'='*80}")
    print("RECOMMENDATION")
    print(f"{'='*80}")

    # Find best overall strategy
    best_strategy = None
    best_score = -1000

    for strategy_name, strategy_results in results:
        avg_outperformance = np.mean([r['outperformance'] for r in strategy_results])
        positive_rate = sum(1 for r in strategy_results if r['outperformance'] > 0) / len(strategy_results)

        # Score = average outperformance * consistency (positive rate)
        score = avg_outperformance * positive_rate

        if score > best_score:
            best_score = score
            best_strategy = strategy_name

    if best_strategy:
        print(f"🏆 RECOMMENDED STRATEGY: {best_strategy}")
        print(".1f")
        print("\nThis strategy shows the best combination of returns and consistency across different market conditions.")

if __name__ == "__main__":
    main()