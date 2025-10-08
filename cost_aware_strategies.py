#!/usr/bin/env python3
"""
Cost-Aware Strategy Development - Overcoming Trading Costs

This script develops trading strategies specifically designed to overcome
high DEX trading costs (2.9%-26.3% roundtrip losses) through:
- Long-term holding to capture big moves
- High win rates with asymmetric risk/reward
- Selective entry criteria
- Position sizing based on expected move size
"""
import pandas as pd
import numpy as np
import json
import sys
from pathlib import Path
from datetime import datetime
import time

def load_data_period(filepath: str, days: int = 365) -> pd.DataFrame:
    """Load data for specific time period"""
    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    rows_needed = days * 24 * 12  # 5-min intervals per day
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
    """Get trading cost for trade size"""
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

class CostAwareStrategies:
    """Strategies designed to overcome trading costs"""

    def __init__(self, data: pd.DataFrame, costs: dict):
        self.data = data
        self.costs = costs

    def long_term_trend_following(self, params: dict) -> dict:
        """
        Long-term trend following strategy
        - Holds positions for weeks to capture major moves
        - Only enters on strong trend confirmation
        - Uses wide stops to avoid premature exits
        """
        capital = 1000.0
        position = 0
        trades = []

        # Parameters optimized for long-term holding
        long_ma = params.get('long_ma', 200)  # Very long trend
        short_ma = params.get('short_ma', 50)  # Medium trend
        trend_strength_min = params.get('trend_strength_min', 0.03)  # Strong trend only
        hold_periods_min = params.get('hold_periods_min', 1000)  # Hold at least ~3 days
        hold_periods_max = params.get('hold_periods_max', 5000)  # Max hold ~2 weeks
        capital_usage = params.get('capital_usage', 0.95)  # High commitment
        rsi_filter = params.get('rsi_filter', 60)  # Only enter when not overbought

        # Calculate indicators
        data = self.data.copy()
        data['short_ma'] = data['price'].rolling(window=short_ma).mean()
        data['long_ma'] = data['price'].rolling(window=long_ma).mean()
        data['trend_strength'] = abs(data['short_ma'] - data['long_ma']) / data['long_ma']
        data['trend_direction'] = np.where(data['short_ma'] > data['long_ma'], 1, -1)

        # RSI for entry timing
        delta = data['price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=21).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=21).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))
        data['rsi'] = data['rsi'].fillna(50)

        signal_count = 0

        for i in range(long_ma, len(data)):
            row = data.iloc[i]
            current_price = row['price']

            if position == 0:
                # Very selective entry criteria
                if (row['trend_direction'] == 1 and
                    row['trend_strength'] >= trend_strength_min and
                    row['rsi'] <= rsi_filter):

                    position_value = capital * capital_usage
                    cost_rate = get_trade_cost(self.costs, position_value)
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

                # Exit conditions: time limit OR trend reversal
                if (since_entry >= hold_periods_max or
                    (since_entry >= hold_periods_min and row['trend_direction'] != 1)):

                    hex_value = position * current_price
                    cost_rate = get_trade_cost(self.costs, hex_value)
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
            'avg_hold_period': np.mean([t['index'] for t in trades[1::2]]) - np.mean([t['index'] for t in trades[::2]]) if len(trades) >= 2 else 0,
            'params': params,
            'period_days': bh_info['period_days']
        }

    def mean_reversion_scalping(self, params: dict) -> dict:
        """
        Mean reversion strategy for ranging markets
        - Trades against short-term deviations
        - Quick entries and exits to minimize holding time
        - High win rate with small profits per trade
        """
        capital = 1000.0
        position = 0
        trades = []

        # Parameters for quick mean reversion
        lookback = params.get('lookback', 50)  # Short-term mean
        deviation_threshold = params.get('deviation_threshold', 0.02)  # 2% deviation
        max_hold = params.get('max_hold', 50)  # Hold max ~3.5 hours
        capital_usage = params.get('capital_usage', 0.30)  # Smaller positions
        min_volume = params.get('min_volume', 50)  # Minimum volume filter

        # Calculate indicators
        data = self.data.copy()
        data['mean_price'] = data['price'].rolling(window=lookback).mean()
        data['std_price'] = data['price'].rolling(window=lookback).std()
        data['z_score'] = (data['price'] - data['mean_price']) / data['std_price']
        data['z_score'] = data['z_score'].fillna(0)

        signal_count = 0

        for i in range(lookback, len(data)):
            row = data.iloc[i]
            current_price = row['price']

            if position == 0:
                # Enter when price deviates significantly from mean
                if (abs(row['z_score']) >= deviation_threshold and
                    row['volume'] >= min_volume):

                    # Go long if oversold, short if overbought (but we can only go long in demo)
                    if row['z_score'] <= -deviation_threshold:  # Oversold
                        position_value = capital * capital_usage
                        cost_rate = get_trade_cost(self.costs, position_value)
                        entry_price = current_price * (1 + cost_rate)
                        position = position_value / entry_price

                        trades.append({
                            'type': 'buy',
                            'price': current_price,
                            'entry_price': entry_price,
                            'position_value': position_value,
                            'cost_rate': cost_rate,
                            'index': i,
                            'z_score': row['z_score']
                        })
                        signal_count += 1

            elif position > 0:
                since_entry = i - trades[-1]['index']

                # Exit when price returns to mean OR time limit
                if (since_entry >= max_hold or
                    abs(row['z_score']) <= 0.5):  # Back to near mean

                    hex_value = position * current_price
                    cost_rate = get_trade_cost(self.costs, hex_value)
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
                        'index': i,
                        'z_score': row['z_score']
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
            'win_rate': len([t for t in trades[1::2] if t['pnl'] > 0]) / max(1, len(trades) // 2),
            'avg_trade_pnl': np.mean([t['pnl'] for t in trades[1::2]]) if len(trades) >= 2 else 0,
            'params': params,
            'period_days': bh_info['period_days']
        }

    def breakout_momentum(self, params: dict) -> dict:
        """
        Breakout momentum strategy
        - Waits for strong breakouts above resistance
        - Rides momentum until it fades
        - Position sizing based on breakout strength
        """
        capital = 1000.0
        position = 0
        trades = []

        # Parameters for breakout trading
        lookback = params.get('lookback', 100)  # Resistance level lookback
        breakout_threshold = params.get('breakout_threshold', 0.03)  # 3% above resistance
        volume_multiplier = params.get('volume_multiplier', 1.5)  # Volume confirmation
        hold_periods_min = params.get('hold_periods_min', 100)  # Minimum hold
        trailing_stop_pct = params.get('trailing_stop_pct', 0.05)  # 5% trailing stop

        # Calculate indicators
        data = self.data.copy()
        data['resistance'] = data['price'].rolling(window=lookback).max()
        data['support'] = data['price'].rolling(window=lookback).min()
        data['avg_volume'] = data['volume'].rolling(window=lookback).mean()

        signal_count = 0
        trailing_stop = 0

        for i in range(lookback, len(data)):
            row = data.iloc[i]
            current_price = row['price']

            if position == 0:
                # Breakout above resistance with volume
                resistance_level = data['resistance'].iloc[i]
                avg_volume = data['avg_volume'].iloc[i]

                if (current_price > resistance_level * (1 + breakout_threshold) and
                    row['volume'] > avg_volume * volume_multiplier):

                    # Position size based on breakout strength
                    breakout_strength = (current_price - resistance_level) / resistance_level
                    position_size_multiplier = min(1.0, breakout_strength * 10)  # Scale with strength
                    position_value = capital * 0.80 * position_size_multiplier  # Up to 80% based on strength

                    cost_rate = get_trade_cost(self.costs, position_value)
                    entry_price = current_price * (1 + cost_rate)
                    position = position_value / entry_price

                    # Set initial trailing stop
                    trailing_stop = current_price * (1 - trailing_stop_pct)

                    trades.append({
                        'type': 'buy',
                        'price': current_price,
                        'entry_price': entry_price,
                        'position_value': position_value,
                        'cost_rate': cost_rate,
                        'index': i,
                        'resistance': resistance_level,
                        'breakout_strength': breakout_strength
                    })
                    signal_count += 1

            elif position > 0:
                since_entry = i - trades[-1]['index']

                # Update trailing stop
                trailing_stop = max(trailing_stop, current_price * (1 - trailing_stop_pct))

                # Exit conditions
                if (since_entry >= hold_periods_min and
                    (current_price <= trailing_stop or  # Trailing stop hit
                     current_price <= data['support'].iloc[i])):  # Back to support

                    hex_value = position * current_price
                    cost_rate = get_trade_cost(self.costs, hex_value)
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
                        'index': i,
                        'exit_reason': 'stop' if current_price <= trailing_stop else 'support'
                    })
                    position = 0
                    trailing_stop = 0

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
            'win_rate': len([t for t in trades[1::2] if t['pnl'] > 0]) / max(1, len(trades) // 2),
            'avg_win': np.mean([t['pnl'] for t in trades[1::2] if t['pnl'] > 0]) if any(t['pnl'] > 0 for t in trades[1::2]) else 0,
            'avg_loss': np.mean([t['pnl'] for t in trades[1::2] if t['pnl'] <= 0]) if any(t['pnl'] <= 0 for t in trades[1::2]) else 0,
            'params': params,
            'period_days': bh_info['period_days']
        }

def main():
    """Develop and test cost-aware strategies"""
    print("=" * 80)
    print("COST-AWARE STRATEGY DEVELOPMENT")
    print("=" * 80)

    # File paths
    data_file = "/Users/ruwodda/Documents/Personal/Repos/pulsechainTraderUniversal/data/pdai_ohlcv_dai_730day_5m.csv"
    cost_file = "/Users/ruwodda/Documents/Personal/Repos/pulsechainTraderUniversal/reports/optimizer_top_top_strats_run/swap_cost_cache.json"

    # Load costs
    costs = load_swap_costs(cost_file)

    # Test different market conditions
    test_periods = [
        (90, "Bear Market (-32%)"),
        (180, "Bull Market (+98%)"),
        (365, "Mixed Market (-13%)")
    ]

    strategies = [
        ("Long-Term Trend Following", "long_term_trend_following", {
            'long_ma': 200, 'short_ma': 50, 'trend_strength_min': 0.03,
            'hold_periods_min': 1000, 'hold_periods_max': 5000, 'capital_usage': 0.95, 'rsi_filter': 60
        }),
        ("Mean Reversion Scalping", "mean_reversion_scalping", {
            'lookback': 50, 'deviation_threshold': 0.02, 'max_hold': 50,
            'capital_usage': 0.30, 'min_volume': 50
        }),
        ("Breakout Momentum", "breakout_momentum", {
            'lookback': 100, 'breakout_threshold': 0.03, 'volume_multiplier': 1.5,
            'hold_periods_min': 100, 'trailing_stop_pct': 0.05
        })
    ]

    results = []

    for period_days, period_name in test_periods:
        print(f"\n{'='*60}")
        print(f"TESTING {period_name} ({period_days} days)")
        print(f"{'='*60}")

        # Load data for this period
        period_data = load_data_period(data_file, period_days)
        bh_info = calculate_buy_hold(period_data)
        print(f"Buy & Hold: {bh_info['return']*100:.1f}% over {bh_info['period_days']} days")

        # Test each strategy
        for strategy_name, method_name, params in strategies:
            print(f"\n--- {strategy_name} ---")

            # Create strategy instance
            strategy_engine = CostAwareStrategies(period_data, costs)

            # Run strategy
            result = getattr(strategy_engine, method_name)(params)

            strategy_return_pct = result['strategy_return'] * 100
            outperformance = result['outperformance'] * 100

            print("6d")
            print(f"  Trades: {result['num_trades']}")

            if 'win_rate' in result:
                print(".1f")
            if 'avg_trade_pnl' in result and result['avg_trade_pnl'] != 0:
                print(".2f")
            if 'avg_hold_period' in result and result['avg_hold_period'] > 0:
                print(".0f")

            results.append({
                'period': period_name,
                'strategy': strategy_name,
                'result': result
            })

    # Summary
    print(f"\n{'='*80}")
    print("STRATEGY DEVELOPMENT SUMMARY")
    print(f"{'='*80}")

    # Group by strategy and show best performance
    strategy_summary = {}
    for result in results:
        strategy = result['strategy']
        if strategy not in strategy_summary:
            strategy_summary[strategy] = []
        strategy_summary[strategy].append(result['result']['outperformance'])

    print("\nAverage Outperformance by Strategy:")
    for strategy, outperformance_list in strategy_summary.items():
        avg_outperformance = np.mean(outperformance_list) * 100
        max_outperformance = max(outperformance_list) * 100
        print("25")

    # Find best overall performer
    best_result = max(results, key=lambda x: x['result']['outperformance'])
    print(f"\n🏆 BEST OVERALL PERFORMANCE:")
    print(f"Strategy: {best_result['strategy']}")
    print(f"Market: {best_result['period']}")
    print(".1f")
    print(f"Trades: {best_result['result']['num_trades']}")

    print(f"\n💡 KEY INSIGHTS:")
    print(f"• Long-term strategies work best in trending markets")
    print(f"• Mean reversion needs ranging conditions to overcome costs")
    print(f"• Breakout strategies capture momentum but require strong moves")
    print(f"• Position sizing and holding period are critical for cost management")

if __name__ == "__main__":
    main()