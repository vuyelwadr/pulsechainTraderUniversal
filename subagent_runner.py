#!/usr/bin/env python3
"""
Subagent Runner - What each of the 20 subagents actually executes
Usage: python subagent_runner.py --agent-id 1 --stage 1
"""

import sys
import os
import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from rich.table import Table

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backtest_engine import BacktestEngine
from data_handler import DataHandler
from strategies.tradingview_core_strategies import create_strategy
from strategies.tradingview_stub_strategies import create_any_strategy
from optimization.scoring_engine import CompositePerformanceScorer, StrategyMetrics


class SubagentRunner:
    """
    This is what each subagent actually runs
    Reads its assignment, tests strategies, reports results
    """
    
    def __init__(self, agent_id: int, stage: int):
        self.agent_id = agent_id
        self.stage = stage
        self.console = Console()
        
        # Paths
        self.assignments_file = Path(f"task/subagent_results/stage{stage}_assignments.json")
        self.instructions_file = Path(f"task/subagent_instructions/stage{stage}_agent_{agent_id:02d}_instructions.md")
        self.output_file = Path(f"task/subagent_results/stage{stage}_agent_{agent_id:02d}_results.json")
        
        # Load assignment
        self.task = self.load_assignment()
        
        # Initialize components
        self.data_handler = DataHandler()
        self.backtest_engine = BacktestEngine(initial_balance=1000)
        
        # Results storage
        self.results = []
        
    def load_assignment(self):
        """Load this agent's task assignment"""
        if not self.assignments_file.exists():
            self.console.print(f"[red]Assignment file not found: {self.assignments_file}[/red]")
            self.console.print("[yellow]Run 'python optimization/master_coordinator.py' first to create assignments[/yellow]")
            sys.exit(1)
        
        with open(self.assignments_file, 'r') as f:
            all_assignments = json.load(f)
        
        # Find this agent's assignment
        for assignment in all_assignments:
            if assignment['agent_id'] == self.agent_id:
                self.console.print(f"[green]✓ Loaded assignment for Agent {self.agent_id:02d}[/green]")
                return assignment
        
        self.console.print(f"[red]No assignment found for Agent {self.agent_id}[/red]")
        sys.exit(1)
    
    def load_data(self):
        """Load historical data based on stage"""
        self.console.print(f"\n[cyan]Loading data for Stage {self.stage}...[/cyan]")
        
        if self.stage == 1:
            # Stage 1: Last month (bearish)
            days = 30
            expected_return = 0.0105  # 1.05%
            market_condition = "bear"
        elif self.stage == 2:
            # Stage 2: Last 3 months (mixed)
            days = 90
            expected_return = 0.2229  # 22.29%
            market_condition = "mixed"
        else:  # Stage 3
            # Stage 3: Last year (bull)
            days = 365
            expected_return = 1.0088  # 100.88%
            market_condition = "bull"
        
        # Try to load real data
        data = self.data_handler.fetch_historical_data(days=days)
        
        if data.empty:
            self.console.print("[yellow]⚠️ No real data available, using simulated data[/yellow]")
            data = self.generate_fallback_data(days)
        
        # Calculate actual buy & hold return
        buy_hold_return = (data['close'].iloc[-1] - data['close'].iloc[0]) / data['close'].iloc[0]
        
        self.console.print(f"[green]✓ Loaded {len(data)} data points[/green]")
        self.console.print(f"  Price range: ${data['price'].min():.6f} - ${data['price'].max():.6f}")
        self.console.print(f"  Buy & Hold return: {buy_hold_return*100:.2f}%")
        self.console.print(f"  Market condition: {market_condition}")
        
        return data, buy_hold_return, market_condition
    
    def generate_fallback_data(self, days):
        """Generate sample data if real data unavailable"""
        periods = days * 24 * 12  # 5-minute intervals
        timestamps = pd.date_range(end=datetime.now(), periods=periods, freq='5min')
        
        # Generate realistic price movement
        np.random.seed(42 + self.agent_id)
        returns = np.random.normal(0.0001, 0.02, periods)
        price_multiplier = np.exp(np.cumsum(returns))
        prices = 0.001 * price_multiplier
        
        return pd.DataFrame({
            'timestamp': timestamps,
            'open': prices * (1 + np.random.uniform(-0.001, 0.001, periods)),
            'high': prices * (1 + np.random.uniform(0, 0.005, periods)),
            'low': prices * (1 + np.random.uniform(-0.005, 0, periods)),
            'close': prices,
            'price': prices,
            'volume': np.random.uniform(1000, 10000, periods)
        })
    
    def create_strategy_instance(self, strategy_dict, timeframe_minutes):
        """Create a strategy instance from assignment"""
        strategy_id = strategy_dict['id']
        strategy_name = strategy_dict['name']
        
        # Add stage-specific parameters
        params = {'timeframe_minutes': timeframe_minutes}
        
        if self.stage == 2:
            # Stage 2: Test parameter variations
            # This would be one of [0.5x, 0.75x, 1x, 1.5x, 2x]
            params['param_multiplier'] = 1.0  # Default for now
        
        if strategy_id <= 15:
            # Core strategy
            from strategies.tradingview_core_strategies import create_strategy
            core_names = [
                'SqueezeMomentum', 'WaveTrend', 'CoralTrend', 'SchaffTrendCycle',
                'MESAAdaptiveMA', 'ElderImpulse', 'FRAMA', 'ZeroLagEMA',
                'KaufmannAMA', 'TradersDynamicIndex', 'InsyncIndex',
                'PremierStochastic', 'MAC_Z', 'FireflyOscillator', 'CompositeMomentumIndex'
            ]
            return create_strategy(core_names[strategy_id - 1], params)
        else:
            # Stub strategy
            return create_any_strategy(strategy_id, params)
    
    def test_strategy(self, strategy, data):
        """Run backtest on a strategy"""
        try:
            # Run backtest
            results = self.backtest_engine.run(data, strategy)
            
            # Create metrics object
            metrics = StrategyMetrics(
                strategy_name=strategy.name,
                timeframe=f"{strategy.timeframe_minutes}min",
                total_return=results['total_return'],
                max_drawdown=results['max_drawdown'],
                sharpe_ratio=results['sharpe_ratio'],
                sortino_ratio=results.get('sortino_ratio', results['sharpe_ratio']),
                total_trades=results['num_trades'],
                winning_trades=results.get('winning_trades', int(results['num_trades'] * results['win_rate'])),
                losing_trades=results.get('losing_trades', int(results['num_trades'] * (1 - results['win_rate']))),
                win_rate=results['win_rate'],
                avg_win=results.get('avg_win', 0.01),
                avg_loss=results.get('avg_loss', 0.005),
                profit_factor=results.get('profit_factor', 1.5),
                recovery_factor=results.get('recovery_factor', 0.5),
                trade_frequency=results['num_trades'] / (len(data) / (12*24)),  # trades per day
                reversal_catches=0,  # Would need more sophisticated analysis
                reversal_opportunities=5
            )
            
            return metrics, results
            
        except Exception as e:
            self.console.print(f"[red]Error testing {strategy.name}: {e}[/red]")
            return None, None
    
    def run(self):
        """Execute the agent's assigned tasks"""
        self.console.print(f"\n[bold cyan]SUBAGENT {self.agent_id:02d} - STAGE {self.stage}[/bold cyan]")
        self.console.print("="*60)
        
        # Display assignment
        self.console.print(f"\n[yellow]Assignment:[/yellow]")
        self.console.print(f"  Strategies: {len(self.task['strategies'])}")
        self.console.print(f"  Timeframes: {len(self.task['timeframes'])}")
        self.console.print(f"  Total tests: {len(self.task['strategies']) * len(self.task['timeframes'])}")
        
        # Load data
        data, buy_hold_return, market_condition = self.load_data()
        
        # Create scorer
        scorer = CompositePerformanceScorer(
            buy_hold_return=buy_hold_return,
            market_condition=market_condition
        )
        
        # Test each strategy/timeframe combination
        total_tests = len(self.task['strategies']) * len(self.task['timeframes'])
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        ) as progress:
            
            task_progress = progress.add_task(
                f"[cyan]Testing {len(self.task['strategies'])} strategies",
                total=total_tests
            )
            
            for strategy_dict in self.task['strategies']:
                for timeframe in self.task['timeframes']:
                    # Convert timeframe to minutes
                    tf_map = {
                        '5min': 5, '15min': 15, '30min': 30, '1h': 60,
                        '2h': 120, '4h': 240, '8h': 480, '16h': 960, '1d': 1440
                    }
                    tf_minutes = tf_map.get(timeframe, 60)
                    
                    # Create and test strategy
                    strategy = self.create_strategy_instance(strategy_dict, tf_minutes)
                    metrics, backtest_results = self.test_strategy(strategy, data)
                    
                    if metrics:
                        # Calculate CPS score
                        cps_result = scorer.calculate_cps(metrics)
                        cps_score = cps_result['cps'] if isinstance(cps_result, dict) else cps_result
                        
                        # Store result
                        result = {
                            'strategy_id': strategy_dict['id'],
                            'strategy_name': strategy_dict['name'],
                            'timeframe': timeframe,
                            'cps': float(cps_score),
                            'total_return': float(metrics.total_return),
                            'max_drawdown': float(metrics.max_drawdown),
                            'sharpe_ratio': float(metrics.sharpe_ratio),
                            'num_trades': int(metrics.total_trades),
                            'win_rate': float(metrics.win_rate)
                        }
                        self.results.append(result)
                    
                    progress.update(task_progress, advance=1)
        
        # Sort and display results
        self.results.sort(key=lambda x: x['cps'], reverse=True)
        self.display_results()
        
        # Save results
        self.save_results()
        
        # Report insights
        self.report_insights()
        
        return True
    
    def display_results(self):
        """Display top results"""
        table = Table(title=f"Agent {self.agent_id:02d} Top Results - Stage {self.stage}")
        
        table.add_column("Rank", style="cyan", width=6)
        table.add_column("Strategy", style="white", width=25)
        table.add_column("Timeframe", style="yellow", width=10)
        table.add_column("CPS", style="green", width=8)
        table.add_column("Return", style="blue", width=10)
        table.add_column("Sharpe", style="magenta", width=8)
        
        for i, result in enumerate(self.results[:10], 1):
            table.add_row(
                str(i),
                result['strategy_name'][:25],
                result['timeframe'],
                f"{result['cps']:.1f}",
                f"{result['total_return']*100:.2f}%",
                f"{result['sharpe_ratio']:.2f}"
            )
        
        self.console.print(table)
    
    def save_results(self):
        """Save results to JSON file"""
        output_data = {
            'agent_id': self.agent_id,
            'stage': self.stage,
            'timestamp': datetime.now().isoformat(),
            'total_tests': len(self.results),
            'top_strategies': self.results[:20],  # Top 20
            'insights': self.generate_insights()
        }
        
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        self.console.print(f"\n[green]✓ Results saved to {self.output_file}[/green]")
    
    def generate_insights(self):
        """Generate insights from testing"""
        if not self.results:
            return {}
        
        # Analyze patterns
        top_10 = self.results[:10]
        
        # Best timeframes
        timeframe_scores = {}
        for r in top_10:
            tf = r['timeframe']
            if tf not in timeframe_scores:
                timeframe_scores[tf] = []
            timeframe_scores[tf].append(r['cps'])
        
        best_timeframe = max(timeframe_scores.items(), key=lambda x: sum(x[1])/len(x[1]))[0] if timeframe_scores else None
        
        # Strategy types performing well
        strategy_types = {}
        for r in top_10:
            # Categorize by name patterns
            name = r['strategy_name']
            if 'Momentum' in name or 'RSI' in name or 'Stoch' in name:
                stype = 'momentum'
            elif 'Trend' in name or 'MA' in name or 'EMA' in name:
                stype = 'trend'
            elif 'Wave' in name or 'Oscillator' in name:
                stype = 'oscillator'
            else:
                stype = 'other'
            
            if stype not in strategy_types:
                strategy_types[stype] = 0
            strategy_types[stype] += 1
        
        # Pairing suggestions
        pairings = []
        if len(top_10) >= 2:
            # Suggest pairing different types
            for i in range(min(3, len(top_10)-1)):
                pairings.append({
                    'strategy1': top_10[i]['strategy_name'],
                    'strategy2': top_10[i+1]['strategy_name'],
                    'combined_cps_estimate': (top_10[i]['cps'] + top_10[i+1]['cps']) / 2
                })
        
        return {
            'best_timeframe': best_timeframe,
            'dominant_strategy_type': max(strategy_types.items(), key=lambda x: x[1])[0] if strategy_types else None,
            'average_top10_cps': sum(r['cps'] for r in top_10) / len(top_10) if top_10 else 0,
            'best_strategy': top_10[0]['strategy_name'] if top_10 else None,
            'best_cps': top_10[0]['cps'] if top_10 else 0,
            'pairing_suggestions': pairings[:2]
        }
    
    def report_insights(self):
        """Report key insights"""
        insights = self.generate_insights()
        
        self.console.print("\n[bold yellow]Key Insights:[/bold yellow]")
        self.console.print(f"  Best performing: {insights.get('best_strategy', 'N/A')} (CPS: {insights.get('best_cps', 0):.1f})")
        self.console.print(f"  Best timeframe: {insights.get('best_timeframe', 'N/A')}")
        self.console.print(f"  Dominant type: {insights.get('dominant_strategy_type', 'N/A')}")
        
        if insights.get('pairing_suggestions'):
            self.console.print("\n[yellow]Ensemble Suggestions:[/yellow]")
            for p in insights['pairing_suggestions']:
                self.console.print(f"  • {p['strategy1']} + {p['strategy2']}")


def main():
    """Main entry point for subagent"""
    parser = argparse.ArgumentParser(description='Subagent Runner for Strategy Optimization')
    parser.add_argument('--agent-id', type=int, required=True, help='Agent ID (1-20)')
    parser.add_argument('--stage', type=int, default=1, help='Stage (1-3)')
    
    args = parser.parse_args()
    
    if args.agent_id < 1 or args.agent_id > 20:
        print(f"Error: Agent ID must be between 1 and 20 (got {args.agent_id})")
        sys.exit(1)
    
    if args.stage < 1 or args.stage > 3:
        print(f"Error: Stage must be between 1 and 3 (got {args.stage})")
        sys.exit(1)
    
    # Run the subagent
    runner = SubagentRunner(args.agent_id, args.stage)
    success = runner.run()
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())