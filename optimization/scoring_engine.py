#!/usr/bin/env python3
"""
Composite Performance Scoring (CPS) Engine
Evaluates trading strategies using multiple weighted metrics
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class StrategyMetrics:
    """Container for strategy performance metrics"""
    strategy_name: str
    timeframe: str
    total_return: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    recovery_factor: float
    trade_frequency: float  # trades per month
    reversal_catches: int
    reversal_opportunities: int
    
    
class CompositePerformanceScorer:
    """
    Calculates Composite Performance Score (CPS) for trading strategies
    
    Weights:
    - 30% Profit Score
    - 25% Capital Preservation Score
    - 20% Risk-Adjusted Score
    - 15% Trade Activity Score (profit-aware)
    - 10% Trend Detection Score
    """
    
    def __init__(self, buy_hold_return: float, market_condition: str = "mixed"):
        """
        Initialize scorer with market context
        
        Args:
            buy_hold_return: Buy & Hold return for the period
            market_condition: "bull", "bear", "mixed", or "sideways"
        """
        self.buy_hold_return = buy_hold_return
        self.market_condition = market_condition
        
        # Adjust expectations based on market
        if market_condition == "bear":
            self.profit_threshold = 0  # Any profit is good in bear market
            self.drawdown_tolerance = 0.15  # Allow more drawdown
        elif market_condition == "sideways":
            self.profit_threshold = buy_hold_return * 0.5
            self.drawdown_tolerance = 0.10
        else:  # bull or mixed
            self.profit_threshold = buy_hold_return
            self.drawdown_tolerance = 0.20
            
    def calculate_cps(self, metrics: StrategyMetrics) -> Dict:
        """
        Calculate Composite Performance Score
        
        Returns:
            Dict with overall CPS and component scores
        """
        profit_score = self._calculate_profit_score(metrics.total_return)
        preservation_score = self._calculate_preservation_score(metrics.max_drawdown)
        risk_score = self._calculate_risk_adjusted_score(metrics.sharpe_ratio, metrics.sortino_ratio)
        activity_score = self._calculate_activity_score(
            metrics.trade_frequency, 
            metrics.total_return,
            metrics.win_rate
        )
        trend_score = self._calculate_trend_detection_score(
            metrics.reversal_catches,
            metrics.reversal_opportunities
        )
        
        # Weighted combination
        cps = (
            0.30 * profit_score +
            0.25 * preservation_score +
            0.20 * risk_score +
            0.15 * activity_score +
            0.10 * trend_score
        )

        # Guardrail: if a strategy executes zero trades, it should not rank as a top performer.
        # Cap CPS for zero-trade results so preservation doesn't dominate.
        if metrics.total_trades == 0:
            cps = min(cps, 50.0)
        
        return {
            'cps': round(cps, 2),
            'profit_score': round(profit_score, 2),
            'preservation_score': round(preservation_score, 2),
            'risk_adjusted_score': round(risk_score, 2),
            'activity_score': round(activity_score, 2),
            'trend_detection_score': round(trend_score, 2),
            'metrics': metrics
        }
        
    def _calculate_profit_score(self, strategy_return: float) -> float:
        """Calculate profit component (30% weight) with stable behavior for flat/negative benchmarks.

        strategy_return and buy_hold_return are decimals (e.g., 0.12 for 12%).
        """
        bh = float(self.buy_hold_return)

        # Handle flat or near-zero benchmark explicitly to avoid divide-by-small and sign issues
        if abs(bh) < 1e-3:
            # Flat market
            if strategy_return >= 0.01:  # >= +1%
                return min(100, 80 + (strategy_return * 100))
            elif strategy_return > 0:
                return 60 + (strategy_return * 200)
            elif strategy_return == 0:
                return 40  # matched flat market (not impressive)
            else:
                return max(0, 40 + (strategy_return * 200))

        # Normal cases with meaningful (possibly negative) benchmark
        if strategy_return >= bh * 1.5:
            return 100
        elif strategy_return >= bh:
            # Linear scale 80..100 using absolute benchmark magnitude
            ratio = (strategy_return - bh) / (abs(bh) * 0.5)
            return 80 + min(20, 20 * ratio)
        elif strategy_return >= 0:
            # Profitable but below benchmark
            return 50 + (30 * strategy_return / max(abs(bh), 0.01))
        else:
            # Losing strategy - penalize proportional to loss magnitude
            return max(0, 50 + (strategy_return * 100))
            
    def _calculate_preservation_score(self, max_drawdown: float) -> float:
        """Calculate capital preservation component (25% weight)"""
        drawdown_abs = abs(max_drawdown)
        
        if drawdown_abs < 0.05:
            return 100
        elif drawdown_abs < 0.10:
            return 80
        elif drawdown_abs < 0.15:
            return 60
        elif drawdown_abs < 0.20:
            return 40
        else:
            return max(0, 40 - (drawdown_abs - 0.20) * 200)
            
    def _calculate_risk_adjusted_score(self, sharpe: float, sortino: float) -> float:
        """Calculate risk-adjusted return component (20% weight)"""
        # Use the better of Sharpe or Sortino
        best_ratio = max(sharpe, sortino * 0.8)  # Sortino slightly discounted
        
        if best_ratio > 2.0:
            return 100
        elif best_ratio > 1.5:
            return 80
        elif best_ratio > 1.0:
            return 60
        elif best_ratio > 0.5:
            return 40
        else:
            return max(0, best_ratio * 40)
            
    def _calculate_activity_score(self, trades_per_month: float, 
                                 strategy_return: float, win_rate: float) -> float:
        """
        Calculate trade activity component (15% weight)
        Profit-aware: profitable strategies can trade as much as they want!
        """
        if trades_per_month == 0:
            return 0  # Completely passive = always bad
            
        if strategy_return > 0:  # PROFITABLE strategies
            if trades_per_month < 3:
                return 70  # Profitable but too conservative
            elif trades_per_month <= 10:
                return 100  # Profitable with moderate activity
            elif trades_per_month <= 30:
                return 100  # Still perfect - profitable high-frequency
            else:  # 30+ trades
                # Even with 100+ trades, if profitable, still good!
                if win_rate > 0.6:  # High win rate
                    return 100  # Perfect - found a real edge
                elif win_rate > 0.5:  # Decent win rate
                    return 90
                else:  # Low win rate but still profitable (big winners)
                    return 80
        else:  # LOSING strategies
            if trades_per_month < 3:
                return 40  # Few losses is better than many
            elif trades_per_month <= 10:
                return 30  # Moderate activity with losses
            elif trades_per_month <= 30:
                return 20  # Active losing
            else:  # 30+ trades while losing
                return max(0, 20 - (trades_per_month - 30) * 0.5)
                
    def _calculate_trend_detection_score(self, catches: int, opportunities: int) -> float:
        """Calculate trend reversal detection component (10% weight)"""
        if opportunities == 0:
            return 50  # No opportunities to judge
            
        detection_rate = catches / opportunities
        
        if detection_rate > 0.8:
            return 100
        elif detection_rate > 0.6:
            return 80
        elif detection_rate > 0.4:
            return 60
        elif detection_rate > 0.2:
            return 40
        else:
            return detection_rate * 200
            

class StrategyRanker:
    """Ranks strategies based on CPS and other criteria"""
    
    def __init__(self, scorer: CompositePerformanceScorer):
        self.scorer = scorer
        self.results = []
        
    def add_result(self, metrics: StrategyMetrics):
        """Add a strategy result for ranking"""
        score_data = self.scorer.calculate_cps(metrics)
        self.results.append(score_data)
        
    def get_rankings(self, top_n: int = None) -> pd.DataFrame:
        """
        Get ranked strategies
        
        Returns:
            DataFrame with rankings and all scores
        """
        if not self.results:
            return pd.DataFrame()
            
        df = pd.DataFrame(self.results)
        df = df.sort_values('cps', ascending=False)
        
        if top_n:
            df = df.head(top_n)
            
        df['rank'] = range(1, len(df) + 1)
        
        return df
        
    def get_diverse_selection(self, total_count: int = 60) -> List[Dict]:
        """
        Get diverse selection ensuring different strategy types
        
        Includes:
        - Top profit makers
        - Best capital preservers
        - Best risk-adjusted
        - Most active traders
        - Best trend detectors
        - Wildcards
        """
        if len(self.results) <= total_count:
            return self.results
            
        df = pd.DataFrame(self.results)
        
        selected = set()
        
        # Top 10 by overall CPS
        top_cps = df.nlargest(10, 'cps')
        selected.update(top_cps.index)
        
        # Top 10 profit makers
        top_profit = df.nlargest(10, 'profit_score')
        selected.update(top_profit.index)
        
        # Top 10 capital preservers
        top_preserve = df.nlargest(10, 'preservation_score')
        selected.update(top_preserve.index)
        
        # Top 10 risk-adjusted
        top_risk = df.nlargest(10, 'risk_adjusted_score')
        selected.update(top_risk.index)
        
        # Top 10 active traders (with positive return)
        active_profitable = df[df['metrics'].apply(lambda x: x.total_return > 0)]
        if len(active_profitable) > 0:
            top_active = active_profitable.nlargest(10, 'activity_score')
            selected.update(top_active.index)
        
        # Top 10 trend detectors
        top_trend = df.nlargest(10, 'trend_detection_score')
        selected.update(top_trend.index)
        
        # Fill remainder with next best CPS scores
        remaining = total_count - len(selected)
        if remaining > 0:
            unused = df[~df.index.isin(selected)]
            if len(unused) > 0:
                wildcards = unused.nlargest(min(remaining, len(unused)), 'cps')
                selected.update(wildcards.index)
        
        return df.loc[list(selected)].to_dict('records')


def calculate_metrics_from_backtest(backtest_result: Dict, timeframe: str) -> StrategyMetrics:
    """
    Convert backtest results to StrategyMetrics
    
    Args:
        backtest_result: Result from BacktestEngine
        timeframe: Timeframe used for the test
    """
    trades = backtest_result.get('trades', [])
    
    # Calculate trade statistics
    total_trades = len(trades)
    winning_trades = sum(1 for t in trades if t.get('profit', 0) > 0)
    losing_trades = total_trades - winning_trades
    
    wins = [t['profit'] for t in trades if t.get('profit', 0) > 0]
    losses = [abs(t['profit']) for t in trades if t.get('profit', 0) < 0]
    
    avg_win = np.mean(wins) if wins else 0
    avg_loss = np.mean(losses) if losses else 0
    
    # Calculate profit factor
    total_wins = sum(wins) if wins else 0
    total_losses = sum(losses) if losses else 0
    profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
    
    # Estimate trade frequency (trades per month)
    days_tested = backtest_result.get('days_tested', 30)
    trade_frequency = (total_trades / days_tested) * 30 if days_tested > 0 else 0
    
    # TODO: Add reversal detection logic
    reversal_catches = 0
    reversal_opportunities = 0

    return StrategyMetrics(
        strategy_name=backtest_result.get('strategy_name', 'Unknown'),
        timeframe=timeframe,
        total_return=backtest_result.get('total_return_pct', 0) / 100,
        max_drawdown=backtest_result.get('max_drawdown_pct', 0) / 100,
        sharpe_ratio=backtest_result.get('sharpe_ratio', 0),
        sortino_ratio=backtest_result.get('sortino_ratio', 0),
        total_trades=total_trades,
        winning_trades=winning_trades,
        losing_trades=losing_trades,
        win_rate=backtest_result.get('win_rate_pct', 0) / 100,
        avg_win=avg_win,
        avg_loss=avg_loss,
        profit_factor=profit_factor,
        recovery_factor=backtest_result.get('recovery_factor', 0),
        trade_frequency=trade_frequency,
        reversal_catches=reversal_catches,
        reversal_opportunities=reversal_opportunities
    )


class DynamicUtilityScorer:
    """
    Profit-led, drawdown-aware continuous utility scorer with soft reliability and
    optional OOS weighting. No hard floors or caps; all penalties are smooth.

    Primary utility (decimals):
        U = R / (1 + lambda_dd * (DD ** dd_power))

    Reliability weighting (optional, soft):
        w(n) = sqrt(n / (n + k))   with k=0 to disable

    OOS blend (optional; if only one window is available, use score() directly):
        Score_final = alpha_oos * (U_IS * w(n_IS)) + (1 - alpha_oos) * (U_OOS * w(n_OOS))
    """

    def __init__(
        self,
        lambda_dd: float = 2.0,
        dd_power: float = 1.25,
        epsilon: float = 0.01,
        reliability_k: float = 5.0,
        alpha_oos: float = 0.25,
    ) -> None:
        self.lambda_dd = float(lambda_dd)
        self.dd_power = float(dd_power)
        self.epsilon = float(epsilon)
        self.reliability_k = float(reliability_k)
        self.alpha_oos = float(alpha_oos)

    def _utility(self, return_dec: float, dd_dec: float) -> float:
        r = float(return_dec)
        dd = abs(float(dd_dec))
        denom = 1.0 + self.lambda_dd * (dd ** self.dd_power)
        return r / denom if denom != 0 else 0.0

    def _reliability(self, n_trades: int) -> float:
        n = max(0.0, float(n_trades))
        k = max(0.0, self.reliability_k)
        if k <= 1e-12:
            return 1.0
        return (n / (n + k)) ** 0.5

    def score(self, metrics: StrategyMetrics) -> Dict[str, float]:
        """Compute utility score for a single window (no OOS split)."""
        u = self._utility(metrics.total_return, metrics.max_drawdown)
        w = self._reliability(metrics.total_trades)
        score = u * w
        # Diagnostics: PDR and Soft-Calmar-like (for reporting only)
        pdr = metrics.total_return / (abs(metrics.max_drawdown) + self.epsilon)
        # For Soft-Calmar diagnostic, we need a period length; not available here. Skip CAGR.
        return {
            'score': float(score),
            'utility': float(u),
            'weight_reliability': float(w),
            'pdr': float(pdr),
        }

    def score_oos(
        self,
        is_metrics: StrategyMetrics,
        oos_metrics: StrategyMetrics,
    ) -> Dict[str, float]:
        """Compute OOS-weighted utility if IS/OOS splits are provided."""
        is_part = self.score(is_metrics)
        oos_part = self.score(oos_metrics)
        s = self.alpha_oos * is_part['score'] + (1.0 - self.alpha_oos) * oos_part['score']
        return {
            'score_final': float(s),
            'is_score': float(is_part['score']),
            'oos_score': float(oos_part['score']),
            'is_weight': float(is_part['weight_reliability']),
            'oos_weight': float(oos_part['weight_reliability']),
            'is_utility': float(is_part['utility']),
            'oos_utility': float(oos_part['utility']),
            'pdr_is': float(is_part['pdr']),
            'pdr_oos': float(oos_part['pdr']),
        }


if __name__ == "__main__":
    # Test the scoring engine
    print("Testing Composite Performance Scorer...")
    
    # Create scorer for bearish market (1.05% Buy & Hold)
    scorer = CompositePerformanceScorer(buy_hold_return=0.0105, market_condition="bear")
    
    # Test strategy that preserved capital in downtrend
    test_metrics = StrategyMetrics(
        strategy_name="TestStrategy",
        timeframe="1h",
        total_return=0.02,  # 2% return in bear market is good!
        max_drawdown=-0.08,  # 8% drawdown
        sharpe_ratio=1.2,
        sortino_ratio=1.5,
        total_trades=15,
        winning_trades=9,
        losing_trades=6,
        win_rate=0.6,
        avg_win=0.005,
        avg_loss=0.003,
        profit_factor=1.67,
        recovery_factor=0.25,
        trade_frequency=15,  # 15 trades per month
        reversal_catches=3,
        reversal_opportunities=5
    )
    
    result = scorer.calculate_cps(test_metrics)
    print(f"\nTest Strategy CPS: {result['cps']}")
    print(f"Component Scores:")
    for key, value in result.items():
        if key not in ['cps', 'metrics']:
            print(f"  {key}: {value}")
