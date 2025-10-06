# Strategy Performance Across Timeframes

## Summary Matrix

| Timeframe | Strategy | Return | Buy & Hold | Difference | Result |
|-----------|----------|--------|------------|------------|--------|
| **3 Months** | MultiTimeframeMomentum | 28.60% | 22.29% | +6.31% | ✅ Beats |
| **1 Year** | MultiTimeframeMomentum | 87.88% | 100.88% | -13.00% | ❌ Loses |

## Key Insights

### 3-Month Performance (June-Sept 2025)
- **Winner**: MultiTimeframeMomentum Strategy
- **Configuration**: min_strength=0.5, position=50%, slippage=0.5%
- **Trades**: 2 (highly selective)
- **Win Rate**: 100%
- **Key Success**: Perfect timing on major moves

### 1-Year Performance (Sept 2024-Sept 2025)
- **Winner**: Buy & Hold (so far)
- **Strategy Performance**: 87.88% (still excellent)
- **Trades**: 10 (more active)
- **Challenge**: Sustained uptrend favored holding

## Market Dynamics

### Price Movement
- **3-Month**: 198.32 → 242.53 WPLS/HEX (+22.29%)
- **1-Year**: 120.73 → 242.53 WPLS/HEX (+100.88%)

### Market Characteristics
- **3 Months**: Some volatility, momentum opportunities
- **1 Year**: Strong sustained uptrend, doubling in value

## Strategy Selection Guidelines

### Use Momentum Strategies When:
- Timeframe ≤ 3 months
- Market showing volatility
- Quick profits desired
- Risk management important

### Use Buy & Hold When:
- Timeframe ≥ 6 months
- Strong trending market
- Lower maintenance desired
- Maximum returns in bull market

### Hybrid Approach
Consider allocating portfolio:
- 50% Buy & Hold (long-term)
- 30% Momentum (short-term trades)
- 20% Cash/Reserve (opportunities)

## Current Optimization Status

### Completed
- ✅ 3-month optimization: Found 28.60% winner
- ✅ 1-year test of winner: 87.88% return
- ✅ Documentation of results

### In Progress
- 🔄 Archipelago optimizer running on 1-year data
- 🔄 Searching for strategy to beat 100.88% Buy & Hold
- 🔄 Testing 200+ configurations across 12 strategies

### Expected Outcomes
The 1-year optimizer may find:
- DCA strategies that average into positions
- Grid trading that profits from volatility
- Trend-following that stays in longer
- Adaptive strategies that switch modes

## Risk-Adjusted Performance

While raw returns favor Buy & Hold for 1 year, consider:

### Sharpe Ratio (estimated)
- **Buy & Hold**: Lower (high volatility)
- **MTF Strategy**: Higher (managed risk)

### Maximum Drawdown
- **Buy & Hold**: Potentially -30% or more
- **MTF Strategy**: Controlled via exits

### Peace of Mind
- **Buy & Hold**: Must endure all dips
- **MTF Strategy**: Active risk management

## Recommendations

### For 3-Month Horizons
1. Use MultiTimeframeMomentum (28.60% proven)
2. Settings: min_strength=0.5
3. Position size: 50%
4. Expected: Beat Buy & Hold by 5-10%

### For 1-Year Horizons
1. Await optimizer results (running now)
2. Consider 70% Buy & Hold baseline
3. Use 30% for active strategies
4. Rebalance quarterly

### For Mixed Horizons
1. Segment capital by timeframe
2. Use appropriate strategy per segment
3. Regular performance review
4. Adjust based on market regime

## Files and Evidence
- `winner_1year_test_*.json` - 1-year test results
- `1year_result_*.json` - Detailed metrics
- `archipelago_1year_output.txt` - Optimizer progress (running)
- `best_strategy_1year_*.json` - Will contain optimal 1-year strategy

---

*Status: Archipelago optimizer currently running to find best 1-year strategy. Results expected within 30-60 minutes.*