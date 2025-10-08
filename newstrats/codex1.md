# Codex1 Optimization Log – Donchian Champion Dynamic Refresh

## Objective
- Analyse the existing high-performing strategies documented in `newstrats/local.md`.
- Improve at least one strategy's cost-aware performance (total return and recency windows) without regressing any of the recorded baselines in `strategy_performance_summary.csv`.
- Respect the production swap-cost model stored in `swap_cost_cache.json` and use the maintained evaluation tooling.

## Dataset & Tooling
- **Dataset:** `data/pdai_ohlcv_dai_730day_5m.csv` (5 min HEX/PDAI candles).
- **Cost model:** `swap_cost_cache.json` (bucketed loss rates + gas adjustments).
- **Evaluation:** `python scripts/evaluate_vost_strategies.py --swap-cost-cache swap_cost_cache.json --trade-size {5000|10000|25000}`.
- **Segment checks:** Custom Python snippets (see below) to measure last‑90/last‑30 day returns off the returned equity curve so that indicators still use the full history.

## Strategy Reviewed
- **Target:** `DonchianChampionDynamicStrategy` (Champion v4) defined in `strategies/donchian_champion_strategy.py`.
- **Baseline defaults:**
  - `dd_base=0.16`
  - `gain_weight=0.10`
  - `dd_max=0.45`
  - Other parameters identical to those documented in `newstrats/local.md`.

## Tuning Process
1. **Manual grid-search** (Python REPL) while importing `run_strategy` from `scripts/evaluate_vost_strategies` so costs/trade execution exactly matched production logic.
2. Explored ranges:
   - `dd_base` ∈ {0.14, 0.16, 0.18}
   - `gain_weight` ∈ {0.08, 0.10, 0.12}
   - `dd_max` ∈ {0.40, 0.45, 0.50}
   - `entry_buffer_frac` ∈ {0, 0.002, …, 0.01}
3. Kept `dd_k=0.5`, `dd_min=0.10`, `atr_days=1` constant to preserve responsiveness and avoid drawdown cliffs described in earlier reports.
4. `entry_buffer_frac>0` produced lower net returns because most breakout thrusts already gap substantially above the previous high, so every basis-point of extra buffer removed high-conviction trades but did not eliminate the choppy ones (confirmed by drop in total return and trade count).
5. The best combination across all three trade-size buckets was:
   - `dd_base=0.14`
   - `gain_weight=0.12`
   - `dd_max=0.40`
   - `entry_buffer_frac=0`
   This configuration allows the dynamic drawdown to tighten faster after shallow rallies (lower base) but relax more aggressively once trades move deep in-the-money (higher gain weight), all while capping the stop loosening at 40 % to protect against prolonged bleed-outs.

## Code Change
- Updated the default parameter block for `DonchianChampionDynamicStrategy` inside `strategies/donchian_champion_strategy.py` to the tuned values above so every consumer (batch evaluator, CSV exporter, notebooks) automatically benefits from the improved configuration.

## Results Summary
| Trade Size | Baseline Total Return | New Total Return | Δ | Notes |
|-----------:|----------------------:|-----------------:|---:|-------|
| 5 k DAI    | 5,434.7 % | **7,148.3 %** | +1,713.6 pp | Sharpe ↑ 2.67 → 2.83; MaxDD improves to −46.97 %. |
| 10 k DAI   | 3,046.6 % | **4,020.8 %** | +974.2 pp  | Maintains 32 trades; slightly deeper DD (−51.9 %) but still better than prior (−53.2 %). |
| 25 k DAI   | 598.8 %  | **815.1 %**   | +216.3 pp  | Turnover unchanged; cost drag greatly reduced. |

**Recent window check (5 k DAI bucket):**
- Last 90 days: −34.78 % (improved from −37.79 %).
- Last 30 days: −21.17 % (unchanged; no trades fired in that window).

_All calculations performed through `run_strategy` with the swap-cost lookups, matching the repo’s production workflow._

## Reproduction Steps
1. Pull latest repo state and ensure `pip install -r requirements.txt`.
2. Run `python scripts/evaluate_vost_strategies.py --swap-cost-cache swap_cost_cache.json --trade-size 5000` to confirm the 7,148 % headline return.
3. Optional: repeat with `--trade-size 10000` and `25000` for cross-bucket validation.
4. To replicate the segment checks, reuse the helper snippet embedded in this document (see appendix).

## Appendix – Helper Snippet
```python
from pathlib import Path
import pandas as pd
from strategies.donchian_champion_strategy import DonchianChampionDynamicStrategy
from scripts.evaluate_vost_strategies import load_dataset, load_swap_costs, run_strategy

params = {'dd_base': 0.14, 'gain_weight': 0.12, 'dd_max': 0.40}
trade_size = 5000

data = load_dataset(Path('data/pdai_ohlcv_dai_730day_5m.csv'))
costs = load_swap_costs(Path('swap_cost_cache.json'))
stats = run_strategy(DonchianChampionDynamicStrategy(params), data, swap_costs=costs, trade_notional=trade_size)

def segment_return(equity, timestamps, days):
    cutoff = timestamps.iloc[-1] - pd.Timedelta(days=days)
    mask = timestamps >= cutoff
    if not mask.any():
        return 0.0
    idx = mask.idxmax()
    start_equity = equity[idx-1] if idx > 0 else 1.0
    return (equity[-1] / start_equity - 1.0) * 100

print('Total return %:', stats['total_return_pct'])
print('Last 90d %:', segment_return(stats['equity_curve'], data['timestamp'], 90))
print('Last 30d %:', segment_return(stats['equity_curve'], data['timestamp'], 30))
```

## Next Ideas
- Explore combining the dynamic trail with a soft profit take (e.g., 60 % of realised ATR gains) to free capital sooner during prolonged sideways phases without sacrificing the big winners.
- Investigate regime gating on the reversion strategies (`CSMARevertStrategy`) to reduce capital lock-up during sustained downtrends while preserving their outsized crash-rebound profits.
- Continue evaluating volume multipliers for MultiWeekBreakout, but add adaptive volatility filters (ATR quantiles) to avoid the stalled periods that still generate residual drawdowns.
