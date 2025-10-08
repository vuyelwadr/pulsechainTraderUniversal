
# Backtesting & Iteration — pro1

This guide shows how to **run, evaluate, and iterate** the `pro1` strategies
until they outperform your baselines by **Return %**, while respecting
`swap_cost_cache.json`.

## One-time setup
1. Ensure the `pro1` strategies are registered (see `pro1.md`).
2. (Optional) Set baseline strategies (comma-separated):
   ```bash
   export PRO1_BASELINE_STRATS="YourExistingStrat1,YourExistingStrat2"
   ```

## Run the search
```bash
python scripts/backtest_runner_pro1.py \
  --bot-script pulsechain_trading_bot.py \
  --python-exe python \
  --timeframes 7,30,90,180,365,730 \
  --samples-per-strategy 120 \
  --out results_pro1.csv
```

- The runner passes parameters to strategies via env vars (`PRO1_PARAMS_<ClassName>`).
- Grids are defined in `configs/grid_pro1.json`. Edit freely.
- It parses stdout heuristically for `Return %`, `Max Drawdown`, `Profit Factor`, `Trades`.

## Inspect results
```bash
# Top few configs per strategy:
cat best_pro1.md

# Full comparison against baselines with filters:
python scripts/analyze_results_pro1.py --csv results_pro1.csv --out report_pro1.md --min_pf 1.1 --min_wins 3
```

## Acceptance criteria (suggested)
- Mean Profit Factor across timeframes ≥ **1.10**.
- Beats the best baseline return in **≥ 3** timeframes.
- Avoid configs with pathological drawdowns even if return is high.

## Keep iterating
- Increase `--samples-per-strategy`.
- Narrow ranges in `configs/grid_pro1.json` around winners.
- Consider smaller `min_edge_multiple` if your real slippage is lower than the cache (or higher if it’s worse).

> All signals remain **cost-gated** via `utils/cost_model_pro1.py`, so cheap noise trades should be filtered automatically.
