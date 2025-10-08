#!/usr/bin/env bash
# Simple batch runner for the new pro1 strategies across common time windows.
# Requires your existing CLI entry point. Adjust paths as needed.
set -euo pipefail

STRATS=("RegimeSwitchingPro1" "MeanReversionZScorePro1" "LiquidityAwareBreakoutPro1")
DAYS=(7 30 90 180 365 730)

for S in "${STRATS[@]}"; do
  for D in "${DAYS[@]}"; do
    echo ">>> Backtesting $S for last $D days"
    python pulsechain_trading_bot.py --backtest --days "$D" --strategy "$S" || true
  done
done
