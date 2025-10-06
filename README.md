# 🚀 HEX Trading Bot + Data/Optimization Pipeline (PulseChain)

An end‑to‑end stack for HEX trading on PulseChain:
- Real on‑chain data collection from PulseX (no synthetic data)
- Walk‑forward (feed‑forward) strategy optimization on a unified 2‑year dataset
- Modular strategy system + backtester and demo live trading

## 🎯 Features

- **Demo Mode Trading**: Safe testing without real money
- **Backtesting Engine**: Test strategies on historical data (OOS‑led metrics)
- **Modular Strategy System**: Easy to add/modify trading strategies
- **Real-time HTML Reports**: Interactive web-based dashboards
- **CLI & Web Interface**: Both command-line and browser access
- **Moving Average Crossover**: Built-in trend-following strategy
- **PulseX Integration**: Direct integration with PulseX DEX
- **Walk‑Forward Optimization**: OOS‑led selection with profit‑first, drawdown‑aware scoring
- **Caching for Speed**: Persistent caches for block timestamps, Sync reserves, and swap events (coverage‑aware)

## 🧭 Real‑Data Policy (No Synthetic Data)

- All prices, volumes, reserves, and candles are derived strictly from on‑chain PulseX Swap/Sync events.
- If a candle has no Sync inside its interval, reserve columns stay NaN (no forward‑fill, no interpolation).
- Backtests use real prices; “demo mode” only simulates execution without spending real funds.

## 📋 Requirements

- Python 3.8+
- PulseChain RPC access
- Dependencies in `requirements.txt`

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or download the project
cd pulsechainTraderUniversal

# Install dependencies
pip install -r requirements.txt

# Create directories
mkdir -p data html_reports data/.cache logs
```

### 2. Configuration

The `.env` file is already set up for demo mode. No additional configuration needed for testing.

### 3. Collect Real OHLCV (strict on‑chain)

The collector builds 5‑minute candles from PulseX swaps and attaches strict inside‑candle reserves from Sync events.

```bash
# 2‑year, 5‑minute dataset (recommended unified source for the optimizer)
python collectors/swap_ohlcv_collector.py \
  --days 730 --interval 5m \
  --workers 6 --chunk-size 12000 \
  --pin-rpc-per-worker \
  --log-file logs/collector_2y.log \
  --out data/<asset>_ohlcv_<quote>_730day_5m.csv  # e.g., data/hex_ohlcv_dai_730day_5m.csv

# Quick 1‑day sanity check
python collectors/swap_ohlcv_collector.py --days 1 --interval 5m --workers 4 --chunk-size 20000
```

Output CSV columns (optimizer‑ready):
- `timestamp, open, high, low, close, volume, reserve_<asset>, reserve_<bridge>, block`

Performance notes:
- Persistent caches under `data/.cache/block_ts.sqlite`:
  - `block_ts` (block → timestamp), `sync_reserves` (block → reserves), `swap_events` (decoded swaps), `coverage` (skip getLogs when range fully cached).
- Re‑running overlapping windows is much faster thanks to caches. All cached values are 100% real on‑chain.

### 4. Run Backtest

```bash
# Run 30-day backtest
python bot/hex_trading_bot.py --backtest

# Custom backtest period
python bot/hex_trading_bot.py --backtest --days 7

# Specific strategy
python bot/hex_trading_bot.py --backtest --strategy MovingAverageCrossover
```

### 5. Live Trading (Demo Mode)

```bash
# Start live demo trading
python bot/hex_trading_bot.py --live

# The bot will:
# - Fetch real HEX price data
# - Generate trading signals
# - Execute simulated trades
# - Create real-time HTML reports
```

## 📊 Web Interface

When running backtests or live trading, HTML reports are automatically generated in the `html_reports/` directory:

- **Backtests**: `backtest_[strategy]_[timestamp].html`
- **Live Trading**: `live_trading.html` (auto-refreshes)

Open these files in your browser to see:
- Portfolio performance charts
- Trading signals and execution
- Performance metrics
- Real-time updates

## 🔧 Strategies

### Moving Average Crossover (Default)

The built-in strategy uses:
- **Short MA**: 10 periods (configurable)
- **Long MA**: 30 periods (configurable)
- **Signal Logic**: Buy when short MA crosses above long MA, sell when it crosses below

### Adding Custom Strategies

1. Create a new file in `strategies/` directory
2. Inherit from `BaseStrategy` class
3. Implement `calculate_indicators()` and `generate_signals()` methods
4. Add to bot in `hex_trading_bot.py`

Example:
```python
from strategies.base_strategy import BaseStrategy

class MyStrategy(BaseStrategy):
    def calculate_indicators(self, data):
        # Your indicator calculations
        return data
    
    def generate_signals(self, data):
        # Your signal generation logic
        return data
```

## ⚙️ Configuration Options

Edit `.env` file to customize:

```bash
# Strategy Parameters
MA_SHORT_PERIOD=10        # Short moving average period
MA_LONG_PERIOD=30         # Long moving average period

# Trading Parameters
INITIAL_BALANCE=1000      # Starting balance in DAI
MAX_TRADE_AMOUNT_PCT=0.5  # Max % of balance per trade
SLIPPAGE_TOLERANCE=0.05   # 5% slippage tolerance

# Data Settings
BACKTEST_DAYS=30          # Default backtest period
DATA_FETCH_INTERVAL=60    # Price update interval (seconds)
```

## 📈 Understanding Results

### Backtest Metrics

- **Total Return**: Overall profit/loss percentage
- **Win Rate**: Percentage of profitable trades
- **Sharpe Ratio**: Risk-adjusted return (higher is better)
- **Max Drawdown**: Largest peak-to-trough decline
- **Profit Factor**: Ratio of gross profit to gross loss

### Signal Strength

All strategies return signal strength (0.0 to 1.0):
- **0.6+**: Execute trade
- **0.8+**: Strong signal
- **Below 0.6**: Hold position

## 🛡️ Safety Features

- **Demo Mode Only**: No real money at risk
- **Slippage Protection**: Built-in price impact simulation
- **Position Sizing**: Configurable trade amounts
- **Error Handling**: Robust error recovery
- **Data Validation**: Price and signal validation

## 🔍 Monitoring

### CLI Output
```
Price: 0.00001234 DAI, Signal: buy, Strength: 0.75
DEMO BUY: 4567.8901 HEX at 0.00001234 DAI
```

### HTML Dashboard
- Real-time price updates
- Portfolio value tracking
- Trade execution history
- Strategy performance metrics

## 📁 Project Structure

```
pulsechainTraderUniversal/
├── hex_trading_bot.py      # Main bot class
├── config.py               # Configuration and constants
├── data_handler.py         # Price data management
├── backtest_engine.py      # Backtesting system
├── html_generator.py       # HTML report generation
├── strategies/
│   ├── base_strategy.py    # Strategy base class
│   └── ma_crossover.py     # Moving average strategy
├── data/                   # Collected data + caches
│   ├── <asset>_ohlcv_<quote>_730day_5m.csv   # Unified 2‑year 5m OHLCV (default: hex_ohlcv_dai_730day_5m.csv)
│   └── .cache/
│       └── block_ts.sqlite             # SQLite caches (block_ts, sync_reserves, swap_events, coverage)
├── src/
│   └── pipelines/
│       └── runner.py                   # Multi‑stage walk‑forward optimizer (OOS‑led)
├── scripts/
│   └── aggregate.py                    # Aggregator (CSV + MD leaderboards)
├── html_reports/           # Generated HTML reports
├── .env                    # Configuration file
└── requirements.txt        # Python dependencies
```

## 🎛️ Command Line Options (Bot)

```bash
# Backtest mode
python bot/hex_trading_bot.py --backtest [--days N] [--strategy NAME]

# Live trading mode  
python bot/hex_trading_bot.py --live [--demo]

# Status check
python bot/hex_trading_bot.py

# Help
python bot/hex_trading_bot.py --help
```

## 🧪 Optimization Pipeline (Walk‑Forward Defaults)

The optimizer always uses the unified 2‑year file. “Stage” controls how many winners to keep and report output paths — not the dataset.

- Timeframes (always): `5min, 15min, 30min, 1h, 4h, 8h, 16h, 1d`
- Objective (default): Utility `U = Return / (1 + λ · DD^p)` (profit‑led, drawdown‑aware)
- OOS‑led selection: mean OOS utility across walk‑forward folds

Walk‑forward per stage (recency‑biased windows):
- Stage 30d: last 90d window; folds train 45d → OOS 15d, step 15d
- Stage 90d: last 270d window; folds train 120d → OOS 30d, step 30d
- Stage 1y: last 365d window; folds train 180d → OOS 30d, step 30d

Run examples:
```bash
# Stage 90d: keep top 40 strategies by OOS score
python -m optimization.runner \
  --stage 90d --top-n2 40 \
  --strategies-file strategies_all.json \
  --workers 12 --calls 60

# Full 3‑stage run (top‑N per stage)
python -m optimization.runner \
  --top-n1 60 --top-n2 40 --top-n3 5 \
  --workers 12 --calls 60

# Gather all strategies and launch the full multi-objective marathon in background
python - <<'PY'
import json
from optimization.runner import collect_strategies
json.dump(collect_strategies(), open('strategies_all.json','w'))
PY

nohup python optimization/runner.py \
  --stage all \
  --strategies-file strategies_all.json \
  --calls 180 \
  --workers 0 \
  --objectives mar,utility,cps,profit_biased,cps_v2,cps_v2_profit_biased \
  --out-dir reports/full_optimizer_run \
  > logs/full_optimizer_run.log 2>&1 &

tail -f logs/full_optimizer_run.log   # monitor progress (Ctrl+C to stop tail)
```

Outputs under `reports/optimizer_pipeline_<ts>_[stage]/`:
- Per strategy/timeframe JSON: includes `folds` (IS/OOS ranges + scores), `selected_params`, final `score` (mean OOS utility)
- `stage_aggregate.csv`: objective‑aware columns (objective, score, utility, pdr, draws, trades)
- `stage_best.csv`: best timeframe per strategy
- `stage_report.md`: human‑readable leaderboards (Top by OOS score, Top by total return, Top by CPS for legacy)

Tip: `--calls` is per fold; runtime ≈ strategies × timeframes × folds × calls. Use lower calls (e.g., 10–30) for quick triage; increase for deeper stages.

## 📊 Token Information

- **HEX Contract**: `0x2b591e99afE9f32eAA6214f7B7629768c40Eeb39`
- **WPLS Contract**: `0xA1077a294dDE1B09bB078844df40758a5D0f9a27`
- **DAI Contract**: `0xefD766cCb38EaF1dfd701853BFCe31359239F305`
- **PulseX Router**: `0x165C3410fC91EF562C50559f7d2289fEbed552d9`
- **Trading Route**: HEX → WPLS → DAI (uses HEX/WPLS + WPLS/DAI pools)

## ⚠️ Disclaimers

- **Educational Purpose**: This bot is for learning and testing only
- **Demo Mode**: No real trading occurs - all trades are simulated
- **No Guarantees**: Past performance doesn't predict future results
- **Risk Warning**: Cryptocurrency trading involves significant risk
- **DYOR**: Do your own research before any real trading

## 🐛 Troubleshooting

### Connection Issues
- Check RPC_URL in `.env`
- Verify internet connection
- Try alternative RPC endpoints

### Data Issues (Collector)
- NaN reserves: strict “inside‑candle only” means no Sync occurred in that 5m window (real‑only). This is expected for some candles.
- Re‑runs slow? The first run builds caches. Overlapping runs are much faster thanks to `data/.cache/block_ts.sqlite`.
- To rebuild fully: remove `data/.cache/` (will re‑fetch on‑chain data).

### Strategy Issues
- Review strategy parameters in `.env`
- Check minimum signal strength settings
- Validate indicator calculations

## 🧠 Notes & Future Enhancements

- OOS‑led selection aims for higher realized returns (lower drawdowns and better Calmar) than single‑window tuning.
- Optional next steps:
- Use reserves in backtests for real AMM impact (x·y=k) — the dataset now exposes `reserve_hex` and `reserve_dai` columns.
  - Add strategy rotation (top‑3 OOS leaders) with a handover threshold to avoid churn.
  - Add Parquet caches for event/candle stores to shrink disk size further while keeping speed.
- [ ] API endpoints

---

**Happy Trading! 🎯**

Remember: This is demo mode only. Always test thoroughly before considering any real trading.




  Quick checklist

  - Dependencies: pip install scikit-optimize pandas numpy
  - Data: ensure data/<asset>_ohlcv_<quote>_30day_5m.csv exists (the runner will fall back to Config.resolve_ohlcv_path() if needed).
  - CPU: adjust --workers to your cores (defaults to 12).

  Run Stage 1 (30d only)

  - python -m optimization.runner --stage 30d --top-n1 60
  - Writes to reports/optimizer_pipeline_<timestamp>_30d/stage1_30d/
  - Selected strategies: stage1_30d/summary.json (key “top”)

  Run Stage 2 using Stage 1’s output

  - python -m optimization.runner --stage 90d --top-n2 20 --from-summary reports/optimizer_pipeline_<ts>_30d/stage1_30d/summary.json

  Notes

  - Uses full GP Bayesian optimizer + CPS scoring.
  - Tunes each strategy’s real parameters (strategy-specific ranges via the new registry).
  - Candlestick patterns are marked unused and won’t run.
  - You can customize: --timeframes 5min,15min,30min,1h,2h,4h,8h,16h,1d, --calls 80, --out-dir path.

token0=0x2b591e99afE9f32eAA6214f7B7629768c40Eeb39 token1=0xA1077a294dDE1B09bB078844df40758a5D0f9a27
  If you want, I can kick off the 30‑day run now or tune workers/calls for your machine.

  ### Example collector run (default configuration)

  ```bash
  python collectors/swap_ohlcv_collector.py \
    --days 365 --interval 5m \
    --pin-rpc-per-worker \
    --out data/<asset>_ohlcv_<quote>_365day_5m.csv
  ```

  The script prints progress for each block range and finishes with the resolved output path. When using the default environment (`ASSET_SYMBOL=HEX`, `QUOTE_SYMBOL=DAI`) the file name becomes `data/hex_ohlcv_dai_365day_5m.csv`.

  ### Handy follow-up commands

  - Aggregate a finished stage directory into CSV/Markdown dashboards:
    ```bash
    python -m optimization.aggregate reports/optimizer_pipeline_<timestamp>_<stage>/stage2_90d
    ```
  - Kick off a targeted stage with a curated strategy list:
    ```bash
    python -m optimization.runner --stage 90d --strategies-file strategies_all.json --workers 12 --calls 80
    ```
  - Backtest a shortlist using your latest dataset:
    ```bash
    python scripts/top_strategy_backtest.py \
      --reports-dir reports \
      --data-path data/<asset>_ohlcv_<quote>_730day_5m.csv \
      --max-backtests 300
    ```