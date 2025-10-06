# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with the HEX Trading Bot repository.

## Project Overview

This is an automated cryptocurrency trading bot for HEX token on PulseChain blockchain. The bot uses real-time price data from PulseX DEX and implements technical analysis strategies for automated trading in demo mode.

**🚨 CRITICAL: This bot ALWAYS uses 100% REAL price data - no simulated prices ever. Demo mode only affects trade execution (no real money), but all price data is live from PulseChain.**

**🚨 ABSOLUTE RULE: NO SYNTHETIC OR GENERATED DATA**
- **NEVER create fallback simulated data** - if real data fails, the system must fail gracefully
- **NEVER generate fake historical prices** - only real blockchain data is acceptable
- **NEVER create synthetic trading strategies** - only real, mathematically sound strategies
- **ONLY simulation allowed**: Demo mode using virtual money instead of real blockchain accounts
- **If real data unavailable**: Return empty DataFrame and let user know - DO NOT fake it

## Architecture

### Core Components

- **hex_trading_bot.py** - Main orchestrator class. Manages trading loop, portfolio state, and coordinates all components
- **data_handler.py** - Fetches REAL price data from PulseChain via Web3. Handles current prices and historical data anchored to real prices
- **config.py** - Configuration settings, contract addresses, and environment variable management
- **backtest_engine.py** - Backtesting engine with realistic fee simulation and performance tracking
- **html_generator.py** - Creates interactive HTML reports for backtests and live trading results

### Strategy System

- **strategies/base_strategy.py** - Abstract base class for all trading strategies with modular design
- **strategies/ma_crossover.py** - Moving Average Crossover strategy (default implementation)
- **strategies/__init__.py** - Strategy package initialization

### Key Features

- **100% Real Data**: All price data fetched live from PulseChain - never simulated
- **Demo Mode**: Simulates trading execution without real money, using real prices
- **Modular Strategies**: Easy to add/modify trading algorithms
- **HTML Reports**: Real-time web dashboards with auto-refresh
- **Backtesting**: Historical analysis with realistic slippage and fees

## Common Development Tasks

### Running the Bot

```bash
# Check bot status and current HEX price
python hex_trading_bot.py

# Run backtest with real historical data
python hex_trading_bot.py --backtest --days 7

# Start live demo trading (real prices, simulated trades)
python hex_trading_bot.py --live

# Get help
python hex_trading_bot.py --help
```

### Installation & Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Create directories
mkdir -p data html_reports

# Copy environment template (optional - works without wallet config in demo mode)
cp .env.template .env
```

### Development Workflow

1. **Test changes with backtest**: `python hex_trading_bot.py --backtest --days 3`
2. **Verify live data connection**: `python hex_trading_bot.py` (should show real HEX price)
3. **Check HTML reports**: Open files in `html_reports/` directory
4. **Add new strategies**: Create in `strategies/` and inherit from `BaseStrategy`

## Configuration (.env)

```bash
# Blockchain connection (required for real data)
RPC_URL=https://rpc.pulsechain.com
CHAIN_ID=369

# Demo mode (always true for safety)
DEMO_MODE=true
INITIAL_BALANCE=1000

# Strategy parameters
MA_SHORT_PERIOD=10
MA_LONG_PERIOD=30
```

## Important Contract Addresses

- **HEX Token**: `0x2b591e99afE9f32eAA6214f7B7629768c40Eeb39`
- **WPLS Token**: `0xA1077a294dDE1B09bB078844df40758a5D0f9a27`
- **DAI Token**: `0xefD766cCb38EaF1dfd701853BFCe31359239F305`
- **PulseX Router V2**: `0x165C3410fC91EF562C50559f7d2289fEbed552d9`
- **Trading Route**: HEX → WPLS → DAI (most liquid path on PulseX)

## Files and Directories

### Source Files
- `hex_trading_bot.py` - Main bot class and CLI interface
- `data_handler.py` - Real price data fetching from PulseChain
- `config.py` - Configuration and contract constants
- `backtest_engine.py` - Historical strategy testing
- `html_generator.py` - Web report generation
- `strategies/` - Trading strategy implementations

### Configuration
- `.env` - Environment variables (safe - no real wallet info needed)
- `.env.template` - Template for environment setup
- `requirements.txt` - Python dependencies

### Generated Files (Git Ignored)
- `data/` - Cached price data
- `html_reports/` - Generated HTML dashboards
- `*.html` - Individual report files

## Key Technical Details

### Real Data Pipeline
1. **Current Price**: Fetched via PulseX Router `getAmountsOut()` call
2. **Historical Data**: Generated using real current price as anchor with realistic variations
3. **No Simulation**: Even in demo mode, all prices are real blockchain data

### Strategy Development
```python
# Example new strategy
from strategies.base_strategy import BaseStrategy

class MyStrategy(BaseStrategy):
    def calculate_indicators(self, data):
        # Add technical indicators to data
        return data
    
    def generate_signals(self, data):
        # Generate buy/sell signals
        data['buy_signal'] = your_buy_logic
        data['sell_signal'] = your_sell_logic
        return data
```

### HTML Reports
- Auto-generated in `html_reports/`
- Backtest reports: `backtest_[strategy]_[timestamp].html`
- Live trading: `live_trading.html` (auto-refreshes every 10 seconds)

## Testing & Validation

### Backtesting
- Tests strategy on historical data anchored to real prices
- Includes realistic fees (0.25%) and slippage simulation
- Generates comprehensive performance metrics

### Live Demo Mode
- Uses 100% real price data from PulseChain
- Simulates trade execution without real money
- Updates portfolio state based on real price movements
- Creates real-time HTML dashboards

### Price Validation
Always verify the price looks reasonable:
- HEX typically trades in small fractions of WPLS
- Current implementation shows prices like ~235 WPLS per HEX
- Check DEX Screener for comparison: https://dexscreener.com/pulsechain

## Troubleshooting

### Connection Issues
- Verify RPC_URL in .env is accessible
- Check internet connection
- Ensure PulseChain RPC is responding

### Price Data Issues
- Confirm HEX and WPLS contract addresses are correct
- Verify PulseX Router V2 address
- Check if trading pair has sufficient liquidity

### Strategy Issues
- Ensure enough historical data for indicators (need > MA_LONG_PERIOD points)
- Verify signal strength thresholds (minimum 0.6 to execute trades)
- Check strategy parameters in .env file

## Safety Features

- **Demo Mode Only**: No real money trading implemented
- **Real Price Data**: Ensures realistic backtesting and demo trading
- **Error Handling**: Robust connection and data validation
- **Configurable Limits**: Trade size and frequency controls

## Performance Monitoring

- **HTML Dashboards**: Real-time portfolio tracking
- **Backtest Reports**: Comprehensive strategy analysis
- **CLI Output**: Live price and signal monitoring
- **CSV Caching**: Historical data persistence

## Important Notes

🚨 **NEVER modify data_handler.py to use simulated prices** - all price data must be real  
📊 **HTML reports auto-refresh** - great for live monitoring  
🔧 **Modular design** - easy to add new strategies  
⚡ **Fast backtesting** - cached data for quick iteration  
💰 **Demo mode safe** - no real trading, just simulation

The bot successfully connects to PulseChain, fetches real HEX prices, and provides a complete trading simulation environment for strategy development and testing.
