# LazyBear Technical Indicators

This directory contains trading strategies based on technical indicators developed by LazyBear.

## Strategy #154: VWAP Bands

**File:** `strategy_154_vwap_bands.py`  
**Source:** http://pastebin.com/6VqZ8DQ3

### Description
VWAP Bands strategy uses Volume Weighted Average Price (VWAP) with standard deviation bands to identify overbought and oversold conditions.

### Key Features
- **VWAP Calculation**: Volume-weighted average price as the central line
- **Standard Deviation Bands**: Multiple levels (1x, 2x, optional 3x multipliers)
- **Volume Confirmation**: Signals require above-average volume
- **Multiple Band Levels**: L1, L2, and optional L3 bands for signal strength

### Trading Logic
- **Buy Signals**: Price bouncing from lower bands (oversold conditions)
- **Sell Signals**: Price hitting upper bands (overbought conditions)
- **Signal Strength**: Enhanced by volume, VWAP trend, and price momentum

### Parameters
- `length` (34): Period for standard deviation calculation
- `l1_multiplier` (1.0): Multiplier for first band level
- `l2_multiplier` (2.0): Multiplier for second band level
- `l3_multiplier` (2.5): Multiplier for optional third band level
- `use_l3_bands` (False): Whether to use third band level
- `min_strength` (0.6): Minimum signal strength to trigger trade
- `volume_threshold` (1.0): Minimum volume multiplier vs average

### Usage Example

```python
from strategies.lazybear.technical_indicators.technical_indicators.strategy_154_vwap_bands import VWAPBandsStrategy

# Create strategy with default parameters
strategy = VWAPBandsStrategy()

# Or with custom parameters
custom_params = {
    'length': 50,
    'l1_multiplier': 1.2,
    'l2_multiplier': 2.5,
    'use_l3_bands': True,
    'min_strength': 0.7
}
strategy = VWAPBandsStrategy(custom_params)

# Use with StrategyManager
from strategies.base_strategy import StrategyManager

manager = StrategyManager()
manager.add_strategy(strategy)
manager.set_active_strategy("VWAPBands")

# Get trading signals
signal_type, signal_strength = strategy.get_signal(your_data)
```

### Testing
The strategy has been thoroughly tested with:
- ✅ Unit tests for all methods
- ✅ Integration tests with StrategyManager
- ✅ Realistic token trading data simulation
- ✅ Multiple market conditions (bull, bear, sideways)

### Performance Notes
- Works well in trending markets with volume confirmation
- Effective for identifying reversal points at extreme band levels
- Signal strength increases with volume and momentum alignment
- Best used with other indicators for confirmation in ranging markets
