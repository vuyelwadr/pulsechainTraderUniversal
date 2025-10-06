#!/usr/bin/env python3
"""
Strategy 012: Constance Brown Derivative Oscillator (wrapper)

TradingView URL: https://www.tradingview.com/v/6wfwJ6To/
Type: momentum/oscillator

Delegates to canonical Strategy036ConstanceBrownDerivativeOscillator to ensure
parity while keeping manifest mapping and class name.
"""

from strategies.lazybear.technical_indicators.technical_indicators.strategy_036_constance_brown_derivative_oscillator import (
    Strategy036ConstanceBrownDerivativeOscillator,
)


class Strategy012ConstanceBrownDerivativeOscillator(Strategy036ConstanceBrownDerivativeOscillator):
    def __init__(self, parameters: dict = None):
        super().__init__(parameters or {})
        self.name = "Strategy_012_ConstanceBrownDerivativeOscillator"
