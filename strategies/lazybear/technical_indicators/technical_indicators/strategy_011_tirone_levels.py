#!/usr/bin/env python3
"""
Strategy 011: Tirone Levels (wrapper)

TradingView URL: https://www.tradingview.com/v/ZdbzUf9B/
Type: levels/support-resistance

Implements Tirone Levels by delegating to the canonical Strategy035TironeLevels
implementation (same math), preserving Pine defaults (midpoint off, mean on).
"""

from strategies.base_strategy import BaseStrategy
from strategies.lazybear.technical_indicators.technical_indicators.strategy_035_tirone_levels import Strategy035TironeLevels


class Strategy011TironeLevels(Strategy035TironeLevels):
    def __init__(self, parameters: dict = None):
        params = {
            'length': 20,
            'use_midpoint_method': False,
            'use_mean_method': True,
            'signal_threshold': 0.6,
            'level_tolerance': 0.002,
        }
        if parameters:
            params.update(parameters)
        super().__init__(params)
        self.name = "Strategy_011_TironeLevels"
