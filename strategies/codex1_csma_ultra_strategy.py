"""Codex1 CSMA Ultra Strategy."""

from __future__ import annotations

from typing import Dict

from .codex1_csma_enhanced_strategy import Codex1CSMAEnhancedStrategy


class Codex1CSMAUltraStrategy(Codex1CSMAEnhancedStrategy):
    """Aggressively tuned CSMA reversion with shallower SMA and wider exits.

    This variant keeps the original Codex1 CSMA structure but switches to a
    432-bar (~1.5 day) simple moving average, allows entries once price drops
    24% beneath that anchor with RSI≤30, and lets positions run until price
    recovers roughly 5.4% above the SMA.  The combination came out of the
    October 2025 codex1 parameter sweep and delivers ~6.8k% net return on the
    cost-aware 5k DAI bucket while also improving the 10k and 25k buckets over
    the prior best-in-class CSMA variants.
    """

    def __init__(self, parameters: Dict | None = None):
        defaults = {
            'n_sma': 432,
            'entry_drop': 0.24,
            'exit_up': 0.054,
            'rsi_period': 14,
            'rsi_max': 30.0,
            'timeframe_minutes': 5,
        }
        if parameters:
            defaults.update(parameters)
        super().__init__(parameters=defaults)
        self.name = 'Codex1CSMAUltraStrategy'

    @classmethod
    def parameter_space(cls) -> Dict[str, tuple[float, float]]:
        base = super().parameter_space()
        base.update(
            {
                'n_sma': (360, 600),
                'entry_drop': (0.18, 0.3),
                'exit_up': (0.03, 0.08),
                'rsi_max': (25, 36),
            }
        )
        return base
