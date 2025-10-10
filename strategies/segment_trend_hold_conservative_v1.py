"""Conservative version of SegmentTrendHold optimized for 1.5% fees."""

from __future__ import annotations

from typing import Dict, Tuple

import pandas as pd

from .base_strategy import BaseStrategy


class SegmentTrendHoldConservativeV1(BaseStrategy):
    """
    Conservative trend holding strategy with moderate selectivity.
    Balances trade frequency with cost efficiency for 1.5% fee environment.
    """

    requires_regime_data: bool = True

    def __init__(self, parameters: Dict | None = None) -> None:
        # Conservative parameters - balanced approach
        defaults = {
            "timeframe_minutes": 240,  
            "trade_amount_pct": 0.6,   # Smaller position size to control risk
            "entry_strength_threshold": 0.12,  # Moderate selectivity
            "min_confirm_strength": 0.20,      # Reasonable confirmation
            "exit_strength_threshold": -0.05,  # Exit on mild weakness
            "exit_delay_bars": 6,              # Moderate exit speed
            "require_confirm": True,
            "confirm_timeframe_minutes": 240,
            "confirm_exit": True,
            "allow_early_entry": False,
            "confirm_grace_bars": 8,           # Some grace for brief dips
            "max_drawdown_pct": 0.20,          # Moderate risk control
            "trail_atr_mult": 2.0,             # Standard trailing stop
            "time_stop_bars": 400,             # ~1 week max hold
            "reentry_cooldown_bars": 8,        # Moderate cooldown
        }
        if parameters:
            defaults.update(parameters)
        super().__init__("SegmentTrendHoldConservativeV1", defaults)
        self.requires_confirmation = bool(self.parameters.get("require_confirm", False))

    @classmethod
    def parameter_space(cls) -> Dict[str, Tuple[float, float]]:
        return {
            "entry_strength_threshold": (0.08, 0.20),
            "min_confirm_strength": (0.15, 0.30),
            "exit_strength_threshold": (-0.15, 0.0),
            "exit_delay_bars": (3, 12),
            "max_drawdown_pct": (0.10, 0.30),
            "trail_atr_mult": (1.0, 3.0),
            "time_stop_bars": (200, 600),
            "reentry_cooldown_bars": (4, 12),
        }

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        if "price" not in df.columns:
            df["price"] = df.get("close", df.get("price", 0)).astype(float)
        df["state_prev"] = df["trend_state"].shift(1).fillna("NONE")
        return df

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy().reset_index(drop=True)
        df["buy_signal"] = False
        df["sell_signal"] = False
        df["signal_strength"] = 0.0
        df["signal_type"] = "hold"

        # Extract parameters
        exit_delay = max(1, int(self.parameters.get("exit_delay_bars", 6)))
        require_confirm = bool(self.parameters.get("require_confirm", True))
        confirm_exit = bool(self.parameters.get("confirm_exit", True))
        entry_strength_threshold = float(self.parameters.get("entry_strength_threshold", 0.12))
        exit_strength_threshold = float(self.parameters.get("exit_strength_threshold", -0.05))
        allow_early_entry = bool(self.parameters.get("allow_early_entry", False))
        confirm_grace_bars = int(self.parameters.get("confirm_grace_bars", 8))
        min_confirm_strength = float(self.parameters.get("min_confirm_strength", 0.20))
        max_drawdown_pct = float(self.parameters.get("max_drawdown_pct", 0.20))
        trail_atr_mult = float(self.parameters.get("trail_atr_mult", 2.0))
        time_stop_bars = int(self.parameters.get("time_stop_bars", 400))
        reentry_cooldown_bars = int(self.parameters.get("reentry_cooldown_bars", 8))

        # State tracking
        prev_states = df.get("state_prev", df["trend_state"].shift(1).fillna("NONE"))
        
        holding = False
        exit_counter = 0
        entry_price = None
        peak_price = None
        bars_in_position = 0
        confirm_grace_remaining = 0
        reentry_cooldown_remaining = 0
        atr_percent = df.get("atr_percent")

        for i in range(len(df)):
            state = df.at[i, "trend_state"]
            prev_state = prev_states.iat[i] if i > 0 else "NONE"
            confirm_state = df.at[i, "confirm_trend_state"] if "confirm_trend_state" in df.columns else "UPTREND"
            strength = df.at[i, "trend_strength_score"] if "trend_strength_score" in df.columns else None
            confirm_strength = df.at[i, "confirm_trend_strength"] if "confirm_trend_strength" in df.columns else None
            price = float(df.at[i, "close"] if "close" in df.columns else df.at[i, "price"])

            # Update confirmation grace
            if confirm_state == "UPTREND" and confirm_strength is not None and confirm_strength >= min_confirm_strength:
                confirm_grace_remaining = confirm_grace_bars
            else:
                confirm_grace_remaining = max(0, confirm_grace_remaining - 1)

            confirm_ok = confirm_state == "UPTREND" or confirm_grace_remaining > 0
            strength_ok = (confirm_strength is None) or (confirm_strength >= min_confirm_strength)

            # ENTRY LOGIC
            if not holding:
                if reentry_cooldown_remaining > 0:
                    reentry_cooldown_remaining -= 1

                can_buy = False
                signal_strength = 0.0

                # Trend transition entry
                if (reentry_cooldown_remaining == 0 and 
                    state == "UPTREND" and prev_state != "UPTREND"):
                    
                    trend_ok = strength is not None and strength >= entry_strength_threshold
                    confirm_ok_layer = not require_confirm or (confirm_ok and strength_ok)

                    if trend_ok and confirm_ok_layer:
                        can_buy = True
                        signal_strength = min(1.0, strength / entry_strength_threshold)

                # Early entry during confirmed uptrend
                if not can_buy and allow_early_entry and reentry_cooldown_remaining == 0:
                    if (confirm_ok and strength_ok and 
                        strength is not None and strength >= entry_strength_threshold):
                        can_buy = True
                        signal_strength = min(0.8, strength / entry_strength_threshold)

                if can_buy:
                    df.at[i, "buy_signal"] = True
                    df.at[i, "signal_strength"] = signal_strength
                    df.at[i, "signal_type"] = "buy"
                    holding = True
                    exit_counter = 0
                    entry_price = price
                    peak_price = price
                    bars_in_position = 0
                    continue

            # EXIT LOGIC
            if holding:
                if peak_price is not None:
                    peak_price = max(peak_price, price)
                bars_in_position += 1

                # Trend-based exit
                if state != "UPTREND":
                    strong_exit = True
                    if strength is not None and strength > exit_strength_threshold:
                        strong_exit = False
                    if confirm_exit and confirm_state == "UPTREND":
                        strong_exit = False
                    if confirm_exit and confirm_strength is not None and confirm_strength > exit_strength_threshold:
                        strong_exit = False
                    
                    if strong_exit:
                        exit_counter += 1
                    else:
                        exit_counter = max(0, exit_counter - 1)
                else:
                    exit_counter = 0

                # Risk management exits
                risk_exit = False
                
                # Drawdown stop
                if max_drawdown_pct > 0 and peak_price:
                    drawdown = (price - peak_price) / peak_price
                    if drawdown <= -abs(max_drawdown_pct):
                        risk_exit = True

                # ATR trailing stop
                if not risk_exit and trail_atr_mult > 0 and peak_price and atr_percent is not None:
                    try:
                        atr_val = atr_percent.iat[i] if i < len(atr_percent) else None
                        if atr_val is not None and not pd.isna(atr_val):
                            atr_val_float = float(atr_val)
                            if atr_val_float > 0:
                                drop_from_peak = (peak_price - price) / peak_price
                                atr_threshold = trail_atr_mult * (atr_val_float / 100.0 if atr_val_float > 1 else atr_val_float)
                                if drop_from_peak >= atr_threshold:
                                    risk_exit = True
                    except (IndexError, TypeError):
                        pass

                # Time stop
                if not risk_exit and time_stop_bars > 0 and bars_in_position >= time_stop_bars:
                    risk_exit = True

                # Execute exit
                if exit_counter >= exit_delay or risk_exit:
                    df.at[i, "sell_signal"] = True
                    df.at[i, "signal_strength"] = 1.0
                    df.at[i, "signal_type"] = "sell"
                    holding = False
                    exit_counter = 0
                    entry_price = None
                    peak_price = None
                    bars_in_position = 0
                    confirm_grace_remaining = 0
                    reentry_cooldown_remaining = reentry_cooldown_bars

        # Force sell at end if still holding
        if holding:
            df.at[len(df) - 1, "sell_signal"] = True
            df.at[len(df) - 1, "signal_strength"] = 1.0
            df.at[len(df) - 1, "signal_type"] = "sell"

        return df
