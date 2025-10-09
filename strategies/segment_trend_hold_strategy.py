"""Hold entire uptrend segments identified by trend_state label."""

from __future__ import annotations

from typing import Dict

import pandas as pd

from .base_strategy import BaseStrategy


class SegmentTrendHoldStrategy(BaseStrategy):
    """Buys at the start of each UPTREND regime and exits when it ends."""

    def __init__(self, parameters: Dict | None = None) -> None:
        defaults = {
            "timeframe_minutes": 60,
            "trade_amount_pct": 1.0,
            "exit_delay_bars": 6,
            "require_confirm": True,
            "confirm_timeframe_minutes": 240,
            "entry_strength_threshold": 0.05,
            "exit_strength_threshold": -0.25,
            "confirm_exit": True,
        }
        if parameters:
            defaults.update(parameters)
        super().__init__("SegmentTrendHoldStrategy", defaults)
        # Flag used by the walk-forward loader so it attaches confirmation states
        self.requires_confirmation = bool(self.parameters.get("require_confirm", False))

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

        exit_delay = max(1, int(self.parameters.get("exit_delay_bars", 1)))
        require_confirm = bool(self.parameters.get("require_confirm", False))
        confirm_exit = bool(self.parameters.get("confirm_exit", True))
        entry_strength_threshold = float(self.parameters.get("entry_strength_threshold", 0.0))
        exit_strength_threshold = float(self.parameters.get("exit_strength_threshold", 0.0))

        holding = False
        exit_counter = 0
        for i in range(len(df)):
            state = df.at[i, "trend_state"]
            prev_state = df.at[i, "state_prev"] if "state_prev" in df.columns else df["trend_state"].shift(1).fillna("NONE")[i]
            confirm_state = df.at[i, "confirm_trend_state"] if "confirm_trend_state" in df.columns else "UPTREND"
            strength = df.at[i, "trend_strength_score"] if "trend_strength_score" in df.columns else None
            confirm_strength = df.at[i, "confirm_trend_strength"] if "confirm_trend_strength" in df.columns else None

            if not holding and state == "UPTREND" and prev_state != "UPTREND":
                if require_confirm and confirm_state != "UPTREND":
                    continue
                if strength is not None and strength < entry_strength_threshold:
                    continue
                df.at[i, "buy_signal"] = True
                df.at[i, "signal_strength"] = 1.0
                df.at[i, "signal_type"] = "buy"
                holding = True
                exit_counter = 0
            elif holding:
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
                if exit_counter >= exit_delay:
                    df.at[i, "sell_signal"] = True
                    df.at[i, "signal_strength"] = 1.0
                    df.at[i, "signal_type"] = "sell"
                    holding = False
                    exit_counter = 0

        if holding:
            df.at[len(df) - 1, "sell_signal"] = True
            df.at[len(df) - 1, "signal_strength"] = 1.0
            df.at[len(df) - 1, "signal_type"] = "sell"

        return df
