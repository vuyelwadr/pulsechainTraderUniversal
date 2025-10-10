"""Ensemble strategy combining best variants for consistent 1.5% fee performance."""

from __future__ import annotations

from typing import Dict, Tuple, List
import pandas as pd

from .base_strategy import BaseStrategy


class SegmentTrendHoldEnsembleV1(BaseStrategy):
    """
    Ensemble strategy that combines Balanced and Conservative approaches.
    Uses position sizing and weighted signals to maximize consistency.
    
    Logic: 
    - Balanced strategy provides signals for trending markets
    - Conservative strategy acts as fallback for choppy markets  
    - Different position sizing based on signal confidence
    """

    requires_regime_data: bool = True

    def __init__(self, parameters: Dict | None = None) -> None:
        defaults = {
            "timeframe_minutes": 240,
            "trade_amount_pct": 0.8,  # Base position size
            "ensemble_weights": {
                "balanced": 0.7,    # Primary strategy weight
                "conservative": 0.3 # Conservative fallback weight
            },
            # Balanced parameters (primary)
            "bal_entry_threshold": 0.10,
            "bal_confirm_strength": 0.18,
            "bal_exit_threshold": -0.06,
            "bal_exit_delay": 8,
            "bal_drawdown": 0.25,
            # Conservative parameters (secondary)
            "cons_entry_threshold": 0.12,
            "cons_confirm_strength": 0.20,
            "cons_exit_threshold": -0.05,
            "cons_exit_delay": 6,
            "cons_drawdown": 0.20,
            # Ensemble controls
            "min_ensemble_score": 0.6,  # Minimum combined signal strength
            "confidence_threshold": 0.8,  # High confidence for full position
            "partial_position_mult": 0.5,  # Position size for lower confidence
            "require_both_entry": False,  # Whether both strategies must agree to enter
            "require_both_exit": True,    # Whether both strategies must agree to exit
        }
        if parameters:
            defaults.update(parameters)
        super().__init__("SegmentTrendHoldEnsembleV1", defaults)

    @classmethod
    def parameter_space(cls) -> Dict[str, Tuple[float, float]]:
        return {
            "bal_entry_threshold": (0.08, 0.15),
            "bal_confirm_strength": (0.15, 0.25),
            "cons_entry_threshold": (0.10, 0.18),
            "cons_confirm_strength": (0.15, 0.28),
            "min_ensemble_score": (0.4, 0.8),
            "confidence_threshold": (0.6, 0.9),
        }

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        if "price" not in df.columns:
            df["price"] = df.get("close", df.get("price", 0)).astype(float)
        df["state_prev"] = df["trend_state"].shift(1).fillna("NONE")
        return df

    def _generate_substrategy_signals(self, data: pd.DataFrame, strategy_type: str) -> pd.DataFrame:
        """Generate signals for a specific sub-strategy"""
        df = data.copy().reset_index(drop=True)
        
        if strategy_type == "balanced":
            entry_threshold = float(self.parameters.get("bal_entry_threshold", 0.10))
            confirm_strength = float(self.parameters.get("bal_confirm_strength", 0.18))
            exit_threshold = float(self.parameters.get("bal_exit_threshold", -0.06))
            exit_delay = int(self.parameters.get("bal_exit_delay", 8))
            max_dd = float(self.parameters.get("bal_drawdown", 0.25))
        else:  # conservative
            entry_threshold = float(self.parameters.get("cons_entry_threshold", 0.12))
            confirm_strength = float(self.parameters.get("cons_confirm_strength", 0.20))
            exit_threshold = float(self.parameters.get("cons_exit_threshold", -0.05))
            exit_delay = int(self.parameters.get("cons_exit_delay", 6))
            max_dd = float(self.parameters.get("cons_drawdown", 0.20))

        df[f"{strategy_type}_buy_signal"] = False
        df[f"{strategy_type}_sell_signal"] = False
        df[f"{strategy_type}_signal_strength"] = 0.0

        state_prev = df["trend_state"].shift(1).fillna("NONE")
        holding = False
        exit_counter = 0
        entry_price = None
        peak_price = None

        for i in range(len(df)):
            state = df.at[i, "trend_state"]
            prev_state = state_prev.iat[i] if i > 0 else "NONE"
            strength = df.at[i, "trend_strength_score"] if "trend_strength_score" in df.columns else None
            confirm_strength_val = df.at[i, "confirm_trend_strength"] if "confirm_trend_strength" in df.columns else None
            price = float(df.at[i, "close"] if "close" in df.columns else df.at[i, "price"])
            
            cur_confirm_strength = confirm_strength_val if confirm_strength_val is not None else confirm_strength

            if not holding:
                # Entry logic
                if (state == "UPTREND" and prev_state != "UPTREND" and
                    strength is not None and strength >= entry_threshold and
                    cur_confirm_strength >= confirm_strength):
                    
                    df.at[i, f"{strategy_type}_buy_signal"] = True
                    df.at[i, f"{strategy_type}_signal_strength"] = min(1.0, strength / entry_threshold)
                    holding = True
                    exit_counter = 0
                    entry_price = price
                    peak_price = price
            else:
                # Exit logic
                peak_price = max(peak_price, price)
                
                # Drawdown check
                if max_dd > 0 and peak_price:
                    drawdown = (price - peak_price) / peak_price
                    if drawdown <= -abs(max_dd):
                        df.at[i, f"{strategy_type}_sell_signal"] = True
                        df.at[i, f"{strategy_type}_signal_strength"] = 1.0
                        holding = False
                        exit_counter = 0
                        entry_price = None
                        peak_price = None
                        continue
                
                # Trend-based exit
                if state != "UPTREND":
                    strength_exit_ok = strength is None or strength <= exit_threshold
                    if strength_exit_ok:
                        exit_counter += 1
                    else:
                        exit_counter = max(0, exit_counter - 1)
                else:
                    exit_counter = 0
                
                if exit_counter >= exit_delay:
                    df.at[i, f"{strategy_type}_sell_signal"] = True
                    df.at[i, f"{strategy_type}_signal_strength"] = 1.0
                    holding = False
                    exit_counter = 0
                    entry_price = None
                    peak_price = None

        # Force sell at end
        if holding:
            df.at[len(df) - 1, f"{strategy_type}_sell_signal"] = True
            df.at[len(df) - 1, f"{strategy_type}_signal_strength"] = 1.0

        return df

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy().reset_index(drop=True)
        df["buy_signal"] = False
        df["sell_signal"] = False
        df["signal_strength"] = 0.0
        df["signal_type"] = "hold"

        # Generate sub-strategy signals
        df_balanced = self._generate_substrategy_signals(df.copy(), "balanced")
        df_conservative = self._generate_substrategy_signals(df.copy(), "conservative")

        # Ensemble parameters
        weights = self.parameters.get("ensemble_weights", {"balanced": 0.7, "conservative": 0.3})
        min_ensemble_score = float(self.parameters.get("min_ensemble_score", 0.6))
        confidence_threshold = float(self.parameters.get("confidence_threshold", 0.8))
        require_both_entry = bool(self.parameters.get("require_both_entry", False))
        require_both_exit = bool(self.parameters.get("require_both_exit", True))

        # Combine signals
        for i in range(len(df)):
            bal_buy = df_balanced.at[i, "balanced_buy_signal"]
            cons_buy = df_conservative.at[i, "conservative_buy_signal"]
            bal_sell = df_balanced.at[i, "balanced_sell_signal"]
            cons_sell = df_conservative.at[i, "conservative_sell_signal"]
            
            bal_strength = df_balanced.at[i, "balanced_signal_strength"]
            cons_strength = df_conservative.at[i, "conservative_signal_strength"]

            # Ensemble buy logic
            ensemble_buy_score = 0.0
            buy_confidence = 0.0
            
            if bal_buy:
                ensemble_buy_score += weights["balanced"] * bal_strength
                buy_confidence += weights["balanced"]
            if cons_buy:
                ensemble_buy_score += weights["conservative"] * cons_strength
                buy_confidence += weights["conservative"]

            # Entry decision
            if ensemble_buy_score >= min_ensemble_score:
                if not require_both_entry or (bal_buy and cons_buy):
                    df.at[i, "buy_signal"] = True
                    
                    # Scale position size based on confidence
                    if buy_confidence >= confidence_threshold:
                        df.at[i, "signal_strength"] = ensemble_buy_score
                    else:
                        # Partial position for lower confidence
                        df.at[i, "signal_strength"] = ensemble_buy_score * float(self.parameters.get("partial_position_mult", 0.5))
                    
                    df.at[i, "signal_type"] = "buy"

            # Ensemble sell logic (more conservative)
            ensemble_sell_score = 0.0
            sell_confidence = 0.0
            
            if bal_sell:
                ensemble_sell_score += weights["balanced"] * bal_strength
                sell_confidence += weights["balanced"]
            if cons_sell:
                ensemble_sell_score += weights["conservative"] * cons_strength
                sell_confidence += weights["conservative"]

            # Exit decision
            if ensemble_sell_score >= 0.5:  # Lower threshold for exits
                if not require_both_exit or (bal_sell or cons_sell):
                    df.at[i, "sell_signal"] = True
                    df.at[i, "signal_strength"] = 1.0
                    df.at[i, "signal_type"] = "sell"

        return df
