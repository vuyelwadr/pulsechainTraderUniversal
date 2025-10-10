from __future__ import annotations

from typing import Dict, Any
import pandas as pd
import numpy as np

from .base_strategy import BaseStrategy


class AdaptiveTrendHoldStrategy(BaseStrategy):
    """Adaptive version of SegmentTrendHoldStrategy that adjusts parameters based on market conditions."""

    requires_regime_data: bool = True

    def __init__(self, parameters: Dict | None = None) -> None:
        # Load adaptive configuration if provided
        if parameters and "adaptive_config_file" in parameters:
            import json
            with open(parameters["adaptive_config_file"], 'r') as f:
                self.adaptive_config = json.load(f)
            # Use first configuration from the file
            if isinstance(self.adaptive_config, list):
                self.adaptive_config = self.adaptive_config[0]
            # Override parameters with config
            base_config = self.adaptive_config.get("base_config", {})
            base_config.update(parameters)
            parameters = base_config
        else:
            self.adaptive_config = None

        # Set default parameters
        defaults = {
            "timeframe_minutes": 60,
            "trade_amount_pct": 1.0,
            "exit_delay_bars": 2,
            "require_confirm": True,
            "confirm_timeframe_minutes": 240,
            "entry_strength_threshold": 0.05,
            "exit_strength_threshold": -0.28,
            "confirm_exit": True,
            "allow_early_entry": True,
            "confirm_grace_bars": 6,
            "min_confirm_strength": 0.12,
            "max_drawdown_pct": 0.18,
            "trail_atr_mult": 1.8,
            "time_stop_bars": 0,
            "reentry_cooldown_bars": 3,
            "adaptive_mode": "volatility_based",
            "volatility_window": 20,
            "trend_strength_window": 20
        }
        
        if parameters:
            defaults.update(parameters)
        super().__init__("AdaptiveTrendHoldStrategy", defaults)
        self.requires_confirmation = bool(self.parameters.get("require_confirm", False))

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        if "price" not in df.columns:
            df["price"] = df.get("close", df.get("price", 0)).astype(float)
        
        # Calculate adaptive indicators
        df["state_prev"] = df["trend_state"].shift(1).fillna("NONE")
        
        # Calculate rolling volatility (using ATR or price changes)
        if "atr_percent" in df.columns:
            df["market_volatility"] = df["atr_percent"].rolling(window=self.parameters.get("volatility_window", 20), min_periods=5).mean()
        else:
            # Simple price change volatility as fallback
            df["price_change"] = df["price"].pct_change()
            df["market_volatility"] = df["price_change"].rolling(window=self.parameters.get("volatility_window", 20), min_periods=5).std()
        
        # Calculate trend strength (rolling trend_strength_score)
        if "trend_strength_score" in df.columns:
            df["trend_strength_avg"] = df["trend_strength_score"].rolling(window=self.parameters.get("trend_strength_window", 20), min_periods=5).mean()
        else:
            df["trend_strength_avg"] = 0.5  # Default moderate trend
        
        return df

    def get_adaptive_parameters(self, row_data: pd.Series) -> Dict[str, Any]:
        """Calculate adaptive parameters based on current market conditions."""
        current_volatility = row_data.get("market_volatility", 0.05)
        current_trend_strength = row_data.get("trend_strength_avg", 0.5)
        
        adaptive_mode = self.parameters.get("adaptive_mode", "volatility_based")
        base_params = {
            "entry_strength_threshold": self.parameters["entry_strength_threshold"],
            "confirm_grace_bars": self.parameters["confirm_grace_bars"],
            "min_confirm_strength": self.parameters["min_confirm_strength"],
            "max_drawdown_pct": self.parameters["max_drawdown_pct"],
            "trail_atr_mult": self.parameters["trail_atr_mult"],
            "trade_amount_pct": self.parameters["trade_amount_pct"],
            "reentry_cooldown_bars": self.parameters["reentry_cooldown_bars"]
        }
        
        # Volatility-based adaptation
        if adaptive_mode == "volatility_based" and self.adaptive_config:
            vol_thresholds = self.adaptive_config.get("volatility_thresholds", {"low": 0.02, "medium": 0.06, "high": 0.12})
            adaptive_params = self.adaptive_config.get("adaptive_parameters", {})
            
            if current_volatility <= vol_thresholds.get("low", 0.02):
                regime = "low_volatility"
            elif current_volatility <= vol_thresholds.get("medium", 0.06):
                regime = "medium_volatility"
            else:
                regime = "high_volatility"
            
            adjustments = adaptive_params.get(regime, {})
            for param, adjustment in adjustments.items():
                if param in base_params:
                    base_params[param] += adjustment
        
        # Trend strength-based adaptation
        elif adaptive_mode == "trend_strength_based" and self.adaptive_config:
            trend_thresholds = self.adaptive_config.get("trend_strength_thresholds", {"weak": 0.2, "moderate": 0.5, "strong": 0.8})
            adaptive_params = self.adaptive_config.get("adaptive_parameters", {})
            
            if current_trend_strength <= trend_thresholds.get("weak", 0.2):
                regime = "weak_trend"
            elif current_trend_strength <= trend_thresholds.get("moderate", 0.5):
                regime = "moderate_trend"
            else:
                regime = "strong_trend"
            
            adjustments = adaptive_params.get(regime, {})
            for param, adjustment in adjustments.items():
                if param in base_params:
                    base_params[param] += adjustment
        
        # Hybrid adaptation
        elif adaptive_mode == "hybrid_volatility_trend" and self.adaptive_config:
            # Volatility adaptation for position sizing and trailing
            vol_adapt = self.adaptive_config.get("volatility_adaptation", {})
            if current_volatility <= 0.03:
                vol_regime = "low_vol"
            elif current_volatility <= 0.08:
                vol_regime = "med_vol"
            else:
                vol_regime = "high_vol"
            
            vol_adjustments = vol_adapt.get(vol_regime, {})
            for param, adjustment in vol_adjustments.items():
                if param == "position_size_adj":
                    base_params["trade_amount_pct"] *= adjustment
                elif param in base_params:
                    base_params[param] += adjustment
            
            # Trend strength adaptation for entry thresholds
            trend_adapt = self.adaptive_config.get("trend_strength_adaptation", {})
            if current_trend_strength <= 0.3:
                trend_regime = "weak_trend"
            elif current_trend_strength <= 0.7:
                trend_regime = "moderate_trend"
            else:
                trend_regime = "strong_trend"
            
            trend_adjustments = trend_adapt.get(trend_regime, {})
            for param, adjustment in trend_adjustments.items():
                if param == "entry_strength_adj":
                    base_params["entry_strength_threshold"] += adjustment
                elif param == "min_confirm_adj":
                    base_params["min_confirm_strength"] += adjustment
        
        # Ensure reasonable bounds
        base_params["entry_strength_threshold"] = max(-0.1, min(1.0, base_params["entry_strength_threshold"]))
        base_params["min_confirm_strength"] = max(0.0, min(1.0, base_params["min_confirm_strength"]))
        base_params["max_drawdown_pct"] = max(0.05, min(0.5, base_params["max_drawdown_pct"]))
        base_params["trail_atr_mult"] = max(0.5, min(10.0, base_params["trail_atr_mult"]))
        base_params["trade_amount_pct"] = max(0.5, min(2.0, base_params["trade_amount_pct"]))
        base_params["confirm_grace_bars"] = max(0, min(20, base_params["confirm_grace_bars"]))
        base_params["reentry_cooldown_bars"] = max(0, min(20, base_params["reentry_cooldown_bars"]))
        
        return base_params

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy().reset_index(drop=True)
        df["buy_signal"] = False
        df["sell_signal"] = False
        df["signal_strength"] = 0.0
        df["signal_type"] = "hold"
        df["adaptive_volatility"] = df["market_volatility"]
        df["adaptive_trend_strength"] = df["trend_strength_avg"]

        if "state_prev" in df.columns:
            prev_states = df["state_prev"]
        else:
            prev_states = df["trend_state"].shift(1).fillna("NONE")

        holding = False
        exit_counter = 0
        entry_price = None
        peak_price = None
        bars_in_position = 0
        confirm_grace_remaining = 0
        reentry_cooldown_remaining = 0
        atr_percent = df.get("atr_percent")

        for i in range(len(df)):
            # Get adaptive parameters for current bar
            adaptive_params = self.get_adaptive_parameters(df.iloc[i])
            
            state = df.at[i, "trend_state"]
            prev_state = prev_states.iat[i]
            confirm_state = df.at[i, "confirm_trend_state"] if "confirm_trend_state" in df.columns else "UPTREND"
            strength = df.at[i, "trend_strength_score"] if "trend_strength_score" in df.columns else None
            confirm_strength = df.at[i, "confirm_trend_strength"] if "confirm_trend_strength" in df.columns else None
            price = float(df.at[i, "close"] if "close" in df.columns else df.at[i, "price"])

            # Use adaptive parameters
            confirm_grace_bars = int(adaptive_params["confirm_grace_bars"])
            min_confirm_strength = adaptive_params["min_confirm_strength"]
            max_drawdown_pct = adaptive_params["max_drawdown_pct"]
            trail_atr_mult = adaptive_params["trail_atr_mult"]
            reentry_cooldown_bars = int(adaptive_params["reentry_cooldown_bars"])
            exit_delay = max(1, int(self.parameters.get("exit_delay_bars", 1)))
            require_confirm = bool(self.parameters.get("require_confirm", False))
            confirm_exit = bool(self.parameters.get("confirm_exit", True))
            entry_strength_threshold = adaptive_params["entry_strength_threshold"]
            exit_strength_threshold = float(self.parameters.get("exit_strength_threshold", 0.0))
            allow_early_entry = bool(self.parameters.get("allow_early_entry", False))
            time_stop_bars = int(self.parameters.get("time_stop_bars", 0) or 0)

            if confirm_state == "UPTREND" and confirm_strength is not None and confirm_strength >= min_confirm_strength:
                confirm_grace_remaining = confirm_grace_bars
            else:
                confirm_grace_remaining = max(0, confirm_grace_remaining - 1)

            confirm_ok = confirm_state == "UPTREND" or confirm_grace_remaining > 0
            strength_ok = (confirm_strength is None) or (confirm_strength >= min_confirm_strength)

            if not holding:
                if reentry_cooldown_remaining > 0:
                    reentry_cooldown_remaining -= 1

                can_buy = False
                if allow_early_entry and reentry_cooldown_remaining == 0:
                    if confirm_ok and strength_ok and strength is not None and strength >= entry_strength_threshold:
                        can_buy = True
                if not can_buy and reentry_cooldown_remaining == 0 and state == "UPTREND" and prev_state != "UPTREND":
                    if (not require_confirm or confirm_ok) and (strength is None or strength >= entry_strength_threshold) and (not require_confirm or strength_ok):
                        can_buy = True

                if can_buy:
                    df.at[i, "buy_signal"] = True
                    df.at[i, "signal_strength"] = max(df.at[i, "signal_strength"], adaptive_params["trade_amount_pct"])
                    df.at[i, "signal_type"] = "buy_adaptive"
                    holding = True
                    exit_counter = 0
                    entry_price = price
                    peak_price = price
                    bars_in_position = 0
                    continue

            if holding:
                if peak_price is not None:
                    peak_price = max(peak_price, price)
                bars_in_position += 1

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

                risk_exit = False
                if max_drawdown_pct > 0 and peak_price:
                    drop_pct = (price - peak_price) / peak_price
                    if drop_pct <= -abs(max_drawdown_pct):
                        risk_exit = True
                if not risk_exit and trail_atr_mult > 0 and peak_price and atr_percent is not None:
                    atr_val = atr_percent.iat[i] if i < len(atr_percent) else None
                    if atr_val is not None and not pd.isna(atr_val):
                        atr_val_float = float(atr_val)
                        if atr_val_float > 0:
                            drop_from_peak = (peak_price - price) / peak_price
                            atr_threshold = trail_atr_mult * (atr_val_float / 100.0 if atr_val_float > 1 else atr_val_float)
                            if drop_from_peak >= atr_threshold:
                                risk_exit = True
                if not risk_exit and time_stop_bars > 0 and bars_in_position >= time_stop_bars:
                    risk_exit = True

                if exit_counter >= exit_delay or risk_exit:
                    df.at[i, "sell_signal"] = True
                    df.at[i, "signal_strength"] = adaptive_params["trade_amount_pct"]
                    df.at[i, "signal_type"] = "sell_adaptive"
                    holding = False
                    exit_counter = 0
                    entry_price = None
                    peak_price = None
                    bars_in_position = 0
                    confirm_grace_remaining = 0
                    reentry_cooldown_remaining = reentry_cooldown_bars

        if holding:
            df.at[len(df) - 1, "sell_signal"] = True
            df.at[len(df) - 1, "signal_strength"] = adaptive_params["trade_amount_pct"]
            df.at[len(df) - 1, "signal_type"] = "sell_adaptive"

        return df
