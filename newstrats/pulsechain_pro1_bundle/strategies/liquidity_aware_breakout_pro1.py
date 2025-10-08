
"""
strategies/liquidity_aware_breakout_pro1.py

Liquidity-aware Donchian breakout that avoids trading when recent on-chain
volume is thin relative to its rolling baseline and when the expected
move is not large enough to clear round-trip costs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from strategies.base_strategy import BaseStrategy  # type: ignore
from utils.cost_model_pro1 import cost_gate, estimate_trade_cost_bps

def _sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(window=n, min_periods=n).mean()

class LiquidityAwareBreakoutPro1(BaseStrategy):
    """
    Parameters:
      donchian_n=55, vol_window=288 (approx one day on 5m), vol_thresh=0.8,
      min_edge_multiple=1.2
    """
    def __init__(
        self,
        donchian_n: int = 55,
        vol_window: int = 288,
        vol_thresh: float = 0.8,
        min_edge_multiple: float = 1.20,
        route_key: str = "HEX/DAI",
        trade_amount_quote: float = 500.0,
    ):
        super().__init__()
        self.donchian_n = donchian_n
        self.vol_window = vol_window
        self.vol_thresh = vol_thresh
        self.min_edge_multiple = min_edge_multiple
        self.route_key = route_key
        self.trade_amount_quote = trade_amount_quote

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in ("open","high","low","close"):
            if col not in df.columns:
                raise ValueError(f"Missing column: {col}")
        upper = df["high"].rolling(window=self.donchian_n, min_periods=self.donchian_n).max()
        lower = df["low"].rolling(window=self.donchian_n, min_periods=self.donchian_n).min()
        df["dc_upper"], df["dc_lower"] = upper, lower
        if "volume" in df.columns:
            df["vol_ma"] = _sma(df["volume"], self.vol_window)
            df["vol_ok"] = df["volume"] >= (self.vol_thresh * df["vol_ma"])
        else:
            # If no volume available, act as if volume is OK to not block strategy.
            df["vol_ok"] = True
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "dc_upper" not in df.columns:
            df = self.calculate_indicators(df)

        close = df["close"]
        breakout_up = (close > df["dc_upper"].shift(1)) & df["vol_ok"]
        breakdown = (close < df["dc_lower"].shift(1)) & df["vol_ok"]

        # Expected move = channel height in bps
        channel_bps = ((df["dc_upper"] - df["dc_lower"]) / close * 1e4).clip(lower=0.0)

        def _gate(mask: pd.Series) -> pd.Series:
            out = pd.Series(False, index=df.index)
            idx = mask[mask].index
            for i in idx:
                if cost_gate(expected_edge_bps=float(channel_bps.loc[i]),
                             amount_quote=float(self.trade_amount_quote),
                             route_key=self.route_key,
                             min_edge_multiple=self.min_edge_multiple):
                    out.loc[i] = True
            return out

        df["buy_signal"] = _gate(breakout_up)  # long HEX
        df["sell_signal"] = _gate(breakdown)   # rotate back to DAI on breakdown

        rt_cost = estimate_trade_cost_bps(amount_quote=float(self.trade_amount_quote),
                                          route_key=self.route_key, single_side=False)
        strength = (channel_bps / max(rt_cost, 1e-6)).clip(0, 2.0) / 2.0
        df["signal_strength"] = strength.fillna(0.0)
        return df
