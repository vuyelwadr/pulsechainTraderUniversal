
"""
strategies/mean_reversion_zscore_pro1.py

Simple, conservative, cost-aware mean-reversion strategy intended to
generalize across timeframes. Trades around a mid-term EMA with dynamic
z-score bands and gates all actions on swap costs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from strategies.base_strategy import BaseStrategy  # type: ignore
from utils.cost_model_pro1 import cost_gate, estimate_trade_cost_bps

def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()

def _sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()

class MeanReversionZScorePro1(BaseStrategy):
    """
    Parameters:
      ema_n=60, z_entry=1.6, z_exit=0.25, vol_window=60
      min_edge_multiple=1.2, trade_amount_quote=500
    """
    def __init__(
        self,
        ema_n: int = 60,
        z_entry: float = 1.6,
        z_exit: float = 0.25,
        vol_window: int = 60,
        min_edge_multiple: float = 1.20,
        route_key: str = "HEX/DAI",
        trade_amount_quote: float = 500.0,
    ):
        super().__init__()
        self.ema_n = ema_n
        self.z_entry = z_entry
        self.z_exit = z_exit
        self.vol_window = vol_window
        self.min_edge_multiple = min_edge_multiple
        self.route_key = route_key
        self.trade_amount_quote = trade_amount_quote

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in ("open","high","low","close"):
            if col not in df.columns:
                raise ValueError(f"Missing column: {col}")
        df["ema"] = _ema(df["close"], self.ema_n)
        resid = (df["close"] - df["ema"])
        vol = resid.rolling(window=self.vol_window, min_periods=self.vol_window).std(ddof=0)
        df["z"] = resid / vol.replace(0, np.nan)
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "ema" not in df.columns:
            df = self.calculate_indicators(df)

        z = df["z"]
        close = df["close"]
        vol = (close - df["ema"]).rolling(window=self.vol_window, min_periods=self.vol_window).std(ddof=0)

        # Entry/exit conditions
        want_long = (z <= -self.z_entry)
        want_exit = (z >= -self.z_exit)

        # Expected edge in bps approximated as |z| * (vol/close) in bps
        expected_edge_bps = (z.abs() * (vol / close) * 1e4).clip(lower=0.0)

        def _gate(mask: pd.Series) -> pd.Series:
            out = pd.Series(False, index=df.index)
            idx = mask[mask].index
            for i in idx:
                if cost_gate(expected_edge_bps=float(expected_edge_bps.loc[i]),
                             amount_quote=float(self.trade_amount_quote),
                             route_key=self.route_key,
                             min_edge_multiple=self.min_edge_multiple):
                    out.loc[i] = True
            return out

        df["buy_signal"] = _gate(want_long)
        df["sell_signal"] = _gate(want_exit)

        rt_cost = estimate_trade_cost_bps(amount_quote=float(self.trade_amount_quote),
                                          route_key=self.route_key, single_side=False)
        strength = (expected_edge_bps / max(rt_cost, 1e-6)).clip(0, 2.0) / 2.0
        df["signal_strength"] = strength.fillna(0.0)

        return df
