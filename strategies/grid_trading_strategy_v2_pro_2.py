
"""
GridTradingStrategyV2 — adaptive, volatility-aware grid with regime filters
-------------------------------------------------------------------------

Drop-in replacement for GridTradingStrategy (same public methods):
  - calculate_indicators(data: pd.DataFrame) -> pd.DataFrame
  - generate_signals(data: pd.DataFrame) -> pd.DataFrame
  - get_grid_info() -> Dict
Also defines: parameter_space() -> Dict[str, Tuple[float,float]] for the optimizer pipeline.

Key improvements vs v1:
- ATR/VOL adaptive grid spacing and range (tightens in calm, widens in stormy markets)
- Trend/regime filter (skip grids in strong trends; bias levels with skew)
- Fee/edge gating (only trade if expected edge > min_edge_bps)
- Recenter on ATR-based drift (recenter when |price - center| > z_atr * ATR)
- Cooldown after fills to reduce whipsaw
- Optional asymmetric level density (more buys below in drawdowns)

Assumptions:
- `data` contains columns: timestamp (optional), open/high/low/close or price. If only price is present, it is used for all.
- Volume may be zero; VWAP will fall back to mid price (H+L+C)/3.
"""
from typing import Dict, Tuple, List, Optional
import numpy as np
import pandas as pd
import logging

try:
    from .base_strategy import BaseStrategy
except Exception:
    # allow ad-hoc import when run outside package
    class BaseStrategy:
        def __init__(self, name: str, parameters: Dict):
            self.name = name
            self.parameters = parameters
        def validate_data(self, data: pd.DataFrame) -> bool:
            return isinstance(data, pd.DataFrame) and len(data) > 0

logger = logging.getLogger(__name__)


class GridTradingStrategyV2Pro2(BaseStrategy):
    def __init__(self, parameters: Optional[Dict] = None):
        # Sensible defaults anchored to your prior bests (grid 2–3%, 18–20 levels, ~24–30% span)
        defaults = {
            # Core grid controls
            "base_step_pct": 2.0,          # baseline step (% of price) before ATR scaling
            "price_range_pct": 26.0,       # target total band (% of price) around center
            "num_grids_min": 10,
            "num_grids_max": 22,
            # Volatility adaptation
            "atr_period": 14,
            "atr_mult_step": 0.65,         # scales base_step by ATR/price
            "atr_mult_range": 10.0,        # scales range by ATR/price
            "vol_floor": 0.003,            # realized vol (std of log returns) lower bound to enable trading
            "vol_ceil": 0.15,              # upper bound to avoid chaos; set higher to allow more
            # Trend/regime filter
            "anchor_tf_minutes": 480,      # 8h anchor for trend slope
            "ema_fast": 9,
            "ema_slow": 36,
            "trend_slope_thresh": 0.0005,  # skip grids when |slope| above this (strong trend)
            "allow_with_trend_sells": True, # allow taking sells in uptrend (profits), buys in downtrend
            # Asymmetry / skew
            "skew_down": 0.35,             # 0..1 more density below center
            "skew_up": 0.15,               # 0..1 more density above center
            # Recenter and cooldown
            "z_atr_recenter": 1.5,         # recenter when |price-center| > z*ATR
            "rebalance_threshold_pct": 2.0, # extra safety; % move since last rebalance
            "cooldown_bars": 2,            # bars to wait after any fill signal
            # Edge & risk
            "fee_bps": 25.0,               # per-side fee in basis points (adjust to venue)
            "min_edge_bps": 90.0,          # require step >= fees * 2 + cushion
            "max_position_levels": 6,      # cap simultaneous ladder fills on one side
            # Signal gating
            "min_strength": 0.40,          # 0..1 signal strength gate
        }
        if parameters:
            defaults.update(parameters)
        super().__init__("GridTradingStrategyV2", defaults)

        # State
        self.grid_center = 0.0
        self.grid_levels: List[Dict] = []
        self.last_rebalance_price = 0.0
        self.cooldown = 0  # bars

    # ---- Optimizer-facing parameter space (used by your runner) ----
    @staticmethod
    def parameter_space() -> Dict[str, Tuple[float, float]]:
        # Keep to Real/Integer ranges; runner converts automatically
        return {
            "base_step_pct": (1.0, 3.5),
            "price_range_pct": (18.0, 34.0),
            "num_grids_min": (8, 16),
            "num_grids_max": (16, 28),
            "atr_period": (10, 28),
            "atr_mult_step": (0.35, 1.25),
            "atr_mult_range": (6.0, 16.0),
            "vol_floor": (0.001, 0.02),
            "vol_ceil": (0.06, 0.25),
            "ema_fast": (6, 16),
            "ema_slow": (20, 80),
            "trend_slope_thresh": (0.0002, 0.0015),
            "skew_down": (0.0, 0.70),
            "skew_up": (0.0, 0.50),
            "z_atr_recenter": (1.0, 2.5),
            "rebalance_threshold_pct": (1.0, 6.0),
            "cooldown_bars": (0, 6),
            "fee_bps": (10.0, 40.0),
            "min_edge_bps": (50.0, 140.0),
            "max_position_levels": (2, 10),
            "min_strength": (0.25, 0.75),
        }

    # ---- Indicators & helpers ----
    @staticmethod
    def _ensure_ohlc(df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        if 'timestamp' in d.columns:
            try:
                d['timestamp'] = pd.to_datetime(d['timestamp'])
            except Exception:
                pass
        if 'close' not in d.columns and 'price' in d.columns:
            d['close'] = d['price']
        for c in ('open', 'high', 'low'):
            if c not in d.columns:
                d[c] = d['close']
        if 'volume' not in d.columns:
            d['volume'] = 0.0
        d['price'] = d['close']
        return d

    @staticmethod
    def _atr(df: pd.DataFrame, period: int) -> pd.Series:
        hi, lo, cl = df['high'], df['low'], df['close']
        tr = np.maximum(hi - lo, np.maximum((hi - cl.shift(1)).abs(), (lo - cl.shift(1)).abs()))
        return tr.rolling(period, min_periods=max(2, period//2)).mean().fillna(method='bfill')

    @staticmethod
    def _ema(x: pd.Series, span: int) -> pd.Series:
        return x.ewm(span=span, adjust=False).mean()

    def _realized_vol(self, cl: pd.Series, lookback: int = 48) -> pd.Series:
        lr = np.log(cl).diff()
        return lr.rolling(lookback, min_periods=max(8, lookback//3)).std().fillna(0.0)

    def _build_center(self, df: pd.DataFrame) -> float:
        # VWAP fallback to typical price mean
        tp = (df['high'] + df['low'] + df['close']) / 3.0
        if (df['volume'] > 0).any():
            vwap = (tp * df['volume']).rolling(24, min_periods=8).sum() / (df['volume'].rolling(24, min_periods=8).sum().replace(0, np.nan))
            center = float(vwap.iloc[-1]) if not np.isnan(vwap.iloc[-1]) else float(tp.tail(24).mean())
        else:
            center = float(tp.tail(24).mean())
        return center

    def _needs_recenter(self, price: float, atr: float) -> bool:
        p = self.parameters
        drift = abs(price - self.grid_center)
        cond_z = atr > 0 and (drift / atr) > p["z_atr_recenter"]
        cond_pct = (self.last_rebalance_price > 0) and (abs(price - self.last_rebalance_price) / self.last_rebalance_price) * 100.0 >= p["rebalance_threshold_pct"]
        return (cond_z or cond_pct or not self.grid_levels)

    def _expected_edge_ok(self, step_pct: float) -> bool:
        p = self.parameters
        # Two sides of fees plus cushion
        min_step_bps = (2.0 * p["fee_bps"]) + p["min_edge_bps"]
        return (step_pct * 10000.0) >= min_step_bps

    def _density_weights(self, n: int, skew_dn: float, skew_up: float) -> Tuple[np.ndarray, np.ndarray]:
        # Geometric-like density: more weight near center; apply asymmetry
        idx = np.arange(1, n + 1, dtype=float)
        base = np.exp(-0.15 * (idx - 1))  # decay away from center
        w_dn = base * (1.0 + skew_dn * (idx / idx.max()))
        w_up = base * (1.0 + skew_up * (idx / idx.max()))
        # normalize to sum to 1
        w_dn /= w_dn.sum()
        w_up /= w_up.sum()
        return w_dn, w_up

    # ---- Public API ----
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        if not self.validate_data(data):
            return data
        p = self.parameters
        df = self._ensure_ohlc(data)

        # Core stats
        atr = self._atr(df, int(p["atr_period"]))
        rv = self._realized_vol(df['close'], lookback=48)
        ema_f = self._ema(df['close'], int(p["ema_fast"]))
        ema_s = self._ema(df['close'], int(p["ema_slow"]))
        # slope per bar (approximate)
        slope = (ema_f - ema_f.shift(1)) / ema_f.shift(1)

        # derive step/range from ATR
        with np.errstate(divide='ignore', invalid='ignore'):
            atr_over_price = (atr / df['close']).fillna(0.0)
        step_pct_dyn = (p["base_step_pct"] / 100.0) * (1.0 + p["atr_mult_step"] * atr_over_price)
        step_pct_dyn = step_pct_dyn.clip(lower=0.002, upper=0.05)  # sanity
        range_pct_dyn = (p["price_range_pct"] / 100.0) * (1.0 + p["atr_mult_range"] * atr_over_price)
        range_pct_dyn = range_pct_dyn.clip(lower=0.10, upper=0.80)

        out = df.copy()
        out["atr"] = atr
        out["realized_vol"] = rv
        out["ema_fast"] = ema_f
        out["ema_slow"] = ema_s
        out["trend_slope"] = slope.fillna(0.0)
        out["step_pct_dyn"] = step_pct_dyn
        out["range_pct_dyn"] = range_pct_dyn

        # compute/update grid for the latest row
        self._setup_grid_levels(out)
        # Distance to nearest level
        out["dist_to_grid"] = out["close"].apply(self._distance_from_nearest_grid)

        return out

    def _setup_grid_levels(self, df: pd.DataFrame) -> None:
        """Update grid center/levels based on latest bar and parameters."""
        if df.empty:
            return
        p = self.parameters
        price = float(df["close"].iloc[-1])
        atr = float(df["atr"].iloc[-1]) if "atr" in df.columns else 0.0
        # center selection
        self.grid_center = self._build_center(df.tail(64))
        # check recenter conditions
        if not self._needs_recenter(price, atr):
            return

        # compute dynamic step/range and grids
        step_pct = float(df["step_pct_dyn"].iloc[-1])
        range_pct = float(df["range_pct_dyn"].iloc[-1])
        # ensure edge vs fees
        if not self._expected_edge_ok(step_pct):
            # widen to pass fee threshold
            bump = ( (2.0 * p["fee_bps"] + p["min_edge_bps"]) / 10000.0 ) / max(1e-9, step_pct)
            step_pct *= max(1.0, bump)

        # derive number of levels
        num_levels_each = int(np.clip(np.floor((range_pct / step_pct) / 2.0), p["num_grids_min"], p["num_grids_max"]))
        # build asymmetric weights
        w_dn, w_up = self._density_weights(num_levels_each, p["skew_down"], p["skew_up"])

        # construct levels
        self.grid_levels = []
        # cumulative step distances using weights
        dn_steps = np.cumsum(w_dn) * range_pct
        up_steps = np.cumsum(w_up) * range_pct
        for i, dp in enumerate(dn_steps, 1):
            lvl = self.grid_center * (1.0 - dp)
            self.grid_levels.append({"level": float(lvl), "type": "buy", "distance": i})
        for i, up in enumerate(up_steps, 1):
            lvl = self.grid_center * (1.0 + up)
            self.grid_levels.append({"level": float(lvl), "type": "sell", "distance": i})

        self.last_rebalance_price = price

    def _distance_from_nearest_grid(self, price: float) -> float:
        if not self.grid_levels:
            return 1e9
        return float(min(abs(price - g["level"]) / price for g in self.grid_levels))

    def _calc_strength(self, *, distance_idx: int, rv: float, slope: float, step_pct: float) -> float:
        # closer levels stronger; moderate vol is good; avoid strong trend counter-trades
        prox = max(0.0, 1.0 - 0.12 * (distance_idx - 1))
        vol_good = np.clip((rv - 0.002) / 0.05, 0.0, 1.0)   # soft window
        trend_soft = np.exp(-8.0 * abs(slope))              # punish strong slopes
        edge = np.clip((step_pct*10000.0) / 120.0, 0.0, 1.0)
        return float(0.45*prox + 0.25*vol_good + 0.20*trend_soft + 0.10*edge)

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        if "atr" not in df.columns or "realized_vol" not in df.columns:
            df = self.calculate_indicators(df)

        p = self.parameters
        df["buy_signal"] = False
        df["sell_signal"] = False
        df["signal_strength"] = 0.0

        if not self.grid_levels:
            return df

        # iterate rows
        for i in range(len(df)):
            price = float(df.iloc[i]["close"] if "close" in df.columns else df.iloc[i].get("price", np.nan))
            if not np.isfinite(price):
                continue
            rv = float(df.iloc[i]["realized_vol"])
            slope = float(df.iloc[i]["trend_slope"])
            step_pct = float(df.iloc[i]["step_pct_dyn"])

            # cooldown
            if self.cooldown > 0:
                self.cooldown -= 1
                continue

            # volatility gates
            if rv < p["vol_floor"] or rv > p["vol_ceil"]:
                continue

            # strong-trend gate (but allow taking profits with trend if configured)
            if abs(slope) > p["trend_slope_thresh"]:
                # allow sells in strong uptrends or buys in strong downtrends to realize profits
                allow_sell = p["allow_with_trend_sells"] and slope > 0
                allow_buy = p["allow_with_trend_sells"] and slope < 0
            else:
                allow_sell = allow_buy = True

            # check levels
            best_strength = 0.0
            do_buy = do_sell = False

            # BUY side
            if allow_buy:
                for lv in (g for g in self.grid_levels if g["type"] == "buy"):
                    # trigger when price within 0.5 * step of level
                    level = lv["level"]
                    thresh = step_pct * 0.5
                    hit = abs(price/level - 1.0) <= thresh
                    if hit:
                        s = self._calc_strength(distance_idx=lv["distance"], rv=rv, slope=slope, step_pct=step_pct)
                        if s > best_strength:
                            best_strength = s
                            do_buy = True
                            do_sell = False

            # SELL side
            if allow_sell:
                for lv in (g for g in self.grid_levels if g["type"] == "sell"):
                    level = lv["level"]
                    thresh = step_pct * 0.5
                    hit = abs(price/level - 1.0) <= thresh
                    if hit:
                        s = self._calc_strength(distance_idx=lv["distance"], rv=rv, slope=slope, step_pct=step_pct)
                        if s > best_strength:
                            best_strength = s
                            do_sell = True
                            do_buy = False

            # Edge/strength gating & cooldown
            if best_strength >= p["min_strength"]:
                if do_buy:
                    df.iat[i, df.columns.get_loc("buy_signal")] = True
                    df.iat[i, df.columns.get_loc("signal_strength")] = best_strength
                    self.cooldown = int(p["cooldown_bars"])
                elif do_sell:
                    df.iat[i, df.columns.get_loc("sell_signal")] = True
                    df.iat[i, df.columns.get_loc("signal_strength")] = best_strength
                    self.cooldown = int(p["cooldown_bars"])

        return df

    def get_grid_info(self) -> Dict:
        return {
            "grid_center": float(self.grid_center),
            "num_levels": len(self.grid_levels),
            "buy_levels": [float(l["level"]) for l in self.grid_levels if l["type"] == "buy"],
            "sell_levels": [float(l["level"]) for l in self.grid_levels if l["type"] == "sell"],
            "parameters": dict(self.parameters),
            "last_rebalance_price": float(self.last_rebalance_price),
        }
