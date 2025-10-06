"""
Regime Grid + Breakout Hybrid Strategy

Purpose
- Monetize range-bound volatility via an adaptive grid while layering a
  breakout overlay to participate in trends. Gating by long/slow regime filters
  aims to reduce adverse exposure in bearish conditions.

Notes
- Derives all indicators purely from provided real price series (no synthetic data).
- Follows BaseStrategy interface used by the bot:
  - calculate_indicators(data) -> returns data with indicators
  - generate_signals(data) -> adds buy_signal/sell_signal/signal_strength
"""
from typing import Dict, List
import numpy as np
import pandas as pd

from strategies.base_strategy import BaseStrategy


class RegimeGridBreakoutHybrid(BaseStrategy):
    """
    Regime-aware grid with breakout overlay.

    Parameters
    - atr_period: ATR lookback used for volatility (default 14)
    - ema_fast: Fast EMA for trend/slope (default 50)
    - ema_slow: Slow EMA for regime filter (default 200)
    - atr_mult: Multiplier for ATR%-based grid step (default 1.0)
    - min_step_pct: Minimum rung spacing as % of price (default 0.012 = 1.2%)
    - num_rungs: Max rungs each side (default 8)
    - min_strength: Minimum signal strength to emit (default 0.5)
    - slope_thresh: Minimum EMA fast slope to qualify as bullish (default 0.0)
    - side_when_bear: 0=both, 1=buy-only, 2=disable in bearish regimes (default 1)
    - breakout_window: Donchian channel window for breakouts (default 20)
    - breakout_slope_min: Min EMA-fast slope for breakout confirm (default 0.0)
    - trail_atr_mult: Not used by signals directly; provided for risk systems (default 2.0)
    - rebalance_threshold: % move from last center to rebuild rungs (default 0.10)
    - center_lookback: Bars to average for grid center (default 20)
    - timeframe_minutes: Analytics timeframe hint (default 60)
    """

    def __init__(self, parameters: Dict = None):
        params = {
            'atr_period': 14,
            'ema_fast': 50,
            'ema_slow': 200,
            'atr_mult': 1.0,
            'min_step_pct': 0.012,
            'num_rungs': 8,
            'min_strength': 0.5,
            'slope_thresh': 0.0,
            'side_when_bear': 1,
            'breakout_window': 20,
            'breakout_slope_min': 0.0,
            'trail_atr_mult': 2.0,
            'rebalance_threshold': 0.10,
            'center_lookback': 20,
            'timeframe_minutes': 60,
        }
        if parameters:
            params.update(parameters)
        super().__init__('RegimeGridBreakoutHybrid', params)

        self._levels: List[float] = []
        self._center: float = 0.0
        self._last_center_price: float = 0.0

    # --- helpers ---
    @staticmethod
    def _ema(series: pd.Series, span: int) -> pd.Series:
        return series.ewm(span=max(2, int(span)), adjust=False).mean()

    @staticmethod
    def _atr(df: pd.DataFrame, period: int) -> pd.Series:
        h, l, c1 = df['high'], df['low'], df['close'].shift(1)
        tr = pd.concat([(h - l).abs(), (h - c1).abs(), (l - c1).abs()], axis=1).max(axis=1)
        return tr.rolling(max(2, int(period))).mean()

    def _rebuild_levels(self, price: float, step_pct: float, num_rungs: int):
        self._center = float(price)
        self._levels = []
        num = max(1, int(num_rungs))
        for i in range(1, num + 1):
            self._levels.append(self._center * (1.0 - step_pct * i))  # buy rung
            self._levels.append(self._center * (1.0 + step_pct * i))  # sell rung

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        if not self.validate_data(data):
            return data
        df = data.copy()
        # Avoid over-restrictive minimum length; EMAs/EWM work on short series too.
        # Keep computation even for shorter TFs so strategy can operate across 4h/8h/16h/1d windows.

        # price field
        df['price'] = df.get('price', df.get('close', df['close']))

        # trend + volatility
        df['ema_fast'] = self._ema(df['price'], int(self.parameters['ema_fast']))
        df['ema_slow'] = self._ema(df['price'], int(self.parameters['ema_slow']))
        df['ema_fast_slope'] = df['ema_fast'].diff()
        atr = self._atr(df.assign(close=df['price']), int(self.parameters['atr_period']))
        df['atr'] = atr
        df['atrp'] = (atr / df['price']).clip(lower=1e-12)

        # regimes
        slope_thresh = float(self.parameters['slope_thresh'])
        df['bullish'] = (df['ema_fast'] >= df['ema_slow']) & (df['ema_fast_slope'] >= slope_thresh)
        df['bearish'] = (df['ema_fast'] < df['ema_slow'])

        # breakout channels (Donchian)
        n = max(2, int(self.parameters['breakout_window']))
        df['donch_hi'] = df['price'].rolling(n).max()
        df['donch_lo'] = df['price'].rolling(n).min()

        # step percent and grid center
        step_pct = np.maximum(float(self.parameters['atr_mult']) * df['atrp'], float(self.parameters['min_step_pct']))
        df['step_pct'] = step_pct

        # center as recent mean
        c_look = max(2, int(self.parameters['center_lookback']))
        current_center = float(df['price'].tail(c_look).mean())
        current_price = float(df['price'].iloc[-1])

        # decide (re)build
        need_build = False
        if not len(self._levels):
            need_build = True
        else:
            thresh = float(self.parameters['rebalance_threshold'])
            ref = self._last_center_price if self._last_center_price > 0 else self._center
            move = abs(current_price - ref) / max(ref, 1e-12)
            if move >= thresh:
                need_build = True
        if need_build:
            self._rebuild_levels(current_center, float(step_pct.iloc[-1]), int(self.parameters['num_rungs']))
            self._last_center_price = current_price

        return df

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        if 'step_pct' not in df.columns:
            df = self.calculate_indicators(df)

        df['buy_signal'] = False
        df['sell_signal'] = False
        df['signal_strength'] = 0.0

        if not len(self._levels):
            return df

        min_strength = float(self.parameters['min_strength'])
        side_bear = int(self.parameters['side_when_bear'])
        slope_min = float(self.parameters['breakout_slope_min'])

        # tolerance around rungs
        tol_lo = 0.01  # 1% below a rung
        tol_hi = 0.005  # 0.5% above a rung

        for i in range(len(df)):
            px = float(df.iloc[i]['price'])
            step = float(df.iloc[i]['step_pct']) if pd.notna(df.iloc[i].get('step_pct', np.nan)) else float(self.parameters['min_step_pct'])
            is_bull = bool(df.iloc[i].get('bullish', False))
            is_bear = bool(df.iloc[i].get('bearish', False))
            ema_slope = float(df.iloc[i].get('ema_fast_slope', 0.0))

            # gate sides by regime
            allow_buy = True
            allow_sell = True
            if is_bear:
                if side_bear == 1:
                    allow_sell = False
                elif side_bear == 2:
                    allow_buy = False
                    allow_sell = False

            best_strength = 0.0
            do_buy = False
            do_sell = False

            # nearest rung checks
            if allow_buy:
                # any buy rung (<= center)
                buy_lvls = [lvl for lvl in self._levels if lvl <= self._center]
                for lvl in buy_lvls:
                    # within tolerance: price between [lvl*(1-tol_lo), lvl*(1+tol_hi)]
                    if px >= lvl * (1.0 - tol_lo) and px <= lvl * (1.0 + tol_hi):
                        dist = abs(px - lvl) / max(px, 1e-12)
                        # Normalize proximity by an effective step that respects fee/slippage scale
                        eff = max(step, 0.008)  # at least ~0.8% of price
                        strength = float(np.clip(1.0 - (dist / eff), 0.0, 1.0))
                        # trend bonus; be conservative if bearish
                        strength *= (1.0 if is_bull else 0.7)
                        if strength > best_strength:
                            best_strength = strength
                            do_buy, do_sell = True, False
                        break

            if allow_sell and best_strength < 1.0:  # check sells only if not maxed
                sell_lvls = [lvl for lvl in self._levels if lvl >= self._center]
                for lvl in sell_lvls:
                    if px <= lvl * (1.0 + tol_hi) and px >= lvl * (1.0 - tol_lo):
                        dist = abs(px - lvl) / max(px, 1e-12)
                        eff = max(step, 0.008)
                        strength = float(np.clip(1.0 - (dist / eff), 0.0, 1.0))
                        # in strong bull, slightly damp sells to avoid cutting trends
                        strength *= (0.9 if is_bull else 0.8)
                        if strength > best_strength:
                            best_strength = strength
                            do_buy, do_sell = False, True
                        break

            # breakout overlay
            # bullish breakout: price crosses above prior Donchian high with positive slope
            d_hi = df.iloc[i].get('donch_hi', np.nan)
            d_lo = df.iloc[i].get('donch_lo', np.nan)
            if pd.notna(d_hi) and allow_buy and ema_slope >= slope_min and px >= float(d_hi):
                # breakout strength scales with slope relative to price
                slope_rel = float(np.clip(abs(ema_slope) / max(px, 1e-12), 0.0, 0.01))
                bo_strength = float(np.clip(0.6 + 40.0 * slope_rel, 0.0, 1.0))
                if bo_strength > best_strength:
                    best_strength = bo_strength
                    do_buy, do_sell = True, False

            # bearish breakout (allowed only if sells are allowed and not disabled in bear)
            if pd.notna(d_lo) and allow_sell and ema_slope <= -slope_min and px <= float(d_lo):
                slope_rel = float(np.clip(abs(ema_slope) / max(px, 1e-12), 0.0, 0.01))
                bo_strength = float(np.clip(0.6 + 40.0 * slope_rel, 0.0, 1.0))
                # in explicit bear regimes you may allow exits more readily
                bo_strength *= (1.0 if is_bear else 0.9)
                if bo_strength > best_strength:
                    best_strength = bo_strength
                    do_buy, do_sell = False, True

            if best_strength >= min_strength:
                if do_buy:
                    df.iat[i, df.columns.get_loc('buy_signal')] = True
                if do_sell:
                    df.iat[i, df.columns.get_loc('sell_signal')] = True
                df.iat[i, df.columns.get_loc('signal_strength')] = best_strength

        return df

    def get_grid_info(self) -> Dict:
        return {
            'center': self._center,
            'last_center_price': self._last_center_price,
            'num_levels': len(self._levels),
            'levels': list(self._levels),
            'parameters': self.parameters,
        }

    # --- Optimizer parameter space (tighter bounds for faster, safer tuning) ---
    @classmethod
    def parameter_space(cls):
        # Return explicit bounds to avoid degenerate choices (e.g., min_step_pct=0, rebalance_threshold=1)
        return {
            'atr_period': (10, 40),
            'ema_fast': (20, 80),
            'ema_slow': (100, 400),
            'atr_mult': (0.6, 1.5),
            'min_step_pct': (0.006, 0.03),
            'num_rungs': (4, 14),
            'min_strength': (0.3, 0.8),
            'slope_thresh': (0.0, 0.5),
            'side_when_bear': (0, 2),
            'breakout_window': (10, 50),
            'breakout_slope_min': (0.0, 0.5),
            'trail_atr_mult': (1.0, 3.0),
            'rebalance_threshold': (0.05, 0.25),
            'center_lookback': (10, 40),
        }
