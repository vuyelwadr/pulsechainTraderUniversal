"""
Grid Trading Strategy V2 — Trend- and Volatility‑Adaptive

Design goals
- Keep grid’s chop-harvesting edge, but reduce premature exits in uptrends
  and avoid shallow dip buys in downtrends.
- Ensure rung spacing clears round‑trip fees + slippage.
- Recenter proactively when price dwells at the range extremes.
- Add a short cooldown to reduce whipsaws on lower TFs.

All indicators are computed from the provided real OHLC/price data — no synthetic data.
"""
from typing import Dict, List
import numpy as np
import pandas as pd
import logging

from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


class GridTradingStrategyV2(BaseStrategy):
    """
    GridTradingStrategyV2

    Key improvements vs V1:
    - ATR%-based step with minimum step ensures cost coverage
    - Trend-aware asymmetric spacing (wider sells in bull, deeper buys in bear)
    - Dwell-based recentering when price lingers at extremes
    - Trade cooldown to reduce whipsaws
    - Explicit parameter_space() for tighter, safer optimization

    Parameters
    - min_step_pct: Minimum rung spacing as % of price (default 0.02 = 2%)
    - atr_mult: ATR%-to-step multiplier (default 1.0)
    - num_grids: Rungs each side (default 10)
    - trend_ema_fast, trend_ema_slow: Trend filter EMAs (50/200)
    - sell_spacing_mult_bull: Sell rung inflation when bullish (default 1.3)
    - buy_spacing_mult_bear: Buy rung inflation when bearish (default 1.2)
    - recenter_threshold_pct: Price drift from center to force recenter (default 0.12)
    - recenter_dwell_bars: Bars in top/bottom third to recenter (default 6)
    - center_lookback: Bars to average for center anchor (default 20)
    - cooldown_bars: Bars to suppress signals after a trade (default 2)
    - min_strength: Minimum strength to emit signal (default 0.55)
    - side_when_bear: 0=both, 1=buy-only, 2=disable (default 1)
    - min_edge_pct: Minimum effective step to clear RT costs (default 0.02 = 2%)
    - fee_pct: Per-side fee (default 0.0025 = 0.25%)
    - slippage_pct: Per-side slippage allowance (default 0.01 = 1%)
    - tol_buy_lo, tol_buy_hi, tol_sell_lo, tol_sell_hi: Proximity bands around rungs
    - timeframe_minutes: analysis timeframe hint (default 60)
    """

    def __init__(self, parameters: Dict = None):
        p = {
            'min_step_pct': 0.02,
            'atr_mult': 1.0,
            'buy_atr_mult': 1.0,
            'sell_atr_mult': 1.0,
            'num_grids': 10,
            'trend_ema_fast': 50,
            'trend_ema_slow': 200,
            'sell_spacing_mult_bull': 1.3,
            'buy_spacing_mult_bear': 1.2,
            'recenter_threshold_pct': 0.12,
            'recenter_dwell_bars': 6,
            'center_lookback': 20,
            'cooldown_bars': 2,
            'min_strength': 0.55,
            'side_when_bear': 1,
            'min_edge_pct': 0.02,
            'fee_pct': 0.0025,
            'slippage_pct': 0.01,
            'tol_buy_lo': 0.01,
            'tol_buy_hi': 0.005,
            'tol_sell_lo': 0.005,
            'tol_sell_hi': 0.01,
            'timeframe_minutes': 60,
            'regime_enabled': 1,
            'regime_timeframe': '4h',
            'regime_ma_period': 120,
            'regime_buffer_pct': 0.01,
            'buy_level_decay': 0.75,
            'sell_level_decay': 0.85,
        }
        if parameters:
            p.update(parameters)
        super().__init__('GridTradingStrategyV2', p)

        self._levels: List[float] = []
        self._center: float = 0.0
        self._last_center_price: float = 0.0
        self._buy_step_samples: List[float] = []
        self._sell_step_samples: List[float] = []

    # --- helpers ---
    @staticmethod
    def _ema(series: pd.Series, span: int) -> pd.Series:
        return series.ewm(span=max(2, int(span)), adjust=False).mean()

    @staticmethod
    def _atr(df: pd.DataFrame, period: int) -> pd.Series:
        h, l, c1 = df['high'], df['low'], df['close'].shift(1)
        tr = pd.concat([(h - l).abs(), (h - c1).abs(), (l - c1).abs()], axis=1).max(axis=1)
        return tr.rolling(max(2, int(period))).mean()

    def _rebuild_levels(self, price: float, step_buy: float, step_sell: float, num: int):
        self._center = float(price)
        self._levels = []
        n = max(1, int(num))
        for i in range(1, n + 1):
            self._levels.append(self._center * (1.0 - step_buy * i))   # buy rungs
            self._levels.append(self._center * (1.0 + step_sell * i))  # sell rungs

    def _need_recenter(self, df: pd.DataFrame, price: float) -> bool:
        # threshold by drift
        drift = abs(price - (self._last_center_price or self._center or price)) / max(self._last_center_price or self._center or price, 1e-12)
        if drift >= float(self.parameters['recenter_threshold_pct']):
            return True
        # dwell at extremes
        n = max(3, int(self.parameters['center_lookback']))
        if len(df) < n:
            return False
        window = df.tail(n)
        lo = window['price'].min(); hi = window['price'].max()
        rng = max(hi - lo, 1e-12)
        pos = (window['price'] - lo) / rng
        dwell = max(2, int(self.parameters['recenter_dwell_bars']))
        tail = pos.tail(dwell)
        if len(tail) >= dwell and ((tail > 0.66).all() or (tail < 0.34).all()):
            return True
        return False

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        if not self.validate_data(data):
            return data
        df = data.copy()
        price = df.get('price', df.get('close', df['close']))
        df['price'] = price
        # trend + ATR%
        df['ema_fast'] = self._ema(price, int(self.parameters['trend_ema_fast']))
        df['ema_slow'] = self._ema(price, int(self.parameters['trend_ema_slow']))
        df['ema_fast_slope'] = df['ema_fast'].diff()
        atr = self._atr(df.assign(close=price), 14)
        df['atrp'] = (atr / price).clip(lower=1e-9)
        # regime flags
        df['bullish'] = (df['ema_fast'] >= df['ema_slow']) & (df['ema_fast_slope'] >= 0)
        df['bearish'] = (df['ema_fast'] < df['ema_slow'])
        # step per bar
        min_step = float(self.parameters['min_step_pct'])
        df['step_pct'] = np.maximum(min_step, float(self.parameters['atr_mult']) * df['atrp'])

        # Higher-timeframe regime filter gating
        df['regime_ma'] = np.nan
        df['regime_on'] = True
        if int(self.parameters.get('regime_enabled', 1)) == 1 and 'timestamp' in df.columns:
            try:
                ts = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
                base = pd.DataFrame({'timestamp': ts, 'price': df['price']}).dropna()
                base = base.set_index('timestamp')
                timeframe = (str(self.parameters.get('regime_timeframe', '4h')) or '4h').lower()
                ht_close = base['price'].resample(timeframe).last().dropna()
                span = max(2, int(self.parameters.get('regime_ma_period', 200)))
                regime_ma = ht_close.ewm(span=span, adjust=False).mean()
                mapped = regime_ma.reindex(ts, method='ffill')
                df['regime_ma'] = mapped.to_numpy()
                buffer_pct = float(self.parameters.get('regime_buffer_pct', 0.0))
                df['regime_on'] = (df['price'] > df['regime_ma'] * (1.0 + buffer_pct)).fillna(False)
            except Exception:
                df['regime_ma'] = np.nan
                df['regime_on'] = True

        # decide center and rebuild levels
        center_lookback = max(2, int(self.parameters['center_lookback']))
        current_center = float(df['price'].tail(center_lookback).mean())
        current_price = float(df['price'].iloc[-1])

        # asymmetric spacing by regime
        last_is_bull = bool(df['bullish'].iloc[-1])
        last_is_bear = bool(df['bearish'].iloc[-1])
        atr_slice = df['atrp'].tail(max(center_lookback, 5)).dropna()
        avg_atrp = float(atr_slice.mean()) if not atr_slice.empty else float(df['atrp'].iloc[-1])
        base_buy = max(min_step, avg_atrp * float(self.parameters.get('buy_atr_mult', self.parameters.get('atr_mult', 1.0))))
        base_sell = max(min_step, avg_atrp * float(self.parameters.get('sell_atr_mult', self.parameters.get('atr_mult', 1.0))))
        base_buy = float(np.clip(base_buy, min_step, 0.5))
        base_sell = float(np.clip(base_sell, min_step, 0.5))
        step_buy = base_buy * (float(self.parameters['buy_spacing_mult_bear']) if last_is_bear else 1.0)
        step_sell = base_sell * (float(self.parameters['sell_spacing_mult_bull']) if last_is_bull else 1.0)

        if not len(self._levels) or self._need_recenter(df, current_price):
            if step_buy > 0:
                self._buy_step_samples.append(step_buy)
            if step_sell > 0:
                self._sell_step_samples.append(step_sell)
            self._rebuild_levels(current_center, step_buy, step_sell, int(self.parameters['num_grids']))
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

        buy_levels = [lvl for lvl in self._levels if lvl <= self._center]
        sell_levels = [lvl for lvl in self._levels if lvl >= self._center]
        buy_decay = float(self.parameters.get('buy_level_decay', 1.0))
        sell_decay = float(self.parameters.get('sell_level_decay', 1.0))
        buy_decay = np.clip(buy_decay, 0.1, 1.0)
        sell_decay = np.clip(sell_decay, 0.1, 1.0)

        min_strength = float(self.parameters['min_strength'])
        cooldown_bars = max(0, int(self.parameters['cooldown_bars']))
        min_edge = float(self.parameters['min_edge_pct'])

        tol_b_lo = float(self.parameters['tol_buy_lo']); tol_b_hi = float(self.parameters['tol_buy_hi'])
        tol_s_lo = float(self.parameters['tol_sell_lo']); tol_s_hi = float(self.parameters['tol_sell_hi'])
        side_bear = int(self.parameters['side_when_bear'])

        cooldown = 0
        for i in range(len(df)):
            px = float(df.iloc[i]['price'])
            step = float(df.iloc[i]['step_pct'])
            is_bull = bool(df.iloc[i].get('bullish', False))
            is_bear = bool(df.iloc[i].get('bearish', False))
            regime_on = bool(df.iloc[i].get('regime_on', True))

            if cooldown > 0:
                cooldown -= 1
                continue
            # require effective step >= min_edge
            eff = max(step, float(self.parameters['min_step_pct']))
            if eff < min_edge:
                continue

            allow_buy = True
            allow_sell = True
            if not regime_on:
                allow_buy = False
            if is_bear:
                if side_bear == 1:
                    allow_sell = False
                elif side_bear == 2:
                    allow_buy = False; allow_sell = False

            best_strength = 0.0
            want_buy = False
            want_sell = False

            # buy rung checks
            if allow_buy:
                for idx, lvl in enumerate(buy_levels):
                    if px >= lvl * (1.0 - tol_b_lo) and px <= lvl * (1.0 + tol_b_hi):
                        dist = abs(px - lvl) / max(px, 1e-12)
                        strength = float(np.clip(1.0 - (dist / max(eff, 1e-6)), 0.0, 1.0))
                        strength *= (1.0 if is_bull else 0.8)  # trend bonus
                        if buy_decay < 1.0:
                            strength *= buy_decay ** idx
                        if strength > best_strength:
                            best_strength = strength; want_buy = True; want_sell = False
                        break

            # sell rung checks
            if allow_sell and best_strength < 1.0:
                for idx, lvl in enumerate(sell_levels):
                    if px <= lvl * (1.0 + tol_s_hi) and px >= lvl * (1.0 - tol_s_lo):
                        dist = abs(px - lvl) / max(px, 1e-12)
                        strength = float(np.clip(1.0 - (dist / max(eff, 1e-6)), 0.0, 1.0))
                        # reduce eager profit taking in strong bull
                        strength *= (0.9 if is_bull else 1.0)
                        if sell_decay < 1.0:
                            strength *= sell_decay ** idx
                        if strength > best_strength:
                            best_strength = strength; want_buy = False; want_sell = True
                        break

            if best_strength >= min_strength:
                if want_buy:
                    df.iat[i, df.columns.get_loc('buy_signal')] = True
                if want_sell:
                    df.iat[i, df.columns.get_loc('sell_signal')] = True
                df.iat[i, df.columns.get_loc('signal_strength')] = best_strength
                cooldown = cooldown_bars

        return df

    @classmethod
    def parameter_space(cls):
        # Tight bounds to keep optimizer from degenerate configs
        return {
            'min_step_pct': (0.012, 0.04),
            'atr_mult': (0.5, 5.0),
            'buy_atr_mult': (0.5, 5.0),
            'sell_atr_mult': (0.5, 5.0),
            'num_grids': (6, 20),
            'trend_ema_fast': (20, 80),
            'trend_ema_slow': (100, 400),
            'sell_spacing_mult_bull': (1.0, 1.8),
            'buy_spacing_mult_bear': (1.0, 1.8),
            'recenter_threshold_pct': (0.08, 0.25),
            'recenter_dwell_bars': (3, 16),
            'center_lookback': (10, 40),
            'cooldown_bars': (0, 6),
            'min_strength': (0.45, 0.8),
            'side_when_bear': (0, 2),
            'min_edge_pct': (0.008, 0.03),
            'tol_buy_lo': (0.005, 0.02),
            'tol_buy_hi': (0.002, 0.02),
            'tol_sell_lo': (0.002, 0.02),
            'tol_sell_hi': (0.005, 0.02),
            'regime_enabled': (0, 1),
            'regime_ma_period': (40, 240),
            'regime_buffer_pct': (0.0, 0.03),
            'buy_level_decay': (0.5, 1.0),
            'sell_level_decay': (0.6, 1.0),
        }
