"""GridTradingStrategyV3Overdrive

Ultra-aggressive grid variant that extends V2Pro with dual anchors, breakout
micro-grids, and pyramiding logic.  All computations rely on real OHLC/volume
inputs supplied by the trading engine – no synthetic data introduced here.
"""
from __future__ import annotations

from .base_strategy import BaseStrategy
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .grid_trading_strategy_v2_pro import GridTradingStrategyV2Pro


@dataclass
class _MicroGridState:
    active: bool = False
    levels: List[Dict[str, float]] = None
    bars_remaining: int = 0


class GridTradingStrategyV3Overdrive(GridTradingStrategyV2Pro, BaseStrategy):
    """Trend leaning grid with breakout harvesting overlay."""

    def __init__(self, parameters: Optional[Dict] = None):
        defaults = dict(
            base_grid_size_percent=1.5,
            num_grids=24,
            price_range_percent=48.0,
            min_strength=0.12,
            vol_threshold=0.006,
            rebalance_threshold=0.6,
            anchor_lookback=18,
            atr_period=17,
            step_vol_mult=1.55,
            skew_strength=0.75,
            zscore_limit=1.9,
            cooldown_bars=0,
            # V3 specific knobs
            dual_anchor_fast=26,
            dual_anchor_slow=96,
            pyramid_multiplier=1.6,
            pyramid_slope_fast=0.0018,
            pyramid_slope_slow=0.0009,
            breakout_window=34,
            breakout_z_threshold=1.9,
            micro_grid_span=3,
            micro_grid_decay=6,
            trailing_atr_mult=2.8,
            volatility_compression_pct=0.008,
            compression_bars=8,
            max_active_levels=5,
        )
        if parameters:
            defaults.update(parameters)
        super().__init__(defaults)
        self._micro_state = _MicroGridState(active=False, levels=[], bars_remaining=0)
        self._combo_anchor: float = 0.0
        self._compression_counter = 0

    # ------------------------------------------------------------------
    # Indicator calculations
    # ------------------------------------------------------------------
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        df = super().calculate_indicators(data)
        if df.empty:
            return df

        price = df['price'] if 'price' in df.columns else df['close']
        p = self.parameters

        # Dual anchors – fast/slow EMA and a volume-weighted anchor
        fast_n = int(max(2, p['dual_anchor_fast']))
        slow_n = int(max(fast_n + 1, p['dual_anchor_slow']))
        df['anchor_fast'] = price.ewm(span=fast_n, adjust=False).mean()
        df['anchor_slow'] = price.ewm(span=slow_n, adjust=False).mean()

        if 'volume' in df.columns and df['volume'].abs().sum() > 0:
            tp = (df['high'] + df['low'] + df['close']) / 3.0
            vol_anchor = ((tp * df['volume']).rolling(window=slow_n, min_periods=3).sum() /
                (df['volume'].rolling(window=slow_n, min_periods=3).sum().replace(0, np.nan)))
            df['anchor_volume'] = vol_anchor.fillna(method='bfill').fillna(df['anchor_slow'])
        else:
            df['anchor_volume'] = df['anchor_slow']

        df['anchor_combo'] = (
            df['anchor_fast'] * 0.4 + df['anchor_slow'] * 0.4 + df['anchor_volume'] * 0.2
        ).fillna(method='bfill').fillna(price)

        # Store latest anchor for grid rebuilding downstream
        self._combo_anchor = float(df['anchor_combo'].iloc[-1])

        # Donchian channel for breakout confirmation
        window = max(5, int(p['breakout_window']))
        df['donch_hi_overdrive'] = price.rolling(window, min_periods=2).max()
        df['donch_lo_overdrive'] = price.rolling(window, min_periods=2).min()

        # Compute EMA slopes for pyramiding decisions (per-bar change)
        df['anchor_fast_slope'] = df['anchor_fast'].diff()
        df['anchor_slow_slope'] = df['anchor_slow'].diff()

        # Recenter grid using combo anchor for the latest bar to lean into trend
        last_row = df.iloc[-1].copy()
        last_row['anchor'] = self._combo_anchor
        df.loc[df.index[-1], 'anchor'] = self._combo_anchor
        try:
            self._setup_grid_levels(last_row, df)
        except Exception:
            # parent handles integrity; if recenter fails we keep previous state
            pass

        return df

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------
    def _activate_micro_grid(self, price: float, step_pct: float) -> None:
        span = max(1, int(self.parameters.get('micro_grid_span', 3)))
        micro_step = max(1e-6, step_pct * 0.6)
        levels: List[Dict[str, float]] = []
        for idx in range(1, span + 1):
            lvl = price * (1.0 + micro_step * idx)
            levels.append({'level': lvl, 'type': 'sell', 'distance': idx})
        self._micro_state = _MicroGridState(active=True, levels=levels, bars_remaining=int(self.parameters.get('micro_grid_decay', 6)))

    def _decay_micro_grid(self) -> None:
        if self._micro_state.bars_remaining <= 0:
            self._micro_state = _MicroGridState(active=False, levels=[], bars_remaining=0)
            return
        self._micro_state.bars_remaining -= 1
        if self._micro_state.bars_remaining <= 0:
            self._micro_state = _MicroGridState(active=False, levels=[], bars_remaining=0)

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        df = self.calculate_indicators(df)

        for col in ('buy_signal', 'sell_signal', 'signal_strength'):
            if col not in df.columns:
                df[col] = False if col.endswith('_signal') else 0.0

        p = self.parameters
        max_levels = int(p.get('max_active_levels', 5))
        compression_threshold = float(p.get('volatility_compression_pct', 0.008))
        compression_bars = max(1, int(p.get('compression_bars', 8)))

        for i in range(len(df)):
            row = df.iloc[i]
            price = float(row['price'] if 'price' in df.columns else row['close'])
            step_pct = self._dynamic_step_pct(row)
            atr_pct = float(row.get('atr_pct', 0.0))
            vol = float(row.get('vol', 0.0))
            zscore = float(row.get('zscore', 0.0))

            # Volatility compression watchdog
            if vol <= compression_threshold:
                self._compression_counter += 1
            else:
                self._compression_counter = 0
            force_profit_take = self._compression_counter >= compression_bars

            # Manage breakout micro-grid lifecycle
            donch_hi = float(row.get('donch_hi_overdrive', np.nan))
            breakout = np.isfinite(donch_hi) and price >= donch_hi and zscore >= float(p.get('breakout_z_threshold', 1.9))
            if breakout:
                self._activate_micro_grid(price, step_pct)
            else:
                if self._micro_state.active:
                    self._decay_micro_grid()

            # Prepare sell level set with optional micro-grid overlay
            sell_levels = [lvl for lvl in self.grid_levels if lvl.get('type') == 'sell']
            if self._micro_state.active and self._micro_state.levels:
                sell_levels = sell_levels + self._micro_state.levels

            buy_levels = [lvl for lvl in self.grid_levels if lvl.get('type') == 'buy']

            min_strength = float(p.get('min_strength', 0.12))
            allow_entries = len(buy_levels) <= max_levels
            allow_sells = True

            # Pyramiding gate
            pyramid_fast = float(row.get('anchor_fast_slope', 0.0)) >= float(p.get('pyramid_slope_fast', 0.0018))
            pyramid_slow = float(row.get('anchor_slow_slope', 0.0)) >= float(p.get('pyramid_slope_slow', 0.0009))
            allow_pyramid = pyramid_fast and pyramid_slow

            tol = float(np.clip(0.25 * step_pct + 0.1 * atr_pct, 0.001, 0.02))

            best_strength = 0.0
            buy_hit = False
            sell_hit = False

            if allow_entries:
                for lvl in buy_levels:
                    rel = (price - lvl['level']) / max(lvl['level'], 1e-9)
                    if -tol <= rel <= tol:
                        strength = 1.0 - abs(rel) / max(step_pct, 1e-6)
                        strength = np.clip(strength, 0.0, 1.0)
                        if allow_pyramid:
                            strength *= float(p.get('pyramid_multiplier', 1.6))
                        if strength > best_strength:
                            best_strength = strength
                            buy_hit = True
                            sell_hit = False
                        break

            if allow_sells and not buy_hit:
                first_sell_skipped = False
                for lvl in sell_levels:
                    rel = (lvl['level'] - price) / max(lvl['level'], 1e-9)
                    if -tol <= rel <= tol:
                        if not first_sell_skipped and breakout:
                            first_sell_skipped = True
                            continue
                        strength = np.clip(1.0 - abs(rel) / max(step_pct, 1e-6), 0.0, 1.0)
                        if force_profit_take:
                            strength = max(strength, 0.8)
                        if strength > best_strength:
                            best_strength = strength
                            buy_hit = False
                            sell_hit = True
                        break

            if best_strength >= min_strength:
                df.iat[i, df.columns.get_loc('signal_strength')] = float(np.clip(best_strength, 0.0, 1.0))
                if buy_hit:
                    df.iat[i, df.columns.get_loc('buy_signal')] = True
                if sell_hit:
                    df.iat[i, df.columns.get_loc('sell_signal')] = True

        return df

    @classmethod
    def parameter_space(cls):
        p = super().parameter_space()
        p.update({
            'dual_anchor_fast': (12, 48),
            'dual_anchor_slow': (64, 160),
            'pyramid_multiplier': (1.2, 2.2),
            'pyramid_slope_fast': (0.0008, 0.003),
            'pyramid_slope_slow': (0.0004, 0.002),
            'breakout_window': (20, 60),
            'breakout_z_threshold': (1.5, 2.6),
            'micro_grid_span': (2, 4),
            'micro_grid_decay': (4, 12),
            'trailing_atr_mult': (2.0, 3.4),
            'volatility_compression_pct': (0.004, 0.012),
            'compression_bars': (4, 12),
            'max_active_levels': (3, 8),
        })
        return p
