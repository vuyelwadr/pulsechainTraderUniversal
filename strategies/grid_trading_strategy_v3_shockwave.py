"""GridTradingStrategyV3Shockwave

State-machine driven aggressive grid that cycles through accumulation,
expansion, and blow-off regimes, all computed from real OHLCV data.
"""
from __future__ import annotations

from .base_strategy import BaseStrategy
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .grid_trading_strategy_v2_aggressive import GridTradingStrategyV2Aggressive


class GridTradingStrategyV3Shockwave(GridTradingStrategyV2Aggressive, BaseStrategy):
    PHASE_ACCUMULATION = 'accumulation'
    PHASE_EXPANSION = 'expansion'
    PHASE_BLOWOFF = 'blowoff'
    PHASE_NEUTRAL = 'neutral'

    def __init__(self, parameters: Optional[Dict] = None):
        defaults = dict(
            min_step_pct=0.011,
            atr_mult=1.48,
            num_grids=18,
            sell_spacing_mult_bull=1.52,
            buy_spacing_mult_bear=1.78,
            recenter_threshold_pct=0.08,
            recenter_dwell_bars=4,
            center_lookback=22,
            cooldown_bars=0,
            min_strength=0.62,
            min_edge_pct=0.0075,
            breakout_window=28,
            breakout_slope_min=0.0006,
            strong_slope_min=0.0012,
            trail_atr_mult=3.3,
            no_sell_above_ema_fast_in_bull=1,
            allow_bear_breakout_sell=1,
            keltner_period=20,
            keltner_mult=2.2,
            bollinger_period=45,
            bollinger_std=2.4,
            macd_fast=26,
            macd_slow=78,
            macd_signal=9,
            blowoff_z=2.4,
            blowoff_timeout=10,
            exposure_cap=2.4,
            rsi_period=9,
        )
        if parameters:
            defaults.update(parameters)
        super().__init__(defaults)
        self._phase = self.PHASE_NEUTRAL
        self._blowoff_timer = 0
        self._exposure_counter = 0.0

    # ------------------------------------------------------------------
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        df = super().calculate_indicators(data)
        if df.empty:
            return df

        p = self.parameters
        price = df['price'] if 'price' in df.columns else df['close']

        # Keltner Channel using ATR from parent
        kc_period = max(2, int(p.get('keltner_period', 20)))
        atr = df['atr'] if 'atr' in df.columns else price.rolling(14).std().fillna(method='bfill')
        ema_mid = price.ewm(span=kc_period, adjust=False).mean()
        df['keltner_mid'] = ema_mid
        df['keltner_up'] = ema_mid + float(p['keltner_mult']) * atr
        df['keltner_down'] = ema_mid - float(p['keltner_mult']) * atr

        # Bollinger for blow-off exits
        bb_period = max(5, int(p.get('bollinger_period', 45)))
        bb_std = float(p.get('bollinger_std', 2.4))
        rolling_mean = price.rolling(bb_period, min_periods=5).mean()
        rolling_std = price.rolling(bb_period, min_periods=5).std().fillna(0.0)
        df['bollinger_up'] = rolling_mean + bb_std * rolling_std

        # MACD histogram for expansion state
        fast = price.ewm(span=max(2, int(p['macd_fast'])), adjust=False).mean()
        slow = price.ewm(span=max(3, int(p['macd_slow'])), adjust=False).mean()
        macd = fast - slow
        signal = macd.ewm(span=max(2, int(p['macd_signal'])), adjust=False).mean()
        df['macd_hist'] = macd - signal

        # RSI for accumulation detection
        delta = price.diff()
        gain = delta.clip(lower=0.0)
        loss = -delta.clip(upper=0.0)
        rsi_period = max(2, int(p.get('rsi_period', 9)))
        avg_gain = gain.rolling(rsi_period).mean()
        avg_loss = loss.rolling(rsi_period).mean().replace(0, np.nan)
        rs = avg_gain / avg_loss
        df['rsi_fast'] = (100 - 100 / (1 + rs)).fillna(50.0)

        return df

    # ------------------------------------------------------------------
    def _update_phase(self, row: pd.Series) -> None:
        price = float(row['price'] if 'price' in row else row['close'])
        z = float(row.get('zscore', 0.0))
        ema_slope = float(row.get('ema_fast_slope', 0.0))
        macd_hist = float(row.get('macd_hist', 0.0))
        rsi = float(row.get('rsi_fast', 50.0))
        k_down = float(row.get('keltner_down', price))
        bb_up = float(row.get('bollinger_up', price * 1.05))

        # Blow-off takes precedence
        donch_hi = float(row.get('donch_hi', price))
        if price >= donch_hi and z >= float(self.parameters.get('blowoff_z', 2.4)):
            self._phase = self.PHASE_BLOWOFF
            self._blowoff_timer = int(self.parameters.get('blowoff_timeout', 10))
            return

        if self._phase == self.PHASE_BLOWOFF:
            self._blowoff_timer -= 1
            if self._blowoff_timer <= 0 or price <= bb_up:
                self._phase = self.PHASE_NEUTRAL
            return

        if price <= k_down and rsi < 32:
            self._phase = self.PHASE_ACCUMULATION
            return

        if ema_slope > 0 and macd_hist > 0:
            self._phase = self.PHASE_EXPANSION
            return

        self._phase = self.PHASE_NEUTRAL

    # ------------------------------------------------------------------
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        df = self.calculate_indicators(df)

        for col in ('buy_signal', 'sell_signal', 'signal_strength'):
            if col not in df.columns:
                df[col] = False if col.endswith('_signal') else 0.0

        exposure_cap = float(self.parameters.get('exposure_cap', 2.4))
        size_schedule = [1.0, 1.35, 1.75, 2.1]
        threshold = float(self.parameters.get('min_strength', 0.62))
        phase_timeout = int(self.parameters.get('blowoff_timeout', 10))

        for i in range(len(df)):
            row = df.iloc[i]
            self._update_phase(row)
            price = float(row['price'] if 'price' in df.columns else row['close'])
            step_pct = self._dynamic_step_pct(row)
            atr_pct = float(row.get('atr_pct', 0.0))
            tol = float(np.clip(0.25 * step_pct + 0.1 * atr_pct, 0.001, 0.02))

            # Reset exposure counter slowly when sells already triggered earlier
            if bool(row.get('sell_signal', False)):
                self._exposure_counter = max(0.0, self._exposure_counter - 1.0)

            allow_buys = self._exposure_counter < exposure_cap
            allow_sells = True

            pending_buy_strength = 0.0
            pending_sell_strength = 0.0
            buy_hit = False
            sell_hit = False

            buy_levels = [lvl for lvl in self._levels if lvl <= self._center]
            sell_levels = [lvl for lvl in self._levels if lvl >= self._center]

            if self._phase == self.PHASE_ACCUMULATION and allow_buys:
                for idx, lvl in enumerate(buy_levels[:len(size_schedule)]):
                    rel = (price - lvl) / max(lvl, 1e-9)
                    if -tol <= rel <= tol:
                        pending_buy_strength = np.clip(1.0 - abs(rel) / max(step_pct, 1e-6), 0.0, 1.0)
                        pending_buy_strength *= size_schedule[idx]
                        buy_hit = True
                        break

            elif self._phase == self.PHASE_EXPANSION:
                allow_sells = False
                if allow_buys and buy_levels:
                    lvl = buy_levels[0]
                    rel = (price - lvl) / max(lvl, 1e-9)
                    if -tol <= rel <= tol:
                        pending_buy_strength = np.clip(1.0 - abs(rel) / max(step_pct, 1e-6), 0.0, 1.0)
                        buy_hit = True

            elif self._phase == self.PHASE_BLOWOFF:
                donch_hi = float(row.get('donch_hi', price))
                z = float(row.get('zscore', 0.0))
                if allow_buys and price >= donch_hi and z >= float(self.parameters.get('blowoff_z', 2.4)):
                    pending_buy_strength = 0.85
                    buy_hit = True
                    self._blowoff_timer = phase_timeout
                for lvl in sell_levels[:3]:
                    rel = (lvl - price) / max(lvl, 1e-9)
                    if -tol <= rel <= tol:
                        pending_sell_strength = max(np.clip(1.0 - abs(rel) / max(step_pct, 1e-6), 0.0, 1.0), 0.7)
                        sell_hit = True
                        break

            else:  # neutral fallback
                if allow_buys and buy_levels:
                    rel = (price - buy_levels[0]) / max(buy_levels[0], 1e-9)
                    if -tol <= rel <= tol:
                        pending_buy_strength = np.clip(1.0 - abs(rel) / max(step_pct, 1e-6), 0.0, 1.0)
                        buy_hit = True
                if not buy_hit and allow_sells:
                    for lvl in sell_levels:
                        rel = (lvl - price) / max(lvl, 1e-9)
                        if -tol <= rel <= tol:
                            pending_sell_strength = np.clip(1.0 - abs(rel) / max(step_pct, 1e-6), 0.0, 1.0)
                            sell_hit = True
                            break

            best_strength = max(pending_buy_strength, pending_sell_strength)
            if best_strength >= threshold:
                df.iat[i, df.columns.get_loc('signal_strength')] = float(np.clip(best_strength, 0.0, 1.0))
                if buy_hit:
                    df.iat[i, df.columns.get_loc('buy_signal')] = True
                    self._exposure_counter += 1.0
                if sell_hit:
                    df.iat[i, df.columns.get_loc('sell_signal')] = True
                    self._exposure_counter = max(0.0, self._exposure_counter - 1.0)

            if self._phase != self.PHASE_BLOWOFF:
                self._blowoff_timer = 0

        return df

    @classmethod
    def parameter_space(cls):
        p = super().parameter_space()
        p.update({
            'keltner_period': (10, 40),
            'keltner_mult': (1.6, 2.8),
            'bollinger_period': (20, 70),
            'bollinger_std': (1.6, 3.2),
            'macd_fast': (12, 40),
            'macd_slow': (50, 120),
            'macd_signal': (5, 15),
            'blowoff_z': (1.8, 3.0),
            'blowoff_timeout': (6, 16),
            'exposure_cap': (1.5, 3.0),
            'rsi_period': (6, 14),
        })
        return p
