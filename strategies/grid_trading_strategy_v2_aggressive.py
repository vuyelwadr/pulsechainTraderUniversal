"""
Grid Trading Strategy V2 — Aggressive Overlay

Adds a momentum breakout overlay and sell‑delay rules to V2 to chase
profits more assertively in trends while keeping grid harvesting in ranges.

Indicators derive solely from provided OHLC/price — no synthetic data.
"""
from typing import Dict, List
import numpy as np
import pandas as pd

from .base_strategy import BaseStrategy
from .grid_trading_strategy_v2 import GridTradingStrategyV2


class GridTradingStrategyV2Aggressive(GridTradingStrategyV2):
    """
    V2 Aggressive: Grid + trend breakout overlay

    Additions vs V2:
    - Donchian breakout buys (and optional downside sells)
    - Delay rung sells in strong bullish conditions (ride trends)
    - ATR‑based trailing exit in bull when momentum fades

    Parameters (extends V2)
    - breakout_window: Donchian channel window (default 20)
    - breakout_slope_min: Min EMA-fast slope for breakout confirm (default 0.0005)
    - strong_slope_min: Slope threshold to treat bull as strong (default 0.001)
    - trail_atr_mult: ATR multiple for trailing exit (default 2.0)
    - no_sell_above_ema_fast_in_bull: 0/1 gate to delay sells in strong bull (default 1)
    - allow_bear_breakout_sell: 0/1 to allow downside breakout sells (default 0)
    """

    def __init__(self, parameters: Dict = None):
        p = {
            'breakout_window': 20,
            'breakout_slope_min': 0.0005,
            'strong_slope_min': 0.001,
            'trail_atr_mult': 2.0,
            'no_sell_above_ema_fast_in_bull': 1,
            'allow_bear_breakout_sell': 0,
            'max_equity_drawdown_pct': 0.45,
        }
        if parameters:
            p.update(parameters)
        # Chain up to V2 with merged params
        super().__init__(p)

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        df = super().calculate_indicators(data)
        if df.empty:
            return df
        n = max(5, int(self.parameters.get('breakout_window', 20)))
        pr = df.get('price', df.get('close', df['close']))
        df['donch_hi'] = pr.rolling(n).max()
        df['donch_lo'] = pr.rolling(n).min()
        # Save ATR (absolute) for trailing use; V2 computed atrp; recompute atr here
        h, l, c1 = df['high'], df['low'], pr.shift(1)
        tr = pd.concat([(h - l).abs(), (h - c1).abs(), (l - c1).abs()], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14).mean()
        return df

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        if 'step_pct' not in df.columns or 'donch_hi' not in df.columns:
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

        # Pull common params from V2
        min_strength = float(self.parameters.get('min_strength', 0.55))
        cooldown_bars = max(0, int(self.parameters.get('cooldown_bars', 2)))
        min_edge = float(self.parameters.get('min_edge_pct', 0.02))
        tol_b_lo = float(self.parameters.get('tol_buy_lo', 0.01))
        tol_b_hi = float(self.parameters.get('tol_buy_hi', 0.005))
        tol_s_lo = float(self.parameters.get('tol_sell_lo', 0.005))
        tol_s_hi = float(self.parameters.get('tol_sell_hi', 0.01))
        side_bear = int(self.parameters.get('side_when_bear', 1))

        # Aggressive knobs
        bo_slope = float(self.parameters.get('breakout_slope_min', 0.0005))
        strong_slope = float(self.parameters.get('strong_slope_min', 0.001))
        trail_k = float(self.parameters.get('trail_atr_mult', 2.0))
        no_sell_above_fast = int(self.parameters.get('no_sell_above_ema_fast_in_bull', 1)) == 1
        allow_bear_breakout_sell = int(self.parameters.get('allow_bear_breakout_sell', 0)) == 1

        cooldown = 0
        in_position = False
        entry_price = None
        position_stack: List[float] = []
        max_drawdown_pct = float(self.parameters.get('max_equity_drawdown_pct', 0.45))
        for i in range(len(df)):
            px = float(df.iloc[i]['price'])
            step = float(df.iloc[i]['step_pct'])
            is_bull = bool(df.iloc[i].get('bullish', False))
            is_bear = bool(df.iloc[i].get('bearish', False))
            ema_fast = float(df.iloc[i].get('ema_fast', px))
            ema_slope = float(df.iloc[i].get('ema_fast_slope', 0.0))
            donch_hi = float(df.iloc[i].get('donch_hi', np.nan)) if pd.notna(df.iloc[i].get('donch_hi', np.nan)) else None
            donch_lo = float(df.iloc[i].get('donch_lo', np.nan)) if pd.notna(df.iloc[i].get('donch_lo', np.nan)) else None
            atr = float(df.iloc[i].get('atr', 0.0))
            regime_on = bool(df.iloc[i].get('regime_on', True))

            if cooldown > 0:
                cooldown -= 1
                continue
            eff = max(step, float(self.parameters.get('min_step_pct', 0.02)))
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

            # 1) Rung buys
            if allow_buy:
                for idx, lvl in enumerate(buy_levels):
                    if px >= lvl * (1.0 - tol_b_lo) and px <= lvl * (1.0 + tol_b_hi):
                        dist = abs(px - lvl) / max(px, 1e-12)
                        strength = float(np.clip(1.0 - (dist / max(eff, 1e-6)), 0.0, 1.0))
                        strength *= (1.0 if is_bull else 0.8)
                        if buy_decay < 1.0:
                            strength *= buy_decay ** idx
                        if strength > best_strength:
                            best_strength = strength; want_buy = True; want_sell = False
                        break

            # 2) Rung sells (with aggressive sell delay in strong bull)
            if allow_sell and best_strength < 1.0:
                # Delay sells in strong bull above EMA-fast unless trailing exit triggers
                strong_bull = is_bull and ema_slope >= strong_slope and px >= ema_fast
                trailing_exit = (atr > 0 and px <= ema_fast - trail_k * atr)
                permit_sell = (not strong_bull) or trailing_exit
                if permit_sell:
                    for idx, lvl in enumerate(sell_levels):
                        if px <= lvl * (1.0 + tol_s_hi) and px >= lvl * (1.0 - tol_s_lo):
                            dist = abs(px - lvl) / max(px, 1e-12)
                            strength = float(np.clip(1.0 - (dist / max(eff, 1e-6)), 0.0, 1.0))
                            # in bull, slightly damp rung sells
                            strength *= (0.9 if is_bull else 1.0)
                            if sell_decay < 1.0:
                                strength *= sell_decay ** idx
                            if strength > best_strength:
                                best_strength = strength; want_buy = False; want_sell = True
                            break
                else:
                    # Explicitly gate sells above EMA-fast in strong bull if configured
                    if not no_sell_above_fast:
                        # fallback to normal rung logic
                        for idx, lvl in enumerate(sell_levels):
                            if px <= lvl * (1.0 + tol_s_hi) and px >= lvl * (1.0 - tol_s_lo):
                                dist = abs(px - lvl) / max(px, 1e-12)
                                strength = float(np.clip(1.0 - (dist / max(eff, 1e-6)), 0.0, 1.0))
                                if sell_decay < 1.0:
                                    strength *= sell_decay ** idx
                                if strength > best_strength:
                                    best_strength = strength; want_buy = False; want_sell = True
                                break

            # 3) Breakout overlay (buy): Donchian high + positive slope
            if allow_buy and donch_hi is not None and px >= donch_hi and ema_slope >= bo_slope:
                # Boost strength with slope magnitude (normalized)
                slope_rel = float(np.clip(abs(ema_slope) / max(px, 1e-12), 0.0, 0.01))
                bo_strength = float(np.clip(0.7 + 40.0 * slope_rel, 0.0, 1.0))
                if bo_strength > best_strength:
                    best_strength = bo_strength; want_buy = True; want_sell = False

            # 4) Optional downside breakout overlay (sell)
            if allow_bear_breakout_sell and allow_sell and donch_lo is not None and px <= donch_lo and ema_slope <= -bo_slope:
                slope_rel = float(np.clip(abs(ema_slope) / max(px, 1e-12), 0.0, 0.01))
                bo_strength = float(np.clip(0.7 + 40.0 * slope_rel, 0.0, 1.0))
                # In bear, sell overlay is un-dampened
                if bo_strength > best_strength:
                    best_strength = bo_strength; want_buy = False; want_sell = True

            if best_strength >= min_strength:
                if want_buy:
                    df.iat[i, df.columns.get_loc('buy_signal')] = True
                if want_sell:
                    df.iat[i, df.columns.get_loc('sell_signal')] = True
                df.iat[i, df.columns.get_loc('signal_strength')] = best_strength
                cooldown = cooldown_bars

            # Global drawdown fail-safe
            if in_position and max_drawdown_pct > 0 and position_stack:
                avg_entry = float(np.mean(position_stack))
                if px <= avg_entry * (1.0 - max_drawdown_pct):
                    df.iat[i, df.columns.get_loc('sell_signal')] = True
                    df.iat[i, df.columns.get_loc('signal_strength')] = max(
                        df.iat[i, df.columns.get_loc('signal_strength')], 0.9
                    )
                    in_position = False
                    entry_price = None
                    position_stack.clear()
                    continue

            if df.iat[i, df.columns.get_loc('sell_signal')]:
                in_position = False
                entry_price = None
                position_stack.clear()
            elif df.iat[i, df.columns.get_loc('buy_signal')]:
                in_position = True
                entry_price = px
                position_stack.append(px)

        return df

    @classmethod
    def parameter_space(cls):
        # Merge V2 base with aggressive knobs
        base = GridTradingStrategyV2.parameter_space()
        base.update({
            'breakout_window': (10, 80),
            'breakout_slope_min': (0.0002, 0.005),
            'strong_slope_min': (0.0005, 0.01),
            'trail_atr_mult': (1.0, 3.5),
            'no_sell_above_ema_fast_in_bull': (0, 1),
            'allow_bear_breakout_sell': (0, 1),
            'max_equity_drawdown_pct': (0.15, 0.40),
        })
        return base
