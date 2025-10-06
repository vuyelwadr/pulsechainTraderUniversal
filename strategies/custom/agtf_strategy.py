"""
Adaptive Grid + Trend Filter (AGTF)

Idea
 - Run a grid only when a slow/long trend filter is not strongly bearish.
 - Set grid step adaptively = max(atr_mult * ATR, min_step_pct * price).
 - Cap inventory via num_rungs and only emit one side of signals per bar.
 - Strength blends proximity to rung, volatility, and trend quality.

This file follows the BaseStrategy contract used across the repo:
 - calculate_indicators(data) -> returns data with extra columns
 - generate_signals(data) -> adds buy_signal/sell_signal/signal_strength

Notes
 - Execution/risk is managed by the backtest engine; we only emit signals.
 - No synthetic data is introduced; indicators derive from provided prices.
"""
from typing import Dict, List
import logging
import numpy as np
import pandas as pd

from strategies.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


class AdaptiveGridTrendStrategy(BaseStrategy):
    """
    Adaptive Grid with Trend Filter

    Parameters
    - atr_period: ATR length for volatility (default 14)
    - ema_fast: Fast EMA for slope/trend (default 50)
    - ema_slow: Slow EMA for regime filter (default 200)
    - atr_mult: Multiplier for ATR-based step (default 0.8)
    - min_step_pct: Minimum grid step as % of price (default 0.006)
    - num_rungs: Max rungs each side (default 6)
    - min_strength: Minimum signal strength to emit (default 0.45)
    - slope_thresh: Minimum EMA slope (bp per bar) to consider bullish (default 0.0)
    - side_when_bear: 0=both, 1=buy-only, 2=disable (default 1)
    - timeframe_minutes: analysis timeframe (default 60)
    """

    def __init__(self, parameters: Dict = None):
        params = {
            'atr_period': 14,
            'ema_fast': 50,
            'ema_slow': 200,
            'atr_mult': 0.8,
            'min_step_pct': 0.006,
            'num_rungs': 6,
            'min_strength': 0.45,
            'slope_thresh': 0.0,
            'side_when_bear': 1,
            'timeframe_minutes': 60,
        }
        if parameters:
            params.update(parameters)
        super().__init__('AdaptiveGridTrendStrategy', params)
        self._levels: List[float] = []
        self._center: float = 0.0

    # --- helpers ---
    @staticmethod
    def _ema(series: pd.Series, span: int) -> pd.Series:
        return series.ewm(span=span, adjust=False).mean()

    @staticmethod
    def _atr(df: pd.DataFrame, period: int) -> pd.Series:
        h, l, c = df['high'], df['low'], df['close'].shift(1)
        tr = pd.concat([(h - l).abs(), (h - c).abs(), (l - c).abs()], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    def _rebuild_levels(self, price: float, step: float, num: int):
        self._center = float(price)
        self._levels = []
        for i in range(1, max(1, int(num)) + 1):
            self._levels.append(self._center * (1.0 - step * i))  # buy rungs
            self._levels.append(self._center * (1.0 + step * i))  # sell rungs

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        if not self.validate_data(data):
            return data
        df = data.copy()
        if len(df) < max(220, self.parameters['ema_slow'] + 10):
            return df
        # Basic fields
        df['price'] = df.get('price', df.get('close', df['close']))
        # Trend + volatility
        df['ema_fast'] = self._ema(df['price'], int(self.parameters['ema_fast']))
        df['ema_slow'] = self._ema(df['price'], int(self.parameters['ema_slow']))
        df['ema_fast_slope'] = df['ema_fast'].diff()
        df['atr'] = self._atr(df.assign(close=df['price']), int(self.parameters['atr_period']))
        df['atrp'] = (df['atr'] / df['price']).clip(lower=1e-9)
        # Regime flags
        df['bullish'] = (df['ema_fast'] >= df['ema_slow']) & (df['ema_fast_slope'] >= float(self.parameters['slope_thresh']))
        df['bearish'] = (df['ema_fast'] < df['ema_slow'])
        # Grid step (percent)
        step_pct = np.maximum(float(self.parameters['atr_mult']) * df['atrp'], float(self.parameters['min_step_pct']))
        df['step_pct'] = step_pct
        # Rebuild levels using recent center
        self._rebuild_levels(df['price'].iloc[-1], float(step_pct.iloc[-1]), int(self.parameters['num_rungs']))
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
        for i in range(len(df)):
            px = float(df.iloc[i]['price'])
            step = float(df.iloc[i]['step_pct'])
            is_bull = bool(df.iloc[i].get('bullish', False))
            is_bear = bool(df.iloc[i].get('bearish', False))
            # decide allowed sides
            allow_buy = True
            allow_sell = True
            if is_bear:
                if side_bear == 1:  # buy-only in bear
                    allow_sell = False
                elif side_bear == 2:  # disable in bear
                    allow_buy = False; allow_sell = False
            # check proximity to levels
            # nearest buy below and sell above
            if allow_buy:
                buys = [lvl for lvl in self._levels if lvl <= self._center and px <= lvl * (1.0 + 0.005) and px >= lvl * (1.0 - 0.01)]
                if buys:
                    dist = abs(px - buys[0]) / max(px, 1e-9)
                    strength = float(np.clip(1.0 - (dist / max(step, 1e-6)), 0.0, 1.0))
                    strength *= 0.6 + 0.4 * (1.0 if is_bull else 0.5)  # trend bonus
                    if strength >= min_strength:
                        df.iat[i, df.columns.get_loc('buy_signal')] = True
                        df.iat[i, df.columns.get_loc('signal_strength')] = strength
                        continue
            if allow_sell:
                sells = [lvl for lvl in self._levels if lvl >= self._center and px >= lvl * (1.0 - 0.005) and px <= lvl * (1.0 + 0.01)]
                if sells:
                    dist = abs(px - sells[0]) / max(px, 1e-9)
                    strength = float(np.clip(1.0 - (dist / max(step, 1e-6)), 0.0, 1.0))
                    strength *= 0.6 + 0.4 * (1.0 if is_bull else 0.5)
                    if strength >= min_strength:
                        df.iat[i, df.columns.get_loc('sell_signal')] = True
                        df.iat[i, df.columns.get_loc('signal_strength')] = strength
        return df

