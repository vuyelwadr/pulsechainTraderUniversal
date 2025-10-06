"""MomentumRegimeV3Fusion

Regime-adaptive extension of TMCCCBCIHybridStrategy that fuses mean-reversion
and breakout logic using real volatility ratios and OBV thrust.
"""
from __future__ import annotations

from .base_strategy import BaseStrategy
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .custom.tmcc_cbci_hybrid_strategy import TMCCCBCIHybridStrategy


class MomentumRegimeV3Fusion(TMCCCBCIHybridStrategy, BaseStrategy):
    def __init__(self, parameters: Optional[Dict] = None):
        defaults = dict(
            ema_len=27,
            atr_len=18,
            mul=1.32,
            rsi_period=12,
            cb_low=24,
            cb_high=76,
            signal_threshold=0.7,
            vol_ratio_lookback_fast=3,
            vol_ratio_lookback_slow=21,
            kama_fast=12,
            kama_slow=30,
            obv_smooth=5,
            chandelier_atr=3.2,
            expansion_vol_ratio=0.65,
            max_position_multiplier=2.4,
        )
        if parameters:
            defaults.update(parameters)
        super().__init__(defaults)

    # ------------------------------------------------------------------
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        df = super().calculate_indicators(data)
        if df.empty:
            return df

        p = self.parameters
        price = df['price'] if 'price' in df.columns else df['close']

        # Volatility ratios
        fast_n = max(2, int(p.get('vol_ratio_lookback_fast', 3)))
        slow_n = max(fast_n + 1, int(p.get('vol_ratio_lookback_slow', 21)))
        returns = price.pct_change().fillna(0.0)
        vol_fast = returns.rolling(fast_n, min_periods=1).std().fillna(0.0)
        vol_slow = returns.rolling(slow_n, min_periods=1).std().replace(0, np.nan)
        df['vol_ratio'] = (vol_fast / vol_slow).fillna(0.0)

        # Adaptive KAMA band
        kama_fast = max(2, int(p.get('kama_fast', 12)))
        kama_slow = max(kama_fast + 1, int(p.get('kama_slow', 30)))
        kama = price.ewm(span=kama_fast, adjust=False).mean()
        kama_slow_val = price.ewm(span=kama_slow, adjust=False).mean()
        df['kama_fast'] = kama
        df['kama_slow'] = kama_slow_val

        atr = df['atr'] if 'atr' in df.columns else price.rolling(int(p['atr_len'])).std().fillna(method='bfill')
        df['breakout_trigger'] = kama + 1.5 * atr.rolling(kama_fast, min_periods=1).mean().fillna(atr)

        # OBV + acceleration
        if 'volume' in df.columns:
            obv = (np.sign(price.diff().fillna(0.0)) * df['volume']).cumsum()
        else:
            obv = price.diff().fillna(0.0).cumsum()
        smooth = max(1, int(p.get('obv_smooth', 5)))
        df['obv'] = obv
        df['obv_ema'] = obv.ewm(span=smooth, adjust=False).mean()
        df['obv_accel'] = df['obv_ema'].diff()

        # Chandelier stop reference
        chand_mult = float(p.get('chandelier_atr', 3.2))
        df['chandelier_stop'] = price - chand_mult * atr

        return df

    # ------------------------------------------------------------------
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        df = self.calculate_indicators(df)

        for col in ('buy_signal', 'sell_signal', 'signal_strength'):
            if col not in df.columns:
                df[col] = False if col.endswith('_signal') else 0.0

        expansion_cutoff = float(self.parameters.get('expansion_vol_ratio', 0.65))
        max_pos_mult = float(self.parameters.get('max_position_multiplier', 2.4))
        base_threshold = float(self.parameters.get('signal_threshold', 0.7))

        for i in range(len(df)):
            row = df.iloc[i]
            price = float(row['price'] if 'price' in df.columns else row['close'])
            vol_ratio = float(row.get('vol_ratio', 0.0))
            cbci = float(row.get('cbci', 50.0))
            cbci_delta = float(df['cbci'].iloc[i] - df['cbci'].iloc[i-1]) if i > 0 else 0.0
            kama_fast = float(row.get('kama_fast', price))
            trigger = float(row.get('breakout_trigger', price))
            obv_accel = float(row.get('obv_accel', 0.0))
            chandelier = float(row.get('chandelier_stop', price * 0.7))
            atr = float(row.get('atr', 0.0))

            buy = False
            sell = False
            strength = 0.0

            if vol_ratio < expansion_cutoff:
                # Contraction regime: mean-reversion using TMCC bands + CBCI slope
                dn = float(row.get('dn', price))
                up = float(row.get('up', price))
                mid = float(row.get('mid', price))
                rsi = float(row.get('rsi', 50.0))
                # Upward slope of CBCI (cbci_delta >0) signals recovery
                if price <= dn and cbci_delta > 0 and rsi < float(self.parameters['cb_low']):
                    buy = True
                    strength = max(base_threshold, min(1.0, abs(dn - price) / max(1e-6, dn)))
                elif price >= up and cbci_delta < 0 and rsi > float(self.parameters['cb_high']):
                    sell = True
                    strength = max(base_threshold, min(1.0, abs(price - up) / max(1e-6, up)))
                # tighten stops if price moves back to mid
                if sell and price <= mid:
                    sell = False
            else:
                # Expansion regime: breakout follow-through with OBV thrust
                if price > trigger and obv_accel > 0:
                    buy = True
                    percentile = np.clip(vol_ratio / max(1e-6, expansion_cutoff), 1.0, max_pos_mult)
                    strength = max(base_threshold, min(1.0, 0.6 + 0.2 * percentile))
                elif price < chandelier:
                    sell = True
                    strength = max(base_threshold, min(1.0, (chandelier - price) / max(atr, 1e-6)))

            if strength >= base_threshold:
                df.iat[i, df.columns.get_loc('signal_strength')] = float(np.clip(strength, 0.0, 1.0))
                if buy:
                    df.iat[i, df.columns.get_loc('buy_signal')] = True
                if sell:
                    df.iat[i, df.columns.get_loc('sell_signal')] = True

        return df

    @classmethod
    def parameter_space(cls):
        base = super().parameter_space()
        base.update({
            'vol_ratio_lookback_fast': (2, 6),
            'vol_ratio_lookback_slow': (14, 35),
            'kama_fast': (8, 18),
            'kama_slow': (22, 40),
            'obv_smooth': (3, 9),
            'chandelier_atr': (2.4, 3.6),
            'expansion_vol_ratio': (0.5, 0.9),
            'max_position_multiplier': (1.6, 2.8),
        })
        return base
