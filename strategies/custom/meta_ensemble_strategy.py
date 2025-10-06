"""
Meta Ensemble Strategy

Combines three sub-strategies (AdaptiveRegimeHybridStrategy, VolatilityBreakoutKAMAStrategy,
VWAPMomentumPullbackStrategy) using regime-aware dynamic weights.
"""
from typing import Dict, List
import numpy as np
import pandas as pd
from strategies.base_strategy import BaseStrategy
from strategies.custom.arh_strategy import AdaptiveRegimeHybridStrategy
from strategies.custom.vbk_strategy import VolatilityBreakoutKAMAStrategy
from strategies.custom.vwap_momentum_pullback_strategy import VWAPMomentumPullbackStrategy


def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=max(2, int(n)), adjust=False).mean()


def _atr(df: pd.DataFrame, n: int) -> pd.Series:
    h, l, c = df['high'], df['low'], df['close']
    tr = np.maximum(h - l, np.maximum((h - c.shift(1)).abs(), (l - c.shift(1)).abs()))
    return pd.Series(tr).rolling(max(2, int(n))).mean()


class MetaEnsembleStrategy(BaseStrategy):
    def __init__(self, parameters: Dict = None):
        p = {
            'ema_regime': 100,
            'atr_len': 14,
            'w_arh': 0.34,
            'w_vbk': 0.33,
            'w_vmp': 0.33,
            'trend_bias': 0.65,
            'range_bias': 0.55,
            'signal_threshold': 0.4,
            'timeframe_minutes': 60,
        }
        if parameters:
            p.update(parameters)
        super().__init__('MetaEnsembleStrategy', p)
        self._subs = [
            AdaptiveRegimeHybridStrategy(),
            VolatilityBreakoutKAMAStrategy(),
            VWAPMomentumPullbackStrategy(),
        ]

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        d = data.copy()
        price = d.get('price', d.get('close', d['close']))
        d['ema_regime'] = _ema(price, int(self.parameters['ema_regime']))
        d['ema_slope'] = d['ema_regime'].diff()
        d['atr'] = _atr(d.assign(close=price), int(self.parameters['atr_len']))
        return d

    def _run_subs(self, data: pd.DataFrame) -> List[pd.DataFrame]:
        outs = []
        for s in self._subs:
            outs.append(s.generate_signals(data))
        return outs

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        d = self.calculate_indicators(data)
        outs = self._run_subs(data)
        # ensure necessary columns exist
        for i, o in enumerate(outs):
            for col in ('buy_signal', 'sell_signal', 'signal_strength'):
                if col not in o.columns:
                    o[col] = False if col != 'signal_strength' else 0.0
            outs[i] = o
        trend_up = d['ema_slope'] > 0
        p = self.parameters
        w_arh = float(p['w_arh']); w_vbk = float(p['w_vbk']); w_vmp = float(p['w_vmp'])
        s = max(1e-9, w_arh + w_vbk + w_vmp)
        w_arh /= s; w_vbk /= s; w_vmp /= s
        # dynamic weights
        w_arh_dyn = np.where(trend_up, w_arh * float(p['trend_bias']), w_arh * float(p['range_bias']))
        w_vbk_dyn = np.where(trend_up, w_vbk * float(p['trend_bias']), w_vbk * (1.0 - float(p['range_bias'])))
        w_vmp_dyn = np.where(trend_up, w_vmp * (1.0 - float(p['trend_bias'])), w_vmp * float(p['range_bias']))
        strength = (
            w_arh_dyn * outs[0]['signal_strength'].values +
            w_vbk_dyn * outs[1]['signal_strength'].values +
            w_vmp_dyn * outs[2]['signal_strength'].values
        )
        buy = (outs[0]['buy_signal'] | outs[1]['buy_signal'] | outs[2]['buy_signal']).values
        sell = (outs[0]['sell_signal'] | outs[1]['sell_signal'] | outs[2]['sell_signal']).values
        thr = float(p['signal_threshold'])
        strength = np.where(buy | sell, np.maximum(strength, thr), 0.0)
        out = d.copy()
        out['buy_signal'] = pd.Series(buy, index=d.index)
        out['sell_signal'] = pd.Series(sell, index=d.index)
        out['signal_strength'] = pd.Series(strength, index=d.index).fillna(0.0)
        return out

