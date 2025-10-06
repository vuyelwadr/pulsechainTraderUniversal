"""
Regime Router Strategy

Switches between trend breakout and range mean-reversion using rolling R² of
log-price and ATR context; aligns with refined pack 399 design.
"""
from typing import Dict
import numpy as np
import pandas as pd
from strategies.base_strategy import BaseStrategy


def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=max(2, int(n)), adjust=False).mean()


def _atr(df: pd.DataFrame, n: int) -> pd.Series:
    h, l, c = df['high'], df['low'], df['close']
    tr = np.maximum(h - l, np.maximum((h - c.shift(1)).abs(), (l - c.shift(1)).abs()))
    return pd.Series(tr).rolling(max(2, int(n))).mean()


def _rolling_r2(logp: pd.Series, n: int) -> pd.Series:
    y = logp.astype(float)
    x = np.arange(len(y))
    r2 = pd.Series(np.nan, index=y.index)
    n = max(5, int(n))
    for i in range(n, len(y)):
        yy = y.iloc[i - n : i]
        xx = x[i - n : i]
        xx = (xx - xx.mean()) / (xx.std() + 1e-12)
        yy = (yy - yy.mean()) / (yy.std() + 1e-12)
        r = np.corrcoef(xx, yy)[0, 1]
        r2.iloc[i] = r * r
    return r2


class RegimeRouterStrategy(BaseStrategy):
    def __init__(self, parameters: Dict = None):
        p = {
            'r2_len': 100,
            'r2_trend_thr': 0.3,
            'ema_base': 50,
            'atr_len': 14,
            'keltner_len': 34,
            'keltner_mult': 1.5,
            'breakout_atr_mult': 0.5,
            'signal_threshold': 0.6,
            'timeframe_minutes': 60,
        }
        if parameters:
            p.update(parameters)
        super().__init__('RegimeRouterStrategy', p)

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        d = data.copy()
        if 'close' not in d and 'price' in d:
            d['close'] = d['price']
        for c in ('high', 'low', 'close'):
            if c not in d:
                d[c] = d.get('close', pd.Series(index=d.index)).fillna(method='ffill')
        atr = _atr(d, int(self.parameters['atr_len']))
        ema_b = _ema(d['close'], int(self.parameters['ema_base']))
        logp = np.log(d['close'].replace(0, np.nan)).replace([-np.inf, np.inf], np.nan).ffill()
        r2 = _rolling_r2(logp, int(self.parameters['r2_len'])).fillna(0.0)
        mid = _ema(d['close'], int(self.parameters['keltner_len']))
        up = mid + float(self.parameters['keltner_mult']) * atr
        dn = mid - float(self.parameters['keltner_mult']) * atr
        d['atr'] = atr; d['ema_base'] = ema_b; d['r2'] = r2; d['kel_up'] = up; d['kel_dn'] = dn; d['kel_mid'] = mid
        return d

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        d = self.calculate_indicators(data)
        price = d.get('price', d.get('close', d['close']))
        trending = d['r2'] > float(self.parameters['r2_trend_thr'])
        ranging = ~trending
        buy_tr = trending & (price > d['ema_base'] + float(self.parameters['breakout_atr_mult']) * d['atr'])
        sell_tr = trending & (price < d['ema_base'] - float(self.parameters['breakout_atr_mult']) * d['atr'])
        buy_rg = ranging & (price <= d['kel_dn'])
        sell_rg = ranging & (price >= d['kel_up'])
        buy = buy_tr | buy_rg
        sell = sell_tr | sell_rg
        span = (d['kel_up'] - d['kel_dn']).replace(0, np.nan)
        band_pos = ((price - d['kel_dn']) / span).clip(0, 1.0)
        st = (0.5 * d['r2'] + 0.5 * (1 - (band_pos - 0.5).abs() * 2)).clip(0, 1.0)
        thr = float(self.parameters['signal_threshold'])
        st = st.where(buy | sell, 0.0)
        st[(buy | sell) & (st < thr)] = thr
        d['buy_signal'] = buy.fillna(False)
        d['sell_signal'] = sell.fillna(False)
        d['signal_strength'] = st.fillna(0.0)
        return d

