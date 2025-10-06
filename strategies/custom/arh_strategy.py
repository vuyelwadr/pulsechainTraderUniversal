"""
Adaptive Regime Hybrid (ARH)

Regime model:
- Trend regime: EMA fast > EMA slow and KAMA slope >= 0 → trade pullbacks in trend direction.
- Range regime: otherwise → mean reversion via RSI bands.

Signals: buy/sell booleans + signal_strength 0..1.
"""
from typing import Dict
import numpy as np
import pandas as pd
from strategies.base_strategy import BaseStrategy


class AdaptiveRegimeHybridStrategy(BaseStrategy):
    def __init__(self, parameters: Dict = None):
        params = {
            'ema_fast': 50,
            'ema_slow': 200,
            'kama_period': 30,
            'rsi_period': 14,
            'rsi_buy': 35,
            'rsi_sell': 65,
            'pullback_z': 0.5,
            'atr_period': 14,
            'min_strength': 0.5,
            'timeframe_minutes': 60,
        }
        if parameters:
            params.update(parameters)
        super().__init__('AdaptiveRegimeHybridStrategy', params)

    @staticmethod
    def _ema(s: pd.Series, span: int) -> pd.Series:
        return s.ewm(span=max(2, int(span)), adjust=False).mean()

    @staticmethod
    def _atr(df: pd.DataFrame, n: int) -> pd.Series:
        h, l, c = df['high'], df['low'], df['close'].shift(1)
        tr = pd.concat([(h-l).abs(), (h-c).abs(), (l-c).abs()], axis=1).max(axis=1)
        return tr.rolling(max(2, int(n))).mean()

    @staticmethod
    def _rsi(s: pd.Series, n: int) -> pd.Series:
        delta = s.diff()
        up = delta.clip(lower=0.0)
        dn = -delta.clip(upper=0.0)
        roll_up = up.rolling(n).mean()
        roll_dn = dn.rolling(n).mean().replace(0, np.nan)
        rs = roll_up / roll_dn
        return (100 - (100/(1+rs))).fillna(50.0)

    def _kama(self, s: pd.Series, n: int) -> pd.Series:
        # Simple KAMA approximation: adaptive EMA with efficiency ratio
        n = max(2, int(n))
        change = s.diff(n).abs()
        volatility = s.diff().abs().rolling(n).sum().replace(0, np.nan)
        er = (change / volatility).clip(0, 1).fillna(0)
        sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
        kama = s.copy()
        for i in range(1, len(s)):
            kama.iat[i] = kama.iat[i-1] + sc.iat[i] * (s.iat[i] - kama.iat[i-1])
        return kama

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        df['price'] = df.get('price', df.get('close', df['close']))
        df['ema_fast'] = self._ema(df['price'], int(self.parameters['ema_fast']))
        df['ema_slow'] = self._ema(df['price'], int(self.parameters['ema_slow']))
        df['kama'] = self._kama(df['price'], int(self.parameters['kama_period']))
        df['kama_slope'] = df['kama'].diff()
        df['rsi'] = self._rsi(df['price'], int(self.parameters['rsi_period']))
        df['atr'] = self._atr(df.assign(close=df['price']), int(self.parameters['atr_period']))
        df['bull'] = (df['ema_fast'] > df['ema_slow']) & (df['kama_slope'] >= 0)
        df['bear'] = (df['ema_fast'] < df['ema_slow']) & (df['kama_slope'] <= 0)
        # pullback z vs EMA fast
        diff = df['price'] - df['ema_fast']
        std = diff.rolling(50).std().replace(0, np.nan)
        df['pb_z'] = (diff / std).fillna(0.0)
        return df

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        if 'pb_z' not in df.columns:
            df = self.calculate_indicators(df)
        df['buy_signal'] = False
        df['sell_signal'] = False
        df['signal_strength'] = 0.0
        z_th = float(self.parameters['pullback_z'])
        rsi_buy = float(self.parameters['rsi_buy'])
        rsi_sell = float(self.parameters['rsi_sell'])
        min_s = float(self.parameters['min_strength'])
        for i in range(len(df)):
            rsi = float(df.iloc[i]['rsi'])
            z = float(df.iloc[i]['pb_z'])
            bull = bool(df.iloc[i]['bull'])
            bear = bool(df.iloc[i]['bear'])
            # Trend regime: buy pullbacks when bull, sell pullups when bear
            if bull and z <= -z_th:
                s = np.clip(abs(z) / (z_th if z_th>0 else 1e-6), 0.0, 1.0)
                if s >= min_s:
                    df.iat[i, df.columns.get_loc('buy_signal')] = True
                    df.iat[i, df.columns.get_loc('signal_strength')] = float(s)
                    continue
            if bear and z >= z_th:
                s = np.clip(abs(z) / (z_th if z_th>0 else 1e-6), 0.0, 1.0)
                if s >= min_s:
                    df.iat[i, df.columns.get_loc('sell_signal')] = True
                    df.iat[i, df.columns.get_loc('signal_strength')] = float(s)
                    continue
            # Range regime: RSI mean reversion
            if not bull and not bear:
                if rsi <= rsi_buy:
                    s = np.clip((rsi_buy - rsi)/rsi_buy, 0.0, 1.0)
                    if s >= min_s:
                        df.iat[i, df.columns.get_loc('buy_signal')] = True
                        df.iat[i, df.columns.get_loc('signal_strength')] = float(s)
                        continue
                if rsi >= rsi_sell:
                    s = np.clip((rsi - rsi_sell)/(100-rsi_sell+1e-6), 0.0, 1.0)
                    if s >= min_s:
                        df.iat[i, df.columns.get_loc('sell_signal')] = True
                        df.iat[i, df.columns.get_loc('signal_strength')] = float(s)
        return df

