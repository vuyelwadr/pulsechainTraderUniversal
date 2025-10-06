#!/usr/bin/env python3
"""
Strategy 181: Insync Index With BB

Pine reference: pine_scripts/181_insync_index_bb.pine

Signals: buy when Insync Index crosses above 50; sell when crosses below 50.
"""

import pandas as pd
import numpy as np
from typing import Dict
import os, sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)
from strategies.base_strategy import BaseStrategy

try:
    sys.path.insert(0, os.path.join(project_root, 'src'))
    from utils.vectorized_helpers import crossover, crossunder
except Exception:
    def crossover(a,b): return (a>b) & (a.shift(1)<=b)
    def crossunder(a,b): return (a<b) & (a.shift(1)>=b)


class Strategy181InsyncIndexWithBB(BaseStrategy):
    def __init__(self, parameters: Dict=None):
        params = {
            'lengthBB': 20,
            'multBB': 2.0,
            'lengthCCI': 14,
            'dpoLength': 18,
            'lengthROC': 10,
            'lengthRSI': 14,
            'lengthStoch': 14,
            'lengthD': 3,
            'lengthK': 1,
            'lengthSMA': 10,
            'signal_threshold': 0.6,
        }
        if parameters:
            params.update(parameters)
        super().__init__('Strategy_181_InsyncIndexWithBB', params)

    @staticmethod
    def _sma(s: pd.Series, n: int) -> pd.Series:
        return s.rolling(max(1,int(n)), min_periods=1).mean()

    @staticmethod
    def _ema(s: pd.Series, n: int) -> pd.Series:
        return s.ewm(span=max(1,int(n)), adjust=False, min_periods=1).mean()

    @staticmethod
    def _rsi(s: pd.Series, n: int) -> pd.Series:
        n=max(1,int(n))
        d=s.diff()
        up=d.clip(lower=0); dn=-d.clip(upper=0)
        roll_up=up.ewm(alpha=1.0/n, adjust=False, min_periods=1).mean()
        roll_dn=dn.ewm(alpha=1.0/n, adjust=False, min_periods=1).mean()
        rs=roll_up/roll_dn.replace(0,np.nan)
        return (100 - 100/(1+rs)).fillna(50)

    @staticmethod
    def _cci(src: pd.Series, high: pd.Series, low: pd.Series, n: int) -> pd.Series:
        tp=(high+low+src)/3.0
        sma_tp=tp.rolling(max(1,int(n)), min_periods=1).mean()
        md=(tp - sma_tp).abs().rolling(max(1,int(n)), min_periods=1).mean()
        return (tp - sma_tp) / (0.015 * md.replace(0,np.nan))

    @staticmethod
    def _stoch(close: pd.Series, high: pd.Series, low: pd.Series, n: int, smoothK: int, smoothD: int) -> (pd.Series, pd.Series):
        hh = high.rolling(max(1,int(n)), min_periods=1).max()
        ll = low.rolling(max(1,int(n)), min_periods=1).min()
        k = 100.0 * (close - ll) / (hh - ll).replace(0,np.nan)
        k = k.rolling(max(1,int(smoothK)), min_periods=1).mean()
        d = k.rolling(max(1,int(smoothD)), min_periods=1).mean()
        return k, d

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        for c in ('high','low','close','volume'):
            if c not in data:
                data[c] = 0 if c=='volume' else data.get('close', pd.Series(index=data.index)).fillna(method='ffill')
        p=self.parameters
        src=data['close']
        # BB normalized position
        basis=self._sma(src, int(p['lengthBB']))
        dev = float(p['multBB']) * src.rolling(int(p['lengthBB']), min_periods=1).std(ddof=0)
        bolinslb=basis - dev
        bolinsub=basis + dev
        bolins2=(src - bolinslb) / (bolinsub - bolinslb).replace(0,np.nan)
        bolinsll = pd.Series(0, index=src.index)
        bolinsll[bolins2 < 0.05] = -5
        bolinsll[bolins2 > 0.95] = 5

        cciins = pd.Series(0, index=src.index)
        cci=self._cci(src, data['high'], data['low'], int(p['lengthCCI']))
        cciins[cci > 100] = 5
        cciins[cci < -100] = -5

        # EMO simplified: use SMA of price change * range / volume
        emo = self._sma(10000 * src.diff().fillna(0) * (data['high'] - data['low']) / data['volume'].replace(0,np.nan), int(p['lengthSMA']))
        emvins2 = emo - self._sma(emo, int(p['lengthSMA']))
        emvinsb = pd.Series(0, index=src.index)
        emvinss = pd.Series(0, index=src.index)
        emvinsb[(emvins2 < 0) & (self._sma(emo, int(p['lengthSMA'])) < 0)] = -5
        emvinss[(emvins2 > 0) & (self._sma(emo, int(p['lengthSMA'])) > 0)] = 5

        # MACD simplified sign relative to its SMA
        macd = self._ema(src,12) - self._ema(src,26)
        macdins2 = macd - self._sma(macd, int(p['lengthSMA']))
        macdinsb = pd.Series(0, index=src.index)
        macdinss = pd.Series(0, index=src.index)
        macdinsb[(macdins2 < 0) & (self._sma(macd, int(p['lengthSMA'])) < 0)] = -5
        macdinss[(macdins2 > 0) & (self._sma(macd, int(p['lengthSMA'])) > 0)] = 5

        # MFI simplified: RSI of hlc3 used as proxy
        mfiins = pd.Series(0, index=src.index)
        mfi = self._rsi((data['high']+data['low']+data['close'])/3.0, int(p['lengthRSI']))
        mfiins[mfi > 80] = 5
        mfiins[mfi < 20] = -5

        # DPO proxy: price minus SMA lagged
        ma = self._sma(src, int(p['dpoLength']))
        barsback = int(p['dpoLength'])//2 + 1
        dpo = src - ma.shift(barsback)
        pdoins2 = dpo - self._sma(dpo, int(p['lengthSMA']))
        pdoinsb = pd.Series(0, index=src.index)
        pdoinss = pd.Series(0, index=src.index)
        pdoinsb[(pdoins2 < 0) & (self._sma(dpo, int(p['lengthSMA'])) < 0)] = -5
        pdoinss[(pdoins2 > 0) & (self._sma(dpo, int(p['lengthSMA'])) > 0)] = 5

        # ROC
        roc = 100.0 * (src - src.shift(int(p['lengthROC']))) / src.shift(int(p['lengthROC']))
        rocins2 = roc - self._sma(roc, int(p['lengthSMA']))
        rocinsb = pd.Series(0, index=src.index)
        rocinss = pd.Series(0, index=src.index)
        rocinsb[(rocins2 < 0) & (self._sma(roc, int(p['lengthSMA'])) < 0)] = -5
        rocinss[(rocins2 > 0) & (self._sma(roc, int(p['lengthSMA'])) > 0)] = 5

        # RSI
        rsi = self._rsi(src, int(p['lengthRSI']))
        rsiins = pd.Series(0, index=src.index)
        rsiins[rsi > 70] = 5
        rsiins[rsi < 30] = -5

        # Stoch
        k, d = self._stoch(data['close'], data['high'], data['low'], int(p['lengthStoch']), int(p['lengthD']), int(p['lengthK']))
        stopdins = pd.Series(0, index=src.index)
        stopkins = pd.Series(0, index=src.index)
        stopdins[d > 80] = 5
        stopdins[d < 20] = -5
        stopkins[k > 80] = 5
        stopkins[k < 20] = -5

        iidx = 50 + cciins + bolinsll + rsiins + stopkins + stopdins + mfiins + emvinsb + emvinss + rocinss + rocinsb + pdoinss.shift(10).fillna(0) + pdoinsb.shift(10).fillna(0) + macdinss + macdinsb
        data['iidx'] = iidx.clip(0,100).fillna(50)
        return data

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        buy = crossover(data['iidx'], 50.0)
        sell = crossunder(data['iidx'], 50.0)
        data['buy_signal'] = buy
        data['sell_signal'] = sell
        st = (data['iidx'] - 50.0).abs() / 50.0
        st = st.clip(0,1).fillna(0).where(buy|sell, 0.0)
        thr=float(self.parameters['signal_threshold'])
        st[(buy|sell) & (st < thr)] = thr
        data['signal_strength'] = st
        return data

