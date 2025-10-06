"""
Volatility-Targeted DCA (VT-DCA)

Idea
 - Enter on statistically significant pullbacks (z-score / RSI blend).
 - Size entries inversely to realized volatility; cap max DCAs and scaling.
 - Exit on mean-reversion to a rolling mid or profit target; cool-down after exits.

Follows BaseStrategy interface (calculate_indicators / generate_signals).
"""
from typing import Dict
import logging
import numpy as np
import pandas as pd

from strategies.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


class VolatilityTargetedDCAStrategy(BaseStrategy):
    """
    Parameters
    - lookback: z-score lookback (default 40)
    - rsi_period: RSI period (default 14)
    - z_entry: negative z-score threshold to buy (default -0.8)
    - z_add: additional threshold per DCA rung (default -0.4)
    - rsi_entry: RSI threshold to allow buys (default 45)
    - target_vol: target monthly-ish vol proxy (percent/100, default 0.20)
    - max_dcas: max additional DCAs after first (default 3)
    - tp_pct: take profit percent (default 0.08)
    - sl_pct: stop loss percent (soft guard; default 0.18)
    - cooloff_bars: bars to cool off after an exit (default 6)
    - timeframe_minutes: analysis timeframe (default 60)
    """
    def __init__(self, parameters: Dict = None):
        params = {
            'lookback': 40,
            'rsi_period': 14,
            'z_entry': -0.8,
            'z_add': -0.4,
            'rsi_entry': 45,
            'target_vol': 0.20,
            'max_dcas': 3,
            'tp_pct': 0.08,
            'sl_pct': 0.18,
            'cooloff_bars': 6,
            'timeframe_minutes': 60,
        }
        if parameters:
            params.update(parameters)
        super().__init__('VolatilityTargetedDCAStrategy', params)

    @staticmethod
    def _rsi(series: pd.Series, period: int) -> pd.Series:
        delta = series.diff()
        up = delta.clip(lower=0.0)
        dn = -delta.clip(upper=0.0)
        roll_up = up.rolling(period).mean()
        roll_dn = dn.rolling(period).mean()
        rs = (roll_up / roll_dn).replace([np.inf, -np.inf], np.nan)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi.fillna(50.0)

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        if not self.validate_data(data):
            return data
        df = data.copy()
        price = df.get('price', df.get('close', df['close']))
        lb = int(self.parameters['lookback'])
        # Z-score on price vs rolling mean (robust to short history)
        mu = price.rolling(lb, min_periods=1).mean()
        sd = price.rolling(lb, min_periods=2).std().replace(0, np.nan)
        z = (price - mu) / sd
        df['zscore'] = z.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        # RSI gate
        df['rsi'] = self._rsi(price, int(self.parameters['rsi_period']))
        # Simple realized vol proxy
        ret = price.pct_change().fillna(0.0)
        df['vol'] = ret.rolling(lb, min_periods=2).std().fillna(ret.std() if ret.std() is not None else 0.0)
        # Rolling mid for exits
        df['mid'] = mu.bfill().ffill().fillna(price)
        return df

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        if 'zscore' not in df.columns or 'mid' not in df.columns or 'rsi' not in df.columns or 'vol' not in df.columns:
            df = self.calculate_indicators(df)
        # If still missing core fields, return empty signals to avoid errors on short windows
        for col in ('zscore','rsi','vol','mid'):
            if col not in df.columns:
                df['buy_signal'] = False
                df['sell_signal'] = False
                df['signal_strength'] = 0.0
                return df
        df['buy_signal'] = False
        df['sell_signal'] = False
        df['signal_strength'] = 0.0

        z_entry = float(self.parameters['z_entry'])
        z_add = float(self.parameters['z_add'])
        rsi_entry = float(self.parameters['rsi_entry'])
        target_vol = float(self.parameters['target_vol'])
        max_dcas = int(self.parameters['max_dcas'])
        tp = float(self.parameters['tp_pct'])
        sl = float(self.parameters['sl_pct'])
        cool = int(self.parameters['cooloff_bars'])

        # Local state trackers across the pass (approximation; engine tracks real P&L)
        dca_count = 0
        last_entry_price = None
        cooldown = 0

        for i in range(len(df)):
            px = float(df.iloc[i].get('price', df.iloc[i].get('close')))
            z = float(df.iloc[i]['zscore'])
            rsi = float(df.iloc[i]['rsi'])
            vol = float(df.iloc[i]['vol'])
            mid = float(df.iloc[i]['mid'])
            if np.isnan(px) or cooldown > 0:
                cooldown = max(0, cooldown - 1)
                continue
            # Dynamic size hint (not executed here; mapped into signal_strength)
            # More size when vol is below target, less when above
            vol_adj = np.clip(target_vol / max(vol, 1e-6), 0.25, 2.0)

            # Entry/DCAs
            want_entry = (z <= z_entry) and (rsi <= rsi_entry)
            want_add = (dca_count > 0) and (z <= (z_entry + dca_count * z_add))
            if (want_entry or want_add) and dca_count <= max_dcas:
                df.iat[i, df.columns.get_loc('buy_signal')] = True
                # strength encodes both statistical edge and size hint
                strength = np.clip((abs(z) / abs(z_entry)) * 0.6 + vol_adj * 0.4, 0.0, 1.0)
                df.iat[i, df.columns.get_loc('signal_strength')] = float(strength)
                last_entry_price = px if last_entry_price is None else (0.5 * last_entry_price + 0.5 * px)
                dca_count += 1
                continue

            # Exit: mean reversion back to mid or target profit, and emergency stop
            if last_entry_price is not None and dca_count > 0:
                up_pnl = (px / last_entry_price) - 1.0
                # Profit take or revert to mid
                if (up_pnl >= tp) or (px >= mid):
                    df.iat[i, df.columns.get_loc('sell_signal')] = True
                    df.iat[i, df.columns.get_loc('signal_strength')] = float(np.clip(up_pnl / max(tp, 1e-6), 0.0, 1.0))
                    dca_count = 0
                    last_entry_price = None
                    cooldown = cool
                    continue
                # Soft stop if extreme adverse move
                if up_pnl <= -sl:
                    df.iat[i, df.columns.get_loc('sell_signal')] = True
                    df.iat[i, df.columns.get_loc('signal_strength')] = 0.9
                    dca_count = 0
                    last_entry_price = None
                    cooldown = cool
        return df
