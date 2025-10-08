
from __future__ import annotations
import json
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List
from pathlib import Path
import pandas as pd
import numpy as np

@dataclass
class SwapCostBuckets:
    step: int
    buckets: Dict[int, Dict[str, Any]]

    @classmethod
    def load(cls, path: str | Path) -> "SwapCostBuckets":
        with open(path, "r") as f:
            data = json.load(f)
        step = int(data["metadata"]["step_notional"])
        buckets = {int(k): v for k,v in data["entries"].items()}
        return cls(step=step, buckets=buckets)

    def bucket_for(self, notional: float) -> int:
        for k in sorted(self.buckets):
            if notional <= k: return k
        return sorted(self.buckets)[-1]

    def per_side_rate(self, notional: float) -> float:
        b = self.bucket_for(notional)
        return float(self.buckets[b]["derived"]["loss_rate"]) / 2.0

    def gas(self, notional: float, side: str) -> float:
        b = self.bucket_for(notional)
        if side.lower().startswith("b"):
            return float(self.buckets[b]["buy"]["gas_use_estimate_usd"])
        return float(self.buckets[b]["sell"]["gas_use_estimate_usd"])

def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def rsi(s: pd.Series, n: int=14) -> pd.Series:
    d = s.diff()
    up = d.clip(lower=0).rolling(n, min_periods=n).mean()
    dn = (-d.clip(upper=0)).rolling(n, min_periods=n).mean()
    rs = up / (dn + 1e-12)
    return 100 - (100/(1+rs))

@dataclass
class HybridV2Params:
    n_sma: int = 576
    entry_drop: float = 0.25
    rsi_n: int = 14
    rsi_max: float = 30.0
    base_exit_up: float = 0.05
    z_mult: float = 0.03
    exit_cap: float = 0.20
    max_hold_bars: int = 720
    ema_fast: int = 96
    ema_slow: int = 288
    gap: float = 0.012
    n_break: int = 96
    trail_up: float = 0.22

class HybridV2:
    def __init__(self, p: HybridV2Params, costs: SwapCostBuckets):
        self.p = p; self.costs = costs

    def run(self, close: pd.Series, high: pd.Series, low: pd.Series, start_cash: float=1000.0):
        s = close.dropna().copy(); idx = s.index
        sma = s.rolling(self.p.n_sma, min_periods=self.p.n_sma).mean()
        std = s.rolling(self.p.n_sma, min_periods=self.p.n_sma).std(ddof=0)
        z = (s - sma) / (std + 1e-12)
        r = rsi(s, self.p.rsi_n)
        ef = ema(s, self.p.ema_fast); es = ema(s, self.p.ema_slow)
        slope = ef - ef.shift(max(2, self.p.ema_fast//2))
        gap = (ef - es) / (es + 1e-12)
        hdc = high.rolling(self.p.n_break, min_periods=self.p.n_break).max()
        ldc = low.rolling(self.p.n_break, min_periods=self.p.n_break).min()

        p = s.to_numpy(); sm = sma.to_numpy(); zz=z.to_numpy(); rr = r.to_numpy()
        efp=ef.to_numpy(); esp=es.to_numpy(); slp=slope.to_numpy(); gp = gap.to_numpy()
        hdp=hdc.to_numpy(); ldp=ldc.to_numpy()

        cash = start_cash; tokens=0.0; position=0; peak=None
        trades=[]; equity=[]; entry_i=None; entry_px=None; capital_in=None; entry_bucket=None

        for i in range(len(p)):
            px = p[i]; equity.append(cash + tokens*px)
            smi=sm[i]; zi=zz[i]; rsi_v=rr[i]
            eff=efp[i]; esl=esp[i]; sl=slp[i]; g=gp[i]; hb=hdp[i]; lb=ldp[i]
            if any(np.isnan(x) for x in (smi,eff,esl,sl,g,hb,lb,rsi_v)): continue
            uptrend = (eff>esl) and (sl>0) and (g>=self.p.gap)

            # ENTRY
            if position==0 and cash>0:
                if uptrend and px >= hb:
                    N=cash; rate=self.costs.per_side_rate(N); buy_g=self.costs.gas(N,"buy")
                    eff_cash = cash - N*rate - buy_g
                    if eff_cash>0: tokens=eff_cash/px; cash-=N; position=1
                    entry_i=i; entry_px=px; capital_in=N; entry_bucket=self.costs.bucket_for(N); peak=px
                    continue
                if (not uptrend) and px <= smi*(1 - self.p.entry_drop) and rsi_v <= self.p.rsi_max:
                    N=cash; rate=self.costs.per_side_rate(N); buy_g=self.costs.gas(N,"buy")
                    eff_cash = cash - N*rate - buy_g
                    if eff_cash>0: tokens=eff_cash/px; cash-=N; position=1
                    entry_i=i; entry_px=px; capital_in=N; entry_bucket=self.costs.bucket_for(N); peak=px
                    continue

            # EXIT
            if position==1:
                peak = px if peak is None else max(peak, px)
                if uptrend:
                    if px <= peak*(1 - self.p.trail_up):
                        gross=tokens*px; rate=self.costs.per_side_rate(gross); sell_g=self.costs.gas(gross,"sell")
                        proceed=max(0.0, gross - gross*rate - sell_g); cash += proceed
                        trades.append({"entry_time":str(idx[entry_i]),"exit_time":str(idx[i]),"entry_px":float(entry_px),"exit_px":float(px),
                                       "entry_bucket":int(entry_bucket),"exit_bucket":int(self.costs.bucket_for(gross)),
                                       "capital_in":float(capital_in),"proceeds":float(proceed),"trade_roi":float(proceed/capital_in - 1.0),"mode":"trend"})
                        tokens=0.0; position=0; peak=None; entry_i=entry_px=capital_in=entry_bucket=None
                else:
                    posz=max(0.0, zi if not np.isnan(zi) else 0.0)
                    exit_up = self.p.base_exit_up + min(self.p.exit_cap - self.p.base_exit_up, self.p.z_mult * posz)
                    exit_level = smi*(1 + exit_up)
                    if px >= exit_level or (entry_i is not None and (i-entry_i) >= self.p.max_hold_bars and px >= smi):
                        gross=tokens*px; rate=self.costs.per_side_rate(gross); sell_g=self.costs.gas(gross,"sell")
                        proceed=max(0.0, gross - gross*rate - sell_g); cash += proceed
                        trades.append({"entry_time":str(idx[entry_i]),"exit_time":str(idx[i]),"entry_px":float(entry_px),"exit_px":float(px),
                                       "entry_bucket":int(entry_bucket),"exit_bucket":int(self.costs.bucket_for(gross)),
                                       "capital_in":float(capital_in),"proceeds":float(proceed),"trade_roi":float(proceed/capital_in - 1.0),"mode":"mr"})
                        tokens=0.0; position=0; peak=None; entry_i=entry_px=capital_in=entry_bucket=None

        if position==1 and tokens>0:
            px=p[-1]; gross=tokens*px; rate=self.costs.per_side_rate(gross); sell_g=self.costs.gas(gross,"sell")
            proceed=max(0.0, gross - gross*rate - sell_g); cash+=proceed
            trades.append({"entry_time":str(idx[entry_i]) if entry_i is not None else str(idx[-1]),
                           "exit_time":str(idx[-1]), "entry_px": float(entry_px) if entry_px is not None else float(px),
                           "exit_px": float(px), "entry_bucket": int(entry_bucket) if entry_bucket is not None else int(self.costs.bucket_for(proceed)),
                           "exit_bucket": int(self.costs.bucket_for(gross)), "capital_in": float(capital_in) if capital_in else 0.0, "proceeds": float(proceed),
                           "trade_roi": float(proceed/capital_in - 1.0) if capital_in else 0.0})
            tokens=0.0

        curve = pd.Series([*equity], index=idx[:len(equity)], name="equity")
        return float(cash), curve, pd.DataFrame(trades)

def adapter():
    # Provide a runner adapter
    name = "hybrid_v2"
    space = {
        "n_sma":[576],
        "entry_drop":[0.25],
        "base_exit_up":[0.05],
        "z_mult":[0.03],
        "exit_cap":[0.2],
        "ema_fast":[96],
        "ema_slow":[288],
        "gap":[0.012],
        "n_break":[96],
        "trail_up":[0.22],
        "rsi_n":[14],
        "rsi_max":[30.0],
    }
    def run_fn(ohlcv: pd.DataFrame, params: Dict[str,Any], swap_cost_json: str, start_cash: float=1000.0):
        close = ohlcv['close'] if 'close' in (c.lower() for c in ohlcv.columns) else ohlcv.iloc[:,1]
        high = ohlcv['high'] if 'high' in (c.lower() for c in ohlcv.columns) else close
        low  = ohlcv['low']  if 'low'  in (c.lower() for c in ohlcv.columns) else close
        close.index = pd.to_datetime(close.index, utc=True, errors='coerce')
        high = high.reindex(close.index); low = low.reindex(close.index)
        costs = SwapCostBuckets.load(swap_cost_json)
        p = HybridV2Params(**{k: params.get(k,v[0] if isinstance(v,list) else v) for k,v in space.items()})
        strat = HybridV2(p, costs)
        final, curve, trades = strat.run(close, high, low, start_cash=start_cash)
        return {"name": name, "final_equity": final, "equity_curve": curve, "trades": trades.to_dict(orient='records'), "params": p.__dict__}
    return name, space, run_fn
