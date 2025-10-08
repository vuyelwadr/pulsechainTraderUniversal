
"""
C-SMA-Revert V2 (tuned): SMA(576) dip-buy + RSI<=30 + exit at SMA + 4.8%.
Cost-aware (bucketed loss_rate/2 per side + side gas).
"""

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
        with open(path, "r") as f: data = json.load(f)
        step = int(data["metadata"]["step_notional"])
        buckets = {int(k): v for k,v in data["entries"].items()}
        return cls(step=step, buckets=buckets)
    def bucket_for(self, n: float)->int:
        for k in sorted(self.buckets):
            if n<=k: return k
        return sorted(self.buckets)[-1]
    def per_side_rate(self, n: float)->float:
        b = self.bucket_for(n); return float(self.buckets[b]["derived"]["loss_rate"])/2.0
    def gas(self, n: float, side: str)->float:
        b = self.bucket_for(n)
        return float(self.buckets[b]["buy"]["gas_use_estimate_usd"] if side.lower().startswith("b") else self.buckets[b]["sell"]["gas_use_estimate_usd"])

def rsi(s: pd.Series, n: int=14)->pd.Series:
    d=s.diff(); up=d.clip(lower=0).rolling(n, min_periods=n).mean()
    dn=(-d.clip(upper=0)).rolling(n, min_periods=n).mean()
    rs=up/(dn+1e-12); return 100 - (100/(1+rs))

@dataclass
class Params:
    n_sma: int = 576
    entry_drop: float = 0.25
    exit_up: float = 0.048  # tuned from sweep
    rsi_n: int = 14
    rsi_max: float = 30.0

class Strategy:
    def __init__(self, p: Params, costs: SwapCostBuckets):
        self.p=p; self.costs=costs
    def run(self, close: pd.Series, start_cash: float=1000.0):
        s = close.dropna().copy(); s.index = pd.to_datetime(s.index, utc=True)
        sma = s.rolling(self.p.n_sma, min_periods=self.p.n_sma).mean()
        r = rsi(s, self.p.rsi_n)
        p = s.to_numpy(); sm = sma.to_numpy(); rr=r.to_numpy(); idx=s.index
        cash=start_cash; tokens=0.0; pos=0; equity=[]; trades=[]
        ei=ep=cap=bucket=None
        for i in range(len(p)):
            px=p[i]; equity.append(cash + tokens*px)
            smi=sm[i]; rv=rr[i]
            if np.isnan(smi) or np.isnan(rv): continue
            if pos==0 and px <= smi*(1 - self.p.entry_drop) and rv <= self.p.rsi_max and cash>0:
                N=cash; rate=self.costs.per_side_rate(N); buy_g=self.costs.gas(N,"buy")
                eff=cash - N*rate - buy_g
                if eff>0: tokens=eff/px; cash-=N; pos=1; ei=i; ep=px; cap=N; bucket=self.costs.bucket_for(N)
            elif pos==1 and px >= smi*(1 + self.p.exit_up):
                gross=tokens*px; rate=self.costs.per_side_rate(gross); sell_g=self.costs.gas(gross,"sell")
                proceed=max(0.0, gross - gross*rate - sell_g); cash+=proceed
                trades.append({"entry_time":str(idx[ei]),"exit_time":str(idx[i]),"entry_px":float(ep),"exit_px":float(px),
                               "entry_bucket":int(bucket),"exit_bucket":int(self.costs.bucket_for(gross)),"capital_in":float(cap),"proceeds":float(proceed),"trade_roi":float(proceed/cap - 1.0)})
                tokens=0.0; pos=0; ei=ep=cap=bucket=None
        if pos==1 and tokens>0:
            px=p[-1]; gross=tokens*px; rate=self.costs.per_side_rate(gross); sell_g=self.costs.gas(gross,"sell"); proceed=max(0.0,gross - gross*rate - sell_g); cash+=proceed
            trades.append({"entry_time":str(idx[ei]) if ei is not None else str(idx[-1]),"exit_time":str(idx[-1]),"entry_px":float(ep) if ep else float(px),"exit_px":float(px),
                           "entry_bucket":int(bucket) if bucket else int(self.costs.bucket_for(proceed)),"exit_bucket":int(self.costs.bucket_for(gross)),"capital_in":float(cap) if cap else 0.0,"proceeds":float(proceed),"trade_roi":float(proceed/cap - 1.0) if cap else 0.0})
            tokens=0.0
        curve = pd.Series([*equity], index=s.index[:len(equity)], name="equity")
        return float(cash), curve, pd.DataFrame(trades)

def adapter():
    name="c_sma_revert_v2"
    space={"n_sma":[576],"entry_drop":[0.25],"exit_up":[0.048,0.05],"rsi_n":[14],"rsi_max":[25,30,35]}
    def run_fn(close: pd.Series, params: Dict[str,Any], swap_cost_json: str, start_cash: float=1000.0):
        costs=SwapCostBuckets.load(swap_cost_json)
        p=Params(
            n_sma=int(params.get("n_sma",576)),
            entry_drop=float(params.get("entry_drop",0.25)),
            exit_up=float(params.get("exit_up",0.048)),
            rsi_n=int(params.get("rsi_n",14)),
            rsi_max=float(params.get("rsi_max",30.0)),
        )
        strat=Strategy(p,costs)
        final, curve, trades = strat.run(close, start_cash=start_cash)
        return {"name":name,"final_equity":final,"equity_curve":curve,"trades":[t for t in trades.to_dict(orient='records')],"params":p.__dict__}
    return name, space, run_fn
