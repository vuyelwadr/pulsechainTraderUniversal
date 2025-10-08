
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
        import json
        with open(path, "r") as f:
            data = json.load(f)
        step = int(data["metadata"]["step_notional"])
        buckets = {int(k): v for k,v in data["entries"].items()}
        return cls(step=step, buckets=buckets)
    def bucket_for(self, n: float)->int:
        for k in sorted(self.buckets):
            if n<=k: return k
        return sorted(self.buckets)[-1]
    def per_side_rate(self, n: float)->float:
        return float(self.buckets[self.bucket_for(n)]["derived"]["loss_rate"])/2.0
    def gas(self, n: float, side: str)->float:
        b=self.bucket_for(n)
        return float(self.buckets[b]["buy"]["gas_use_estimate_usd"] if side.lower().startswith("b") else self.buckets[b]["sell"]["gas_use_estimate_usd"])

def ema(s: pd.Series, n: int)->pd.Series:
    return s.ewm(span=n, adjust=False).mean()

@dataclass
class TTFV2Params:
    fast: int = 96
    slow: int = 288
    slope_lb: int = 12
    gap: float = 0.01
    n_break: int = 96
    trail: float | None = 0.10

class TTFV2:
    def __init__(self, p: TTFV2Params, costs: SwapCostBuckets):
        self.p=p; self.costs=costs
    def run(self, s: pd.Series, start_cash: float=1000.0):
        ef=ema(s,self.p.fast); es=ema(s,self.p.slow)
        slope = ef - ef.shift(self.p.slope_lb)
        rel_gap = (ef - es)/(es+1e-12)
        dc = s.rolling(self.p.n_break, min_periods=self.p.n_break).max()

        p = s.to_numpy(); efp=ef.to_numpy(); esp=es.to_numpy()
        sl=slope.to_numpy(); rg=rel_gap.to_numpy(); dcp=dc.to_numpy(); idx = s.index

        cash=start_cash; tokens=0.0; position=0; peak=None; equity=[]; trades=[]; entry_i=None; capital_in=None; entry_px=None; entry_bucket=None
        for i in range(len(p)):
            px=p[i]; equity.append(cash + tokens*px)
            if any(np.isnan(x) for x in (efp[i],esp[i],sl[i],rg[i],dcp[i])): continue
            up=(efp[i]>esp[i]) and (sl[i]>0) and (rg[i]>=self.p.gap) and (px>=dcp[i])
            down=(efp[i]<esp[i]) or (sl[i]<0)
            if position==0 and up and cash>0:
                N=cash; rate=self.costs.per_side_rate(N); buy_g=self.costs.gas(N,"buy")
                eff=cash - N*rate - buy_g
                if eff>0:
                    tokens = eff/px; cash -= N; position=1; peak=px
                    entry_i=i; capital_in=N; entry_px=px; entry_bucket=self.costs.bucket_for(N)
            if position==1:
                peak=max(peak or px, px)
                trig = down or (self.p.trail is not None and px<=peak*(1-self.p.trail))
                if trig and tokens>0:
                    gross=tokens*px; rate=self.costs.per_side_rate(gross); sell_g=self.costs.gas(gross,"sell")
                    proceed=max(0.0,gross - gross*rate - sell_g); cash+=proceed
                    trades.append({"entry_time":str(idx[entry_i]),"exit_time":str(idx[i]),"entry_px":float(entry_px),"exit_px":float(px),
                                   "entry_bucket":int(entry_bucket),"exit_bucket":int(self.costs.bucket_for(gross)),"capital_in":float(capital_in),"proceeds":float(proceed),"trade_roi":float(proceed/capital_in - 1.0)})
                    tokens=0.0; position=0; peak=None; entry_i=entry_px=capital_in=entry_bucket=None
        if position==1 and tokens>0:
            px=p[-1]; gross=tokens*px; rate=self.costs.per_side_rate(gross); sell_g=self.costs.gas(gross,"sell"); proceed=max(0.0,gross - gross*rate - sell_g); cash+=proceed
            trades.append({"entry_time":str(idx[entry_i]) if entry_i is not None else str(idx[-1]),"exit_time":str(idx[-1]),"entry_px":float(entry_px) if entry_px else float(px),"exit_px":float(px),"entry_bucket":int(entry_bucket) if entry_bucket else int(self.costs.bucket_for(proceed)),"exit_bucket":int(self.costs.bucket_for(gross)),"capital_in":float(capital_in) if capital_in else 0.0,"proceeds":float(proceed),"trade_roi":float(proceed/capital_in - 1.0) if capital_in else 0.0})
            tokens=0.0
        curve = pd.Series([*equity], index=s.index[:len(equity)], name="equity")
        return float(cash), curve, pd.DataFrame(trades)

def adapter():
    name="trend_ttf_v2"
    space={"fast":[96],"slow":[288],"slope_lb":[12],"gap":[0.01,0.015],"n_break":[96],"trail":[0.10]}
    def run_fn(close: pd.Series, params: Dict[str,Any], swap_cost_json: str, start_cash: float=1000.0):
        costs=SwapCostBuckets.load(swap_cost_json)
        p=TTFV2Params(**{k: params.get(k,v[0] if isinstance(v,list) else v) for k,v in space.items()})
        strat=TTFV2(p,costs)
        final,curve,trades=strat.run(close,start_cash=start_cash)
        return {"name":name,"final_equity":final,"equity_curve":curve,"trades":trades.to_dict(orient='records'),"params":p.__dict__}
    return name, space, run_fn
