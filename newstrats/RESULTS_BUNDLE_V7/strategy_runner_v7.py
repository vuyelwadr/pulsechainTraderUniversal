
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
        buckets = {int(k): v for k, v in data["entries"].items()}
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

def rsi(s: pd.Series, n: int=14) -> pd.Series:
    d = s.diff()
    up = d.clip(lower=0).rolling(n, min_periods=n).mean()
    dn = (-d.clip(upper=0)).rolling(n, min_periods=n).mean()
    rs = up / (dn + 1e-12)
    return 100 - (100/(1+rs))

def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

@dataclass
class Params:
    n_sma: int = 576
    entry_drop: float = 0.25
    exit_up: float = 0.048
    rsi_n: int = 14
    rsi_max: float = 30.0
    ema_fast: int = 96
    ema_slow: int = 288
    runner_frac: float = 0.10  # constant runner portion when uptrend at TP
    runner_trail: float = 0.30
    max_hold_bars: int = 1440

class StrategyRunnerV7:
    def __init__(self, p: Params, costs: SwapCostBuckets):
        self.p = p; self.costs = costs

    def run(self, close: pd.Series, start_cash: float = 1000.0):
        s = close.dropna().copy()
        s.index = pd.to_datetime(s.index, utc=True)

        sma = s.rolling(self.p.n_sma, min_periods=self.p.n_sma).mean()
        r = rsi(s, self.p.rsi_n)
        ef = ema(s, self.p.ema_fast)
        es = ema(s, self.p.ema_slow)
        slope = ef - ef.shift(max(2, self.p.ema_fast//2))

        p = s.to_numpy(); sm = sma.to_numpy(); rr = r.to_numpy()
        efp = ef.to_numpy(); esp = es.to_numpy(); slp = slope.to_numpy()
        idx = s.index

        cash=start_cash; tokens=0.0; runner=0.0; pos=0
        equity=[]; trades=[]
        ei=ep=cap=entry_bucket=None
        peak=None

        for i in range(len(p)):
            px = p[i]; equity.append(cash + (tokens+runner)*px)
            smi=sm[i]; rv=rr[i]; efi=efp[i]; esi=esp[i]; sl=slp[i]
            if any(np.isnan(x) for x in (smi, rv, efi, esi, sl)): continue

            # Entry (flat only)
            if tokens==0.0 and runner==0.0 and px <= smi*(1 - self.p.entry_drop) and rv <= self.p.rsi_max and cash>0:
                N=cash; rate=self.costs.per_side_rate(N); buy_g=self.costs.gas(N,"buy")
                eff=cash - N*rate - buy_g
                if eff>0:
                    tokens=eff/px; cash-=N; pos=1; ei=i; ep=px; cap=N; entry_bucket=self.costs.bucket_for(N); peak=px
                continue

            if (tokens+runner)>0: peak = px if peak is None else max(peak, px)

            uptrend = (efi > esi) and (sl > 0)

            # TP on main leg
            if pos==1 and px >= smi*(1 + self.p.exit_up):
                if uptrend and self.p.runner_frac>0:
                    keep = self.p.runner_frac * tokens
                    sold = tokens - keep
                    proceed=0.0; b2=None
                    if sold>0:
                        gross=sold*px; rate=self.costs.per_side_rate(gross); sell_g=self.costs.gas(gross,"sell")
                        proceed=max(0.0, gross - gross*rate - sell_g); cash+=proceed; b2=self.costs.bucket_for(gross)
                    trades.append({"entry_time": str(idx[ei]), "exit_time": str(idx[i]),
                                   "entry_px": float(ep), "exit_px": float(px),
                                   "entry_bucket": int(entry_bucket), "exit_bucket": int(b2) if b2 is not None else int(entry_bucket),
                                   "capital_in": float(cap), "proceeds": float(proceed),
                                   "trade_roi": float(proceed/cap - 1.0) if cap else 0.0,
                                   "partial": f"TP{int((1-self.p.runner_frac)*100)}%_keep{int(self.p.runner_frac*100)}%"})
                    runner += keep; tokens=0.0; pos=0
                else:
                    gross=tokens*px; rate=self.costs.per_side_rate(gross); sell_g=self.costs.gas(gross,"sell")
                    proceed=max(0.0, gross - gross*rate - sell_g); cash+=proceed
                    trades.append({"entry_time": str(idx[ei]), "exit_time": str(idx[i]),
                                   "entry_px": float(ep), "exit_px": float(px),
                                   "entry_bucket": int(entry_bucket), "exit_bucket": int(self.costs.bucket_for(gross)),
                                   "capital_in": float(cap), "proceeds": float(proceed),
                                   "trade_roi": float(proceed/cap - 1.0) if cap else 0.0, "partial":"TP100%"})
                    tokens=0.0; pos=0; ei=ep=cap=entry_bucket=None
                continue

            # Runner exits
            if runner>0 and ((efi < esi) or px <= (peak or px)*(1 - self.p.runner_trail)):
                gross=runner*px; rate=self.costs.per_side_rate(gross); sell_g=self.costs.gas(gross,"sell")
                proceed=max(0.0, gross - gross*rate - sell_g); cash+=proceed
                trades.append({"entry_time": str(idx[ei]) if ei is not None else str(idx[i]),
                               "exit_time": str(idx[i]), "entry_px": float(ep) if ep else float(px),
                               "exit_px": float(px), "entry_bucket": int(entry_bucket) if entry_bucket else int(self.costs.bucket_for(proceed)),
                               "exit_bucket": int(self.costs.bucket_for(gross)), "capital_in": 0.0,
                               "proceeds": float(proceed), "trade_roi": 0.0, "partial":"RunnerExit"})
                runner=0.0; peak=None; ei=ep=cap=entry_bucket=None
                continue

        if tokens>0 or runner>0:
            gross=(tokens+runner)*p[-1]; rate=self.costs.per_side_rate(gross); sell_g=self.costs.gas(gross,"sell")
            proceed=max(0.0, gross - gross*rate - sell_g); cash+=proceed
            trades.append({"entry_time": str(idx[ei]) if ei is not None else str(idx[-1]), "exit_time": str(idx[-1]),
                           "entry_px": float(ep) if ep else float(p[-1]), "exit_px": float(p[-1]),
                           "entry_bucket": int(entry_bucket) if entry_bucket else int(self.costs.bucket_for(proceed)),
                           "exit_bucket": int(self.costs.bucket_for(gross)), "capital_in": float(cap) if cap else 0.0,
                           "proceeds": float(proceed), "trade_roi": 0.0, "partial":"FinalClose"})
            tokens=0.0; runner=0.0

        curve = pd.Series([*equity], index=s.index[:len(equity)], name="equity")
        return float(cash), curve, pd.DataFrame(trades)

def adapter():
    name="runner_v7"
    space={"n_sma":[576],"entry_drop":[0.25],"exit_up":[0.048],"rsi_n":[14],"rsi_max":[30.0],
           "ema_fast":[96],"ema_slow":[288],"runner_frac":[0.10,0.075,0.05],"runner_trail":[0.30,0.35]}
    def run_fn(close: pd.Series, params: Dict[str,Any], swap_cost_json: str, start_cash: float=1000.0):
        costs=SwapCostBuckets.load(swap_cost_json)
        p=Params(
            n_sma=int(params.get("n_sma",576)),
            entry_drop=float(params.get("entry_drop",0.25)),
            exit_up=float(params.get("exit_up",0.048)),
            rsi_n=int(params.get("rsi_n",14)),
            rsi_max=float(params.get("rsi_max",30.0)),
            ema_fast=int(params.get("ema_fast",96)),
            ema_slow=int(params.get("ema_slow",288)),
            runner_frac=float(params.get("runner_frac",0.10)),
            runner_trail=float(params.get("runner_trail",0.30)),
        )
        strat=StrategyRunnerV7(p,costs)
        final, curve, trades = strat.run(close, start_cash=start_cash)
        return {"name": name, "final_equity": final, "equity_curve": curve, "trades": trades.to_dict(orient='records'), "params": p.__dict__}
    return name, space, run_fn

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Runner V7 (constant micro-runner on uptrend TP)")
    ap.add_argument("--csv", required=True, help="OHLCV CSV with timestamp + close/price")
    ap.add_argument("--swap-cost-cache", required=True, help="swap_cost_cache.json path")
    ap.add_argument("--start-cash", type=float, default=1000.0)
    args = ap.parse_args()
    print("Use adapter() from your optimization.runner to integrate this strategy.")
