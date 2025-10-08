
#!/usr/bin/env python3
"""
scripts/backtest_runner_pro1.py

Runs parameter searches for pro1 strategies using your existing CLI backtester.
It relies on ENV-based parameter overrides baked into the pro1 strategies,
so you don't need to change your CLI.

Usage (from repo root):
  python scripts/backtest_runner_pro1.py \
      --bot-script pulsechain_trading_bot.py \
      --python-exe python \
      --timeframes 7,30,90,180,365,730 \
      --samples-per-strategy 120 \
      --out results_pro1.csv

Optionally set baseline strategies to compare against:
  export PRO1_BASELINE_STRATS="MyOldHexStrat,AnotherStrat"

Results:
- CSV at --out (default: results_pro1.csv)
- Markdown summary at best_pro1.md
"""
import argparse, json, os, random, re, subprocess, sys, time, itertools, csv, math
from pathlib import Path

DEFAULT_TIMEFRAMES = [7, 30, 90, 180, 365, 730]
STRATS = ["RegimeSwitchingPro1", "MeanReversionZScorePro1", "LiquidityAwareBreakoutPro1"]

METRIC_PATTERNS = {
    "return_pct": re.compile(r"(?i)(total\s*return|return\s*%|roi)\s*[:= ]\s*(-?\d+(?:\.\d+)?)\s*%"),
    "max_dd_pct": re.compile(r"(?i)(max\s*drawdown|mdd)\s*[:= ]\s*(-?\d+(?:\.\d+)?)\s*%"),
    "profit_factor": re.compile(r"(?i)(profit\s*factor|pf)\s*[:= ]\s*(\d+(?:\.\d+)?)"),
    "trades": re.compile(r"(?i)(trades|#\s*trades)\s*[:= ]\s*(\d+)")
}

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bot-script", default="pulsechain_trading_bot.py")
    ap.add_argument("--python-exe", default=sys.executable)
    ap.add_argument("--timeframes", default=",".join(map(str, DEFAULT_TIMEFRAMES)))
    ap.add_argument("--out", default="results_pro1.csv")
    ap.add_argument("--samples-per-strategy", type=int, default=120)
    ap.add_argument("--grid", default="configs/grid_pro1.json")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--extra-cli", default="", help="extra CLI flags to pass to the bot")
    ap.add_argument("--workdir", default=".")
    return ap.parse_args()

def load_grid(path):
    with open(path, "r") as f:
        return json.load(f)

def sample_params(grid_for_strat, n_samples, rng):
    # Build the full Cartesian product then random sample (if > n)
    keys = list(grid_for_strat.keys())
    choices = [grid_for_strat[k] for k in keys]
    combos = list(itertools.product(*choices))
    rng.shuffle(combos)
    combos = combos[:n_samples] if n_samples < len(combos) else combos
    params_dicts = []
    for tup in combos:
        d = {k: v for k, v in zip(keys, tup)}
        params_dicts.append(d)
    return params_dicts

def run_backtest(python_exe, bot_script, strategy, days, params_json, extra_cli, workdir):
    env = os.environ.copy()
    env[f"PRO1_PARAMS_{strategy}"] = json.dumps(params_json)
    cmd = [python_exe, bot_script, "--backtest", "--days", str(days), "--strategy", strategy]
    if extra_cli.strip():
        cmd += extra_cli.split()
    t0 = time.time()
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=workdir, env=env, text=True)
    dt = time.time() - t0
    out = proc.stdout or ""
    return out, proc.returncode, dt

def extract_metrics(text):
    def m(pat):
        m = METRIC_PATTERNS[pat].search(text)
        return float(m.group(2 if pat!="profit_factor" and pat!="trades" else 2 if pat=="profit_factor" else 2)) if m else float("nan")
    # Safer, explicit extraction
    ret = float("nan"); dd = float("nan"); pf = float("nan"); tr = float("nan")
    m_ret = METRIC_PATTERNS["return_pct"].search(text); 
    if m_ret: ret = float(m_ret.group(2))
    m_dd = METRIC_PATTERNS["max_dd_pct"].search(text); 
    if m_dd: dd = float(m_dd.group(2))
    m_pf = METRIC_PATTERNS["profit_factor"].search(text); 
    if m_pf: pf = float(m_pf.group(2))
    m_tr = METRIC_PATTERNS["trades"].search(text); 
    if m_tr: tr = float(m_tr.group(2))
    return {"return_pct": ret, "max_dd_pct": dd, "profit_factor": pf, "trades": tr}

def write_csv(path, rows, fieldnames):
    new = not Path(path).exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if new:
            w.writeheader()
        for r in rows:
            w.writerow(r)

def summarize_best(csv_path, out_md="best_pro1.md"):
    import pandas as pd
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return
    # aggregate by strategy+param_hash
    grp = df.groupby(["strategy","param_hash"])
    agg = grp.agg(
        mean_return=("return_pct","mean"),
        median_return=("return_pct","median"),
        mean_pf=("profit_factor","mean"),
        timeframes=("days","count")
    ).reset_index().sort_values(by=["mean_return","mean_pf"], ascending=[False, False])
    # choose top 5 per strategy
    tops = agg.groupby("strategy").head(5)
    lines = ["# Best configs — pro1",
             "",
             tops.to_markdown(index=False)]
    Path(out_md).write_text("\n".join(lines))

def main():
    args = parse_args()
    rng = random.Random(args.seed)
    grid = load_grid(args.grid)
    timeframes = [int(x) for x in args.timeframes.split(",") if x.strip()]
    baseline = os.environ.get("PRO1_BASELINE_STRATS","").strip()
    baseline_list = [s.strip() for s in baseline.split(",") if s.strip()]

    fieldnames = ["ts","strategy","days","return_pct","max_dd_pct","profit_factor","trades",
                  "param_json","param_hash","runtime_s","exit_code","is_baseline"]
    results_path = args.out

    # Baseline runs (defaults)
    for strat in baseline_list:
        for d in timeframes:
            out, code, dt = run_backtest(args.python_exe, args.bot_script, strat, d, {}, args.extra_cli, args.workdir)
            metrics = extract_metrics(out)
            row = {
                "ts": time.time(),
                "strategy": strat,
                "days": d,
                "return_pct": metrics["return_pct"],
                "max_dd_pct": metrics["max_dd_pct"],
                "profit_factor": metrics["profit_factor"],
                "trades": metrics["trades"],
                "param_json": "{}",
                "param_hash": "baseline",
                "runtime_s": dt,
                "exit_code": code,
                "is_baseline": 1
            }
            write_csv(results_path, [row], fieldnames)

    # Pro1 searches
    for strat in STRATS:
        params_list = sample_params(grid[strat], args.samples_per_strategy, rng)
        for params in params_list:
            phash = str(hash(json.dumps(params, sort_keys=True)))
            for d in timeframes:
                out, code, dt = run_backtest(args.python_exe, args.bot_script, strat, d, params, args.extra_cli, args.workdir)
                metrics = extract_metrics(out)
                row = {
                    "ts": time.time(),
                    "strategy": strat,
                    "days": d,
                    "return_pct": metrics["return_pct"],
                    "max_dd_pct": metrics["max_dd_pct"],
                    "profit_factor": metrics["profit_factor"],
                    "trades": metrics["trades"],
                    "param_json": json.dumps(params, sort_keys=True),
                    "param_hash": phash,
                    "runtime_s": dt,
                    "exit_code": code,
                    "is_baseline": 0
                }
                write_csv(results_path, [row], fieldnames)

    summarize_best(results_path, out_md="best_pro1.md")
    print(f"Done. Results -> {results_path} ; Summary -> best_pro1.md")

if __name__ == "__main__":
    main()
