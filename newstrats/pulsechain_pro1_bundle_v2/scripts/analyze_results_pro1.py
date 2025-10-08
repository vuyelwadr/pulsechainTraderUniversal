
#!/usr/bin/env python3
"""
scripts/analyze_results_pro1.py

Reads results_pro1.csv and compares pro1 configs against baseline across timeframes.
Outputs a ranked Markdown report.
"""
import argparse, json
import pandas as pd
from pathlib import Path

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results_pro1.csv")
    ap.add_argument("--out", default="report_pro1.md")
    ap.add_argument("--min_pf", type=float, default=1.10)
    ap.add_argument("--min_wins", type=int, default=3, help="min timeframes beating baseline")
    return ap.parse_args()

def main():
    args = parse_args()
    df = pd.read_csv(args.csv)
    # split baseline and candidates
    base = df[df["is_baseline"]==1]
    cand = df[df["is_baseline"]==0]

    # baseline pivot: strategy x days -> return
    base_ret = base.pivot_table(index="days", columns="strategy", values="return_pct", aggfunc="mean")

    cand["param_json_sorted"] = cand["param_json"].astype(str)
    group_cols = ["strategy","param_hash","param_json_sorted","days"]
    agg = cand.groupby(group_cols).agg(return_pct=("return_pct","mean"),
                                       pf=("profit_factor","mean")).reset_index()

    # compute win vs baseline if a matching baseline strategy exists with same 'strategy' name
    # If baseline has different names, user should map manually.
    # We compare against the best baseline per day (optimistic baseline).
    best_base_by_day = base.groupby("days")["return_pct"].max().to_dict()
    agg["beats_baseline"] = agg.apply(lambda r: r["return_pct"] >= best_base_by_day.get(r["days"], float("-inf")), axis=1)

    # aggregate across days
    final = agg.groupby(["strategy","param_hash","param_json_sorted"]).agg(
        mean_return=("return_pct","mean"),
        median_return=("return_pct","median"),
        mean_pf=("pf","mean"),
        wins=("beats_baseline","sum"),
        tests=("days","count")
    ).reset_index()

    # filter
    keep = final[(final["mean_pf"] >= args.min_pf) & (final["wins"] >= args.min_wins)]
    keep = keep.sort_values(by=["wins","mean_return","mean_pf"], ascending=[False, False, False])

    lines = ["# pro1 Backtest Report",
             "",
             f"Filtered with PF ≥ {args.min_pf} and wins ≥ {args.min_wins} timeframes.",
             "",
             keep.to_markdown(index=False)]
    Path(args.out).write_text("\n".join(lines))
    print(f"Wrote {args.out}")

if __name__ == "__main__":
    main()
