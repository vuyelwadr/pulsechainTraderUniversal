#!/usr/bin/env python3
"""Generate an interactive HTML that compares baseline vs new trend segments."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

TIMEFRAMES: List[str] = [
    "5min",
    "15min",
    "30min",
    "1h",
    "2h",
    "4h",
    "8h",
    "16h",
    "1d",
    "2d",
]

STATE_COLORS_NEW: Dict[str, str] = {
    "UPTREND": "rgba(46, 204, 113, 0.12)",
    "DOWNTREND": "rgba(231, 76, 60, 0.12)",
    "RANGE": "rgba(149, 165, 166, 0.08)",
    "UNKNOWN": "rgba(189, 195, 199, 0.08)",
}

STATE_COLORS_BASELINE: Dict[str, str] = {
    "UPTREND": "rgba(46, 204, 113, 0.08)",
    "DOWNTREND": "rgba(231, 76, 60, 0.08)",
    "RANGE": "rgba(149, 165, 166, 0.05)",
    "UNKNOWN": "rgba(189, 195, 199, 0.05)",
}


def build_highlight_series(
    state_df: Optional[pd.DataFrame],
    segments_df: Optional[pd.DataFrame],
) -> tuple[list, list]:
    """Return timestamp/value lists that only include prices inside the segments."""
    if state_df is None or segments_df is None or segments_df.empty:
        return ([], [])

    df = state_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
    df["highlight"] = False

    for _, seg in segments_df.iterrows():
        start = seg["start_time"]
        end = seg["end_time"]
        if pd.isna(start) or pd.isna(end):
            continue
        df.loc[(df["timestamp"] >= start) & (df["timestamp"] <= end), "highlight"] = True

    highlighted = df.loc[:, ["timestamp", "close", "highlight"]].copy()
    highlighted["close"] = highlighted.apply(
        lambda row: float(row["close"]) if row["highlight"] else None,
        axis=1,
    )

    return highlighted["timestamp"].tolist(), highlighted["close"].tolist()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate trend segment comparison HTML")
    parser.add_argument(
        "--analysis-dir",
        default="analysis",
        help="Directory containing the updated trend state/segment CSVs",
    )
    parser.add_argument(
        "--baseline-analysis-dir",
        default="analysis_baseline",
        help="Directory containing the baseline trend files (optional)",
    )
    parser.add_argument(
        "--output-html",
        default="html_reports/trend_segments_overview.html",
        help="Output HTML path",
    )
    parser.add_argument(
        "--summary-csv",
        default="reports/trend_segments_backtest_summary.csv",
        help="Where to write the segment summary CSV",
    )
    parser.add_argument(
        "--cost",
        type=float,
        default=0.015,
        help="Round-trip trading cost used in segment simulation (default 1.5%)",
    )
    parser.add_argument(
        "--baseline-wf-csv",
        default="reports/wf_uptrend_summary_median_fixed.csv",
        help="Walk-forward CSV for the baseline (old) configuration",
    )
    parser.add_argument(
        "--new-wf-csv",
        default="reports/wf_uptrend_summary_grid_atr_cooldown_tfslow.csv",
        help="Walk-forward CSV for the new configuration",
    )
    parser.add_argument(
        "--baseline-trades-csv",
        default="reports/wf_uptrend_trades_median_fixed.csv",
        help="Trade log CSV for the baseline configuration (optional)",
    )
    parser.add_argument(
        "--new-trades-csv",
        default="reports/wf_uptrend_trades_grid_atr_cooldown_tfslow.csv",
        help="Trade log CSV for the new configuration",
    )
    return parser.parse_args()


def to_naive(ts: pd.Series) -> pd.Series:
    if ts.dt.tz is not None:
        return ts.dt.tz_convert(None)
    return ts


def load_price_series(analysis_dir: Path) -> pd.DataFrame:
    base_path = analysis_dir / "trend_states_5min.csv"
    if not base_path.exists():
        raise FileNotFoundError(f"Missing price anchor file: {base_path}")
    df = pd.read_csv(base_path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp")
    price_col = "price" if "price" in df.columns else "close"
    df["price"] = pd.to_numeric(df[price_col], errors="coerce")
    df = df[["timestamp", "price"]]
    df["timestamp"] = df["timestamp"].dt.tz_localize(None)
    return df.dropna()


def load_segments(analysis_dir: Path, label: str) -> Optional[pd.DataFrame]:
    path = analysis_dir / f"trend_segments_{label}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["start_time", "end_time"])
    df = df.sort_values("start_time").reset_index(drop=True)
    return df


def load_state_prices(analysis_dir: Path, label: str, start: pd.Timestamp, end: pd.Timestamp) -> Optional[pd.DataFrame]:
    path = analysis_dir / f"trend_states_{label}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["timestamp"])
    price_col = "close" if "close" in df.columns else "price"
    if price_col not in df.columns:
        return None
    df["timestamp"] = df["timestamp"].dt.tz_localize(None)
    df = df.sort_values("timestamp")
    df = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)].copy()
    df.rename(columns={price_col: "close"}, inplace=True)
    return df[["timestamp", "close"]]


def clamp_segments(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    df = df.copy()
    df["start_time"] = to_naive(df["start_time"])
    df["end_time"] = to_naive(df["end_time"])
    df["start_time"] = df["start_time"].clip(lower=start)
    df["end_time"] = df["end_time"].clip(upper=end)
    df = df[df["end_time"] > df["start_time"]]
    return df


def load_trades(path: Path) -> Optional[pd.DataFrame]:
    if not path or not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["timestamp", "test_start", "test_end"])
    if "timeframe" not in df.columns or "action" not in df.columns:
        return None
    df["timestamp"] = df["timestamp"].dt.tz_localize(None)
    df["timeframe"] = df["timeframe"].astype(str)
    if "phase" in df.columns:
        df["phase"] = df["phase"].astype(str)
    else:
        df["phase"] = "test"
    df["action"] = df["action"].astype(str).str.upper()
    return df


def prepare_trade_markers(trades: Optional[pd.DataFrame]) -> Dict[str, Dict[str, tuple[list, list]]]:
    markers: Dict[str, Dict[str, tuple[list, list]]] = {
        tf: {"buy": ([], []), "sell": ([], [])} for tf in TIMEFRAMES
    }
    if trades is None or trades.empty:
        return markers

    df = trades.copy()
    df = df[df["phase"].str.lower() == "test"]
    if df.empty:
        return markers

    for label, sub in df.groupby("timeframe"):
        if label not in markers:
            continue
        buys = sub[sub["action"] == "BUY"].sort_values("timestamp")
        sells = sub[sub["action"] == "SELL"].sort_values("timestamp")
        markers[label]["buy"] = (buys["timestamp"].tolist(), buys["price"].tolist())
        markers[label]["sell"] = (sells["timestamp"].tolist(), sells["price"].tolist())

    return markers


def build_shapes(
    df: pd.DataFrame,
    y0: float,
    y1: float,
    yref: str,
    color_map: Dict[str, str],
) -> List[dict]:
    shapes: List[dict] = []
    for _, row in df.iterrows():
        color = color_map.get(row.get("state", "UNKNOWN"), color_map["UNKNOWN"])
        shapes.append(
            {
                "type": "rect",
                "xref": "x",
                "yref": yref,
                "x0": row["start_time"],
                "x1": row["end_time"],
                "y0": y0,
                "y1": y1,
                "fillcolor": color,
                "line": {"width": 0},
                "layer": "below",
            }
        )
    return shapes


def compute_segment_summary(
    states_map: Dict[str, pd.DataFrame],
    segments_map: Dict[str, pd.DataFrame],
    cost: float,
) -> pd.DataFrame:
    records: List[Dict[str, float]] = []
    for label, seg_df in segments_map.items():
        if seg_df.empty or label not in states_map:
            records.append(
                {
                    "timeframe": label,
                    "segments": 0,
                    "final_balance": 1000.0,
                    "total_return_pct": 0.0,
                    "mean_segment_return_pct": 0.0,
                    "median_segment_return_pct": 0.0,
                }
            )
            continue

        state_df = states_map[label].set_index("timestamp")["close"]
        capital = 1000.0
        seg_returns: List[float] = []
        for _, row in seg_df.iterrows():
            start = row["start_time"]
            end = row["end_time"]
            if start not in state_df.index or end not in state_df.index:
                continue
            start_price = float(state_df.loc[start])
            end_price = float(state_df.loc[end])
            if start_price <= 0:
                continue
            gross_return = end_price / start_price - 1.0
            seg_returns.append(gross_return)
            capital = capital * (1.0 + gross_return)
            capital *= 1.0 - cost

        if seg_returns:
            records.append(
                {
                    "timeframe": label,
                    "segments": len(seg_returns),
                    "final_balance": capital,
                    "total_return_pct": (capital / 1000.0 - 1.0) * 100.0,
                    "mean_segment_return_pct": np.mean(seg_returns) * 100.0,
                    "median_segment_return_pct": np.median(seg_returns) * 100.0,
                }
            )
        else:
            records.append(
                {
                    "timeframe": label,
                    "segments": 0,
                    "final_balance": 1000.0,
                    "total_return_pct": 0.0,
                    "mean_segment_return_pct": 0.0,
                    "median_segment_return_pct": 0.0,
                }
            )

    summary = pd.DataFrame.from_records(records)
    summary.sort_values("total_return_pct", ascending=False, inplace=True)
    summary.reset_index(drop=True, inplace=True)
    return summary


def load_walkforward_table(new_csv: Path, baseline_csv: Path) -> Optional[pd.DataFrame]:
    if not new_csv.exists() or not baseline_csv.exists():
        return None
    new_df = pd.read_csv(new_csv)
    base_df = pd.read_csv(baseline_csv)
    required_cols = {"timeframe", "total_return_pct", "buy_hold_total_return_pct"}
    if not required_cols.issubset(new_df.columns) or "timeframe" not in base_df.columns:
        return None

    order = [tf for tf in TIMEFRAMES if tf in new_df["timeframe"].unique()]
    base_mean = base_df.groupby("timeframe")["total_return_pct"].mean()
    new_mean = new_df.groupby("timeframe")["total_return_pct"].mean()
    buy_hold_mean = new_df.groupby("timeframe")["buy_hold_total_return_pct"].mean()

    rows = []
    for label in order:
        if label not in new_mean.index or label not in buy_hold_mean.index:
            continue
        rows.append(
            {
                "Timeframe": label,
                "Buy & Hold Mean %": buy_hold_mean.get(label, np.nan),
                "Baseline Mean %": base_mean.get(label, np.nan),
                "New Mean %": new_mean[label],
                "New − B&H Excess %": new_mean[label] - buy_hold_mean.get(label, 0.0),
                "New − Baseline Δ %": new_mean[label] - base_mean.get(label, 0.0),
            }
        )

    if not rows:
        return None

    df = pd.DataFrame(rows)
    return df


def load_fold_details(new_csv: Path, baseline_csv: Path) -> Optional[pd.DataFrame]:
    """Load fold-by-fold details with buy-hold comparison."""
    if not new_csv.exists() or not baseline_csv.exists():
        return None

    new_df = pd.read_csv(new_csv, parse_dates=["test_start", "test_end"])
    base_df = pd.read_csv(baseline_csv, parse_dates=["test_start", "test_end"])

    required = {"timeframe", "total_return_pct", "buy_hold_total_return_pct", "test_start", "test_end"}
    if not required.issubset(new_df.columns) or not required.issubset(base_df.columns):
        return None

    merged = new_df.merge(
        base_df,
        on=["timeframe", "test_start", "test_end"],
        suffixes=("_new", "_baseline"),
    )

    merged["diff_new_vs_baseline_pct"] = merged["total_return_pct_new"] - merged["total_return_pct_baseline"]
    merged["diff_new_vs_buyhold_pct"] = merged["total_return_pct_new"] - merged["buy_hold_total_return_pct_new"]
    merged["diff_baseline_vs_buyhold_pct"] = (
        merged["total_return_pct_baseline"] - merged["buy_hold_total_return_pct_baseline"]
    )

    merged["fold_start"] = merged["test_start"].dt.strftime("%Y-%m-%d %H:%M")
    merged["fold_end"] = merged["test_end"].dt.strftime("%Y-%m-%d %H:%M")

    merged.sort_values(["timeframe", "test_start"], inplace=True)
    merged.reset_index(drop=True, inplace=True)
    return merged


def dataframe_to_fold_html(df: pd.DataFrame) -> str:
    """Create fold-by-fold HTML tables by timeframe"""
    if df.empty:
        return ""
    
    html_parts = []
    
    timeframes = df["timeframe"].unique()
    
    for tf in sorted(timeframes):
        tf_data = df[df["timeframe"] == tf]
        
        table_html = f"""
        <div class="fold-details" id="fold-details-{tf.replace(' ', '_').replace(':', '_')}">
            <h3>Fold Details - {tf} ({len(tf_data)} Folds)</h3>
            <table border="1" class="dataframe fold-table">
                <thead>
                    <tr style="text-align: right; background: #3498db; color: white;">
                        <th>Fold Period</th>
                        <th>Buy & Hold %</th>
                        <th>Baseline %</th>
                        <th>New %</th>
                        <th>New - Baseline Δ</th>
                        <th>New - B&H Excess</th>
                        <th>Baseline - B&H Δ</th>
                        <th>Winner</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        for _, row in tf_data.iterrows():
            fold_period = f"{row['fold_start'][:10]} - {row['fold_end'][:10]}"
            
            # Determine winner
            if row['total_return_pct_new'] > row['total_return_pct_baseline']:
                if row['total_return_pct_new'] > row['buy_hold_total_return_pct_new']:
                    winner = "New ✓"
                    winner_style = "background: #d5f4e6;"
                else:
                    winner = "B&H↑"
                    winner_style = "background: #fff3cd;"
            elif row['total_return_pct_baseline'] > row['buy_hold_total_return_pct_baseline']:
                winner = "Baseline"
                winner_style = "background: #f8d7da;"
            else:
                winner = "B&H↑"
                winner_style = "background: #fff3cd;"
            
            table_html += f"""
                    <tr style="{winner_style}">
                        <td><span class="fold-toggle" data-target="fold-{tf.replace(' ', '_').replace(':', '_')}-{row.name}">{fold_period}</span></td>
                        <td>{row['buy_hold_total_return_pct_new']:.2f}</td>
                        <td>{row['total_return_pct_baseline']:.2f}</td>
                        <td>{row['total_return_pct_new']:.2f}</td>
                        <td>{row['diff_new_vs_baseline_pct']:+.2f}</td>
                        <td>{row['diff_new_vs_buyhold_pct']:+.2f}</td>
                        <td>{row['diff_baseline_vs_buyhold_pct']:+.2f}</td>
                        <td><strong>{winner}</strong></td>
                    </tr>
            """
        
        # Add summary row
        summary_new_mean = tf_data["total_return_pct_new"].mean()
        summary_base_mean = tf_data["total_return_pct_baseline"].mean()
        summary_buyhold_mean = tf_data["buy_hold_total_return_pct_new"].mean()
        
        table_html += f"""
                    <tr style="background: #f8f9fa; font-weight: bold;">
                        <td>Summary</td>
                        <td>{summary_buyhold_mean:.2f}</td>
                        <td>{summary_base_mean:.2f}</td>
                        <td>{summary_new_mean:.2f}</td>
                        <td>{summary_new_mean - summary_base_mean:+.2f}</td>
                        <td>{summary_new_mean - summary_buyhold_mean:+.2f}</td>
                        <td>{summary_base_mean - summary_buyhold_mean:+.2f}</td>
                        <td>New {'✓' if summary_new_mean > max(summary_base_mean, summary_buyhold_mean) else '≈'}</td>
                    </tr>
                </tbody>
            </table>
        </div>
        """
        
        html_parts.append(table_html)
    
    return "".join(html_parts)


def dataframe_to_html(df: pd.DataFrame) -> str:
    return df.to_html(
        index=False,
        float_format=lambda x: f"{x:,.2f}",
        classes="data-table",
    )


def ensure_order(df: pd.DataFrame, col: str, order: Sequence[str]) -> pd.DataFrame:
    return df.set_index(col).reindex(order).reset_index()


def main() -> None:
    args = parse_args()
    analysis_dir = Path(args.analysis_dir)
    baseline_dir = Path(args.baseline_analysis_dir)
    output_path = Path(args.output_html)
    summary_csv = Path(args.summary_csv)

    if not analysis_dir.exists():
        raise FileNotFoundError(f"Analysis directory not found: {analysis_dir}")

    price_df = load_price_series(analysis_dir)
    global_start = price_df["timestamp"].min()
    global_end = price_df["timestamp"].max()

    price_min = float(price_df["price"].min())
    price_max = float(price_df["price"].max())
    y_padding = (price_max - price_min) * 0.05 or 1.0
    y0 = price_min - y_padding
    y1 = price_max + y_padding

    new_segments: Dict[str, pd.DataFrame] = {}
    new_states: Dict[str, pd.DataFrame] = {}
    baseline_segments: Dict[str, pd.DataFrame] = {}
    baseline_states: Dict[str, pd.DataFrame] = {}

    for label in TIMEFRAMES:
        seg_df = load_segments(analysis_dir, label)
        if seg_df is not None:
            seg_df = clamp_segments(seg_df, global_start, global_end)
            seg_df = seg_df[seg_df["state"] == "UPTREND"].reset_index(drop=True)
            new_segments[label] = seg_df
            state_df = load_state_prices(analysis_dir, label, global_start, global_end)
            if state_df is not None:
                new_states[label] = state_df

        if baseline_dir.exists():
            base_seg = load_segments(baseline_dir, label)
            if base_seg is not None:
                base_seg = clamp_segments(base_seg, global_start, global_end)
                base_seg = base_seg[base_seg["state"] == "UPTREND"].reset_index(drop=True)
                baseline_segments[label] = base_seg
                base_state = load_state_prices(baseline_dir, label, global_start, global_end)
                if base_state is not None:
                    baseline_states[label] = base_state

    new_highlights: Dict[str, tuple[list, list]] = {}
    baseline_highlights: Dict[str, tuple[list, list]] = {}
    for label in TIMEFRAMES:
        new_highlights[label] = build_highlight_series(new_states.get(label), new_segments.get(label))
        baseline_highlights[label] = build_highlight_series(
            baseline_states.get(label),
            baseline_segments.get(label),
        )

    summary_df = compute_segment_summary(new_states, new_segments, cost=args.cost)
    if not summary_df.empty:
        price_series = price_df.sort_values("timestamp")
        start_price = float(price_series.iloc[0]["price"])
        end_price = float(price_series.iloc[-1]["price"])
        buy_hold_final = 1000.0 * (end_price / start_price)
        summary_df["buy_hold_final_balance"] = buy_hold_final
        summary_df["buy_hold_total_return_pct"] = (buy_hold_final / 1000.0 - 1.0) * 100.0
        summary_csv.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(summary_csv, index=False)

    new_trades_df = load_trades(Path(args.new_trades_csv))
    baseline_trades_df = load_trades(Path(args.baseline_trades_csv))
    new_has_trades = new_trades_df is not None and not new_trades_df.empty
    baseline_has_trades = baseline_trades_df is not None and not baseline_trades_df.empty
    new_markers = prepare_trade_markers(new_trades_df)
    baseline_markers = prepare_trade_markers(baseline_trades_df)

    shapes_by_tf: Dict[str, List[dict]] = {label: [] for label in TIMEFRAMES}

    def pick_initial_timeframe() -> str:
        preferred = "1h"
        if preferred in TIMEFRAMES:
            seg = new_segments.get(preferred)
            base_seg = baseline_segments.get(preferred)
            if (seg is not None and not seg.empty) or (base_seg is not None and not base_seg.empty):
                return preferred
        for tf in TIMEFRAMES:
            seg = new_segments.get(tf)
            base_seg = baseline_segments.get(tf)
            if (seg is not None and not seg.empty) or (base_seg is not None and not base_seg.empty):
                return tf
        return TIMEFRAMES[0]

    initial_label = pick_initial_timeframe()

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.6, 0.4],
        subplot_titles=("New Segments", "Baseline Segments" if baseline_segments else "Baseline Segments (not available)"),
    )

    price_x = price_df["timestamp"].tolist()
    price_y = price_df["price"].tolist()

    fig.add_trace(
        go.Scatter(
            x=price_x,
            y=price_y,
            mode="lines",
            name="Price",
            line=dict(color="#34495e", width=1.6),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=price_x,
            y=price_y,
            mode="lines",
            name="Price (baseline)",
            line=dict(color="#7f8c8d", width=1.2, dash="dot"),
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    highlight_initial = new_highlights.get(initial_label, ([], []))
    marker_indices: Dict[str, Optional[int]] = {
        "new_highlight": None,
        "baseline_highlight": None,
        "new_buy": None,
        "new_sell": None,
        "baseline_buy": None,
        "baseline_sell": None,
    }

    marker_indices["new_highlight"] = len(fig.data)
    fig.add_trace(
        go.Scatter(
            x=highlight_initial[0],
            y=highlight_initial[1],
            mode="lines",
            name="New Uptrend (price)",
            line=dict(color="#27ae60", width=2.4),
            hoverinfo="skip",
            showlegend=True,
        ),
        row=1,
        col=1,
    )

    baseline_highlight_initial = baseline_highlights.get(initial_label, ([], []))
    marker_indices["baseline_highlight"] = len(fig.data)
    fig.add_trace(
        go.Scatter(
            x=baseline_highlight_initial[0],
            y=baseline_highlight_initial[1],
            mode="lines",
            name="Baseline Uptrend (price)",
            line=dict(color="#5dade2", width=2.0),
            hoverinfo="skip",
            showlegend=baseline_has_trades or bool(baseline_highlight_initial[0]),
        ),
        row=2,
        col=1,
    )

    marker_indices["new_buy"] = len(fig.data)
    fig.add_trace(
        go.Scatter(
            x=[],
            y=[],
            mode="markers",
            name="New Buy",
            marker=dict(symbol="triangle-up", color="#2ecc71", size=9, line=dict(color="#145a32", width=1)),
            hovertemplate="Buy<br>%{x}<br>%{y:.6f}<extra></extra>",
            showlegend=new_has_trades,
        ),
        row=1,
        col=1,
    )

    marker_indices["new_sell"] = len(fig.data)
    fig.add_trace(
        go.Scatter(
            x=[],
            y=[],
            mode="markers",
            name="New Sell",
            marker=dict(symbol="triangle-down", color="#e74c3c", size=9, line=dict(color="#922b21", width=1)),
            hovertemplate="Sell<br>%{x}<br>%{y:.6f}<extra></extra>",
            showlegend=new_has_trades,
        ),
        row=1,
        col=1,
    )

    marker_indices["baseline_buy"] = len(fig.data)
    fig.add_trace(
        go.Scatter(
            x=[],
            y=[],
            mode="markers",
            name="Baseline Buy",
            marker=dict(symbol="triangle-up", color="#5dade2", size=8, line=dict(color="#1f618d", width=1)),
            hovertemplate="Baseline Buy<br>%{x}<br>%{y:.6f}<extra></extra>",
            showlegend=baseline_has_trades,
        ),
        row=2,
        col=1,
    )

    marker_indices["baseline_sell"] = len(fig.data)
    fig.add_trace(
        go.Scatter(
            x=[],
            y=[],
            mode="markers",
            name="Baseline Sell",
            marker=dict(symbol="triangle-down", color="#af7ac5", size=8, line=dict(color="#76448a", width=1)),
            hovertemplate="Baseline Sell<br>%{x}<br>%{y:.6f}<extra></extra>",
            showlegend=baseline_has_trades,
        ),
        row=2,
        col=1,
    )

    fig.update_yaxes(title_text="Price", tickformat=".6f", row=1, col=1, range=[y0, y1])
    fig.update_yaxes(title_text="Price", tickformat=".6f", row=2, col=1, range=[y0, y1])
    fig.update_xaxes(title_text="Time", row=2, col=1)

    new_initial = new_markers.get(initial_label, {"buy": ([], []), "sell": ([], [])})
    baseline_initial = baseline_markers.get(initial_label, {"buy": ([], []), "sell": ([], [])})

    if marker_indices["new_buy"] is not None:
        idx = marker_indices["new_buy"]
        fig.data[idx].x, fig.data[idx].y = new_initial["buy"]
    if marker_indices["new_sell"] is not None:
        idx = marker_indices["new_sell"]
        fig.data[idx].x, fig.data[idx].y = new_initial["sell"]
    if marker_indices["baseline_buy"] is not None:
        idx = marker_indices["baseline_buy"]
        fig.data[idx].x, fig.data[idx].y = baseline_initial["buy"]
    if marker_indices["baseline_sell"] is not None:
        idx = marker_indices["baseline_sell"]
        fig.data[idx].x, fig.data[idx].y = baseline_initial["sell"]
    fig.update_layout(
        title=f"Trend Segment Comparison – {initial_label}",
        shapes=shapes_by_tf.get(initial_label, []),
        hovermode="x unified",
        template="plotly_white",
        margin=dict(l=60, r=30, t=70, b=40),
    )

    annotation_top = fig.layout.annotations[0].to_plotly_json() if fig.layout.annotations else {}
    annotation_bottom = (
        fig.layout.annotations[1].to_plotly_json()
        if len(fig.layout.annotations) > 1
        else {}
    )

    buttons = []
    for label in TIMEFRAMES:
        shapes = shapes_by_tf.get(label, [])
        annotation_list = []
        if annotation_top:
            annotation_list.append({**annotation_top, "text": f"New Segments – {label}"})
        if annotation_bottom and baseline_segments:
            annotation_list.append({**annotation_bottom, "text": f"Baseline Segments – {label}"})

        x_update = [None] * len(fig.data)
        y_update = [None] * len(fig.data)

        x_update[0] = price_x
        y_update[0] = price_y
        x_update[1] = price_x
        y_update[1] = price_y
        marker_values = new_markers.get(label, {"buy": ([], []), "sell": ([], [])})
        baseline_values = baseline_markers.get(label, {"buy": ([], []), "sell": ([], [])})
        highlight_values = new_highlights.get(label, ([], []))
        baseline_highlight_values = baseline_highlights.get(label, ([], []))

        if marker_indices["new_highlight"] is not None:
            idx = marker_indices["new_highlight"]
            x_update[idx], y_update[idx] = highlight_values
        if marker_indices["baseline_highlight"] is not None:
            idx = marker_indices["baseline_highlight"]
            x_update[idx], y_update[idx] = baseline_highlight_values

        if marker_indices["new_buy"] is not None:
            idx = marker_indices["new_buy"]
            x_update[idx], y_update[idx] = marker_values["buy"]
        if marker_indices["new_sell"] is not None:
            idx = marker_indices["new_sell"]
            x_update[idx], y_update[idx] = marker_values["sell"]
        if marker_indices["baseline_buy"] is not None:
            idx = marker_indices["baseline_buy"]
            x_update[idx], y_update[idx] = baseline_values["buy"]
        if marker_indices["baseline_sell"] is not None:
            idx = marker_indices["baseline_sell"]
            x_update[idx], y_update[idx] = baseline_values["sell"]

        button = dict(
            label=label,
            method="update",
            args=[
                {"x": x_update, "y": y_update},
                {
                    "shapes": shapes,
                    "title": f"Trend Segment Comparison – {label}",
                    "annotations": annotation_list if annotation_list else fig.layout.annotations,
                },
            ],
        )
        buttons.append(button)

    initial_annotations = []
    if annotation_top:
        initial_annotations.append({**annotation_top, "text": f"New Segments – {initial_label}"})
    if annotation_bottom and baseline_segments:
        initial_annotations.append({**annotation_bottom, "text": f"Baseline Segments – {initial_label}"})

    fig.update_layout(
        updatemenus=[
            dict(
                type="dropdown",
                x=0.01,
                y=1.18,
                showactive=True,
                buttons=buttons,
            )
        ],
        annotations=initial_annotations if initial_annotations else fig.layout.annotations,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=0.98),
    )

    plot_html = fig.to_html(include_plotlyjs="cdn", full_html=False)

    summary_html = ""
    if not summary_df.empty:
        summary_html = dataframe_to_html(summary_df)

    comparison_df = load_walkforward_table(Path(args.new_wf_csv), Path(args.baseline_wf_csv))
    comparison_html = ""
    if comparison_df is not None:
        comparison_df = ensure_order(comparison_df, "Timeframe", TIMEFRAMES)
        comparison_html = dataframe_to_html(comparison_df)

    timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")

    page = f"""<!DOCTYPE html>
<html>
<head>
<meta charset=\"utf-8\" />
<title>Trend Segments Overview</title>
<style>
body {{
    font-family: Arial, sans-serif;
    margin: 20px;
    background-color: #f9fafb;
}}
h1 {{
    margin-bottom: 4px;
}}
p.meta {{
    color: #555;
    margin-top: 0;
}}
.controls {{
    margin: 12px 0 18px;
}}
.controls button {{
    background-color: #2c3e50;
    color: #ecf0f1;
    border: none;
    padding: 8px 14px;
    border-radius: 4px;
    cursor: pointer;
    font-size: 14px;
}}
.controls button:hover {{
    background-color: #1b2733;
}}
.chart-wrapper {{
    position: relative;
    background: #ffffff;
    border: 1px solid #d5d8dc;
    border-radius: 6px;
    padding: 12px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.05);
    height: 75vh;
}}
.chart-wrapper .plotly-graph-div,
.chart-wrapper .plot-container {{
    width: 100% !important;
    height: 100% !important;
}}
.chart-wrapper:fullscreen,
.chart-wrapper:-webkit-full-screen {{
    width: 100vw;
    height: 100vh;
    border-radius: 0;
    border: none;
    margin: 0;
    padding: 0;
    background: #ffffff;
    top: 0;
    left: 0;
    position: fixed;
}}
.chart-wrapper:fullscreen .plotly-graph-div,
.chart-wrapper:fullscreen .plot-container,
.chart-wrapper:-webkit-full-screen .plotly-graph-div,
.chart-wrapper:-webkit-full-screen .plot-container {{
    width: 100% !important;
    height: 100% !important;
    position: absolute !important;
    top: 0 !important;
    left: 0 !important;
}}
.data-table {{
    border-collapse: collapse;
    margin-top: 18px;
    min-width: 680px;
    background: #ffffff;
}}
.data-table th, .data-table td {{
    border: 1px solid #ccd1d9;
    padding: 6px 10px;
    text-align: right;
}}
.data-table th {{
    background: #e9ecef;
}}
.data-table td:first-child, .data-table th:first-child {{
    text-align: left;
}}
.fold-table {{
    margin-top: 20px;
    border-collapse: collapse;
    width: 100%;
}}
.fold-table th, .fold-table td {{
    border: 1px solid #dee2e6;
    padding: 8px 12px;
    text-align: right;
    font-size: 14px;
}}
.fold-table th {{
    background: #3498db;
    color: white;
    font-weight: bold;
}}
.fold-table td:first-child {{
    text-align: left;
    max-width: 200px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}}
.fold-details {{
    margin: 30px 0;
    padding: 20px;
    border: 1px solid #e3e6ea;
    border-radius: 8px;
    background: #ffffff;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}}
.fold-toggle {{
    cursor: pointer;
    color: #007bff;
    font-family: monospace;
    font-size: 12px;
}}
.fold-toggle:hover {{
    text-decoration: underline;
}}
.performance-delta {{
    font-weight: bold;
}}
.positive-delta {{
    color: #28a745;
}}
.negative-delta {{
    color: #dc3545;
}}
.neutral-delta {{
    color: #6c757d;
}}
.win-new {{
    background: #d4edda !important;
}}
.win-baseline {{
    background: #f8d7da !important;
}}
.win-buyhold {{
    background: #fff3cd !important;
}}
</style>
</head>
<body>
<h1>Trend Segments Overview</h1>
<p class=\"meta\">Generated on {timestamp}</p>
<div class=\"controls\">
  <button id=\"fullscreen-btn\" type=\"button\">Enter Fullscreen</button>
</div>
<div id=\"chart-wrapper\" class=\"chart-wrapper\">
{plot_html}
</div>
"""

    # Load fold details if available (using summary files with fold info)
    fold_df = None
    fold_html = ""
    if args.new_wf_csv and args.baseline_wf_csv:
        fold_df = load_fold_details(Path(args.new_wf_csv), Path(args.baseline_wf_csv))
        if fold_df is not None:
            fold_html = dataframe_to_fold_html(fold_df)

    if comparison_html:
        page += """
<h2>Walk-Forward Performance (1.5% fee)</h2>
<p>Comparison of mean returns across folds for baseline vs new configuration.</p>
"""
        # Add enhanced delta styling to comparison table
        comparison_html = comparison_html.replace('New −', '<span class="performance-delta">New −')
        comparison_html = comparison_html.replace(' Δ %</span>', '%</span>')
        
        page += comparison_html

    if summary_html:
        page += """
<h2>Segment Backtest Summary (Sequential, 1.5% fee)</h2>
"""
        page += summary_html
    
    if fold_html:
        page += """
<h2>Fold-by-Fold Details</h2>
<p>Complete breakdown of individual fold performance with buy & hold comparison.</p>
"""
        page += fold_html

    page += """
<script>
const fullscreenBtn = document.getElementById('fullscreen-btn');
const chartWrapper = document.getElementById('chart-wrapper');

function updateFullscreenLabel() {
  if (document.fullscreenElement || document.webkitFullscreenElement) {
    fullscreenBtn.textContent = 'Exit Fullscreen';
  } else {
    fullscreenBtn.textContent = 'Enter Fullscreen';
  }
}

function resizeChart() {
  const graph = chartWrapper.querySelector('.plotly-graph-div');
  if (graph && window.Plotly && window.Plotly.Plots) {
    window.Plotly.Plots.resize(graph);
  }
}

fullscreenBtn.addEventListener('click', () => {
  if (!document.fullscreenElement && !document.webkitFullscreenElement) {
    if (chartWrapper.requestFullscreen) {
      chartWrapper.requestFullscreen();
    } else if (chartWrapper.webkitRequestFullscreen) {
      chartWrapper.webkitRequestFullscreen();
    }
  } else {
    if (document.exitFullscreen) {
      document.exitFullscreen();
    } else if (document.webkitExitFullscreen) {
      document.webkitExitFullscreen();
    }
  }
});

document.addEventListener('fullscreenchange', () => {
  updateFullscreenLabel();
  setTimeout(resizeChart, 100);
});

document.addEventListener('webkitfullscreenchange', () => {
  updateFullscreenLabel();
  setTimeout(resizeChart, 100);
});

window.addEventListener('resize', () => {
  requestAnimationFrame(resizeChart);
});

updateFullscreenLabel();
setTimeout(resizeChart, 200);

// Fold toggle functionality
document.querySelectorAll('.fold-toggle').forEach(toggle => {
  toggle.addEventListener('click', function() {
    const targetId = this.getAttribute('data-target');
    const target = document.getElementById(targetId);
    if (target) {
      target.style.display = target.style.display === 'none' ? 'table-row' : 'none';
    }
  });
});

// Add color coding to performance deltas
document.querySelectorAll('.performance-delta').forEach(delta => {
  const text = delta.textContent;
  const value = parseFloat(text.match(/[-+]?\d+\.?\d+/)?.[0]);
  if (!isNaN(value)) {
    if (value > 0) {
      delta.classList.add('positive-delta');
    } else if (value < 0) {
      delta.classList.add('negative-delta');
    } else {
      delta.classList.add('neutral-delta');
    }
  }
});

</script>
</body>
</html>
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(page, encoding="utf-8")
    print(f"Trend overview written to {output_path}")


if __name__ == "__main__":
    main()
