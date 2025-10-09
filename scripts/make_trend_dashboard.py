#!/usr/bin/env python3
"""Render a zoomable HTML dashboard with price + trend regime overlays."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


OUTPUT_HTML = Path("analysis/trend_dashboard.html")
TIMEFRAMES: Tuple[str, ...] = (
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
)
TREND_COLORS = {
    "UPTREND": "rgba(34, 197, 94, 0.28)",
    "DOWNTREND": "rgba(239, 68, 68, 0.30)",
    "RANGE": "rgba(148, 163, 184, 0.18)",
}
MAX_POINTS = 4500
MAX_TABLE_ROWS = 200


def _load_timeframe_data(tf: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    price_path = Path(f"analysis/trend_states_{tf}.csv")
    summary_path = Path(f"analysis/trend_segments_{tf}.csv")
    if not price_path.exists() or not summary_path.exists():
        raise FileNotFoundError(f"Missing trend data for timeframe {tf}")

    price_df = pd.read_csv(price_path, parse_dates=["timestamp"])
    summary_df = pd.read_csv(summary_path, parse_dates=["start_time", "end_time"])

    if price_df.empty or summary_df.empty:
        raise ValueError(f"No records found for timeframe {tf}")

    price_df = price_df.sort_values("timestamp").reset_index(drop=True)
    summary_df = summary_df.sort_values("start_time").reset_index(drop=True)

    return price_df, summary_df


def _decimate(df: pd.DataFrame, max_points: int = MAX_POINTS) -> pd.DataFrame:
    if len(df) <= max_points:
        return df.copy()
    step = max(len(df) // max_points, 1)
    sampled = df.iloc[::step].copy()
    if sampled.iloc[-1]["timestamp"] != df.iloc[-1]["timestamp"]:
        sampled = pd.concat([sampled, df.iloc[[-1]]], ignore_index=True)
    return sampled


def _format_table(summary: pd.DataFrame) -> Dict[str, List[str]]:
    display = summary.copy()
    display["start_time"] = display["start_time"].dt.strftime("%Y-%m-%d %H:%M")
    display["end_time"] = display["end_time"].dt.strftime("%Y-%m-%d %H:%M")
    display["net_return"] = display["net_return_pct"].map(lambda v: f"{v:+.2f}%")
    display["max_gain"] = display["max_gain_pct"].map(lambda v: f"{v:+.2f}%")
    display["max_loss"] = display["max_loss_pct"].map(lambda v: f"{v:+.2f}%")

    def _format_duration(row: pd.Series) -> str:
        mins = row["timeframe_minutes"]
        if mins < 720:
            return f"{row['duration_hours']:.1f} h"
        return f"{row['duration_days']:.1f} d"

    display["duration"] = display.apply(_format_duration, axis=1)

    columns = {
        "State": display["state"].tolist(),
        "Start": display["start_time"].tolist(),
        "End": display["end_time"].tolist(),
        "Duration": display["duration"].tolist(),
        "Net %": display["net_return"].tolist(),
        "Max Gain %": display["max_gain"].tolist(),
        "Max Loss %": display["max_loss"].tolist(),
    }
    return columns


def _build_shapes(summary: pd.DataFrame) -> List[Dict[str, object]]:
    shapes: List[Dict[str, object]] = []
    for _, row in summary.iterrows():
        fill = TREND_COLORS.get(row["state"], "rgba(148, 163, 184, 0.15)")
        shapes.append(
            dict(
                type="rect",
                xref="x",
                yref="paper",
                x0=row["start_time"],
                x1=row["end_time"],
                y0=0,
                y1=1,
                fillcolor=fill,
                opacity=1.0,
                layer="below",
                line={"width": 0},
            )
        )
    return shapes


def build_dashboard(default_tf: str = "1h") -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        row_heights=[0.68, 0.32],
        specs=[[{"type": "scatter"}], [{"type": "table"}]],
    )

    trace_visibility: List[bool] = []
    shapes_map: Dict[str, List[Dict[str, object]]] = {}

    for tf in TIMEFRAMES:
        price_df, summary_df = _load_timeframe_data(tf)
        price_plot = _decimate(price_df)

        line_trace = go.Scatter(
            x=price_plot["timestamp"],
            y=price_plot["close"],
            mode="lines",
            name=f"{tf} close",
            line=dict(color="#0f62fe", width=1.8),
            hovertemplate="%{x|%Y-%m-%d %H:%M}<br>Close: %{y:.6f}<extra></extra>",
            visible=(tf == default_tf),
        )
        fig.add_trace(line_trace, row=1, col=1)

        summary_tail = summary_df.tail(MAX_TABLE_ROWS)
        table_columns = _format_table(summary_tail)
        table_trace = go.Table(
            header=dict(
                values=list(table_columns.keys()),
                fill_color="#0b1220",
                font=dict(color="#f8fafc", size=12),
                align="left",
            ),
            cells=dict(
                values=list(table_columns.values()),
                fill_color="#111827",
                font=dict(color="#e5e7eb", size=11),
                align="left",
            ),
            visible=(tf == default_tf),
            name=f"{tf} regimes",
        )
        fig.add_trace(table_trace, row=2, col=1)

        shapes_map[tf] = _build_shapes(summary_df)

    total_traces = len(fig.data)
    traces_per_tf = 2

    buttons = []
    for idx, tf in enumerate(TIMEFRAMES):
        visible = [False] * total_traces
        visible_idx = idx * traces_per_tf
        visible[visible_idx] = True
        visible[visible_idx + 1] = True

        buttons.append(
            dict(
                label=tf,
                method="update",
                args=[
                    {"visible": visible},
                    {
                        "title": f"Trend Dashboard — {tf}",
                        "shapes": shapes_map[tf],
                    },
                ],
            )
        )

    fig.update_layout(
        title=f"Trend Dashboard — {default_tf}",
        updatemenus=[
            dict(
                type="dropdown",
                active=TIMEFRAMES.index(default_tf),
                buttons=buttons,
                x=0.01,
                xanchor="left",
                y=1.25,
                yanchor="top",
            )
        ],
        hovermode="x unified",
        template="plotly_dark",
        plot_bgcolor="#050910",
        paper_bgcolor="#050910",
        margin=dict(l=70, r=30, t=80, b=40),
    )

    fig.update_layout(shapes=shapes_map[default_tf])

    fig.update_yaxes(
        title="Price (DAI)",
        showgrid=True,
        gridcolor="#1f2937",
        zeroline=False,
        row=1,
        col=1,
    )
    fig.update_xaxes(
        title="Timestamp",
        showgrid=True,
        gridcolor="#1f2937",
        rangeselector=dict(
            buttons=[
                dict(count=7, label="7d", step="day", stepmode="backward"),
                dict(count=30, label="30d", step="day", stepmode="backward"),
                dict(count=90, label="90d", step="day", stepmode="backward"),
                dict(step="all", label="All"),
            ]
        ),
        rangeslider=dict(visible=True),
        row=1,
        col=1,
    )

    fig.update_yaxes(visible=False, row=2, col=1)
    fig.update_xaxes(matches="x", row=2, col=1)

    return fig


def main() -> None:
    fig = build_dashboard(default_tf="1h")
    OUTPUT_HTML.parent.mkdir(exist_ok=True)
    fig.write_html(OUTPUT_HTML, include_plotlyjs="cdn")
    print(f"Saved dashboard to {OUTPUT_HTML}")


if __name__ == "__main__":
    main()
