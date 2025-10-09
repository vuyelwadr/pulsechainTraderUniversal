"""Generate trend classification for PLS DAI dataset using hourly resampled data."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List
import csv

import numpy as np
import pandas as pd

DATA_PATH = Path("data/pdai_ohlcv_dai_730day_5m.csv")
HOURLY_SEGMENTS_CSV = Path("reports/pdai_trend_segments_1h.csv")
HOURLY_LABELS_CSV = Path("reports/pdai_trend_labels_1h.csv")

WINDOW = 24  # 1 day of hourly candles to react quicker to emerging moves
RSQUARED_THRESHOLD = 0.55
SLOPE_THRESHOLD = 0.0015  # log-price slope per hour that indicates a meaningful drift


def _linear_regression_slope(values: np.ndarray) -> float:
    """Return slope of a linear regression with R^2 filtering."""
    if np.any(np.isnan(values)):
        return np.nan

    x = np.arange(values.size)
    slope, intercept = np.polyfit(x, values, 1)
    fitted = slope * x + intercept
    ss_tot = np.sum((values - values.mean()) ** 2)
    if ss_tot == 0:
        return np.nan
    ss_res = np.sum((values - fitted) ** 2)
    r_squared = 1 - ss_res / ss_tot
    return float(slope if r_squared >= RSQUARED_THRESHOLD else 0.0)


@dataclass
class TrendSegment:
    label: str
    start: pd.Timestamp
    end: pd.Timestamp
    bars: int
    start_price: float
    end_price: float
    pct_change: float
    avg_slope: float

    @property
    def duration_hours(self) -> float:
        return self.bars


def classify_trend(slope: float) -> str:
    if slope >= SLOPE_THRESHOLD:
        return "uptrend"
    if slope <= -SLOPE_THRESHOLD:
        return "downtrend"
    return "range"


def build_trend_map() -> None:
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"]).sort_values("timestamp")
    df = df.set_index("timestamp")

    hourly = df.resample("1h").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    ).dropna()

    log_close = np.log(hourly["close"])
    slopes = (
        log_close.rolling(WINDOW, min_periods=WINDOW, center=True)
        .apply(_linear_regression_slope, raw=True)
        .rename("log_slope")
    )

    classifications = slopes.apply(classify_trend).rename("trend")
    labels_df = pd.concat([hourly[["open", "high", "low", "close", "volume"]], slopes, classifications], axis=1)
    labels_df = labels_df.dropna(subset=["log_slope"])
    labels_df.to_csv(
        HOURLY_LABELS_CSV,
        index_label="timestamp",
        float_format="%.12f",
        quoting=csv.QUOTE_ALL,
    )

    segments: List[TrendSegment] = []
    if not labels_df.empty:
        current_label = labels_df["trend"].iloc[0]
        start_idx = 0
        for idx, label in enumerate(labels_df["trend"].values):
            if label != current_label:
                segment = _build_segment(labels_df.iloc[start_idx:idx], current_label)
                segments.append(segment)
                current_label = label
                start_idx = idx
        # add final segment
        segments.append(_build_segment(labels_df.iloc[start_idx:], current_label))

    segments_df = pd.DataFrame(
        [
            {
                "trend": seg.label,
                "start": seg.start,
                "end": seg.end,
                "hours": seg.duration_hours,
                "start_price": seg.start_price,
                "end_price": seg.end_price,
                "pct_change": seg.pct_change,
                "avg_log_slope": seg.avg_slope,
            }
            for seg in segments
        ]
    )
    segments_df.to_csv(
        HOURLY_SEGMENTS_CSV,
        index=False,
        float_format="%.12f",
        quoting=csv.QUOTE_ALL,
    )


def _build_segment(window: pd.DataFrame, label: str) -> TrendSegment:
    start_ts = window.index[0]
    end_ts = window.index[-1]
    start_price = float(window["open"].iloc[0])
    end_price = float(window["close"].iloc[-1])
    pct_change = (end_price / start_price - 1) * 100
    avg_slope = float(window["log_slope"].mean())
    return TrendSegment(
        label=label,
        start=start_ts,
        end=end_ts,
        bars=len(window),
        start_price=start_price,
        end_price=end_price,
        pct_change=pct_change,
        avg_slope=avg_slope,
    )


if __name__ == "__main__":
    build_trend_map()
