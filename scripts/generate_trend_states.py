#!/usr/bin/env python3
"""Generate cleaned multi-timeframe trend states backed by percentage checks."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd


DATA_SOURCE = Path("data/pdai_ohlcv_dai_730day_5m.csv")
OUTPUT_DIR = Path("analysis")


TIMEFRAME_ORDER: List[str] = [
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


TIMEFRAME_CONFIG: Dict[str, Dict[str, Any]] = {
    "5min": {
        "rule": "5min",
        "minutes": 5,
        "slope_init_up_pct": 5.0,
        "slope_init_down_pct": -5.0,
        "slope_confirm_up_pct": 1.5,
        "slope_confirm_down_pct": 1.5,
        "net_up_pct": 1.2,
        "net_down_pct": 1.2,
        "range_tolerance_pct": 0.9,
        "range_excursion_pct": 2.5,
        "adx_trend": 22.0,
        "adx_range_max": 16.0,
        "min_bars": 6,
    },
    "15min": {
        "rule": "15min",
        "minutes": 15,
        "slope_init_up_pct": 4.0,
        "slope_init_down_pct": -4.0,
        "slope_confirm_up_pct": 1.0,
        "slope_confirm_down_pct": 1.0,
        "net_up_pct": 2.0,
        "net_down_pct": 2.0,
        "range_tolerance_pct": 1.2,
        "range_excursion_pct": 3.5,
        "adx_trend": 22.0,
        "adx_range_max": 16.0,
        "min_bars": 6,
    },
    "30min": {
        "rule": "30min",
        "minutes": 30,
        "slope_init_up_pct": 3.2,
        "slope_init_down_pct": -3.2,
        "slope_confirm_up_pct": 0.8,
        "slope_confirm_down_pct": 0.8,
        "net_up_pct": 3.0,
        "net_down_pct": 3.0,
        "range_tolerance_pct": 1.8,
        "range_excursion_pct": 4.5,
        "adx_trend": 21.0,
        "adx_range_max": 15.0,
        "min_bars": 5,
    },
    "1h": {
        "rule": "1h",
        "minutes": 60,
        "slope_init_up_pct": 1.0,
        "slope_init_down_pct": -2.2,
        "slope_confirm_up_pct": 0.25,
        "slope_confirm_down_pct": 0.5,
        "net_up_pct": 2.5,
        "net_down_pct": 5.0,
        "range_tolerance_pct": 3.0,
        "range_excursion_pct": 6.0,
        "adx_trend": 20.0,
        "adx_range_max": 14.0,
        "min_bars": 4,
        "panic_loss_pct": 15.0,
        "panic_atr_pct": 6.5,
    },
    "2h": {
        "rule": "2h",
        "minutes": 120,
        "slope_init_up_pct": 0.9,
        "slope_init_down_pct": -1.8,
        "slope_confirm_up_pct": 0.25,
        "slope_confirm_down_pct": 0.4,
        "net_up_pct": 3.5,
        "net_down_pct": 6.5,
        "range_tolerance_pct": 3.5,
        "range_excursion_pct": 7.5,
        "adx_trend": 20.0,
        "adx_range_max": 14.0,
        "min_bars": 4,
        "panic_loss_pct": 16.0,
        "panic_atr_pct": 7.0,
    },
    "4h": {
        "rule": "4h",
        "minutes": 240,
        "slope_init_up_pct": 0.8,
        "slope_init_down_pct": -1.4,
        "slope_confirm_up_pct": 0.25,
        "slope_confirm_down_pct": 0.35,
        "net_up_pct": 4.5,
        "net_down_pct": 8.0,
        "range_tolerance_pct": 4.0,
        "range_excursion_pct": 9.0,
        "adx_trend": 19.0,
        "adx_range_max": 13.0,
        "min_bars": 3,
        "panic_loss_pct": 18.0,
        "panic_atr_pct": 8.0,
    },
    "8h": {
        "rule": "8h",
        "minutes": 480,
        "slope_init_up_pct": 1.1,
        "slope_init_down_pct": -1.1,
        "slope_confirm_up_pct": 0.3,
        "slope_confirm_down_pct": 0.3,
        "net_up_pct": 10.0,
        "net_down_pct": 10.0,
        "range_tolerance_pct": 5.0,
        "range_excursion_pct": 11.0,
        "adx_trend": 18.0,
        "adx_range_max": 12.0,
        "min_bars": 3,
    },
    "16h": {
        "rule": "16h",
        "minutes": 960,
        "slope_init_up_pct": 0.9,
        "slope_init_down_pct": -0.9,
        "slope_confirm_up_pct": 0.25,
        "slope_confirm_down_pct": 0.25,
        "net_up_pct": 12.0,
        "net_down_pct": 12.0,
        "range_tolerance_pct": 5.5,
        "range_excursion_pct": 12.5,
        "adx_trend": 18.0,
        "adx_range_max": 12.0,
        "min_bars": 3,
    },
    "1d": {
        "rule": "1d",
        "minutes": 1_440,
        "slope_init_up_pct": 0.6,
        "slope_init_down_pct": -0.6,
        "slope_confirm_up_pct": 0.2,
        "slope_confirm_down_pct": 0.2,
        "net_up_pct": 8.0,
        "net_down_pct": 8.0,
        "range_tolerance_pct": 4.0,
        "range_excursion_pct": 10.0,
        "adx_trend": 18.0,
        "adx_range_max": 11.0,
        "min_bars": 3,
    },
    "2d": {
        "rule": "2d",
        "minutes": 2_880,
        "slope_init_up_pct": 0.45,
        "slope_init_down_pct": -0.45,
        "slope_confirm_up_pct": 0.15,
        "slope_confirm_down_pct": 0.15,
        "net_up_pct": 10.0,
        "net_down_pct": 10.0,
        "range_tolerance_pct": 5.0,
        "range_excursion_pct": 12.0,
        "adx_trend": 17.0,
        "adx_range_max": 11.0,
        "min_bars": 3,
    },
}


TREND_LOOKBACK_MINUTES = 1_440
SMOOTH_SPAN_BARS = 5


@dataclass
class TrendOutputs:
    timeframe: str
    row_level: pd.DataFrame
    segment_summary: pd.DataFrame


def _load_base_dataframe(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    if df.empty:
        raise ValueError(f"No data found in {csv_path}")

    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
    else:
        df["timestamp"] = df["timestamp"].dt.tz_convert("UTC")

    df = df.sort_values("timestamp").set_index("timestamp")
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Data missing required columns: {sorted(missing)}")

    return df


def _resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    if rule == "5min":
        resampled = df.copy()
    else:
        agg = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
        if "price" in df.columns:
            agg["price"] = "last"
        resampled = df.resample(rule, label="right", closed="right").agg(agg).dropna(how="any")

    resampled = resampled[~resampled["close"].isna()].copy()
    resampled.index.name = "timestamp"
    return resampled


def _rolling_linear_slope(values: pd.Series, window: int) -> pd.Series:
    if window < 2:
        return pd.Series(np.nan, index=values.index)

    x = np.arange(window)
    x_mean = x.mean()
    denom = ((x - x_mean) ** 2).sum()

    def _calc(arr: np.ndarray) -> float:
        if np.any(~np.isfinite(arr)):
            return np.nan
        y = arr
        y_mean = y.mean()
        num = ((x - x_mean) * (y - y_mean)).sum()
        return num / denom if denom else np.nan

    return values.rolling(window, min_periods=window).apply(_calc, raw=True)


def _compute_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    period = max(period, 3)
    high = high.astype(float)
    low = low.astype(float)
    close = close.astype(float)

    up_move = high.diff()
    down_move = (-low.diff()).clip(lower=0.0)

    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    prev_close = close.shift(1)
    tr_components = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    )
    true_range = tr_components.max(axis=1).fillna(0.0)

    atr = true_range.ewm(alpha=1 / period, adjust=False).mean()

    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean().divide(atr.where(atr != 0, np.nan))
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean().divide(atr.where(atr != 0, np.nan))

    di_sum = plus_di + minus_di
    dx = (plus_di.subtract(minus_di).abs().divide(di_sum.where(di_sum != 0, np.nan))) * 100
    adx = dx.ewm(alpha=1 / period, adjust=False).mean()

    return adx.fillna(0.0)


def _compute_atr_percent(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    period = max(period, 3)
    prev_close = close.shift(1)
    tr_components = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    )
    true_range = tr_components.max(axis=1).fillna(0.0)
    atr = true_range.ewm(alpha=1 / period, adjust=False).mean()
    atr_pct = atr.divide(close.where(close != 0, np.nan)) * 100
    return atr_pct.fillna(0.0)


def _segment_boundaries(states: List[str]) -> List[Dict[str, int]]:
    if not states:
        return []
    segments: List[Dict[str, int]] = []
    start_idx = 0
    current = states[0]
    for idx in range(1, len(states)):
        if states[idx] != current:
            segments.append({"state": current, "start_idx": start_idx, "end_idx": idx - 1})
            start_idx = idx
            current = states[idx]
    segments.append({"state": current, "start_idx": start_idx, "end_idx": len(states) - 1})
    return segments


def _merge_short_segments(segments: List[Dict[str, int]], min_bars: int) -> List[Dict[str, int]]:
    if not segments or len(segments) == 1:
        return segments

    merged = [seg.copy() for seg in segments]
    idx = 0
    while idx < len(merged):
        seg = merged[idx]
        length = seg["end_idx"] - seg["start_idx"] + 1
        if length >= min_bars:
            idx += 1
            continue

        if idx == 0:
            merged[1]["start_idx"] = seg["start_idx"]
            merged.pop(0)
        elif idx == len(merged) - 1:
            merged[-2]["end_idx"] = seg["end_idx"]
            merged.pop()
            idx -= 1
        else:
            prev_len = merged[idx - 1]["end_idx"] - merged[idx - 1]["start_idx"] + 1
            next_len = merged[idx + 1]["end_idx"] - merged[idx + 1]["start_idx"] + 1
            if prev_len >= next_len:
                merged[idx - 1]["end_idx"] = seg["end_idx"]
                merged.pop(idx)
                idx -= 1
            else:
                merged[idx + 1]["start_idx"] = seg["start_idx"]
                merged.pop(idx)
        if idx < 0:
            idx = 0

    return merged


def _segment_metrics(segment_df: pd.DataFrame, minutes: int) -> Dict[str, Any]:
    start_price = float(segment_df["close"].iloc[0])
    end_price = float(segment_df["close"].iloc[-1])
    max_price = float(segment_df["close"].max())
    min_price = float(segment_df["close"].min())

    duration_bars = int(len(segment_df))
    duration_minutes = duration_bars * minutes
    duration_hours = duration_minutes / 60.0
    duration_days = duration_hours / 24.0

    def _pct(value: float, base: float) -> float:
        return ((value / base) - 1.0) * 100.0 if base else 0.0

    net_return_pct = _pct(end_price, start_price)
    max_gain_pct = _pct(max_price, start_price)
    max_loss_pct = _pct(min_price, start_price)
    range_pct = _pct(max_price, min_price) if min_price else 0.0

    slope_series = segment_df["log_slope_per_day_pct"].dropna()
    adx_series = segment_df["adx"].dropna()
    atr_series = segment_df["atr_percent"].dropna()

    avg_slope = float(slope_series.mean()) if not slope_series.empty else 0.0
    median_slope = float(slope_series.median()) if not slope_series.empty else 0.0
    avg_adx = float(adx_series.mean()) if not adx_series.empty else 0.0
    median_adx = float(adx_series.median()) if not adx_series.empty else 0.0
    max_adx = float(adx_series.max()) if not adx_series.empty else 0.0
    avg_atr_pct = float(atr_series.mean()) if not atr_series.empty else 0.0

    return {
        "start_time": segment_df["timestamp"].iloc[0],
        "end_time": segment_df["timestamp"].iloc[-1],
        "start_price": start_price,
        "end_price": end_price,
        "max_price": max_price,
        "min_price": min_price,
        "duration_bars": duration_bars,
        "duration_minutes": duration_minutes,
        "duration_hours": duration_hours,
        "duration_days": duration_days,
        "net_return_pct": net_return_pct,
        "max_gain_pct": max_gain_pct,
        "max_loss_pct": max_loss_pct,
        "range_pct": range_pct,
        "avg_slope_pct": avg_slope,
        "median_slope_pct": median_slope,
        "avg_adx": avg_adx,
        "median_adx": median_adx,
        "max_adx": max_adx,
        "avg_atr_pct": avg_atr_pct,
    }


def _classify_segment(metrics: Dict[str, Any], config: Dict[str, Any]) -> str:
    net = metrics["net_return_pct"]
    slope_mean = metrics["avg_slope_pct"]
    adx_mean = metrics["avg_adx"]
    adx_max = metrics["max_adx"]

    if net >= config["net_up_pct"] and (
        slope_mean >= config["slope_confirm_up_pct"] or adx_mean >= config["adx_trend"] or adx_max >= config["adx_trend"]
    ):
        return "UPTREND"

    if net <= -config["net_down_pct"] and (
        slope_mean <= -config["slope_confirm_down_pct"] or adx_mean >= config["adx_trend"] or adx_max >= config["adx_trend"]
    ):
        return "DOWNTREND"

    if (
        abs(net) <= config["range_tolerance_pct"]
        and metrics["range_pct"] <= config["range_excursion_pct"]
        and adx_mean <= config["adx_range_max"]
    ):
        return "RANGE"

    if net > 0:
        return "UPTREND"
    if net < 0:
        return "DOWNTREND"
    return "RANGE"


def _reclassify_segments(row_df: pd.DataFrame, segments: List[Dict[str, int]], config: Dict[str, Any]) -> List[Dict[str, Any]]:
    minutes = config["minutes"]
    working = [seg.copy() for seg in segments]

    while True:
        updated: List[Dict[str, Any]] = []
        merged_flag = False

        for seg in working:
            seg_slice = row_df.iloc[seg["start_idx"] : seg["end_idx"] + 1]
            metrics = _segment_metrics(seg_slice, minutes)
            state = _classify_segment(metrics, config)
            seg_data = {"start_idx": seg["start_idx"], "end_idx": seg["end_idx"], "state": state}

            if updated and updated[-1]["state"] == state:
                updated[-1]["end_idx"] = seg_data["end_idx"]
                merged_flag = True
            else:
                updated.append(seg_data)

        if not merged_flag:
            final_segments: List[Dict[str, Any]] = []
            for seg in updated:
                seg_slice = row_df.iloc[seg["start_idx"] : seg["end_idx"] + 1]
                metrics = _segment_metrics(seg_slice, minutes)
                state = _classify_segment(metrics, config)
                final_segments.append({
                    "state": state,
                    "start_idx": seg["start_idx"],
                    "end_idx": seg["end_idx"],
                    "metrics": metrics,
                })
            return final_segments

        working = updated


def _trend_for_timeframe(df: pd.DataFrame, label: str, config: Dict[str, Any]) -> TrendOutputs:
    minutes = config["minutes"]
    resampled = _resample_ohlcv(df, config["rule"])
    if resampled.empty:
        raise ValueError(f"No data after resampling for timeframe {label}")

    resampled = resampled.copy()
    resampled["timestamp"] = resampled.index

    window_minutes = TREND_LOOKBACK_MINUTES
    window_bars = max(int(math.ceil(window_minutes / minutes)), config["min_bars"])
    smoothing_span = min(max(SMOOTH_SPAN_BARS, 2), len(resampled))

    smooth_close = resampled["close"].ewm(span=smoothing_span, adjust=False).mean()
    log_close = np.log(smooth_close.clip(lower=1e-12))

    slope = _rolling_linear_slope(log_close, window_bars)
    bars_per_day = max(1440.0 / minutes, 1.0)
    slope_per_day_pct = slope * bars_per_day * 100.0

    adx_period = max(min(window_bars // 2, 60), 10)
    adx = _compute_adx(resampled["high"], resampled["low"], resampled["close"], adx_period)
    atr_pct = _compute_atr_percent(resampled["high"], resampled["low"], resampled["close"], adx_period)

    ema_fast_span = max(2, window_bars // 4)
    ema_slow_span = max(ema_fast_span * 2, window_bars)
    ema_fast = resampled["close"].ewm(span=ema_fast_span, adjust=False).mean()
    ema_slow = resampled["close"].ewm(span=ema_slow_span, adjust=False).mean()

    initial_state = np.full(len(resampled), "RANGE", dtype=object)
    up_mask = (
        (slope_per_day_pct >= config["slope_init_up_pct"])
        & (adx >= config["adx_trend"])
        & (ema_fast >= ema_slow)
    )
    down_mask = (
        (slope_per_day_pct <= config["slope_init_down_pct"])
        & (adx >= config["adx_trend"])
        & (ema_fast <= ema_slow)
    )
    initial_state[up_mask] = "UPTREND"
    initial_state[down_mask] = "DOWNTREND"

    row_df = pd.DataFrame(
        {
            "timestamp": resampled["timestamp"],
            "open": resampled["open"],
            "high": resampled["high"],
            "low": resampled["low"],
            "close": resampled["close"],
            "volume": resampled["volume"],
            "initial_state": initial_state,
            "log_slope_per_day_pct": slope_per_day_pct,
            "adx": adx,
            "atr_percent": atr_pct,
            "ema_fast": ema_fast,
            "ema_slow": ema_slow,
        }
    )

    segments = _segment_boundaries(row_df["initial_state"].tolist())
    segments = _merge_short_segments(segments, config["min_bars"])
    final_segments = _reclassify_segments(row_df, segments, config)

    row_df["trend_state"] = "RANGE"
    row_df["segment_index"] = 0
    for idx, seg in enumerate(final_segments, start=1):
        slice_index = row_df.index[seg["start_idx"] : seg["end_idx"] + 1]
        row_df.loc[slice_index, "trend_state"] = seg["state"]
        row_df.loc[slice_index, "segment_index"] = idx

    row_df["timeframe"] = label
    row_df["timeframe_minutes"] = minutes
    row_df["trend_strength_score"] = row_df["log_slope_per_day_pct"] * (row_df["adx"] / max(config["adx_trend"], 1))

    summary_rows: List[Dict[str, Any]] = []
    for idx, seg in enumerate(final_segments, start=1):
        metrics = seg["metrics"]
        summary_rows.append(
            {
                "segment_id": idx,
                "timeframe": label,
                "timeframe_minutes": minutes,
                "state": seg["state"],
                "start_time": metrics["start_time"],
                "end_time": metrics["end_time"],
                "start_price": metrics["start_price"],
                "end_price": metrics["end_price"],
                "max_price": metrics["max_price"],
                "min_price": metrics["min_price"],
                "net_return_pct": metrics["net_return_pct"],
                "max_gain_pct": metrics["max_gain_pct"],
                "max_loss_pct": metrics["max_loss_pct"],
                "range_pct": metrics["range_pct"],
                "duration_bars": metrics["duration_bars"],
                "duration_minutes": metrics["duration_minutes"],
                "duration_hours": metrics["duration_hours"],
                "duration_days": metrics["duration_days"],
                "avg_slope_pct": metrics["avg_slope_pct"],
                "median_slope_pct": metrics["median_slope_pct"],
                "avg_adx": metrics["avg_adx"],
                "median_adx": metrics["median_adx"],
                "max_adx": metrics["max_adx"],
                "avg_atr_pct": metrics["avg_atr_pct"],
            }
        )

    summary = pd.DataFrame(summary_rows)
    return TrendOutputs(label, row_df, summary)


def generate_trend_states() -> Dict[str, TrendOutputs]:
    base_df = _load_base_dataframe(DATA_SOURCE)
    outputs: Dict[str, TrendOutputs] = {}

    for label in TIMEFRAME_ORDER:
        config = TIMEFRAME_CONFIG[label]
        outputs[label] = _trend_for_timeframe(base_df, label, config)

    return outputs


def save_outputs(outputs: Dict[str, TrendOutputs]) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    detail_frames: List[pd.DataFrame] = []
    summary_frames: List[pd.DataFrame] = []

    for label, result in outputs.items():
        detail_path = OUTPUT_DIR / f"trend_states_{label}.csv"
        summary_path = OUTPUT_DIR / f"trend_segments_{label}.csv"
        result.row_level.to_csv(detail_path, index=False)
        result.segment_summary.to_csv(summary_path, index=False)

        detail_frames.append(result.row_level)
        summary_frames.append(result.segment_summary)

    combined_detail = pd.concat(detail_frames, ignore_index=True)
    combined_summary = pd.concat(summary_frames, ignore_index=True)

    combined_detail.to_csv(OUTPUT_DIR / "trend_states_all_timeframes.csv", index=False)
    combined_summary.to_csv(OUTPUT_DIR / "trend_segments_all_timeframes.csv", index=False)


def main() -> None:
    outputs = generate_trend_states()
    save_outputs(outputs)


if __name__ == "__main__":
    main()
