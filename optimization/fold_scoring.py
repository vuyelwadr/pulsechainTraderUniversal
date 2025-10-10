"""Utilities for scoring walk-forward folds with worst-case awareness."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd


def _ensure_columns(df: pd.DataFrame, required: Sequence[str]) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Walk-forward DataFrame missing columns: {', '.join(missing)}")


def compute_fold_scores(
    walkforward_df: pd.DataFrame,
    *,
    group_fields: Sequence[str] = ("timeframe", "strategy"),
    lambda_penalty: float = 5.0,
    gamma_penalty: float = 0.1,
) -> pd.DataFrame:
    """
    Aggregate walk-forward results and compute a fold-aware score.

    Score = 0.6 * median_excess + 0.4 * Q25_excess - lambda_penalty * CVaR10(losses) - gamma_penalty * median_fee

    Args:
        walkforward_df: DataFrame containing hold-out metrics.
        group_fields: Columns to group by (defaults to timeframe + strategy).
        lambda_penalty: Penalty factor applied to CVaR tail losses.
        gamma_penalty: Penalty factor for fee drag (median total cost pct).

    Returns:
        DataFrame with per-group statistics and the computed score.
    """
    if walkforward_df.empty:
        return pd.DataFrame(columns=list(group_fields) + [
            "min_excess",
            "median_excess",
            "max_excess",
            "positive_fold_ratio",
            "q25_excess",
            "cvar_10_loss",
            "median_fee_pct",
            "median_return_pct",
            "median_buy_hold_pct",
            "median_max_drawdown_pct",
            "folds",
            "score",
        ])

    _ensure_columns(
        walkforward_df,
        [
            "holdout_total_return_pct",
            "holdout_buy_hold_return_pct",
            "holdout_max_drawdown_pct",
        ],
    )

    working = walkforward_df.copy()
    # Ensure grouping fields exist; if not, create defaults.
    for field in group_fields:
        if field not in working.columns:
            working[field] = "global"

    working["excess"] = (
        working["holdout_total_return_pct"] - working["holdout_buy_hold_return_pct"]
    )

    agg_map = {
        "excess": ["min", "median", "max", "mean", "count"],
        "holdout_total_cost_pct": "median",
        "holdout_total_return_pct": "median",
        "holdout_buy_hold_return_pct": "median",
        "holdout_max_drawdown_pct": "median",
    }

    grouped = working.groupby(list(group_fields)).agg(agg_map)
    # Flatten MultiIndex columns
    grouped.columns = [
        "_".join(col).rstrip("_")
        if isinstance(col, tuple)
        else col
        for col in grouped.columns.to_flat_index()
    ]
    grouped = grouped.reset_index()

    grouped.rename(
        columns={
            "excess_min": "min_excess",
            "excess_median": "median_excess",
            "excess_max": "max_excess",
            "excess_mean": "mean_excess",
            "excess_count": "folds",
            "holdout_total_cost_pct_median": "median_fee_pct",
            "holdout_total_return_pct_median": "median_return_pct",
            "holdout_buy_hold_return_pct_median": "median_buy_hold_pct",
            "holdout_max_drawdown_pct_median": "median_max_drawdown_pct",
        },
        inplace=True,
    )

    def _tail_stats(group: pd.DataFrame) -> pd.Series:
        values = group["excess"].dropna().to_numpy()
        if values.size == 0:
            return pd.Series({"q25_excess": float("nan"), "cvar_10_loss": 0.0})

        q25 = float(np.percentile(values, 25))
        losses = -values[values < 0]
        if losses.size == 0:
            cvar_loss = 0.0
        else:
            tail_count = max(1, int(np.ceil(losses.size * 0.1)))
            worst_losses = np.sort(losses)[-tail_count:]
            cvar_loss = float(np.mean(worst_losses))

        return pd.Series({"q25_excess": q25, "cvar_10_loss": cvar_loss})

    tail_stats = (
        working.groupby(list(group_fields))
        .apply(_tail_stats)
        .reset_index()
    )
    grouped = grouped.merge(tail_stats, on=list(group_fields), how="left")

    # Positive fold ratio
    positive_ratio = (
        working.assign(positive=lambda row: row["excess"] > 0)
        .groupby(list(group_fields))["positive"]
        .mean()
        .reset_index(name="positive_fold_ratio")
    )
    grouped = grouped.merge(positive_ratio, on=list(group_fields), how="left")

    # Score calculation (robust mix: median + Q25, penalise CVaR losses and fees)
    q25 = grouped["q25_excess"].where(~grouped["q25_excess"].isna(), grouped["median_excess"])
    robust_component = 0.6 * grouped["median_excess"] + 0.4 * q25
    cvar_penalty = grouped.get("cvar_10_loss", pd.Series(0, index=grouped.index)).fillna(0.0)
    fee_penalty = grouped.get("median_fee_pct", pd.Series(0, index=grouped.index)).fillna(0.0)
    grouped["score"] = robust_component - lambda_penalty * cvar_penalty - gamma_penalty * fee_penalty

    # Sort best to worst
    grouped = grouped.sort_values(
        ["score", "median_excess"], ascending=[False, False]
    ).reset_index(drop=True)

    return grouped
