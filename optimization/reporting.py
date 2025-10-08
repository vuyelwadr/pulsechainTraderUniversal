"""Reporting utilities for fast optimizer runner."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd


def _format_params(params: Dict) -> str:
    if not params:
        return "{}"
    return json.dumps(params, sort_keys=True)


def _collect_timeframe_rows(
    trials: Sequence[Dict],
    label: str,
) -> pd.DataFrame:
    rows: List[Dict] = []
    for trial in trials:
        metrics = (trial.get('timeframe_metrics') or {}).get(label) or {}
        if not metrics:
            continue
        best_objective = None
        best_score = None
        for objective, payload in (trial.get('objective_scores') or {}).items():
            score = payload.get('score')
            if score is None:
                continue
            try:
                score_value = float(score)
            except (TypeError, ValueError):
                continue
            if best_score is None or score_value > best_score:
                best_score = score_value
                best_objective = objective
        rows.append(
            {
                'trial_id': trial['trial_id'],
                'strategy': trial['strategy'],
                'parameters': _format_params(trial.get('parameters', {})),
                'best_objective': best_objective,
                'best_objective_score': best_score if best_score is not None else 0.0,
                'total_return_pct': metrics.get('total_return_pct', 0.0),
                'buy_hold_return_pct': metrics.get('buy_hold_return_pct', 0.0),
                'buy_hold_max_drawdown_pct': metrics.get('buy_hold_max_drawdown_pct', 0.0),
                'max_drawdown_pct': metrics.get('max_drawdown_pct', 0.0),
                'cagr_pct': metrics.get('cagr_pct', 0.0),
                'sharpe_ratio': metrics.get('sharpe_ratio', metrics.get('sharpe', 0.0)),
                'sortino_ratio': metrics.get('sortino_ratio', metrics.get('sortino', 0.0)),
                'win_rate_pct': metrics.get('win_rate_pct', 0.0),
                'total_trades': metrics.get('total_trades', metrics.get('num_trades', 0)),
                'profit_factor': metrics.get('profit_factor', 0.0),
                'recovery_factor': metrics.get('recovery_factor', 0.0),
                'total_cost_pct': metrics.get('total_cost_pct', 0.0),
                'avg_cost_per_trade_pct': metrics.get('avg_cost_per_trade_pct', 0.0),
                'final_balance': metrics.get('final_balance', 0.0),
            }
        )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = df.sort_values('total_return_pct', ascending=False).reset_index(drop=True)
    return df


def _collect_objective_rows(
    trials: Sequence[Dict],
    objective: str,
    stage_label: str,
) -> pd.DataFrame:
    rows: List[Dict] = []
    for trial in trials:
        obj = (trial.get('objective_scores') or {}).get(objective)
        stage_metrics = (trial.get('timeframe_metrics') or {}).get(stage_label) or {}
        if not obj or not stage_metrics:
            continue
        rows.append(
            {
                'trial_id': trial['trial_id'],
                'strategy': trial['strategy'],
                'parameters': _format_params(trial.get('parameters', {})),
                'score': float(obj.get('score', 0.0)),
                'total_return_pct': stage_metrics.get('total_return_pct', 0.0),
                'buy_hold_return_pct': stage_metrics.get('buy_hold_return_pct', 0.0),
                'buy_hold_max_drawdown_pct': stage_metrics.get('buy_hold_max_drawdown_pct', 0.0),
                'max_drawdown_pct': stage_metrics.get('max_drawdown_pct', 0.0),
                'cagr_pct': stage_metrics.get('cagr_pct', 0.0),
                'sharpe_ratio': stage_metrics.get('sharpe_ratio', stage_metrics.get('sharpe', 0.0)),
                'sortino_ratio': stage_metrics.get('sortino_ratio', stage_metrics.get('sortino', 0.0)),
                'total_trades': stage_metrics.get('total_trades', stage_metrics.get('num_trades', 0)),
                'final_balance': stage_metrics.get('final_balance', 0.0),
            }
        )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = df.sort_values('score', ascending=False).reset_index(drop=True)
    return df


def _render_html(
    output_path: Path,
    objectives: Sequence[str],
    objective_tables: Dict[str, pd.DataFrame],
    timeframe_tables: Dict[str, Tuple[str, pd.DataFrame]],
) -> None:
    sections: List[str] = []
    sections.append("<h1>Optimizer Run Summary</h1>")

    sections.append("<h2>Top Strategies by Objective</h2>")
    for objective in objectives:
        df = objective_tables.get(objective)
        if df is None or df.empty:
            sections.append(f"<h3>{objective}</h3><p>No data.</p>")
            continue
        sections.append(f"<h3>{objective}</h3>")
        sections.append(df.head(10).to_html(index=False, classes='table table-striped'))

    sections.append("<h2>Performance by Timeframe</h2>")
    for label, (title, df) in timeframe_tables.items():
        if df is None or df.empty:
            sections.append(f"<h3>{title}</h3><p>No data.</p>")
            continue
        sections.append(f"<h3>{title}</h3>")
        sections.append(df.head(10).to_html(index=False, classes='table table-striped'))

    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>Optimizer Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 2rem; }}
        h1, h2, h3 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 100%; margin-bottom: 1.5rem; }}
        th, td {{ border: 1px solid #ddd; padding: 0.6rem; text-align: right; }}
        th {{ background-color: #f2f2f2; }}
        td:first-child, th:first-child {{ text-align: left; }}
    </style>
</head>
<body>
{''.join(sections)}
</body>
</html>
"""
    output_path.write_text(html, encoding='utf-8')


def generate_reports(
    run_dir: Path,
    trials: Sequence[Dict],
    objectives: Sequence[str],
    timeframe_labels: Sequence[Tuple[str, str]],
) -> None:
    """Produce CSV + HTML reports summarising optimisation results."""
    run_dir.mkdir(parents=True, exist_ok=True)
    timeframe_tables: Dict[str, Tuple[str, pd.DataFrame]] = {}
    primary_df: Optional[pd.DataFrame] = None
    for label, title in timeframe_labels:
        df = _collect_timeframe_rows(trials, label)
        if not df.empty:
            df.to_csv(run_dir / f'metrics_{label}.csv', index=False)
            if primary_df is None:
                primary_df = df.copy()
        timeframe_tables[label] = (title, df)

    stage_label = timeframe_labels[0][0] if timeframe_labels else 'stage'
    objective_tables: Dict[str, pd.DataFrame] = {}
    for objective in objectives:
        df = _collect_objective_rows(trials, objective, stage_label)
        if not df.empty:
            df.to_csv(run_dir / f'objective_{objective}.csv', index=False)
        objective_tables[objective] = df

    # Overall summary sorted by total return on the primary stage metrics
    if primary_df is not None and not primary_df.empty:
        summary_path = run_dir / 'summary_total_return.csv'
        primary_df.to_csv(summary_path, index=False)

    _render_html(
        run_dir / 'report.html',
        objectives,
        objective_tables,
        timeframe_tables,
    )
