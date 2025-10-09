"""Reporting utilities for fast optimizer runner."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from optimization.runner import load_strategy_class
from scripts.evaluate_vost_strategies import load_dataset, load_swap_costs, run_strategy

_STAGE_TO_DAYS = {
    '30d': 30,
    '90d': 90,
    '1y': 365,
}


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
                'avg_buy_step_pct': metrics.get('avg_buy_step_pct', 0.0),
                'avg_sell_step_pct': metrics.get('avg_sell_step_pct', 0.0),
                'max_drawdown_duration_days': metrics.get('max_drawdown_duration_days', 0.0),
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
                'avg_buy_step_pct': stage_metrics.get('avg_buy_step_pct', 0.0),
                'avg_sell_step_pct': stage_metrics.get('avg_sell_step_pct', 0.0),
                'max_drawdown_duration_days': stage_metrics.get('max_drawdown_duration_days', 0.0),
            }
        )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = df.sort_values('score', ascending=False).reset_index(drop=True)
    return df


def _load_run_config(run_dir: Path) -> Optional[Dict[str, Any]]:
    config_path = run_dir / 'config.json'
    if not config_path.exists():
        return None
    try:
        return json.loads(config_path.read_text())
    except Exception:
        return None


def _slice_stage_window(df: pd.DataFrame, stage_label: str) -> pd.DataFrame:
    days = _STAGE_TO_DAYS.get(stage_label)
    if days is None:
        return df
    if 'timestamp' in df.columns:
        cutoff = df['timestamp'].max() - pd.Timedelta(days=days)
        return df[df['timestamp'] >= cutoff].reset_index(drop=True)
    return df


def _maybe_generate_equity_plot(
    run_dir: Path,
    stage_label: str,
    top_row: Optional[pd.Series],
    config_payload: Optional[Dict[str, Any]],
) -> Optional[Path]:
    if top_row is None or config_payload is None:
        return None
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None
    strategy_name = top_row.get('strategy')
    if not strategy_name:
        return None
    params_raw = top_row.get('parameters', '{}')
    try:
        parameters = json.loads(params_raw) if isinstance(params_raw, str) else (params_raw or {})
    except json.JSONDecodeError:
        parameters = {}
    data_path = Path(config_payload.get('data_path', ''))
    swap_cost_path = Path(config_payload.get('swap_cost_cache', ''))
    if not data_path.exists() or not swap_cost_path.exists():
        return None
    try:
        df = load_dataset(data_path)
        df = _slice_stage_window(df, stage_label)
        swap_costs = load_swap_costs(swap_cost_path)
        trade_size = float(config_payload.get('trade_size', 1000.0) or 1000.0)
        strategy_cls = load_strategy_class(strategy_name)
        if strategy_cls is None:
            return None
        strategy_instance = strategy_cls(parameters=parameters)
        stats = run_strategy(
            strategy_instance,
            df,
            swap_costs=swap_costs,
            trade_notional=trade_size,
            capture_trades=False,
        )
        equity_curve = np.asarray(stats.get('equity_curve', []), dtype=float)
        if equity_curve.size == 0:
            return None
        if 'timestamp' in df.columns:
            timestamps = pd.to_datetime(df['timestamp'])
        else:
            timestamps = pd.RangeIndex(start=0, stop=len(df))
        price_series = df['price'] if 'price' in df.columns else df['close']
        initial_price = float(price_series.iloc[0])
        buy_hold_equity = trade_size * (price_series / max(initial_price, 1e-9))
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(timestamps, equity_curve, label='Strategy Equity', linewidth=2)
        ax.plot(timestamps, buy_hold_equity, label='Buy & Hold', linestyle='--', linewidth=1.5)
        ax.set_title(f"Top Trial Equity Curve — {strategy_name}")
        ax.set_ylabel("Equity (DAI)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.autofmt_xdate()
        output_path = run_dir / 'top_trial_equity.png'
        fig.savefig(output_path, bbox_inches='tight')
        plt.close(fig)
        return output_path
    except Exception:
        return None


def _generate_trade_plots(
    run_dir: Path,
    stage_label: str,
    primary_df: Optional[pd.DataFrame],
    config_payload: Optional[Dict[str, Any]],
) -> None:
    if primary_df is None or primary_df.empty or config_payload is None:
        return
    data_path = Path(config_payload.get('data_path', ''))
    swap_cost_path = Path(config_payload.get('swap_cost_cache', ''))
    if not data_path.exists() or not swap_cost_path.exists():
        return
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    try:
        df = load_dataset(data_path)
        df = _slice_stage_window(df, stage_label)
        swap_costs = load_swap_costs(swap_cost_path)
    except Exception:
        return

    if 'timestamp' not in df.columns:
        return
    timestamps = pd.to_datetime(df['timestamp'])
    price_series = df['price'] if 'price' in df.columns else df['close']
    trade_size = float(config_payload.get('trade_size', 1000.0) or 1000.0)

    visual_dir = run_dir / 'visualizations'
    visual_dir.mkdir(parents=True, exist_ok=True)

    for strategy_name in primary_df['strategy'].unique():
        subset = primary_df[primary_df['strategy'] == strategy_name]
        if subset.empty:
            continue
        row = subset.iloc[0]
        params_raw = row.get('parameters', '{}')
        try:
            parameters = json.loads(params_raw) if isinstance(params_raw, str) else (params_raw or {})
        except json.JSONDecodeError:
            parameters = {}
        strategy_cls = load_strategy_class(strategy_name)
        if strategy_cls is None:
            continue
        try:
            strategy_instance = strategy_cls(parameters=parameters)
        except Exception:
            continue
        try:
            stats = run_strategy(
                strategy_instance,
                df.copy(),
                swap_costs=swap_costs,
                trade_notional=trade_size,
                capture_trades=True,
            )
        except Exception:
            continue
        trades = stats.get('trades') or []
        if not trades:
            continue
        trades_df = pd.DataFrame(trades)
        trades_df.to_csv(visual_dir / f'{strategy_name}_trades.csv', index=False)

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(timestamps, price_series, label='Price', color='black', linewidth=1.0)

        if 'entry_index' in trades_df:
            entry_idx = trades_df['entry_index'].dropna().astype(int)
            entry_idx = entry_idx[(entry_idx >= 0) & (entry_idx < len(timestamps))]
            if not entry_idx.empty:
                ax.scatter(
                    timestamps.iloc[entry_idx],
                    price_series.iloc[entry_idx],
                    marker='^',
                    color='green',
                    s=40,
                    label='Buy',
                )
        if 'exit_index' in trades_df:
            exit_idx = trades_df['exit_index'].dropna().astype(int)
            exit_idx = exit_idx[(exit_idx >= 0) & (exit_idx < len(timestamps))]
            if not exit_idx.empty:
                ax.scatter(
                    timestamps.iloc[exit_idx],
                    price_series.iloc[exit_idx],
                    marker='v',
                    color='red',
                    s=40,
                    label='Sell',
                )

        ax.set_title(f'{strategy_name} — Trades')
        ax.set_ylabel('Price (DAI)')
        ax.grid(alpha=0.3)
        ax.legend()
        fig.autofmt_xdate()
        fig.savefig(visual_dir / f'{strategy_name}_trades.png', bbox_inches='tight')
        plt.close(fig)


def _render_html(
    output_path: Path,
    objectives: Sequence[str],
    objective_tables: Dict[str, pd.DataFrame],
    timeframe_tables: Dict[str, Tuple[str, pd.DataFrame]],
    equity_plot_path: Optional[Path],
) -> None:
    sections: List[str] = []
    sections.append("<h1>Optimizer Run Summary</h1>")

    if equity_plot_path and equity_plot_path.exists():
        sections.append("<h2>Top Trial Equity Curve</h2>")
        sections.append(f"<img src=\"{equity_plot_path.name}\" alt=\"Top trial equity curve\" style=\"max-width:100%;height:auto;\"/>")

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
    config_payload = _load_run_config(run_dir)
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

    equity_plot_path: Optional[Path] = None
    if primary_df is not None and not primary_df.empty and timeframe_labels:
        stage_label = timeframe_labels[0][0]
        top_row = primary_df.iloc[0]
        equity_plot_path = _maybe_generate_equity_plot(run_dir, stage_label, top_row, config_payload)
        _generate_trade_plots(run_dir, stage_label, primary_df, config_payload)

    _render_html(
        run_dir / 'report.html',
        objectives,
        objective_tables,
        timeframe_tables,
        equity_plot_path,
    )
