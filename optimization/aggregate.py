#!/usr/bin/env python3
"""
Aggregate Optimizer Runner Results

Usage:
  python scripts/aggregate_optimizer_results.py <stage_dir>

Example:
  python scripts/aggregate_optimizer_results.py \
    reports/optimizer_pipeline_20250913_003640_30d/stage1_30d

Outputs:
  - stage_aggregate.csv: one row per strategy/timeframe with CPS and key metrics
  - stage_best.csv: best timeframe per strategy
  - stage_report.md: quick leaderboard and summary counts

Behavior:
  - If a per-run JSON is missing 'cps', this script recomputes CPS on-the-fly
    using optimization.scoring_engine.CompositePerformanceScorer.
"""

import sys
import json
import csv
import gzip
from pathlib import Path
from typing import Any, Tuple
import math
import pandas as pd
from datetime import datetime
import concurrent.futures


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bot.config import Config


def _compute_cps_fallback(row: dict) -> float:
    """Compute CPS when missing using available metrics; returns 0.0 on error."""
    try:
        # Ensure repo root on path for import
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from optimization.scoring_engine import CompositePerformanceScorer, StrategyMetrics
        # Attempt to enrich from the original JSON file when available
        duration_days = 30
        total_trades = int(row.get('total_trades', 0) or 0)
        win_rate_pct = 0.0
        rev_catches = 0
        rev_opps = 0
        path = row.get('path') or row.get('file')
        try:
            if path:
                p = Path(path)
                if not p.exists():
                    # Try resolving relative to current stage dir
                    p = Path(row.get('stage_dir', '')).joinpath(path)
                if p.exists():
                    jd = _read_stage_json(p)
                    if isinstance(jd, dict):
                        res = jd.get('results', {}) or {}
                        duration_days = int(res.get('duration_days', duration_days) or duration_days)
                        total_trades = int(res.get('total_trades', total_trades) or total_trades)
                        win_rate_pct = float(res.get('win_rate_pct', 0.0) or 0.0)
                        rev_catches = int(res.get('reversal_catches', 0) or 0)
                        rev_opps = int(res.get('reversal_opportunities', 0) or 0)
        except Exception:
            pass

        # Convert percentages to decimals for scorer
        total_return = float(row.get('total_return_pct') or 0.0) / 100.0
        max_dd = float(row.get('max_drawdown_pct') or 0.0) / 100.0
        sharpe = float(row.get('sharpe_ratio') or 0.0)
        # Normalize trades to trades per month
        trades_per_month = (total_trades / max(1, duration_days)) * 30.0

        metrics = StrategyMetrics(
            strategy_name=row.get('strategy',''),
            timeframe=row.get('timeframe',''),
            total_return=total_return,
            max_drawdown=max_dd,
            sharpe_ratio=sharpe,
            sortino_ratio=0.0,
            total_trades=total_trades,
            winning_trades=0,
            losing_trades=0,
            win_rate=win_rate_pct / 100.0,
            avg_win=0.0,
            avg_loss=0.0,
            profit_factor=0.0,
            recovery_factor=0.0,
            trade_frequency=trades_per_month,
            reversal_catches=rev_catches,
            reversal_opportunities=rev_opps,
        )
        # We cannot reconstruct dynamic B&H here (no OHLCV in scope); keep a neutral-ish default
        scorer = CompositePerformanceScorer(buy_hold_return=0.2229)
        r = scorer.calculate_cps(metrics)
        return float(r['cps'] if isinstance(r, dict) else r)
    except Exception:
        return 0.0


def _compute_utility_fallback(row: dict) -> float:
    """Compute Utility score with default parameters when missing."""
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from optimization.scoring_engine import StrategyMetrics, DynamicUtilityScorer
    except Exception:
        return 0.0
    try:
        duration_days = 30
        total_trades = int(row.get('total_trades', 0) or 0)
        path = row.get('path') or row.get('file')
        try:
            if path:
                p = Path(path)
                if not p.exists():
                    p = Path(row.get('stage_dir', '')).joinpath(path)
                if p.exists():
                    jd = _read_stage_json(p)
                    if isinstance(jd, dict):
                        res = jd.get('results', {}) or {}
                        duration_days = int(res.get('duration_days', duration_days) or duration_days)
                        total_trades = int(res.get('total_trades', total_trades) or total_trades)
        except Exception:
            pass
        total_return = float(row.get('total_return_pct') or 0.0) / 100.0
        max_dd = float(row.get('max_drawdown_pct') or 0.0) / 100.0
        sharpe = float(row.get('sharpe_ratio') or 0.0)
        trades_per_month = (total_trades / max(1, duration_days)) * 30.0
        metrics = StrategyMetrics(
            strategy_name=row.get('strategy',''),
            timeframe=row.get('timeframe',''),
            total_return=total_return,
            max_drawdown=max_dd,
            sharpe_ratio=sharpe,
            sortino_ratio=0.0,
            total_trades=total_trades,
            winning_trades=0,
            losing_trades=0,
            win_rate=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            profit_factor=0.0,
            recovery_factor=0.0,
            trade_frequency=trades_per_month,
            reversal_catches=0,
            reversal_opportunities=0,
        )
        util = DynamicUtilityScorer(lambda_dd=2.0, dd_power=1.25, epsilon=0.01, reliability_k=5.0, alpha_oos=0.25)
        out = util.score(metrics)
        return float(out.get('score', 0.0))
    except Exception:
        return 0.0


def _read_stage_json(path: Path) -> Any:
    try:
        if path.suffix == '.gz' or path.name.endswith('.json.gz'):
            with gzip.open(path, 'rt', encoding='utf-8') as fh:
                return json.load(fh)
        with path.open('r', encoding='utf-8') as fh:
            return json.load(fh)
    except Exception:
        return None


def load_results(stage_dir: Path):
    """Load per-strategy/timeframe results with robust fallbacks.

    - Reads top-level 'results_agg' (OOS metrics) for returns/DD/sharpe/trades.
    - Pulls 'objective' and 'score' from the JSON when available.
    - If 'score' is missing, falls back to MAR-like score = return/DD (safe).
    - Ensures all fields are concrete (no NaN).
    """
    rows = []
    seen = set()
    for pattern in ("*.json", "*.json.gz"):
        for p in stage_dir.glob(pattern):
            if p.name.startswith('_'):
                continue
            if p.name.startswith('summary.json'):
                continue
            base = p.name.split('.json')[0]
            if base in seen and p.suffix == '.gz':
                continue
            seen.add(base)
            d = _read_stage_json(p)
            if not isinstance(d, dict):
                continue
            objective = d.get('objective', '') or ''
            res = d.get('results_agg')
            if not res:
                continue
            s = d.get('strategy')
            tf = d.get('timeframe')
            ret_source = res.get('oos_return_pct', res.get('total_return_pct'))
            ret = float(ret_source or 0.0)
            if math.isnan(ret):
                continue
            dd = float(res.get('max_drawdown_pct') or 0.0)
            sh = float(res.get('sharpe_ratio') or 0.0)
            nt = int(res.get('total_trades') or 0)
            score_val = d.get('score', None)
            try:
                score = float(score_val) if score_val is not None and score_val != '' else (ret / (dd if dd > 0.0 else 1e-9))
            except Exception:
                score = ret / (dd if dd > 0.0 else 1e-9)
            row = {
                'strategy': s,
                'timeframe': tf,
                'objective': objective or 'mar',
                'score': float(score),
                'score_cv': '',
                'utility': '',
                'weight_reliability': '',
                'pdr': '',
                'cps': '',
                'cps_cv': '',
                'profit_score': '',
                'preservation_score': '',
                'risk_adjusted_score': '',
                'activity_score': '',
                'trend_detection_score': '',
                'oos_return_pct': float(res.get('oos_return_pct') or ret),
                'total_return_pct': ret,
                'max_drawdown_pct': dd,
                'sharpe_ratio': sh,
                'total_trades': nt,
                'bh_return_pct': float(res.get('bh_return_pct') or 0.0),
                'file': p.name,
                'path': str(p),
            }
            rows.append(row)
    return rows


def write_csv(rows, out_path: Path):
    if not rows:
        return

    def _safe_float(val: Any, default: float = float('-inf')) -> float:
        try:
            return float(val)
        except (TypeError, ValueError):
            return default

    def _sort_key(row: dict) -> Tuple[float, float]:
        ret = _safe_float(row.get('oos_return_pct'), default=float('-inf'))
        score = _safe_float(row.get('score'), default=float('-inf'))
        return (ret, score)

    ordered = sorted(rows, key=_sort_key, reverse=True)

    keys = [
        'strategy','timeframe','objective','score','score_cv','utility','weight_reliability','pdr','cps','cps_cv',
        'profit_score','preservation_score','risk_adjusted_score','activity_score','trend_detection_score',
        'oos_return_pct','total_return_pct','max_drawdown_pct','sharpe_ratio','total_trades','bh_return_pct','file'
    ]
    with out_path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in ordered:
            w.writerow({k: r.get(k) for k in keys})


def best_per_strategy(rows):
    best = {}
    for r in rows:
        s = r['strategy']
        if s is None:
            continue
        # Prefer objective score if available
        try:
            current_score = float(r.get('score')) if r.get('score') not in (None, '') else float('-inf')
        except Exception:
            current_score = float('-inf')
        if s not in best:
            best[s] = r
        else:
            prev = best[s]
            try:
                prev_score = float(prev.get('score')) if prev.get('score') not in (None, '') else float('-inf')
            except Exception:
                prev_score = float('-inf')
            if current_score > prev_score:
                best[s] = r
    return best


def write_report(stage_dir: Path, best_map: dict):
    def _key(x):
        try:
            return float(x.get('score')) if x.get('score') not in (None, '') else float('-inf')
        except Exception:
            return float('-inf')
    top = sorted(best_map.values(), key=_key, reverse=True)
    lines = []
    lines.append(f"# Stage Summary — {stage_dir}")
    lines.append("")
    lines.append(f"Total strategies evaluated: {len(best_map)}")
    pos = sum(1 for v in best_map.values() if (v.get('score') not in (None, '')))
    lines.append(f"Strategies with objective score present: {pos}/{len(best_map)}")
    lines.append("")
    lines.append("## Top 20 (best timeframe by objective score)")
    for r in top[:20]:
        def _fmt(v):
            try:
                return f"{float(v):.1f}"
            except Exception:
                return ""
        obj = (r.get('objective') or '').lower()
        try:
            score_disp = f"{float(r.get('score', 0.0)):.2f}"
        except Exception:
            score_disp = ""
        # Build a short, objective-aware component string
        if obj == 'utility':
            comp_str = (
                f" U={_fmt(r.get('utility'))}"
                f"/w={_fmt(r.get('weight_reliability'))}"
                f"/PDR={_fmt(r.get('pdr'))}"
            )
        else:
            comp_str = (
                f" P={_fmt(r.get('profit_score'))}"
                f"/Pres={_fmt(r.get('preservation_score'))}"
                f"/R={_fmt(r.get('risk_adjusted_score'))}"
                f"/A={_fmt(r.get('activity_score'))}"
                f"/T={_fmt(r.get('trend_detection_score'))}"
            )
        lines.append(
            f"- {r['strategy']} [{r['timeframe']}] ({obj or 'objective'}) score={score_disp}{comp_str}, "
            f"ret={float(r.get('total_return_pct') or 0):.2f}%, dd={float(r.get('max_drawdown_pct') or 0):.2f}%, "
            f"trades={int(r.get('total_trades') or 0)}, sh={float(r.get('sharpe_ratio') or 0):.2f}, file {r.get('file','')}"
        )

    # Additional leaderboards
    lines.append("")
    lines.append("## Top 10 by Total Return (all columns)")
    top_ret = sorted(best_map.values(), key=lambda x: float(x.get('total_return_pct') or 0.0), reverse=True)[:10]
    for r in top_ret:
        try:
            cps_disp = f"{float(r.get('cps',0.0) or 0):.2f}"
        except Exception:
            cps_disp = ""
        lines.append(
            f"- {r['strategy']} [{r['timeframe']}]\n"
            f"    - cps {cps_disp}, total_return_pct {float(r.get('total_return_pct') or 0):.4f}, "
            f"max_drawdown_pct {float(r.get('max_drawdown_pct') or 0):.4f}, sharpe_ratio {float(r.get('sharpe_ratio') or 0):.4f}, "
            f"total_trades {int(r.get('total_trades') or 0)}, file {r.get('file','')}"
        )

    lines.append("")
    lines.append("## Top 10 by CPS (all columns)")
    top_cps = sorted(best_map.values(), key=lambda x: float(x.get('cps') or 0.0), reverse=True)[:10]
    for r in top_cps:
        lines.append(
            f"- {r['strategy']} [{r['timeframe']}]\n"
            f"    - cps {float(r.get('cps') or 0):.2f}, total_return_pct {float(r.get('total_return_pct') or 0):.4f}, "
            f"max_drawdown_pct {float(r.get('max_drawdown_pct') or 0):.4f}, sharpe_ratio {float(r.get('sharpe_ratio') or 0):.4f}, "
            f"total_trades {int(r.get('total_trades') or 0)}, file {r.get('file','')}"
        )
    (stage_dir / 'stage_report.md').write_text('\n'.join(lines))


def _resample_ohlc(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    if 'timestamp' in df.columns:
        d = df.copy()
        d['timestamp'] = pd.to_datetime(d['timestamp'])
        d = d.set_index('timestamp')
    else:
        d = df.copy()
        if not isinstance(d.index, pd.DatetimeIndex):
            d.index = pd.to_datetime(d.index)
    # ensure required columns
    if 'close' not in d.columns and 'price' in d.columns:
        d['close'] = d['price']
    if 'open' not in d.columns:
        d['open'] = d['close'].shift(1).fillna(d['close'])
    if 'high' not in d.columns:
        d['high'] = d['close']
    if 'low' not in d.columns:
        d['low'] = d['close']
    if 'volume' not in d.columns:
        d['volume'] = 0
    # Pandas resample accepts '5min','15min','30min','1h','4h','8h','16h','1d'
    rule = timeframe
    out = d.resample(rule).agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
    return out.reset_index()


def _process_dataset_row(r, stage_dir, price_df):
    """Process a single row to build dataset dict for HTML dashboard."""
    # Load per-run JSON for trades and OOS windows
    path = r.get('path') or ''
    if not path:
        return None
    try:
        d = json.loads(Path(path).read_text())
    except Exception:
        try:
            d = json.loads((stage_dir / Path(path).name).read_text())
        except Exception:
            return None
    folds = d.get('folds') or []
    if not folds:
        return None
    # gather OOS window bounds
    ts_list = []
    trades = []
    equity_pts = []
    for f in folds:
        try:
            os = pd.to_datetime(f.get('oos_start'))
            oe = pd.to_datetime(f.get('oos_end'))
            if os is not None and oe is not None:
                ts_list.append(os)
                ts_list.append(oe)
        except Exception:
            pass
        om = f.get('oos_metrics') or {}
        for t in (om.get('trades') or []):
            tt = t.get('timestamp')
            try:
                tt = pd.to_datetime(tt).isoformat()
            except Exception:
                tt = str(tt)
            trades.append({
                'timestamp': tt,
                'type': t.get('type'),
                'price': t.get('price'),
                'signal_strength': t.get('signal_strength'),
            })
        for ph in (om.get('portfolio_history') or []):
            tt = ph.get('timestamp')
            try:
                tt = pd.to_datetime(tt).isoformat()
            except Exception:
                tt = str(tt)
            tv = ph.get('total_value')
            if tv is not None:
                equity_pts.append([tt, float(tv)])
    if not ts_list:
        return None
    start = min(ts_list)
    end = max(ts_list)
    tf = r.get('timeframe') or d.get('timeframe')
    ohlc = []
    bh_line = []
    eq_line = []
    if price_df is not None and tf:
        try:
            # slice 5m source by window and resample to tf
            p = price_df.copy()
            if 'timestamp' in p.columns:
                p['timestamp'] = pd.to_datetime(p['timestamp'])
            else:
                p.index = pd.to_datetime(p.index)
                p = p.reset_index().rename(columns={'index':'timestamp'})
            p = p[(p['timestamp']>=start) & (p['timestamp']<=end)]
            p_tf = _resample_ohlc(p, tf)
            if not p_tf.empty:
                c0 = float(p_tf.iloc[0]['close'])
                for _, row in p_tf.iterrows():
                    ts = pd.to_datetime(row['timestamp']).isoformat()
                    ohlc.append([ts, float(row['open']), float(row['high']), float(row['low']), float(row['close'])])
                    bh_line.append([ts, ((float(row['close'])/c0)-1.0)*100.0 if c0>0 else 0.0])
        except Exception:
            pass
    # Build equity curve (percent)
    if equity_pts:
        equity_pts.sort(key=lambda x: x[0])
        e0 = equity_pts[0][1]
        if e0 and e0 != 0:
            for ts, tv in equity_pts:
                eq_line.append([ts, ((tv/e0)-1.0)*100.0])
        else:
            for ts, tv in equity_pts:
                eq_line.append([ts, 0.0])
    return {
        'id': f"{r.get('strategy')}_{tf}",
        'strategy': r.get('strategy'),
        'timeframe': tf,
        'objective': r.get('objective',''),
        'score': r.get('score'),
        'total_return_pct': r.get('total_return_pct'),
        'max_drawdown_pct': r.get('max_drawdown_pct'),
        'sharpe_ratio': r.get('sharpe_ratio'),
        'total_trades': r.get('total_trades'),
        'bh_return_pct': r.get('bh_return_pct', 0.0),
        'ohlc': ohlc,
        'bh_line': bh_line,
        'eq_line': eq_line,
        'trades': trades,
        'file': r.get('file')
    }


def generate_html_dashboard(stage_dir: Path, rows_sorted: list, top_n: int = 100):
    # Try to locate a 5m OHLCV source
    asset = Config.ASSET_SYMBOL.lower()
    quote = Config.QUOTE_SYMBOL.lower()
    ohlc_candidates = [
        Path(Config.DATA_DIR) / f"{asset}_ohlcv_{quote}_730day_5m.csv",
        Path(Config.DATA_DIR) / f"{asset}_ohlcv_{quote}_365day_5m.csv",
        Path(Config.DATA_DIR) / f"{asset}_ohlcv_{quote}_180day_5m.csv",
    ]
    for candidate in Config.ohlcv_candidates():
        path_obj = Path(candidate)
        if path_obj not in ohlc_candidates:
            ohlc_candidates.append(path_obj)
    ohlc_path = None
    for p in ohlc_candidates:
        if (stage_dir.parents[2] / p).exists():
            ohlc_path = (stage_dir.parents[2] / p)
            break
        if p.exists():
            ohlc_path = p
            break
    price_df = None
    if ohlc_path:
        try:
            price_df = pd.read_csv(ohlc_path)
        except Exception:
            price_df = None
    
    # Process top_n rows in parallel
    top_rows = rows_sorted[:top_n]
    datasets = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(_process_dataset_row, r, stage_dir, price_df) for r in top_rows]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result:
                datasets.append(result)
    
    # Sort datasets by score descending
    datasets.sort(key=lambda x: float(x.get('score') or 0), reverse=True)
    # Build HTML
    html_path = stage_dir / 'stage_dashboard.html'
    options = [{'id':d['id'], 'label': f"{d['strategy']} [{d['timeframe']}] - OBJ {d.get('objective','') or '?'} {d['score']:.3f}"} for d in datasets]
    import html as _html
    opts_html = ''.join([f"<option value=\"{_html.escape(o['id'])}\">{_html.escape(o['label'])}</option>" for o in options])
    max_n = len(datasets)
    html_tmpl = """
<!doctype html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Stage Dashboard</title>
  <script src=\"https://cdn.plot.ly/plotly-2.32.0.min.js\"></script>
  <style>
    body {{ background:#0b0b0f; color:#e0e0e0; font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, 'Helvetica Neue', Arial, 'Noto Sans', 'Liberation Sans', sans-serif; margin:0; }}
    header {{ display:flex; align-items:center; justify-content:space-between; padding:10px 14px; border-bottom:1px solid #222; position: sticky; top: 0; background:#0b0b0f; z-index: 100; }}
    .brand {{ font-weight: 600; letter-spacing: 0.3px; color:#9ecbff }}
    .panel {{ display:flex; gap:8px; align-items:center; padding:8px 12px; flex-wrap: wrap; }}
    select, input, button {{ background:#12121a; color:#ddd; border:1px solid #2b2b35; border-radius:6px; padding:6px 8px; }}
    button.primary {{ background: #1a74e2; border-color: #1a74e2; color:white; }}
    .grid {{ display:grid; grid-template-columns: 300px 1fr; gap:10px; padding:10px; }}
    .sidebar {{ border-right:1px solid #222; padding-right:10px; max-height: calc(100vh - 110px); overflow:auto; }}
    .list {{ font-size: 13px; }}
    .row {{ padding:8px; border-radius:6px; cursor:pointer; display:flex; justify-content:space-between; border:1px solid #1b1b22; margin-bottom:6px; }}
    .row:hover {{ background:#12121a; }}
    .row.active {{ outline: 1px solid #2e7dd8; }}
    .row .meta {{ color:#8aa1b4; }}
    .kpis {{ display:flex; gap:14px; padding:8px 12px; border-top:1px solid #111; border-bottom:1px solid #111; background:#0e0e14; }}
    .kpi {{ font-size:12px; color:#aebad4; }}
    .kpi b {{ color:#e6eefc; }}
    .section-title {{ font-size:12px; letter-spacing:.5px; color:#9fb7d3; margin:10px 0 4px; text-transform:uppercase; }}
    .charts {{ padding:0 10px 10px; }}
    .metrics {{ padding:8px 12px; font-size:13px; color:#cdd9f3; }}
  </style>
</head>
<body>
  <header>
    <div class=\"brand\">Stage Dashboard — __STAGE__</div>
    <div class=\"panel\">
      <label for=\"sel\">Strategy/TF</label>
      <select id=\"sel\"></select>
      <button id=\"prev\">◀</button>
      <button id=\"next\">▶</button>
    </div>
  </header>
  <div class=\"panel\" style=\"gap:12px;\">
    <label>Search <input id=\"search\" placeholder=\"strategy…\" style=\"width:160px\"/></label>
    <label>TF <select id=\"tf\"><option value=\"\">All</option><option>5min</option><option>15min</option><option>30min</option><option>1h</option><option>4h</option><option>8h</option><option>16h</option><option>1d</option></select></label>
    <label>Top N <input id=\"topn\" type=\"number\" value=\"__TOPN__\" step=\"1\" min=\"1\" max=\"__MAXN__\" style=\"width:80px\"/></label>
    <label>Score ≥ <input id=\"fScoreMin\" type=\"number\" step=\"0.01\" style=\"width:90px\"/></label>
    <label>Score ≤ <input id=\"fScoreMax\" type=\"number\" step=\"0.01\" style=\"width:90px\"/></label>
    <label>Trades ≥ <input id=\"fTradesMin\" type=\"number\" step=\"1\" style=\"width:90px\"/></label>
    <label>Trades ≤ <input id=\"fTradesMax\" type=\"number\" step=\"1\" style=\"width:90px\"/></label>
    <label>Return ≥ <input id=\"fRetMin\" type=\"number\" step=\"0.01\" style=\"width:90px\"/></label>
    <label>Return ≤ <input id=\"fRetMax\" type=\"number\" step=\"0.01\" style=\"width:90px\"/></label>
    <label>MaxDD ≤ <input id=\"fDDMax\" type=\"number\" step=\"0.01\" style=\"width:90px\"/></label>
    <button id=\"applyFilters\" class=\"primary\">Apply</button>
    <button id=\"resetFilters\">Reset</button>
  </div>
  <div class=\"kpis\" id=\"kpis\"></div>
  <div class=\"grid\">
    <div class=\"sidebar\">
      <div class=\"section-title\">Leaderboard</div>
      <div class=\"list\" id=\"list\"></div>
    </div>
    <div class=\"charts\">
      <div id=\"chart\" style=\"width: 100%; height: 540px;\"></div>
      <div id=\"eqchart\" style=\"width: 100%; height: 260px; margin-top: 8px;\"></div>
      <div class=\"metrics\" id=\"metrics\"></div>
    </div>
  </div>
  <script>
    const DATASETS = __DATASETS__;
    const max_n = __MAXN__;
    function fmt(n) { return (n===null||n===undefined)?'':(+n).toFixed(2); }
    function filtered() {
      const q = (document.getElementById('search').value||'').toLowerCase();
      const tf = document.getElementById('tf').value;
      const v = (x)=> x===null||x===undefined||x===''||Number.isNaN(x);
      const smin = parseFloat(document.getElementById('fScoreMin').value);
      const smax = parseFloat(document.getElementById('fScoreMax').value);
      const tmin = parseFloat(document.getElementById('fTradesMin').value);
      const tmax = parseFloat(document.getElementById('fTradesMax').value);
      const rmin = parseFloat(document.getElementById('fRetMin').value);
      const rmax = parseFloat(document.getElementById('fRetMax').value);
      const dmax = parseFloat(document.getElementById('fDDMax').value);
      return DATASETS.filter(d=> (
        (!q || d.strategy.toLowerCase().includes(q)) &&
        (!tf || d.timeframe===tf) &&
        (v(smin) || (+d.score)>=smin) &&
        (v(smax) || (+d.score)<=smax) &&
        (v(tmin) || (+d.total_trades)>=tmin) &&
        (v(tmax) || (+d.total_trades)<=tmax) &&
        (v(rmin) || (+d.total_return_pct)>=rmin) &&
        (v(rmax) || (+d.total_return_pct)<=rmax) &&
        (v(dmax) || (+d.max_drawdown_pct)<=dmax)
      )).sort((a,b)=> (+b.score)-(+a.score));
    }
    function updateKpis() {
      const topn = +document.getElementById('topn').value || max_n;
      const rows = filtered().slice(0, topn);
      const avg = (key) => rows.length? (rows.reduce((a,b)=> a+(+b[key]||0),0)/rows.length): 0;
      const html = `
        <div class=\"kpi\">Showing <b>${rows.length}</b> of <b>${DATASETS.length}</b></div>
        <div class=\"kpi\">Avg Return <b>${fmt(avg('total_return_pct'))}%</b></div>
        <div class=\"kpi\">Avg MaxDD <b>${fmt(avg('max_drawdown_pct'))}%</b></div>
        <div class=\"kpi\">Avg Sharpe <b>${fmt(avg('sharpe_ratio'))}</b></div>`;
      document.getElementById('kpis').innerHTML = html;
    }
    function refreshList() {
      const sel = document.getElementById('sel');
      const list = document.getElementById('list');
      const topn = +document.getElementById('topn').value || max_n;
      const rows = filtered().slice(0, topn);
      list.innerHTML = rows.map(d=> `
        <div class=\"row\" data-id=\"${d.id}\" onclick=\"render('${d.id}')\"> 
          <div><b>${d.strategy}</b> <span class=\"meta\">[${d.timeframe}]</span></div>
          <div class=\"meta\">score ${fmt(d.score)} · R ${fmt(d.total_return_pct)}% · DD ${fmt(d.max_drawdown_pct)}%</div>
        </div>`).join('');
      sel.innerHTML = rows.map(d=> `<option value=\"${d.id}\">${d.strategy} [${d.timeframe}]</option>`).join('');
      updateKpis();
    }
    function render(id) {
      const rows = filtered();
      const d = rows.find(x=> x.id===id) || rows[0] || DATASETS[0];
      if (!d) return;
      const times = d.ohlc.map(x=>x[0]);
      const open = d.ohlc.map(x=>x[1]);
      const high = d.ohlc.map(x=>x[2]);
      const low  = d.ohlc.map(x=>x[3]);
      const close= d.ohlc.map(x=>x[4]);
      const buys = d.trades.filter(t=>t.type==='buy');
      const sells= d.trades.filter(t=>t.type==='sell');
      const traceCandle = { x: times, open, high, low, close, type:'candlestick', name:'Price', increasing:{line:{color:'#26a69a'}}, decreasing:{line:{color:'#ef5350'}} };
      const traceBH = { x: d.bh_line.map(x=>x[0]), y: d.bh_line.map(x=>x[1]), type:'scatter', mode:'lines', name:'Buy&Hold %', line:{color:'#ffca28'} };
      const traceEqMain = { x: d.eq_line.map(x=>x[0]), y: d.eq_line.map(x=>x[1]), type:'scatter', mode:'lines', name:'Equity %', line:{color:'#42a5f5'} };
      const traceBuy = { x: buys.map(t=>t.timestamp), y: buys.map(t=>t.price), mode:'markers', name:'Buys', marker:{symbol:'triangle-up', color:'#00e676', size:9} };
      const traceSell= { x: sells.map(t=>t.timestamp), y: sells.map(t=>t.price), mode:'markers', name:'Sells', marker:{symbol:'triangle-down', color:'#ff1744', size:9} };
      const layout = { paper_bgcolor:'#0b0b0f', plot_bgcolor:'#0b0b0f', font:{color:'#e0e0e0'}, xaxis:{gridcolor:'#2a2a35', rangeslider:{visible:true}, rangeselector:{buttons:[{count:7,label:'7d',step:'day',stepmode:'backward'},{count:30,label:'30d',step:'day',stepmode:'backward'},{step:'all'}]} }, yaxis:{gridcolor:'#2a2a35'}, legend:{orientation:'h', y:-0.15}, margin:{l:40,r:15,t:20,b:20} };
      Plotly.newPlot('chart', [traceCandle, traceBH, traceEqMain, traceBuy, traceSell], layout, {responsive:true, displayModeBar:true});
      const traceEq = { x: d.eq_line.map(x=>x[0]), y: d.eq_line.map(x=>x[1]), type:'scatter', mode:'lines', name:'Equity %', line:{color:'#42a5f5'} };
      Plotly.newPlot('eqchart', [traceEq], {paper_bgcolor:'#0b0b0f', plot_bgcolor:'#0b0b0f', font:{color:'#e0e0e0'}, xaxis:{gridcolor:'#2a2a35'}, yaxis:{gridcolor:'#2a2a35'}, margin:{l:40,r:15,t:8,b:32} }, {responsive:true});
      document.getElementById('metrics').innerHTML = `<div><b>${d.strategy}</b> [${d.timeframe}] — objective: ${d.objective}</div><div>Score <b>${fmt(d.score)}</b> · Return <b>${fmt(d.total_return_pct)}%</b> · MaxDD <b>${fmt(d.max_drawdown_pct)}%</b> · Sharpe <b>${fmt(d.sharpe_ratio)}</b> · Trades <b>${d.total_trades}</b> · BH <b>${fmt(d.bh_return_pct)}%</b></div>`;
      const sel = document.getElementById('sel'); if (sel.value !== d.id) sel.value = d.id; Array.from(document.querySelectorAll('.row')).forEach(n=> n.dataset.id===d.id ? n.classList.add('active') : n.classList.remove('active'));
      updateKpis();
    }
    document.getElementById('applyFilters').addEventListener('click', ()=> { refreshList(); const rows = filtered(); if (rows.length) render(rows[0].id); });
    document.getElementById('resetFilters').addEventListener('click', ()=> { document.getElementById('search').value=''; document.getElementById('tf').value=''; for (const id of ['fScoreMin','fScoreMax','fTradesMin','fTradesMax','fRetMin','fRetMax','fDDMax']) document.getElementById(id).value=''; refreshList(); const rows = filtered(); if (rows.length) render(rows[0].id); });
    document.getElementById('topn').addEventListener('change', ()=> { refreshList(); });
    document.getElementById('tf').addEventListener('change', ()=> { refreshList(); });
    document.getElementById('search').addEventListener('input', ()=> { refreshList(); });
    const sel = document.getElementById('sel'); sel.addEventListener('change', (e)=> render(e.target.value));
    document.getElementById('prev').addEventListener('click', ()=> { const rows = filtered(); const i = rows.findIndex(x=> x.id===sel.value); const j = (i<=0? rows.length-1: i-1); if (rows[j]) render(rows[j].id); });
    document.getElementById('next').addEventListener('click', ()=> { const rows = filtered(); const i = rows.findIndex(x=> x.id===sel.value); const j = (i>=rows.length-1? 0: i+1); if (rows[j]) render(rows[j].id); });
    refreshList(); const rows0 = filtered(); if (rows0.length) render(rows0[0].id);
  </script>
</body>
</html>
"""
    html = (
        html_tmpl
        .replace('__DATASETS__', json.dumps(datasets))
        .replace('__MAXN__', str(max_n))
        .replace('__TOPN__', str(max(5, min(top_n, len(datasets)))))
        .replace('__STAGE__', stage_dir.name)
    )
    html_path.write_text(html)
    return html_path


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    stage_dir = Path(sys.argv[1]).resolve()
    if not stage_dir.exists():
        print(f"Not found: {stage_dir}")
        sys.exit(1)
    rows = load_results(stage_dir)
    # Sort full table by score descending for easier scanning (robust to NaN/blank)
    import math
    def _score_key(r: dict) -> float:
        try:
            val = float(r.get('score') if r.get('score') not in (None, '') else 'nan')
            if math.isnan(val):
                return -1e30  # push NaNs to the bottom
            return val
        except Exception:
            return -1e30
    rows_sorted = sorted(rows, key=_score_key, reverse=True)
    write_csv(rows_sorted, stage_dir / 'stage_aggregate.csv')
    best = best_per_strategy(rows_sorted)
    # write best per strategy
    best_rows = list(best.values())
    best_rows_sorted = sorted(best_rows, key=_score_key, reverse=True)
    write_csv(best_rows_sorted, stage_dir / 'stage_best.csv')
    write_report(stage_dir, best)
    # Generate interactive HTML dashboard (top 100 by score)
    try:
        html_path = generate_html_dashboard(stage_dir, rows_sorted, top_n=100)
        print(f"Wrote: {stage_dir/'stage_aggregate.csv'}, {stage_dir/'stage_best.csv'}, {stage_dir/'stage_report.md'}, {html_path}")
    except Exception as e:
        print(f"Wrote: {stage_dir/'stage_aggregate.csv'}, {stage_dir/'stage_best.csv'}, {stage_dir/'stage_report.md'}")
        print(f"Dashboard generation failed: {e}")


if __name__ == '__main__':
    main()
