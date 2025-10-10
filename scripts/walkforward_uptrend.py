import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import pandas as pd

ANALYSIS_DIR = Path("analysis")
OUTPUT_CSV = Path("reports/wf_uptrend_summary.csv")

TIMEFRAMES = ["5min", "15min", "30min", "1h", "2h", "4h", "8h", "16h", "1d", "2d"]
START_CAPITAL = 1000.0
COST_RATE = 0.015

# Trailing stop percentages (as fraction of peak) per timeframe for the fixed-stop mode
TRAIL_STOP = {
    "5min": 0.05,
    "15min": 0.06,
    "30min": 0.07,
    "1h": 0.08,
    "2h": 0.10,
    "4h": 0.12,
    "8h": 0.14,
    "16h": 0.16,
    "1d": 0.18,
    "2d": 0.20,
}


@dataclass
class SimulationConfig:
    """Configuration toggles for a single walk-forward run."""

    name: str = "default"
    threshold_mode: str = "median"  # median, quantile, quantile_grid
    threshold_quantile: float = 0.5
    threshold_grid: Sequence[float] = field(default_factory=lambda: (0.2, 0.35, 0.5, 0.65, 0.8))
    trailing_mode: str = "fixed"  # fixed, atr
    atr_mult: float = 3.0
    atr_floor: float = 0.02
    cooldown_bars: int = 0
    require_strength_positive: bool = False
    analysis_dir: Path = ANALYSIS_DIR
    output_csv: Path = OUTPUT_CSV
    timeframes: Sequence[str] = field(default_factory=lambda: tuple(TIMEFRAMES))
    cost_rate: float = COST_RATE
    per_timeframe: dict = field(default_factory=dict)


@dataclass
class FoldResult:
    timeframe: str
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    final_balance: float
    total_return_pct: float
    buy_hold_final_balance: float
    buy_hold_total_return_pct: float
    trades: int
    strength_threshold: float
    config_name: str
    threshold_mode: str
    trailing_mode: str
    cooldown_bars: int
    fold_index: int


@dataclass
class TradeEvent:
    config_name: str
    timeframe: str
    fold_index: int
    phase: str
    timestamp: pd.Timestamp
    action: str
    price: float
    strength: float
    threshold: float
    reason: str
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def load_states(label: str, analysis_dir: Path) -> pd.DataFrame:
    path = analysis_dir / f"trend_states_{label}.csv"
    df = pd.read_csv(path, parse_dates=["timestamp"])
    price_col = "close" if "close" in df.columns else "price"
    columns = ["timestamp", price_col, "trend_state", "trend_strength_score", "atr_percent"]
    available = [c for c in columns if c in df.columns]
    df = df[available].rename(columns={price_col: "close", "trend_state": "state"})
    df["timestamp"] = df["timestamp"].dt.tz_localize(None)
    if "trend_strength_score" in df.columns:
        df["trend_strength_score"] = pd.to_numeric(df["trend_strength_score"], errors="coerce")
    if "atr_percent" in df.columns:
        df["atr_percent"] = pd.to_numeric(df["atr_percent"], errors="coerce")
    return df.sort_values("timestamp").reset_index(drop=True)


def generate_folds(states: pd.DataFrame) -> List[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    folds = []
    start_ts = states["timestamp"].min()
    end_ts = states["timestamp"].max()
    step = pd.DateOffset(days=30)
    train_length = pd.DateOffset(days=180)
    test_length = pd.DateOffset(days=30)
    current_train_start = start_ts
    while current_train_start + train_length + test_length <= end_ts:
        train_start = current_train_start
        train_end = train_start + train_length
        test_start = train_end
        test_end = test_start + test_length
        folds.append((train_start, train_end, test_start, test_end))
        current_train_start += step
    return folds


def compute_strength_threshold(
    states: pd.DataFrame,
    train_start,
    train_end,
    label: str,
    config: SimulationConfig,
) -> float:
    mask = (
        (states["timestamp"] >= train_start)
        & (states["timestamp"] < train_end)
        & (states["state"] == "UPTREND")
    )
    train_up = states.loc[mask, "trend_strength_score"].dropna()
    if train_up.empty:
        return float("-inf")

    mode = str(_resolve_param(config, label, "threshold_mode")).lower()
    if mode == "median":
        return float(train_up.median())

    if mode == "quantile":
        quantile = float(_resolve_param(config, label, "threshold_quantile"))
        return float(train_up.quantile(quantile))

    if mode == "quantile_grid":
        # Evaluate each quantile on the training window and pick the best performer.
        best_threshold = float(train_up.median())
        best_balance = float("-inf")
        grid = _resolve_param(config, label, "threshold_grid")
        for q in grid:
            threshold = float(train_up.quantile(q))
            if pd.isna(threshold):
                continue
            balance, _, _ = simulate_period(
                states,
                train_start,
                train_end,
                label,
                threshold,
                config,
            )
            if balance > best_balance:
                best_balance = balance
                best_threshold = threshold
        return best_threshold

    raise ValueError(f"Unsupported threshold_mode={config.threshold_mode}")


def _compute_trailing_stop(row, label: str, config: SimulationConfig) -> float:
    mode = _resolve_param(config, label, "trailing_mode")
    if mode == "atr":
        atr_mult = float(_resolve_param(config, label, "atr_mult"))
        atr_floor = float(_resolve_param(config, label, "atr_floor"))
        atr_pct = row.atr_percent if hasattr(row, "atr_percent") else None
        if pd.isna(atr_pct):
            atr_pct = 0.0
        trailing = (atr_pct / 100.0) * atr_mult
        return max(trailing, atr_floor)
    # fallback to fixed thresholds
    return TRAIL_STOP.get(label, TRAIL_STOP["1h"])  # default to 1h if missing


def simulate_period(
    states: pd.DataFrame,
    start,
    end,
    label: str,
    threshold: float,
    config: SimulationConfig,
    trade_log: Optional[List[TradeEvent]] = None,
    *,
    phase: str = "test",
    fold_index: Optional[int] = None,
    config_name: Optional[str] = None,
    test_start: Optional[pd.Timestamp] = None,
    test_end: Optional[pd.Timestamp] = None,
) -> tuple[float, float, int]:
    mask = (states["timestamp"] >= start) & (states["timestamp"] < end)
    window = states.loc[mask].reset_index(drop=True)
    if window.empty:
        return START_CAPITAL, 0.0, 0

    capital = START_CAPITAL
    shares = 0.0
    trades = 0
    in_position = False
    peak_price = None
    cooldown = 0
    cooldown_setting = int(_resolve_param(config, label, "cooldown_bars"))
    require_positive = bool(_resolve_param(config, label, "require_strength_positive"))
    threshold = float(threshold)

    for row in window.itertuples(index=False):
        price = float(row.close)
        state = getattr(row, "state", "RANGE")
        strength = float(getattr(row, "trend_strength_score", 0.0))

        if cooldown > 0:
            cooldown -= 1

        can_enter = not in_position and cooldown == 0
        meets_state = state == "UPTREND"
        meets_strength = strength >= threshold
        if require_positive:
            meets_strength &= strength > 0

        if can_enter and meets_state and meets_strength:
            capital_after_fee = capital * (1 - config.cost_rate)
            if capital_after_fee <= 0:
                continue
            shares = capital_after_fee / price
            capital = 0.0
            in_position = True
            peak_price = price
            trades += 1
            if trade_log is not None:
                event_ts = pd.Timestamp(getattr(row, "timestamp"))
                trade_log.append(
                    TradeEvent(
                        config_name=config_name or config.name,
                        timeframe=label,
                        fold_index=fold_index if fold_index is not None else -1,
                        phase=phase,
                        timestamp=event_ts,
                        action="BUY",
                        price=price,
                        strength=strength,
                        threshold=threshold,
                        reason="entry",
                        test_start=test_start,
                        test_end=test_end,
                    )
                )
            continue

        if not in_position:
            continue

        # Update trailing stop context while in position
        peak_price = max(peak_price, price)
        trailing = _compute_trailing_stop(row, label, config)
        drawdown = (peak_price - price) / peak_price if peak_price else 0.0

        exit_signal = False
        exit_reason = ""
        if state != "UPTREND" or strength < threshold:
            exit_signal = True
            exit_reason = "state_flip" if state != "UPTREND" else "weak_strength"
        elif drawdown >= trailing:
            exit_signal = True
            exit_reason = "trailing_stop"

        if exit_signal:
            proceeds = shares * price
            capital += proceeds * (1 - config.cost_rate)
            shares = 0.0
            in_position = False
            peak_price = None
            cooldown = cooldown_setting
            trades += 1
            if trade_log is not None:
                event_ts = pd.Timestamp(getattr(row, "timestamp"))
                trade_log.append(
                    TradeEvent(
                        config_name=config_name or config.name,
                        timeframe=label,
                        fold_index=fold_index if fold_index is not None else -1,
                        phase=phase,
                        timestamp=event_ts,
                        action="SELL",
                        price=price,
                        strength=strength,
                        threshold=threshold,
                        reason=exit_reason or "exit",
                        test_start=test_start,
                        test_end=test_end,
                    )
                )

    if in_position:
        price = float(window.iloc[-1].close)
        proceeds = shares * price
        capital += proceeds * (1 - config.cost_rate)
        trades += 1
        if trade_log is not None:
            last_ts = pd.Timestamp(window.iloc[-1].timestamp)
            strength_value = (
                float(window.iloc[-1].trend_strength_score)
                if "trend_strength_score" in window.columns
                else float("nan")
            )
            trade_log.append(
                TradeEvent(
                    config_name=config_name or config.name,
                    timeframe=label,
                    fold_index=fold_index if fold_index is not None else -1,
                    phase=phase,
                    timestamp=last_ts,
                    action="SELL",
                    price=price,
                    strength=strength_value,
                    threshold=threshold,
                    reason="end_of_window",
                    test_start=test_start,
                    test_end=test_end,
                )
            )

    return capital, (capital / START_CAPITAL - 1.0) * 100.0, trades


def simulate_fold(
    states: pd.DataFrame,
    test_start,
    test_end,
    label: str,
    threshold: float,
    config: SimulationConfig,
    trade_log: Optional[List[TradeEvent]] = None,
    *,
    fold_index: Optional[int] = None,
) -> tuple[float, float, int]:
    return simulate_period(
        states,
        test_start,
        test_end,
        label,
        threshold,
        config,
        trade_log,
        phase="test",
        fold_index=fold_index,
        config_name=config.name,
        test_start=test_start,
        test_end=test_end,
    )


def buy_hold_metrics(states: pd.DataFrame, test_start, test_end) -> tuple[float, float]:
    mask = (states["timestamp"] >= test_start) & (states["timestamp"] < test_end)
    test_df = states.loc[mask]
    if test_df.empty:
        return START_CAPITAL, 0.0
    start_price = float(test_df.iloc[0]["close"])
    end_price = float(test_df.iloc[-1]["close"])
    if start_price <= 0:
        return START_CAPITAL, 0.0
    final_balance = START_CAPITAL * (end_price / start_price)
    return final_balance, (final_balance / START_CAPITAL - 1.0) * 100.0


def run_walkforward(
    label: str,
    config: SimulationConfig,
    trade_log: Optional[List[TradeEvent]] = None,
) -> List[FoldResult]:
    states = load_states(label, config.analysis_dir)
    folds = generate_folds(states)
    results = []
    for idx, (train_start, train_end, test_start, test_end) in enumerate(folds):
        threshold = compute_strength_threshold(states, train_start, train_end, label, config)
        final_balance, total_return_pct, trades = simulate_fold(
            states,
            test_start,
            test_end,
            label,
            threshold,
            config,
            trade_log,
            fold_index=idx,
        )
        bh_balance, bh_return = buy_hold_metrics(states, test_start, test_end)
        results.append(
            FoldResult(
                timeframe=label,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                final_balance=final_balance,
                total_return_pct=total_return_pct,
                buy_hold_final_balance=bh_balance,
                buy_hold_total_return_pct=bh_return,
                trades=trades,
                strength_threshold=threshold,
                config_name=config.name,
                threshold_mode=config.threshold_mode,
                trailing_mode=config.trailing_mode,
                cooldown_bars=config.cooldown_bars,
                fold_index=idx,
            )
        )
    return results


def _ensure_sequence(value) -> Sequence[str]:
    if isinstance(value, str):
        return tuple(part.strip() for part in value.split(",") if part.strip())
    return tuple(value)


def _load_config_dicts(path: Path) -> List[dict]:
    with path.open() as fp:
        data = json.load(fp)
    if isinstance(data, dict):
        return [data]
    if not isinstance(data, list):
        raise ValueError("Config JSON must be a dict or list of dicts")
    return data


def _config_from_dict(base: SimulationConfig, payload: dict) -> SimulationConfig:
    params = {**base.__dict__, **payload}
    params["analysis_dir"] = Path(params.get("analysis_dir", base.analysis_dir))
    params["output_csv"] = Path(params.get("output_csv", base.output_csv))
    params["timeframes"] = _ensure_sequence(params.get("timeframes", base.timeframes))
    grid_value = params.get("threshold_grid", base.threshold_grid)
    if isinstance(grid_value, str):
        params["threshold_grid"] = tuple(float(q) for q in grid_value.split(",") if q)
    else:
        params["threshold_grid"] = tuple(grid_value)
    per_tf = params.get("per_timeframe", {})
    if isinstance(per_tf, dict):
        canonical = {}
        for tf, overrides in per_tf.items():
            mapped = dict(overrides)
            if "threshold_grid" in mapped:
                value = mapped["threshold_grid"]
                if isinstance(value, str):
                    mapped["threshold_grid"] = tuple(float(q) for q in value.split(",") if q)
                else:
                    mapped["threshold_grid"] = tuple(float(q) for q in value)
            if "threshold_quantile" in mapped:
                mapped["threshold_quantile"] = float(mapped["threshold_quantile"])
            if "cooldown_bars" in mapped:
                mapped["cooldown_bars"] = int(mapped["cooldown_bars"])
            canonical[tf] = mapped
        params["per_timeframe"] = canonical
    return SimulationConfig(**params)


def _resolve_param(config: SimulationConfig, label: str, attribute: str):
    overrides = config.per_timeframe.get(label, {})
    value = overrides.get(attribute, getattr(config, attribute))
    if attribute == "threshold_grid":
        if isinstance(value, str):
            return tuple(float(q) for q in value.split(",") if q)
        return tuple(float(q) for q in value)
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Walk-forward test for labelled uptrend segments")
    parser.add_argument("--analysis-dir", default=str(ANALYSIS_DIR), help="Directory containing trend_state CSVs")
    parser.add_argument("--output", default=str(OUTPUT_CSV), help="Base output CSV path")
    parser.add_argument("--timeframes", default=",".join(TIMEFRAMES), help="Comma-separated timeframes to evaluate")
    parser.add_argument(
        "--threshold-mode",
        choices=["median", "quantile", "quantile_grid"],
        default="median",
        help="How to derive the trend strength threshold per fold",
    )
    parser.add_argument("--threshold-quantile", type=float, default=0.5, help="Quantile when threshold-mode=quantile")
    parser.add_argument(
        "--threshold-grid",
        default="0.2,0.35,0.5,0.65,0.8",
        help="Comma-separated quantiles when threshold-mode=quantile_grid",
    )
    parser.add_argument(
        "--trailing-mode",
        choices=["fixed", "atr"],
        default="fixed",
        help="Trailing-stop regime",
    )
    parser.add_argument("--atr-mult", type=float, default=3.0, help="ATR multiple when trailing-mode=atr")
    parser.add_argument("--atr-floor", type=float, default=0.02, help="Floor drawdown fraction when trailing-mode=atr")
    parser.add_argument("--cooldown", type=int, default=0, help="Bars to wait after exit before re-entry")
    parser.add_argument(
        "--require-strength-positive",
        action="store_true",
        help="Disallow entries when strength is negative even if above threshold",
    )
    parser.add_argument("--config-json", help="Optional JSON (dict or list) with additional config overrides")
    parser.add_argument("--tag", default="default", help="Label for this run; appended to output filename")
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Optional cap on parallel workers when multiple configs are provided",
    )
    return parser.parse_args()


def build_configs(args: argparse.Namespace) -> List[SimulationConfig]:
    base = SimulationConfig(
        name=args.tag,
        threshold_mode=args.threshold_mode,
        threshold_quantile=args.threshold_quantile,
        threshold_grid=tuple(float(q) for q in args.threshold_grid.split(",") if q),
        trailing_mode=args.trailing_mode,
        atr_mult=args.atr_mult,
        atr_floor=args.atr_floor,
        cooldown_bars=args.cooldown,
        require_strength_positive=args.require_strength_positive,
        analysis_dir=Path(args.analysis_dir),
        output_csv=Path(args.output),
        timeframes=_ensure_sequence(args.timeframes),
    )

    if not args.config_json:
        return [base]

    config_path = Path(args.config_json)
    config_dicts = _load_config_dicts(config_path)
    return [_config_from_dict(base, cfg) for cfg in config_dicts]


def _evaluate_config(config: SimulationConfig) -> Tuple[SimulationConfig, pd.DataFrame, pd.DataFrame]:
    all_results: List[FoldResult] = []
    trade_events: List[TradeEvent] = []
    for label in config.timeframes:
        all_results.extend(run_walkforward(label, config, trade_events))

    if not all_results:
        return config, pd.DataFrame(), pd.DataFrame()

    df = pd.DataFrame([r.__dict__ for r in all_results])
    trades_df = pd.DataFrame([event.__dict__ for event in trade_events]) if trade_events else pd.DataFrame()
    return config, df, trades_df


def main():
    args = parse_args()
    configs = build_configs(args)
    combined_frames = []
    combined_trades = []

    if len(configs) == 1:
        evaluated = [_evaluate_config(configs[0])]
    else:
        cpu_total = os.cpu_count() or 1
        target = max(1, int(cpu_total * 0.9))
        max_workers = args.max_workers or min(target, len(configs))
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            evaluated = list(executor.map(_evaluate_config, configs))

    for config, df, trades in evaluated:
        if df.empty:
            print(f"No folds evaluated for config {config.name}.")
            continue

        suffix = f"_{config.name}" if config.name and config.name != "default" else ""
        output_path = config.output_csv
        base_output = Path(args.output)
        if suffix and config.output_csv == base_output:
            output_path = base_output.with_name(f"{base_output.stem}{suffix}{base_output.suffix}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

        summary = df.groupby("timeframe")["total_return_pct"].mean().sort_values(ascending=False)
        print(f"=== {config.name} ===")
        print(summary)
        combined_frames.append(df)

        if trades is not None and not trades.empty:
            config_slug = (config.name or "default").replace(" ", "_")
            trades_output = output_path.parent / f"wf_uptrend_trades_{config_slug}.csv"
            trades.to_csv(trades_output, index=False)
            combined_trades.append(trades)

    if len(combined_frames) > 1:
        merged = pd.concat(combined_frames, ignore_index=True)
        merged_output = Path(args.output).with_name(
            f"{Path(args.output).stem}_all{Path(args.output).suffix}"
        )
        merged_output.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(merged_output, index=False)

    if combined_trades:
        merged_trades = pd.concat(combined_trades, ignore_index=True)
        trades_output = Path(args.output).with_name(
            f"{Path(args.output).stem}_trades_all{Path(args.output).suffix}"
        )
        trades_output.parent.mkdir(parents=True, exist_ok=True)
        merged_trades.to_csv(trades_output, index=False)


if __name__ == "__main__":
    main()
