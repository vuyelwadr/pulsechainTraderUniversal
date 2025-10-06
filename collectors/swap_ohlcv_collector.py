"""CLI wrapper to pull asset→quote OHLCV candles via HEXDataCollector."""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import logging
from pandas.tseries.frequencies import to_offset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from collectors.asset_data_collector import AssetDataCollector
from bot.config import Config

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"Collect real {Config.ASSET_SYMBOL.upper()}→{Config.QUOTE_SYMBOL.upper()} OHLCV candles from PulseChain"
    )
    parser.add_argument("--days", type=float, default=1.0, help="Number of days back from now to collect (ignored if --start provided).")
    parser.add_argument("--start", type=str, default=None, help="ISO timestamp start (UTC). Overrides --days.")
    parser.add_argument("--end", type=str, default=None, help="ISO timestamp end (UTC). Defaults to now.")
    parser.add_argument("--interval", type=str, default="5m", help="Candle interval (e.g. 5m, 15m, 1h, 4h, 1d).")
    volume_choices = [Config.ASSET_SYMBOL.upper(), Config.QUOTE_SYMBOL.upper()]
    parser.add_argument("--volume-asset", type=str, default=Config.QUOTE_SYMBOL.upper(), choices=volume_choices, help="Which asset volume to aggregate.")
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help=(
            "CSV output path. Defaults to data/{asset}_ohlcv_{quote}_<interval>_<start>_<end>.csv"
            .format(asset=Config.ASSET_SYMBOL.lower(), quote=Config.QUOTE_SYMBOL.lower())
        ),
    )
    parser.add_argument("--log-file", type=str, default=None, help="Optional log file (ignored for compatibility).")
    parser.add_argument("--workers", type=int, default=1, help="Compatibility flag (ignored; collector handles concurrency internally).")
    parser.add_argument("--chunk-size", type=int, default=12000, help="Compatibility flag (ignored).")
    parser.add_argument("--pin-rpc-per-worker", action="store_true", help="Compatibility flag (ignored).")
    parser.add_argument("--max-seconds", type=int, default=None, help="Compatibility flag (ignored).")
    return parser.parse_args()


def parse_date(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def interval_to_minutes(interval: str) -> int:
    normalized = interval.strip()
    if normalized.lower().endswith('m') and not normalized.lower().endswith('min'):
        normalized = normalized[:-1] + 'min'
    offset = to_offset(normalized)
    seconds = pd.to_timedelta(offset).total_seconds()
    if seconds == 0:
        raise ValueError(f"Unsupported interval '{interval}'")
    return int(round(seconds / 60))


def main() -> None:
    args = parse_args()

    start_dt = parse_date(args.start)
    end_dt = parse_date(args.end) or datetime.now(tz=timezone.utc)
    if start_dt is None:
        start_dt = end_dt - timedelta(days=args.days)

    interval_minutes = interval_to_minutes(args.interval)

    collector = AssetDataCollector()
    logging.info(
        "Collecting %s→%s candles from %s to %s at %s-minute resolution",
        Config.ASSET_SYMBOL.upper(),
        Config.QUOTE_SYMBOL.upper(),
        start_dt,
        end_dt,
        interval_minutes,
    )
    df = collector.collect_ohlcv_from_swaps(
        start_time=start_dt,
        end_time=end_dt,
        interval_minutes=interval_minutes,
        volume_asset=args.volume_asset,
    )

    if df is None or df.empty:
        print("No swap data found for the requested window.")
        return

    if args.out:
        out_path = Path(args.out)
    else:
        filename = (
            f"{Config.ASSET_SYMBOL.lower()}_ohlcv_{Config.QUOTE_SYMBOL.lower()}_{args.interval}_"
            f"{start_dt.strftime('%Y%m%d')}_{end_dt.strftime('%Y%m%d')}.csv"
        )
        out_path = Path(Config.DATA_DIR) / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} candles to {out_path}")


if __name__ == "__main__":
    main()
