#!/usr/bin/env python3
"""Fetch and persist swap-cost cache data to data/swap_cost_cache.json."""

from __future__ import annotations

import argparse
import sys
import time
from decimal import Decimal, ROUND_UP
from pathlib import Path
from typing import Union

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bot.config import Config
from utils.swap_cost_cache import SwapCostCache, SwapCostCacheError, _to_decimal  # type: ignore


def _normalize(amount: Union[Decimal, float, int], step: Decimal) -> int:
    """Round the amount up to the nearest rung."""
    amt = _to_decimal(amount)
    if amt <= 0:
        return int(step)
    multiplier = (amt / step).to_integral_value(rounding=ROUND_UP)
    return int(multiplier) * int(step)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('data/swap_cost_cache.json'),
        help='Destination JSON file for the swap-cost cache (default: data/swap_cost_cache.json).',
    )
    parser.add_argument(
        '--max-notional',
        type=float,
        default=100000.0,
        help='Maximum DAI notional to fetch (rounded up to nearest step).',
    )
    parser.add_argument(
        '--step',
        type=float,
        default=5000.0,
        help='Notional step size in DAI (default 5000).',
    )
    parser.add_argument(
        '--print-progress',
        action='store_true',
        help='Log progress for each rung as it is cached.',
    )
    args = parser.parse_args()

    output_path = args.output.resolve()
    run_dir = output_path.parent
    run_dir.mkdir(parents=True, exist_ok=True)

    step = Decimal(str(args.step))
    max_notional = Decimal(str(args.max_notional))
    normalized_target = Decimal(_normalize(max_notional, step))

    cache = SwapCostCache(
        run_dir=run_dir,
        config=Config(),
        producer=True,
        initial_target=normalized_target,
        step_notional=step,
    )

    start = time.time()
    try:
        cache.ensure_target(normalized_target)
        rung = cache.step_int
        while rung <= int(normalized_target):
            cache.wait_for_entry(rung)
            if args.print_progress:
                print(f"[swap-cost-fetch] cached rung {rung}")
            rung += cache.step_int
    except KeyboardInterrupt:
        print("Interrupted before completing cache fetch.")
        raise SystemExit(1)
    except SwapCostCacheError as exc:
        print(f"Failed to fetch swap cost cache: {exc}")
        raise SystemExit(2)
    finally:
        cache.stop()

    cache_file = run_dir / cache.cache_path.name
    if cache_file != output_path:
        output_path.write_bytes(cache_file.read_bytes())

    elapsed = time.time() - start
    print(
        f"Swap cost cache written to {output_path} "
        f"(step={step}, ceiling={normalized_target}, elapsed={elapsed:.1f}s)."
    )


if __name__ == '__main__':
    main()
