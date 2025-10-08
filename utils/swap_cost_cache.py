"""
Piteas swap cost cache with background fetching and persistence.

This module maintains a per-run cache of swap quotes for swapping the configured
asset against the quote currency on PulseChain using the public Piteas SDK API.
Quotes are fetched in 5k USD increments (both buy and sell directions) up to a
required ceiling. Fetching happens on a dedicated background thread so it does
not block the runner/backtester, while consumers block only when a requested
notional has not yet been cached.
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, ROUND_HALF_EVEN, ROUND_UP, getcontext
from pathlib import Path
from typing import Any, Dict, Optional, Union
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from bot.config import Config

# High precision is required when working with 18-decimal assets.
getcontext().prec = 50


API_ENDPOINT = "https://sdk.piteas.io/quote"
DEFAULT_ALLOWED_SLIPPAGE_BPS = 50  # 0.5%
STEP_NOTIONAL = Decimal("5000")
MAX_RINGS_PER_CYCLE = 4  # 4 rings * 2 quotes each = 8 requests/min (rate-safe)
REQUEST_LOG_NAME = "swap_cost_requests.log"
CACHE_FILENAME = "swap_cost_cache.json"


class SwapCostCacheError(RuntimeError):
    """Base exception for swap cost cache failures."""


class SwapCostCacheTimeout(SwapCostCacheError):
    """Raised when waiting for a rung exceeds the deadline."""


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    """Atomically persist JSON to avoid corruption during concurrent reads."""
    tmp_fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w") as handle:
            json.dump(payload, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        Path(tmp_name).replace(path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _decimal_from_raw(value: int, decimals: int) -> Decimal:
    """Convert integer amount to Decimal respecting token decimals."""
    if value == 0:
        return Decimal("0")
    scale = Decimal(10) ** decimals
    return Decimal(value) / scale


def _to_decimal(value: Union[str, float, int, Decimal]) -> Decimal:
    """Robust decimal conversion from assorted primitive types."""
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


def _quantize(value: Decimal, decimals: int) -> Decimal:
    """Quantize a Decimal to the token precision."""
    if decimals <= 0:
        return value.to_integral_value(rounding=ROUND_HALF_EVEN)
    quantum = Decimal("1").scaleb(-decimals)
    return value.quantize(quantum, rounding=ROUND_HALF_EVEN)


@dataclass
class SwapCostEntry:
    """Cached information for a single notional rung."""

    notional_dai: Decimal
    token_amount: Decimal
    buy_amount_in_wei: int
    buy_dest_amount_raw: int
    buy_gas_estimate: int
    buy_gas_usd: float
    sell_amount_in_raw: int
    sell_dest_amount_raw: int
    sell_gas_estimate: int
    sell_gas_usd: float
    token_per_dai: Decimal
    dai_per_token: Decimal
    roundtrip_loss_dai: Decimal
    loss_rate: Decimal
    generated_at: datetime

    def to_json(self) -> Dict[str, Any]:
        return {
            "notional_dai": str(self.notional_dai),
            "token_amount": str(self.token_amount),
            "buy": {
                "amount_in_wei": str(self.buy_amount_in_wei),
                "dest_amount_raw": str(self.buy_dest_amount_raw),
                "gas_use_estimate": self.buy_gas_estimate,
                "gas_use_estimate_usd": self.buy_gas_usd,
            },
            "sell": {
                "amount_in_raw": str(self.sell_amount_in_raw),
                "dest_amount_raw": str(self.sell_dest_amount_raw),
                "gas_use_estimate": self.sell_gas_estimate,
                "gas_use_estimate_usd": self.sell_gas_usd,
            },
            "derived": {
                "token_per_dai": str(self.token_per_dai),
                "dai_per_token": str(self.dai_per_token),
                "roundtrip_loss_dai": str(self.roundtrip_loss_dai),
                "loss_rate": str(self.loss_rate),
            },
            "generated_at": self.generated_at.isoformat(),
        }

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "SwapCostEntry":
        derived = data.get("derived", {})
        buy = data.get("buy", {})
        sell = data.get("sell", {})
        return cls(
            notional_dai=_to_decimal(data["notional_dai"]),
            token_amount=_to_decimal(data["token_amount"]),
            buy_amount_in_wei=int(buy.get("amount_in_wei", "0")),
            buy_dest_amount_raw=int(buy.get("dest_amount_raw", "0")),
            buy_gas_estimate=int(buy.get("gas_use_estimate", 0)),
            buy_gas_usd=float(buy.get("gas_use_estimate_usd", 0.0)),
            sell_amount_in_raw=int(sell.get("amount_in_raw", "0")),
            sell_dest_amount_raw=int(sell.get("dest_amount_raw", "0")),
            sell_gas_estimate=int(sell.get("gas_use_estimate", 0)),
            sell_gas_usd=float(sell.get("gas_use_estimate_usd", 0.0)),
            token_per_dai=_to_decimal(derived.get("token_per_dai", "0")),
            dai_per_token=_to_decimal(derived.get("dai_per_token", "0")),
            roundtrip_loss_dai=_to_decimal(derived.get("roundtrip_loss_dai", "0")),
            loss_rate=_to_decimal(derived.get("loss_rate", "0")),
            generated_at=datetime.fromisoformat(data.get("generated_at")),
        )


class SwapCostCache:
    """Manage swap quote caching, persistence, and background fetching."""

    def __init__(
        self,
        run_dir: Union[str, Path],
        config: Optional[Config] = None,
        *,
        producer: bool = False,
        initial_target: Decimal = Decimal("100000"),
        step_notional: Optional[Union[Decimal, float, int]] = None,
    ) -> None:
        self.run_dir = Path(run_dir).resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.cache_path = self.run_dir / CACHE_FILENAME
        self.request_log_path = self.run_dir / REQUEST_LOG_NAME
        self.config = config or Config()
        self.producer = producer
        step_value = _to_decimal(step_notional) if step_notional is not None else STEP_NOTIONAL
        if step_value <= 0:
            raise SwapCostCacheError(f"step_notional must be positive; got {step_value}")
        self.step = step_value
        self.step_int = int(self.step.to_integral_value(rounding=ROUND_UP))
        self.initial_target = initial_target
        self._lock = threading.RLock()
        self._wake_event = threading.Event()
        self._stop_event = threading.Event()
        self._fetch_thread: Optional[threading.Thread] = None
        self._entries: Dict[int, SwapCostEntry] = {}
        self._last_loaded_mtime: float = 0.0
        self._required_ceiling: Decimal = initial_target
        self._queued_targets: set[int] = set()
        self._load_cache()
        if self._entries:
            current_max = max(self._entries.keys()) * Decimal("1")
            if current_max > self._required_ceiling:
                self._required_ceiling = current_max
        self._ensure_request_log()
        if self.producer:
            self._fetch_thread = threading.Thread(
                target=self._fetch_loop, name="PiteasSwapFetcher", daemon=True
            )
            self._fetch_thread.start()

    # ------------------------------------------------------------------ helpers
    def _ensure_request_log(self) -> None:
        """Create the request log file if missing so appends never fail."""
        if not self.request_log_path.exists():
            self.request_log_path.write_text("", encoding="utf-8")

    def _load_cache(self) -> None:
        """Load cached entries from disk into memory."""
        if not self.cache_path.exists():
            return
        try:
            payload = json.loads(self.cache_path.read_text())
        except Exception:
            return
        entries = payload.get("entries", {})
        loaded: Dict[int, SwapCostEntry] = {}
        for key, entry in entries.items():
            try:
                rung = int(key)
                loaded[rung] = SwapCostEntry.from_json(entry)
            except Exception:
                continue
        with self._lock:
            self._entries = loaded
            try:
                self._last_loaded_mtime = self.cache_path.stat().st_mtime
            except OSError:
                self._last_loaded_mtime = time.time()

    def _persist_cache(self) -> None:
        """Flush in-memory entries to disk."""
        with self._lock:
            entries = {str(k): v.to_json() for k, v in self._entries.items()}
        payload = {
            "metadata": {
                "asset_address": self.config.ASSET_ADDRESS,
                "asset_decimals": self.config.ASSET_DECIMALS,
                "quote_address": self.config.QUOTE_ADDRESS,
                "quote_decimals": self.config.QUOTE_DECIMALS,
                "chain_id": self.config.CHAIN_ID,
                "step_notional": str(self.step),
                "generated_at": datetime.utcnow().isoformat(),
            },
            "entries": entries,
        }
        _atomic_write_json(self.cache_path, payload)
        try:
            self._last_loaded_mtime = self.cache_path.stat().st_mtime
        except OSError:
            self._last_loaded_mtime = time.time()

    def _normalize_notional(self, value: Union[Decimal, float, int]) -> int:
        """Round amount up to nearest 5k step (returns integer USD)."""
        amt = _to_decimal(value)
        if amt <= 0:
            return self.step_int
        multiplier = (amt / self.step).to_integral_value(rounding=ROUND_UP)
        rung = int(multiplier) * self.step_int
        return max(self.step_int, rung)

    def _iter_missing_up_to(self, target: Decimal) -> Dict[int, SwapCostEntry]:
        ceiling = self._normalize_notional(target)
        with self._lock:
            present = set(self._entries.keys())
        missing = []
        rung = self.step_int
        while rung <= ceiling:
            if rung not in present:
                missing.append(rung)
            rung += self.step_int
        return missing

    # ---------------------------------------------------------------- fetch loop
    def _fetch_loop(self) -> None:
        """Background loop that fetches quotes until targets are satisfied."""
        while not self._stop_event.is_set():
            self._consume_requests()
            with self._lock:
                target = self._required_ceiling
            missing = self._iter_missing_up_to(target)
            if not missing:
                # No work pending; wait for new requests or stop signal.
                self._wake_event.wait(timeout=5.0)
                self._wake_event.clear()
                continue
            batch = missing[:MAX_RINGS_PER_CYCLE]
            cycle_start = time.time()
            self._fetch_batch(batch)
            elapsed = time.time() - cycle_start
            sleep_seconds = max(0.0, 60.0 - elapsed)
            if sleep_seconds:
                self._stop_event.wait(timeout=sleep_seconds)

    def _fetch_batch(self, batch: Any) -> None:
        """Fetch a batch of ring quotes in parallel."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        results: Dict[int, Optional[SwapCostEntry]] = {rung: None for rung in batch}
        with ThreadPoolExecutor(max_workers=len(batch)) as executor:
            future_map = {
                executor.submit(self._fetch_rung, rung): rung for rung in batch
            }
            for future in as_completed(future_map):
                rung = future_map[future]
                try:
                    entry = future.result()
                    results[rung] = entry
                except Exception as exc:
                    # Log and continue; retry will happen on next cycle.
                    print(f"[swap-cost-cache] Failed to fetch rung {rung}: {exc}")
        new_entries = {r: e for r, e in results.items() if e is not None}
        if not new_entries:
            return
        with self._lock:
            self._entries.update(new_entries)
        self._persist_cache()
        self._wake_event.set()

    def _fetch_rung(self, rung: int) -> Optional[SwapCostEntry]:
        """Fetch buy/sell quotes for the given rung."""
        amount_dai = Decimal(rung)
        amount_in_wei = self._to_base_units(amount_dai, self.config.QUOTE_DECIMALS)
        buy_quote = self._request_quote(
            token_in=self.config.QUOTE_ADDRESS,
            token_out=self.config.ASSET_ADDRESS,
            amount=amount_in_wei,
        )
        dest_amount_raw = int(buy_quote.get("destAmount", "0"), 16)
        token_amount = _decimal_from_raw(dest_amount_raw, self.config.ASSET_DECIMALS)
        if token_amount <= 0:
            raise SwapCostCacheError(
                f"Piteas returned zero token amount for rung {rung}"
            )
        sell_quote = self._request_quote(
            token_in=self.config.ASSET_ADDRESS,
            token_out=self.config.QUOTE_ADDRESS,
            amount=dest_amount_raw,
        )
        dest_amount_sell_raw = int(sell_quote.get("destAmount", "0"), 16)
        dest_amount_dai = _decimal_from_raw(
            dest_amount_sell_raw, self.config.QUOTE_DECIMALS
        )
        roundtrip_loss = amount_dai - dest_amount_dai
        loss_rate = (
            roundtrip_loss / amount_dai if amount_dai > 0 else Decimal("0")
        )
        entry = SwapCostEntry(
            notional_dai=amount_dai,
            token_amount=token_amount,
            buy_amount_in_wei=amount_in_wei,
            buy_dest_amount_raw=dest_amount_raw,
            buy_gas_estimate=int(buy_quote.get("gasUseEstimate", 0)),
            buy_gas_usd=float(buy_quote.get("gasUseEstimateUSD", 0.0)),
            sell_amount_in_raw=dest_amount_raw,
            sell_dest_amount_raw=dest_amount_sell_raw,
            sell_gas_estimate=int(sell_quote.get("gasUseEstimate", 0)),
            sell_gas_usd=float(sell_quote.get("gasUseEstimateUSD", 0.0)),
            token_per_dai=token_amount / amount_dai,
            dai_per_token=(
                dest_amount_dai / token_amount if token_amount > 0 else Decimal("0")
            ),
            roundtrip_loss_dai=roundtrip_loss,
            loss_rate=loss_rate,
            generated_at=datetime.utcnow(),
        )
        print(
            f"[swap-cost-cache] cached rung ${rung/1000:,.0f}k "
            f"buy→token={token_amount:.6f} sell→dai={dest_amount_dai:.6f}"
        )
        return entry

    def _request_quote(self, token_in: str, token_out: str, amount: int) -> Dict[str, Any]:
        """Fetch a single quote from Piteas with retries."""
        params = {
            "chainId": self.config.CHAIN_ID,
            "tokenInAddress": token_in,
            "tokenOutAddress": token_out,
            "amount": str(amount),
            "allowedSlippage": str(DEFAULT_ALLOWED_SLIPPAGE_BPS),
        }
        query = urlencode(params)
        url = f"{API_ENDPOINT}?{query}"
        last_error = None
        for attempt in range(5):
            try:
                req = Request(url, headers={"Accept": "application/json"})
                with urlopen(req, timeout=20) as resp:
                    if resp.status != 200:
                        raise SwapCostCacheError(
                            f"Piteas HTTP {resp.status} for {url}"
                        )
                    data = json.loads(resp.read().decode("utf-8"))
                    if "destAmount" not in data:
                        raise SwapCostCacheError(
                            f"Piteas response missing destAmount: {data}"
                        )
                    return data
            except (HTTPError, URLError, TimeoutError) as exc:
                last_error = exc
            except Exception as exc:
                last_error = exc
            sleep_for = min(8.0, 1.5 * (attempt + 1))
            time.sleep(sleep_for)
        raise SwapCostCacheError(f"Piteas quote failed after retries: {last_error}")

    def _to_base_units(self, amount: Decimal, decimals: int) -> int:
        """Convert a Decimal amount to integer base units for API calls."""
        scaled = amount * (Decimal(10) ** decimals)
        return int(scaled.to_integral_value(rounding=ROUND_HALF_EVEN))

    def _consume_requests(self) -> None:
        """Process queued requests from other processes."""
        try:
            with self.request_log_path.open("r+", encoding="utf-8") as handle:
                lines = handle.readlines()
                handle.seek(0)
                handle.truncate()
        except FileNotFoundError:
            return
        except OSError:
            return
        if not lines:
            return
        updated_target = self._required_ceiling
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
                target = _to_decimal(payload.get("target"))
            except Exception:
                continue
            normalized = self._normalize_notional(target)
            if Decimal(normalized) > updated_target:
                updated_target = Decimal(normalized)
        with self._lock:
            if updated_target > self._required_ceiling:
                self._required_ceiling = updated_target
                self._wake_event.set()

    # ----------------------------------------------------------------- public API
    def ensure_target(self, amount: Union[Decimal, float, int]) -> None:
        """Ensure background fetcher will cover the requested notional."""
        normalized = self._normalize_notional(amount)
        with self._lock:
            if normalized in self._entries:
                self._queued_targets.discard(normalized)
                return
            if self.producer:
                if Decimal(normalized) > self._required_ceiling:
                    self._required_ceiling = Decimal(normalized)
                    self._wake_event.set()
                return
            if normalized in self._queued_targets:
                return
            self._queued_targets.add(normalized)
        payload = {
            "target": str(normalized),
            "ts": datetime.utcnow().isoformat(),
            "pid": os.getpid(),
        }
        try:
            with self.request_log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload) + "\n")
        except Exception:
            pass

    def _refresh_if_stale(self) -> None:
        """Reload cache file if another process updated it."""
        try:
            mtime = self.cache_path.stat().st_mtime
        except OSError:
            return
        if mtime <= self._last_loaded_mtime:
            return
        self._load_cache()

    def wait_for_entry(
        self, amount: Union[Decimal, float, int], timeout: Optional[float] = None
    ) -> SwapCostEntry:
        """Block until the rung for the requested amount is available."""
        normalized = self._normalize_notional(amount)
        start = time.time()
        while True:
            with self._lock:
                entry = self._entries.get(normalized)
            if entry is not None:
                with self._lock:
                    self._queued_targets.discard(normalized)
                return entry
            self.ensure_target(normalized)
            self._refresh_if_stale()
            if timeout is not None and (time.time() - start) > timeout:
                raise SwapCostCacheTimeout(
                    f"Timed out waiting for swap cost rung {normalized}"
                )
            time.sleep(1.5)

    def compute_buy(self, amount_dai: Union[Decimal, float, int]) -> Dict[str, Any]:
        """Compute net asset received and cost for a DAI→token trade."""
        amount = _to_decimal(amount_dai)
        entry = self.wait_for_entry(amount)
        asset_received = entry.token_per_dai * amount
        asset_received = _quantize(asset_received, self.config.ASSET_DECIMALS)
        implied_sell_dai = entry.dai_per_token * asset_received
        implied_sell_dai = _quantize(implied_sell_dai, self.config.QUOTE_DECIMALS)
        total_cost = amount - implied_sell_dai
        if total_cost < Decimal("0"):
            total_cost = Decimal("0")
        return {
            "rung": entry.notional_dai,
            "amount_in_dai": amount,
            "asset_received": asset_received,
            "implied_sell_dai": implied_sell_dai,
            "cost_dai": total_cost,
            "loss_rate": entry.loss_rate,
            "token_per_dai": entry.token_per_dai,
            "dai_per_token": entry.dai_per_token,
            "roundtrip_loss_dai": entry.roundtrip_loss_dai,
        }

    def compute_sell(
        self,
        token_amount: Union[Decimal, float, int],
        estimated_notional_dai: Union[Decimal, float, int],
    ) -> Dict[str, Any]:
        """Compute net DAI received and cost for a token→DAI trade."""
        tokens = _to_decimal(token_amount)
        estimate = _to_decimal(estimated_notional_dai)
        entry = self.wait_for_entry(estimate)
        net_dai = entry.dai_per_token * tokens
        net_dai = _quantize(net_dai, self.config.QUOTE_DECIMALS)
        approx_input_dai = tokens / entry.token_per_dai
        approx_input_dai = _quantize(approx_input_dai, self.config.QUOTE_DECIMALS)
        total_cost = approx_input_dai - net_dai
        if total_cost < Decimal("0"):
            total_cost = Decimal("0")
        return {
            "rung": entry.notional_dai,
            "token_amount": tokens,
            "net_dai": net_dai,
            "approx_input_dai": approx_input_dai,
            "cost_dai": total_cost,
            "loss_rate": entry.loss_rate,
            "token_per_dai": entry.token_per_dai,
            "dai_per_token": entry.dai_per_token,
            "roundtrip_loss_dai": entry.roundtrip_loss_dai,
        }

    def stop(self) -> None:
        """Stop background fetching (for graceful shutdown/testing)."""
        if not self.producer:
            return
        self._stop_event.set()
        self._wake_event.set()
        if self._fetch_thread and self._fetch_thread.is_alive():
            self._fetch_thread.join(timeout=5.0)


# ---------------------------------------------------------------- module-level API
_CACHE_LOCK = threading.Lock()
_CACHE_INSTANCE: Optional[SwapCostCache] = None


def initialize_swap_cost_cache(
    run_dir: Union[str, Path],
    *,
    producer: bool,
    initial_target: Decimal = Decimal("100000"),
    step_notional: Optional[Union[Decimal, float, int]] = None,
) -> SwapCostCache:
    """Initialise the swap cost cache singleton."""
    global _CACHE_INSTANCE
    with _CACHE_LOCK:
        if _CACHE_INSTANCE is None:
            _CACHE_INSTANCE = SwapCostCache(
                run_dir=run_dir,
                config=Config(),
                producer=producer,
                initial_target=initial_target,
                step_notional=step_notional,
            )
        else:
            # Update target if necessary; keep existing instance.
            if producer:
                _CACHE_INSTANCE.ensure_target(initial_target)
    return _CACHE_INSTANCE


def get_swap_cost_cache() -> SwapCostCache:
    """Return the global cache instance, initialising from env if needed."""
    global _CACHE_INSTANCE
    if _CACHE_INSTANCE is None:
        run_dir = os.environ.get("SWAP_COST_CACHE_DIR")
        if not run_dir:
            raise SwapCostCacheError(
                "SWAP_COST_CACHE_DIR not set; initialise swap cost cache first"
            )
        producer = os.environ.get("SWAP_COST_CACHE_PRODUCER", "0") == "1"
        step_env = os.environ.get("SWAP_COST_CACHE_STEP")
        step_value = _to_decimal(step_env) if step_env else None
        initialize_swap_cost_cache(
            run_dir,
            producer=producer,
            step_notional=step_value,
        )
    return _CACHE_INSTANCE


def ensure_worker_cache_initialized() -> None:
    """Ensure a worker process has a cache instance (non-producer)."""
    run_dir = os.environ.get("SWAP_COST_CACHE_DIR")
    if not run_dir:
        raise SwapCostCacheError(
            "SWAP_COST_CACHE_DIR not set in worker environment"
        )
    step_env = os.environ.get("SWAP_COST_CACHE_STEP")
    step_value = _to_decimal(step_env) if step_env else None
    initialize_swap_cost_cache(run_dir, producer=False, step_notional=step_value)
