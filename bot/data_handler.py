"""Data handler for the PulseChain trading bot."""
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
import pytz
from web3 import Web3
from typing import Optional, Dict, List, Tuple
import os, sys
# Ensure repo root on sys.path so sibling packages import when running bot/*.py directly
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
from bot.config import Config, PULSEX_ROUTER_ABI, ERC20_ABI
from collectors.asset_data_collector import AssetDataCollector
import os

logger = logging.getLogger(__name__)

class DataHandler:
    """Handles data fetching and management for the configured asset."""

    def __init__(self):
        self.config = Config()
        self.asset_symbol = self.config.ASSET_SYMBOL
        self.bridge_symbol = self.config.BRIDGE_SYMBOL
        self.quote_symbol = self.config.QUOTE_SYMBOL
        self.asset_decimals = self.config.ASSET_DECIMALS
        self.w3 = None
        self.router_contract = None
        self.asset_contract = None
        self.bridge_contract = None
        self.quote_contract = None
        self._dai_decimals = 18
        self.price_history = pd.DataFrame()
        self.swap_collector = None
        self._last_price_update: Optional[datetime] = None
        self._last_persist_update: Optional[datetime] = None
        
        # Ensure data directory exists
        os.makedirs(self.config.DATA_DIR, exist_ok=True)
        
        self._connect_to_blockchain()
        self._initialize_swap_collector()

    def _price_cache_path(self) -> str:
        path = os.path.join(
            self.config.DATA_DIR,
            f"{self.asset_symbol.lower()}_price_history_{self.quote_symbol.lower()}.csv",
        )
        legacy = os.path.join(self.config.DATA_DIR, "hex_price_history_dai.csv")
        if not os.path.exists(path) and os.path.exists(legacy) and legacy != path:
            try:
                os.replace(legacy, path)
                logger.info("Migrated legacy HEX price cache to %s", path)
            except Exception as exc:
                logger.warning("Failed to migrate legacy price cache from %s: %s", legacy, exc)
        return path

    def _recent_cache_path(self) -> str:
        path = os.path.join(
            self.config.DATA_DIR,
            f"{self.asset_symbol.lower()}_ohlcv_{self.quote_symbol.lower()}_recent365d_5m.csv",
        )
        legacy = os.path.join(self.config.DATA_DIR, "hex_ohlcv_dai_recent365d_5m.csv")
        if not os.path.exists(path) and os.path.exists(legacy) and legacy != path:
            try:
                os.replace(legacy, path)
                logger.info("Migrated legacy HEX recent OHLCV cache to %s", path)
            except Exception as exc:
                logger.warning("Failed to migrate legacy recent cache from %s: %s", legacy, exc)
        return path

    @staticmethod
    def _utc_now() -> datetime:
        """Return timezone-aware UTC now for consistent timestamping."""
        return datetime.now(tz=pytz.UTC)

    def _normalize_price_history(self):
        """Ensure price history timestamps are UTC and sorted."""
        if self.price_history.empty or 'timestamp' not in self.price_history.columns:
            return
        try:
            self.price_history['timestamp'] = pd.to_datetime(
                self.price_history['timestamp'], utc=True, errors='coerce'
            )
            self.price_history.dropna(subset=['timestamp'], inplace=True)
            self.price_history.sort_values('timestamp', inplace=True)
            self.price_history.reset_index(drop=True, inplace=True)
        except Exception as exc:
            logger.warning(f"Failed to normalize price history timestamps: {exc}")

    def _detect_and_fix_price_anomalies(
        self,
        max_jump: float = 0.35,
        window_minutes: int = 60,
        max_ranges: int = 10,
    ) -> None:
        """Identify unrealistic price jumps and rebuild those windows from on-chain data."""

        if (
            self.swap_collector is None
            or self.price_history.empty
            or 'timestamp' not in self.price_history.columns
            or 'price' not in self.price_history.columns
        ):
            return

        df = self.price_history.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        df.dropna(subset=['timestamp', 'price'], inplace=True)
        df.sort_values('timestamp', inplace=True)

        if df.empty:
            return

        df['pct_change'] = df['price'].pct_change()
        anomaly_indices = df.index[df['pct_change'].abs() > max_jump].tolist()
        if not anomaly_indices:
            return

        processed_ranges: List[Tuple[datetime, datetime]] = []
        base_columns = list(self.price_history.columns)
        replacements = 0

        for idx in anomaly_indices:
            ts = df.loc[idx, 'timestamp']
            if pd.isna(ts):
                continue

            window_start = ts - timedelta(minutes=window_minutes)
            window_end = ts + timedelta(minutes=window_minutes)

            # Skip if this timestamp is already covered by a repaired window
            if any(start <= ts <= end for start, end in processed_ranges):
                continue

            try:
                refreshed = self.fetch_historical_data_range(
                    window_start,
                    window_end,
                    interval_minutes=5,
                )
            except Exception as exc:
                logger.warning(
                    "Failed to refetch data around %s for anomaly correction: %s",
                    ts,
                    exc,
                )
                processed_ranges.append((window_start, window_end))
                continue

            if refreshed is None or refreshed.empty:
                processed_ranges.append((window_start, window_end))
                continue

            refreshed['timestamp'] = pd.to_datetime(
                refreshed['timestamp'], utc=True, errors='coerce'
            )
            refreshed.dropna(subset=['timestamp'], inplace=True)
            refreshed.sort_values('timestamp', inplace=True)

            # Align refreshed frame to existing schema
            for col in base_columns:
                if col not in refreshed.columns:
                    refreshed[col] = None
            refreshed = refreshed[base_columns]

            working = self.price_history.copy()
            working['timestamp'] = pd.to_datetime(
                working['timestamp'], utc=True, errors='coerce'
            )
            working.dropna(subset=['timestamp'], inplace=True)

            mask = (working['timestamp'] < window_start) | (working['timestamp'] > window_end)
            working = working[mask]

            combined = pd.concat([working, refreshed], ignore_index=True)
            combined.drop_duplicates(subset=['timestamp'], keep='last', inplace=True)
            combined.sort_values('timestamp', inplace=True)

            self.price_history = combined[base_columns].reset_index(drop=True)
            processed_ranges.append((window_start, window_end))
            replacements += 1

            if replacements >= max_ranges:
                logger.warning(
                    "Reached maximum anomaly correction windows (%s); stopping further fixes",
                    max_ranges,
                )
                break

        if replacements:
            logger.info("Corrected %d anomalous price segment(s) using on-chain refetch", replacements)
            self._normalize_price_history()
            # Force a persistence cycle so downstream caches reflect corrected prices
            self._persist_live_history(self._utc_now(), force=True)

    def _connect_to_blockchain(self):
        """Connect to PulseChain blockchain"""
        try:
            self.w3 = Web3(Web3.HTTPProvider(self.config.RPC_URL))
            
            if not self.w3.is_connected():
                logger.error("Failed to connect to PulseChain")
                raise ConnectionError("Cannot connect to PulseChain RPC")
            
            # Initialize contracts
            self.router_contract = self.w3.eth.contract(
                address=self.config.PULSEX_ROUTER_V2,
                abi=PULSEX_ROUTER_ABI
            )
            
            self.asset_contract = self.w3.eth.contract(
                address=self.config.ASSET_ADDRESS,
                abi=ERC20_ABI
            )
            
            self.bridge_contract = self.w3.eth.contract(
                address=self.config.BRIDGE_ADDRESS,
                abi=ERC20_ABI
            )

            self.quote_contract = self.w3.eth.contract(
                address=self.config.QUOTE_ADDRESS,
                abi=ERC20_ABI
            )

            try:
                self._dai_decimals = self.dai_contract.functions.decimals().call()
            except Exception:
                self._dai_decimals = 18
            
            logger.info("Successfully connected to PulseChain")
            
        except Exception as e:
            logger.error(f"Failed to connect to blockchain: {e}")
            # Even in demo mode, we need real price data
            logger.error("Cannot run without blockchain connection - need real price data")
            raise ConnectionError(f"Blockchain connection required for real price data: {e}")
    
    def _initialize_swap_collector(self):
        """Initialize PulseChain swap collector for real historical data"""
        try:
            self.swap_collector = AssetDataCollector()
            logger.info("Swap data collector initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize swap data collector: {e}")
            logger.warning("Will fall back to simulated historical data if needed")
            self.swap_collector = None
    
    def get_current_price(self) -> Optional[float]:
        """Get current asset price from PulseX using the configured route."""
        try:
            if not self.w3 or not self.w3.is_connected():
                logger.error("No blockchain connection available - cannot get real price data")
                return None
            
            path = [
                self.config.ASSET_ADDRESS,
                self.config.BRIDGE_ADDRESS,
                self.config.QUOTE_ADDRESS,
            ]
            if self.config.ASSET_SYMBOL == self.bridge_symbol:
                path = [self.config.BRIDGE_ADDRESS, self.config.QUOTE_ADDRESS]

            asset_amount_in = 10 ** self.asset_decimals
            
            amounts_out = self.router_contract.functions.getAmountsOut(
                asset_amount_in, path
            ).call()
            
            quote_amount_out = amounts_out[-1]
            try:
                quote_decimals = self.dai_contract.functions.decimals().call()
            except Exception:
                quote_decimals = self.config.QUOTE_DECIMALS

            price = quote_amount_out / (10 ** quote_decimals)
            
            logger.debug(
                "Current %s price: %.12f %s",
                self.asset_symbol,
                price,
                self.quote_symbol,
            )
            return price
            
        except Exception as e:
            logger.error(f"Error fetching current price: {e}")
            # NEVER return simulated data - we need real price data always
            return None
    
    
    def fetch_historical_data(self, days: int = None, use_incremental: bool = True) -> pd.DataFrame:
        """Fetch historical OHLCV data (real swaps; no synthetic)."""
        if days is None:
            days = self.config.BACKTEST_DAYS

        logger.info("Fetching %s days of historical data", days)

        cache_file = self._price_cache_path()

        if not self.w3 or not self.w3.is_connected():
            logger.error("Cannot fetch historical data without blockchain connection")
            return pd.DataFrame()

        if use_incremental and self.swap_collector is not None:
            logger.info("Using incremental OHLCV collection from swap events")
            existing_data = None
            last_ts = None
            if os.path.exists(cache_file):
                try:
                    existing_data = pd.read_csv(cache_file)
                    if 'timestamp' in existing_data.columns:
                        existing_data['timestamp'] = pd.to_datetime(
                            existing_data['timestamp'], utc=True, errors='coerce'
                        )
                        existing_data.dropna(subset=['timestamp'], inplace=True)
                        if not existing_data.empty:
                            last_ts = existing_data['timestamp'].max()
                        else:
                            existing_data = None
                    else:
                        logger.warning(
                            "Cached price history missing timestamp column at %s", cache_file
                        )
                        existing_data = None
                except Exception as exc:
                    logger.warning("Error reading cache file: %s", exc)
                    existing_data = None

            end_time = datetime.now(tz=pytz.UTC)
            if last_ts is None:
                start_time = end_time - timedelta(days=min(days, 365))
                logger.info("No cache present; collecting fresh OHLCV")
            else:
                start_time = last_ts + timedelta(minutes=5)
                if start_time >= end_time:
                    logger.info("Cache is up to date; returning cached OHLCV")
                    self.price_history = existing_data if existing_data is not None else pd.DataFrame()
                    self._normalize_price_history()
                    return self.price_history

            new_df = self.swap_collector.collect_ohlcv_from_swaps(
                start_time=start_time,
                end_time=end_time,
                interval_minutes=5,
                volume_asset=self.asset_symbol,
            )
            if new_df is None or new_df.empty:
                logger.warning("No new OHLCV data collected; using cache if available")
                self.price_history = existing_data if existing_data is not None else pd.DataFrame()
                return self.price_history

            if existing_data is not None and not existing_data.empty:
                combined = pd.concat([existing_data, new_df], ignore_index=True)
                combined.drop_duplicates(subset=['timestamp'], keep='last', inplace=True)
                combined.sort_values('timestamp', inplace=True)
                self.price_history = combined.reset_index(drop=True)
            else:
                self.price_history = new_df.reset_index(drop=True)

            if days is not None:
                cutoff = datetime.now(tz=pytz.UTC) - timedelta(days=days)
                self.price_history = self.price_history[
                    pd.to_datetime(self.price_history['timestamp'], utc=True) >= cutoff
                ]

            self._normalize_price_history()
            self._detect_and_fix_price_anomalies()
            self.refresh_recent_cache(persist_if_empty=False)
            logger.info("Incremental OHLCV collection complete: %s candles", len(self.price_history))

            try:
                self.price_history.to_csv(cache_file, index=False)
            except Exception as exc:
                logger.warning("Failed to write cache: %s", exc)
            return self.price_history

        self.price_history = self._fetch_real_historical_data(days)
        self._normalize_price_history()
        self._detect_and_fix_price_anomalies()

        try:
            self.price_history.to_csv(cache_file, index=False)
            logger.info("Cached historical data to %s", cache_file)
        except Exception as exc:
            logger.warning("Error caching data: %s", exc)

        self.refresh_recent_cache(persist_if_empty=False)

        return self.price_history

    def fetch_historical_data_range(
        self,
        start: datetime,
        end: datetime,
        interval_minutes: int = 5,
    ) -> pd.DataFrame:
        """Fetch real OHLCV data for a specific UTC interval."""

        if start is None or end is None:
            raise ValueError("Both start and end datetime values are required")

        start_ts = pd.to_datetime(start, utc=True, errors='coerce')
        end_ts = pd.to_datetime(end, utc=True, errors='coerce')

        if pd.isna(start_ts) or pd.isna(end_ts):
            raise ValueError("Invalid start or end datetime provided")

        if end_ts <= start_ts:
            raise ValueError("End datetime must be after start datetime")

        if self.swap_collector is None:
            logger.error("Swap data collector not available; cannot fetch explicit range")
            return pd.DataFrame()

        logger.info(
            "Collecting %s OHLCV from %s to %s (interval=%s minutes)",
            self.asset_symbol,
            start_ts,
            end_ts,
            interval_minutes,
        )

        try:
            df = self.swap_collector.collect_ohlcv_from_swaps(
                start_time=start_ts.to_pydatetime(),
                end_time=end_ts.to_pydatetime(),
                interval_minutes=interval_minutes,
                volume_asset=self.asset_symbol,
            )
        except Exception as exc:
            logger.error("Failed to collect OHLCV for range: %s", exc)
            return pd.DataFrame()

        if df is None or df.empty:
            logger.warning("No OHLCV data returned for requested interval")
            return pd.DataFrame()

        if 'price' not in df.columns and 'close' in df.columns:
            df['price'] = df['close']

        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
        df.dropna(subset=['timestamp'], inplace=True)
        df.sort_values('timestamp', inplace=True)
        df.reset_index(drop=True, inplace=True)

        return df

    def refresh_recent_cache(self, persist_if_empty: bool = True):
        """Persist live/in-memory history to disk and update derived OHLCV cache."""

        if self.price_history.empty:
            if not persist_if_empty:
                return
            self.fetch_historical_data(days=self.config.BACKTEST_DAYS, use_incremental=False)
            if self.price_history.empty:
                return

        self._persist_live_history(self._utc_now())
    
    
    def _fetch_real_historical_data(self, days: int) -> pd.DataFrame:
        """Fetch real historical OHLCV from blockchain swap events"""
        logger.info(f"Fetching real historical data from PulseChain for {days} days")
        
        try:
            if self.swap_collector is not None:
                end_time = datetime.now(tz=pytz.UTC)
                start_time = end_time - timedelta(days=days)
                logger.info(
                    "Collecting real %s OHLCV from %s to %s",
                    self.asset_symbol,
                    start_time,
                    end_time,
                )
                df = self.swap_collector.collect_ohlcv_from_swaps(
                    start_time=start_time,
                    end_time=end_time,
                    interval_minutes=5,
                    volume_asset=self.asset_symbol
                )
                if df is not None and not df.empty:
                    logger.info(f"Successfully collected {len(df)} OHLCV candles")
                    # Ensure required columns
                    if 'price' not in df.columns and 'close' in df.columns:
                        df['price'] = df['close']
                    return df
                else:
                    logger.error("Failed to collect real OHLCV - returning empty DataFrame")
                    return pd.DataFrame()
            else:
                logger.error("Swap data collector not available - cannot proceed without real data")
                return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error fetching real historical data: {e}")
            return pd.DataFrame()
    
    def add_price_point(self, price: float, volume: float = 0):
        """Add a new price point to the historical data"""
        timestamp = self._utc_now()
        new_point = pd.DataFrame({
            'timestamp': [timestamp],
            'price': [price],
            'volume': [volume],
            'high': [price],
            'low': [price],
            'open': [price],
            'close': [price]
        })

        if self.price_history.empty:
            self.price_history = new_point
        else:
            self.price_history = pd.concat([self.price_history, new_point], ignore_index=True)

            # Keep only recent data to manage memory
            if len(self.price_history) > 100000:  # Keep last 100k points
                self.price_history = self.price_history.tail(50000)

        self._normalize_price_history()
        self._last_price_update = timestamp
        self._persist_live_history(timestamp)

    def get_latest_data(self, periods: int = 100) -> pd.DataFrame:
        """Get the latest N periods of data"""
        if self.price_history.empty:
            self.fetch_historical_data()

        self._normalize_price_history()
        return self.price_history.tail(periods)

    def get_last_price_update(self) -> Optional[datetime]:
        """Return the timestamp of the most recent live price update."""
        return self._last_price_update

    def get_history_bounds(self) -> Tuple[Optional[datetime], Optional[datetime]]:
        """Return the earliest and latest timestamps currently available."""

        if self.price_history.empty or 'timestamp' not in self.price_history.columns:
            cached = self.load_cached_price_history()
            if cached.empty:
                return None, None
            self.price_history = cached
            self._normalize_price_history()

        if self.price_history.empty or 'timestamp' not in self.price_history.columns:
            return None, None

        try:
            timestamps = pd.to_datetime(self.price_history['timestamp'], utc=True, errors='coerce').dropna()
        except Exception:
            return None, None

        if timestamps.empty:
            return None, None

        start_ts = timestamps.min().to_pydatetime()
        end_ts = timestamps.max().to_pydatetime()
        return start_ts, end_ts

    def _persist_live_history(self, now: datetime, force: bool = False):
        """Persist in-memory history to disk and maintain recent 5m OHLCV cache."""
        if self.price_history.empty:
            return

        if (
            not force
            and self._last_persist_update
            and (now - self._last_persist_update).total_seconds() < 300
        ):
            return

        self._last_persist_update = now

        cache_file = self._price_cache_path()
        recent_file = self._recent_cache_path()

        try:
            self.price_history.to_csv(cache_file, index=False)
        except Exception as exc:
            logger.warning(f"Failed to persist live cache to {cache_file}: {exc}")

        try:
            df = self.price_history.copy()
            if 'timestamp' not in df.columns:
                return

            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
            df.dropna(subset=['timestamp'], inplace=True)

            cutoff = self._utc_now() - timedelta(days=365)
            df = df[df['timestamp'] >= cutoff]
            if df.empty:
                return

            df.set_index('timestamp', inplace=True)
            agg_map = {}
            for col, fn in (
                ('open', 'first'),
                ('high', 'max'),
                ('low', 'min'),
                ('close', 'last'),
                ('price', 'last'),
                ('volume', 'sum'),
            ):
                if col in df.columns:
                    agg_map[col] = fn

            if 'close' not in agg_map:
                return

            ohlcv = df.resample('5T').agg(agg_map)
            ohlcv.dropna(subset=['close'], inplace=True)
            if ohlcv.empty:
                return

            ohlcv.reset_index(inplace=True)
            if 'price' not in ohlcv.columns and 'close' in ohlcv.columns:
                ohlcv['price'] = ohlcv['close']

            ohlcv.to_csv(recent_file, index=False)
            logger.debug(
                "Persisted live OHLCV snapshot to %s (%d rows)",
                recent_file,
                len(ohlcv),
            )
        except Exception as exc:
            logger.warning(f"Failed to update recent OHLCV cache: {exc}")

    def load_cached_price_history(self) -> pd.DataFrame:
        """Load the full cached price history from disk (real data only)."""
        cache_dir = self.config.DATA_DIR
        asset = self.asset_symbol.lower()
        quote = self.quote_symbol.lower()

        candidate_files = [
            f"{asset}_price_history_{quote}.csv",
            f"{asset}_ohlcv_{quote}_730day_5m.csv",
            f"{asset}_ohlcv_{quote}_prior365d_5m.csv",
            f"{asset}_ohlcv_{quote}_recent365d_5m.csv",
        ]

        # Allow legacy HEX cache names to be picked up when operating on other assets
        if asset != "hex":
            candidate_files.extend(
                [
                    "hex_price_history_dai.csv",
                    "hex_ohlcv_dai_730day_5m.csv",
                    "hex_ohlcv_dai_prior365d_5m.csv",
                    "hex_ohlcv_dai_recent365d_5m.csv",
                ]
            )

        best_df = None
        best_span = None

        for filename in candidate_files:
            path = os.path.join(cache_dir, filename)
            if not os.path.exists(path):
                continue

            try:
                df = pd.read_csv(path)
            except Exception as exc:
                logger.warning("Failed to read cached price history from %s: %s", path, exc)
                continue

            if 'timestamp' not in df.columns:
                logger.warning("Cached price history at %s missing timestamp column", path)
                continue

            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
            df.dropna(subset=['timestamp'], inplace=True)
            if df.empty:
                continue

            df.sort_values('timestamp', inplace=True)
            df.reset_index(drop=True, inplace=True)

            span = df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]
            if best_span is None or span > best_span:
                best_span = span
                best_df = df

        if best_df is None:
            return pd.DataFrame()

        self.price_history = best_df.copy()
        self._normalize_price_history()
        self._detect_and_fix_price_anomalies()

        # Persist cleaned history to canonical cache if we made adjustments
        if not self.price_history.empty:
            try:
                canonical_cache = self._price_cache_path()
                self.price_history.to_csv(canonical_cache, index=False)
            except Exception as exc:
                logger.warning("Failed to persist cleaned cache to %s: %s", canonical_cache, exc)

        return self.price_history.copy()
    
    def save_data(self, filename: str = None):
        """Save current price history to file"""
        if filename is None:
            filename = os.path.join(
                self.config.DATA_DIR,
                f"{self.asset_symbol.lower()}_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            )
        
        if not self.price_history.empty:
            self.price_history.to_csv(filename, index=False)
            logger.info(f"Data saved to {filename}")
        else:
            logger.warning("No data to save")
    
    def load_data(self, filename: str):
        """Load price history from file"""
        try:
            self.price_history = pd.read_csv(filename)
            self.price_history['timestamp'] = pd.to_datetime(self.price_history['timestamp'])
            logger.info(f"Loaded {len(self.price_history)} data points from {filename}")
        except Exception as e:
            logger.error(f"Error loading data from {filename}: {e}")
            raise
    
    def get_price_stats(self) -> Dict:
        """Get basic price statistics"""
        if self.price_history.empty:
            return {}
        
        recent_data = self.price_history.tail(1440)  # Last 24 hours if minute data
        
        return {
            'current_price': self.price_history['price'].iloc[-1],
            '24h_high': recent_data['high'].max(),
            '24h_low': recent_data['low'].min(),
            '24h_change': (self.price_history['price'].iloc[-1] / recent_data['price'].iloc[0] - 1) * 100,
            '24h_volume': recent_data['volume'].sum(),
            'data_points': len(self.price_history)
        }
