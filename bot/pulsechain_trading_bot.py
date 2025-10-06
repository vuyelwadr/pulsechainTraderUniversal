"""
PulseChain Trading Bot - Main Bot Class
Orchestrates all components for automated asset trading on PulseChain
"""
import argparse
import math
import os, sys
from pathlib import Path
# Ensure repo root on sys.path so sibling packages import correctly when running this file directly
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
import logging
import time
import threading
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Type
import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer

from urllib.parse import urlparse

import pandas as pd

from bot.config import Config
from bot.data_handler import DataHandler
from strategies.base_strategy import StrategyManager, BaseStrategy
from strategies.ma_crossover import MovingAverageCrossover
from strategies.grid_trading_strategy_v2_aggressive import GridTradingStrategyV2Aggressive
from strategies.custom.regime_grid_breakout_hybrid import RegimeGridBreakoutHybrid
from bot.backtest_engine import BacktestEngine
from bot.html_generator import HTMLGenerator
from bot.trade_executor import TradeExecutor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

TOP_STRATEGY_REPORT = (
    Path(REPO_ROOT)
    / "reports"
    / "gridTradeAggressive"
    / "visualization"
    / "01_gridAggressive_top.json"
)

STRATEGY_CLASS_REGISTRY: Dict[str, Type[BaseStrategy]] = {
    "GridTradingStrategyV2Aggressive": GridTradingStrategyV2Aggressive,
    "RegimeGridBreakoutHybrid": RegimeGridBreakoutHybrid,
}

class PulseChainTradingBot:
    """Main PulseChain Trading Bot class"""
    
    def __init__(self, demo_mode: bool = True):
        self.config = Config()
        self.demo_mode = demo_mode or self.config.DEMO_MODE
        
        # Initialize components
        self.data_handler = DataHandler()
        self.strategy_manager = StrategyManager()
        self.backtest_engine = BacktestEngine()
        self.html_generator = HTMLGenerator()
        self.trade_executor: Optional[TradeExecutor] = None
        
        # Trading state
        self.is_running = False
        self.portfolio_history = []
        self.current_position = None
        self.quote_balance = float(self.config.INITIAL_BALANCE)
        self.asset_balance = 0.0
        self.bridge_balance = 0.0
        self.bridge_value_quote = 0.0
        self.strategy_metadata: Dict[str, Dict] = {}
        self.live_trades: List[Dict] = []
        self.current_position_cost: float = 0.0
        self.latest_backtest_results: Optional[Dict] = None
        self.latest_dashboard_path: Optional[str] = None
        self.dashboard_host = os.environ.get('HEXBOT_DASHBOARD_HOST', '127.0.0.1')
        self.dashboard_port = int(os.environ.get('HEXBOT_DASHBOARD_PORT', '8787'))
        self._dashboard_server: Optional[HTTPServer] = None

        # Setup strategies
        self._initialize_strategies()

        # Preload recent history so dashboards have data immediately and caches stay fresh
        try:
            self.data_handler.fetch_historical_data(days=self.config.BACKTEST_DAYS)
        except Exception as exc:
            logger.warning("Initial historical data fetch failed: %s", exc)
        else:
            self.data_handler.refresh_recent_cache(persist_if_empty=False)

        if not self.demo_mode:
            try:
                self.trade_executor = TradeExecutor(self.data_handler, logger=logger)
                balances = self._refresh_onchain_balances()
                if balances:
                    logger.info(
                        "Loaded on-chain balances – DAI %.4f, HEX %.4f, PLS %.4f",
                        float(balances['dai']),
                        float(balances['hex']),
                        float(balances['pls']),
                    )
            except Exception as exc:
                logger.error("Failed to prepare trade executor: %s", exc)
                raise

        # Run a baseline backtest so the dashboard ships with historical context
        self._run_initial_backtest()

        # Generate an initial dashboard snapshot
        if not self.latest_dashboard_path or not os.path.exists(self.latest_dashboard_path):
            self._generate_live_report()

        # Start lightweight HTTP server for dashboard/API controls
        self._start_dashboard_server()

        logger.info(f"PulseChain Trading Bot initialized - Demo Mode: {self.demo_mode}")
    
    def _initialize_strategies(self):
        """Initialize available trading strategies"""
        
        # Attempt to load the best-performing strategy discovered during optimization.
        best_strategy = self._load_best_strategy()
        if best_strategy:
            self.strategy_manager.add_strategy(best_strategy)
            logger.info(
                "Using %s as primary strategy from %s",
                best_strategy.name,
                self.strategy_metadata.get('best_strategy', {}).get('source_file', 'unknown report')
            )
        else:
            logger.warning(
                "Falling back to default MovingAverageCrossover strategy; could not load top strategy report"
            )

        # Always add Moving Average Crossover strategy for comparison/fallback.
        ma_strategy = MovingAverageCrossover({
            'short_period': self.config.MA_SHORT_PERIOD,
            'long_period': self.config.MA_LONG_PERIOD,
            'ma_type': 'ema',
            'min_strength': 0.6
        })
        self.strategy_manager.add_strategy(ma_strategy)
        
        # Ensure we have an active strategy if no best strategy was registered.
        if not self.strategy_manager.get_active_strategy():
            self.strategy_manager.set_active_strategy(ma_strategy.name)

        logger.info(f"Initialized {len(self.strategy_manager.list_strategies())} strategies")

    def _start_dashboard_server(self):
        """Launch lightweight HTTP server for dashboard + API controls."""
        if self._dashboard_server is not None:
            return

        handler_cls = self._build_dashboard_handler()

        host = self.dashboard_host
        port = self.dashboard_port
        server = None

        for attempt in range(5):
            try:
                server = HTTPServer((host, port), handler_cls)
                break
            except OSError as exc:
                logger.warning("Dashboard port %s unavailable (%s); trying next port", port, exc)
                port += 1

        if server is None:
            logger.error("Could not start dashboard server; no ports available")
            return

        self.dashboard_port = port
        self._dashboard_server = server

        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        logger.info(
            "Dashboard server running at http://%s:%s/dashboard",
            self.dashboard_host,
            self.dashboard_port,
        )

    def _build_dashboard_handler(self):
        """Factory for HTTP request handler bound to this bot instance."""
        bot = self
        logger_local = logger

        class DashboardHandler(BaseHTTPRequestHandler):
            server_version = "HEXBotDashboard/1.0"

            def _set_headers(self, status_code=200, content_type="application/json"):
                self.send_response(status_code)
                self.send_header("Content-Type", content_type)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "Content-Type")
                self.end_headers()

            def do_OPTIONS(self):
                self._set_headers()

            def do_GET(self):
                parsed = urlparse(self.path)
                path = parsed.path

                if path in ("/", "/dashboard"):
                    html_path = bot.latest_dashboard_path
                    if not html_path or not os.path.exists(html_path):
                        bot._generate_live_report()
                        html_path = bot.latest_dashboard_path

                    if not html_path or not os.path.exists(html_path):
                        self._set_headers(503)
                        self.wfile.write(json.dumps({"error": "Dashboard not ready"}).encode("utf-8"))
                        return

                    try:
                        with open(html_path, "rb") as handle:
                            content = handle.read()
                        self._set_headers(200, "text/html; charset=utf-8")
                        self.wfile.write(content)
                    except Exception as exc:
                        logger_local.error("Failed to serve dashboard HTML: %s", exc)
                        self._set_headers(500)
                        self.wfile.write(json.dumps({"error": "Failed to load dashboard"}).encode("utf-8"))
                    return

                if path == "/status":
                    payload = bot.get_status()
                    payload["dashboard_port"] = bot.dashboard_port
                    payload["dashboard_host"] = bot.dashboard_host
                    payload["live_trades"] = len(bot.live_trades)
                    payload["latest_backtest"] = bool(bot.latest_backtest_results)
                    self._set_headers()
                    self.wfile.write(json.dumps(payload, default=str).encode("utf-8"))
                    return

                if path == "/trades":
                    self._set_headers()
                    self.wfile.write(json.dumps(bot.live_trades, default=str).encode("utf-8"))
                    return

                self._set_headers(404)
                self.wfile.write(json.dumps({"error": "Not found"}).encode("utf-8"))

            def do_POST(self):
                parsed = urlparse(self.path)
                if parsed.path != "/api/backtest":
                    self._set_headers(404)
                    self.wfile.write(json.dumps({"error": "Not found"}).encode("utf-8"))
                    return

                length = int(self.headers.get("Content-Length", "0"))
                raw_body = self.rfile.read(length) if length > 0 else b"{}"
                try:
                    payload = json.loads(raw_body.decode("utf-8")) if raw_body else {}
                except json.JSONDecodeError:
                    self._set_headers(400)
                    self.wfile.write(json.dumps({"error": "Invalid JSON payload"}).encode("utf-8"))
                    return

                start_iso = payload.get("start_date")
                end_iso = payload.get("end_date")
                days = payload.get("days")
                strategy_name = payload.get("strategy")
                full_dataset_raw = payload.get("full_dataset")

                if isinstance(full_dataset_raw, bool):
                    full_dataset = full_dataset_raw
                elif isinstance(full_dataset_raw, str):
                    full_dataset = full_dataset_raw.strip().lower() in {"true", "1", "yes", "on"}
                else:
                    full_dataset = bool(full_dataset_raw)

                try:
                    start_dt = pd.to_datetime(start_iso, utc=True, errors='coerce') if start_iso and not full_dataset else None
                    if start_dt is not None and pd.isna(start_dt):
                        raise ValueError("Invalid start_date")
                    end_dt = pd.to_datetime(end_iso, utc=True, errors='coerce') if end_iso and not full_dataset else None
                    if end_dt is not None and pd.isna(end_dt):
                        raise ValueError("Invalid end_date")
                    start_dt = start_dt.to_pydatetime() if start_dt is not None else None
                    end_dt = end_dt.to_pydatetime() if end_dt is not None else None
                except ValueError as exc:
                    self._set_headers(400)
                    self.wfile.write(json.dumps({"error": str(exc)}).encode("utf-8"))
                    return

                if not full_dataset and start_dt and end_dt and end_dt <= start_dt:
                    self._set_headers(400)
                    self.wfile.write(json.dumps({"error": "end_date must be after start_date"}).encode("utf-8"))
                    return

                try:
                    days_val = None if full_dataset else (int(days) if days is not None else None)
                except (TypeError, ValueError):
                    self._set_headers(400)
                    self.wfile.write(json.dumps({"error": "days must be an integer"}).encode("utf-8"))
                    return

                logger_local.info(
                    "HTTP backtest request: strategy=%s, days=%s, start=%s, end=%s, full=%s",
                    strategy_name,
                    days_val,
                    start_dt,
                    end_dt,
                    full_dataset,
                )

                results = bot.run_backtest(
                    days=days_val,
                    strategy_name=strategy_name,
                    start_date=start_dt,
                    end_date=end_dt,
                    use_full_history=full_dataset,
                )

                status_code = 200 if 'error' not in results else 400
                self._set_headers(status_code)
                safe_payload = json.loads(json.dumps(results, default=str))
                self.wfile.write(json.dumps(safe_payload).encode("utf-8"))

            def log_message(self, format, *args):
                logger_local.debug("Dashboard server: " + format, *args)

        return DashboardHandler

    def _run_initial_backtest(self):
        """Populate dashboard with an all-history backtest before serving the UI."""
        try:
            def prepare_history(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
                if df is None or df.empty or 'timestamp' not in df.columns:
                    return None
                prepared = df.copy()
                prepared['timestamp'] = pd.to_datetime(prepared['timestamp'], utc=True, errors='coerce')
                prepared.dropna(subset=['timestamp'], inplace=True)
                if prepared.empty:
                    return None
                prepared.sort_values('timestamp', inplace=True)
                prepared.reset_index(drop=True, inplace=True)
                return prepared

            candidates: List[pd.DataFrame] = []
            prepared_live = prepare_history(self.data_handler.price_history)
            if prepared_live is not None:
                candidates.append(prepared_live)

            prepared_cached = prepare_history(self.data_handler.load_cached_price_history())
            if prepared_cached is not None:
                candidates.append(prepared_cached)

            if not candidates:
                logger.warning("No cached historical data available for initial dashboard backtest")
                results = self.run_backtest(days=self.config.BACKTEST_DAYS)
                if results.get('error'):
                    logger.warning("Initial dashboard backtest failed: %s", results['error'])
                return

            history = max(
                candidates,
                key=lambda df: df['timestamp'].iloc[-1] - df['timestamp'].iloc[0],
            )

            start_ts = history['timestamp'].iloc[0]
            end_ts = history['timestamp'].iloc[-1]
            if pd.isna(start_ts) or pd.isna(end_ts) or end_ts <= start_ts:
                logger.warning("Historical dataset has invalid time span; skipping initial backtest")
                return

            span_days = max(1, int(math.ceil((end_ts - start_ts).total_seconds() / 86400.0)))

            results = self.run_backtest(
                days=span_days,
                start_date=start_ts.to_pydatetime(),
                end_date=end_ts.to_pydatetime(),
                historical_data=history,
            )

            if results.get('error'):
                logger.warning("Initial dashboard backtest failed: %s", results['error'])
        except Exception as exc:
            logger.warning("Initial dashboard backtest failed: %s", exc)
    def _load_best_strategy(self) -> Optional[BaseStrategy]:
        """Load the top-performing strategy configuration produced by the optimizer."""
        report_path = TOP_STRATEGY_REPORT
        if not report_path.exists():
            logger.warning("Top strategy report not found at %s", report_path)
            return None

        try:
            with report_path.open('r') as handle:
                payload = json.load(handle)

            strategy_name = payload.get('strategy', 'GridTradingStrategyV2Aggressive')
            strategy_cls = STRATEGY_CLASS_REGISTRY.get(strategy_name)
            if not strategy_cls:
                logger.error(
                    "Strategy %s from %s is not registered; update STRATEGY_CLASS_REGISTRY",
                    strategy_name,
                    report_path,
                )
                return None

            params = payload.get('selected_params', {}) or {}
            strategy = strategy_cls(params)

            metrics = payload.get('metrics_row', {})
            self.strategy_metadata['best_strategy'] = {
                'name': strategy.name,
                'timeframe_minutes': params.get('timeframe_minutes'),
                'objective': metrics.get('objective'),
                'total_return_pct': metrics.get('total_return_pct'),
                'max_drawdown_pct': metrics.get('max_drawdown_pct'),
                'sharpe_ratio': metrics.get('sharpe_ratio'),
                'total_trades': metrics.get('total_trades'),
                'source_file': str(report_path),
                'label': payload.get('label'),
            }

            # Share context in logs for quick verification.
            logger.info(
                "Loaded top strategy %s (%s) with timeframe %s minutes",
                strategy.name,
                payload.get('label', report_path.name),
                params.get('timeframe_minutes')
            )

            return strategy
        except Exception as exc:
            logger.error("Failed to load top strategy definition from %s: %s", report_path, exc)
            return None

    def run_backtest(
        self,
        days: int = None,
        strategy_name: str = None,
        start_date: datetime = None,
        end_date: datetime = None,
        historical_data: Optional[pd.DataFrame] = None,
        use_full_history: bool = False,
    ) -> Dict:
        """Run backtest on historical data"""
        
        if use_full_history:
            days = None
        else:
            days = days or self.config.BACKTEST_DAYS

        if historical_data is not None:
            historical_data = historical_data.copy()
            if not historical_data.empty and 'timestamp' in historical_data.columns:
                historical_data['timestamp'] = pd.to_datetime(
                    historical_data['timestamp'], utc=True, errors='coerce'
                )
                historical_data.dropna(subset=['timestamp'], inplace=True)
                historical_data.sort_values('timestamp', inplace=True)
                historical_data.reset_index(drop=True, inplace=True)

                if start_date is None:
                    start_ts = historical_data['timestamp'].iloc[0]
                    start_date = start_ts.to_pydatetime()
                if end_date is None:
                    end_ts = historical_data['timestamp'].iloc[-1]
                    end_date = end_ts.to_pydatetime()
            else:
                logger.warning("Provided historical data is empty or missing timestamps")
        else:
            if use_full_history:
                logger.info("Starting backtest for entire available dataset")
                historical_data = self.data_handler.load_cached_price_history()
                if historical_data.empty:
                    logger.warning("Full-history dataset is empty; falling back to default fetch window")
                    historical_data = self.data_handler.fetch_historical_data(days)
            else:
                historical_data = pd.DataFrame()

            if not use_full_history and start_date:
                logger.info(
                    "Starting backtest for custom range %s → %s",
                    start_date,
                    end_date or "now",
                )
                if end_date is None:
                    end_date = (
                        datetime.now(tz=start_date.tzinfo)
                        if start_date.tzinfo
                        else datetime.utcnow()
                    )
                try:
                    historical_data = self.data_handler.fetch_historical_data_range(start_date, end_date)
                except ValueError as exc:
                    return {'error': str(exc)}
            elif not use_full_history:
                logger.info(f"Starting backtest for last {days} days")
                historical_data = self.data_handler.fetch_historical_data(days)

        if historical_data.empty:
            return {'error': 'No historical data available'}

        range_start_iso = None
        range_end_iso = None
        if 'timestamp' in historical_data.columns and not historical_data.empty:
            ts_series = pd.to_datetime(historical_data['timestamp'], utc=True, errors='coerce').dropna()
            if not ts_series.empty:
                range_start_iso = ts_series.min().isoformat()
                range_end_iso = ts_series.max().isoformat()

        # Select strategy
        if strategy_name:
            if strategy_name not in self.strategy_manager.strategies:
                return {'error': f'Strategy {strategy_name} not found'}
            strategy = self.strategy_manager.strategies[strategy_name]
        else:
            strategy = self.strategy_manager.get_active_strategy()
            if not strategy:
                return {'error': 'No active strategy selected'}
        # Run backtest
        results = self.backtest_engine.run_backtest(
            strategy,
            historical_data,
        )

        # Generate HTML report
        if not results.get('error'):
            if 'strategy_name' not in results:
                results['strategy_name'] = strategy.name
            html_file = self.html_generator.generate_backtest_report(results, strategy.name)
            results['html_report'] = html_file
            if start_date is None:
                start_date = pd.to_datetime(range_start_iso).to_pydatetime() if range_start_iso else None
            if end_date is None:
                end_date = pd.to_datetime(range_end_iso).to_pydatetime() if range_end_iso else None

            range_days = days
            if range_days is None and range_start_iso and range_end_iso:
                start_ts = pd.to_datetime(range_start_iso)
                end_ts = pd.to_datetime(range_end_iso)
                if not pd.isna(start_ts) and not pd.isna(end_ts):
                    delta_days = (end_ts - start_ts).total_seconds() / 86400.0
                    range_days = max(1, int(math.ceil(delta_days)))

            results['requested_range'] = {
                'start': start_date.isoformat() if start_date else range_start_iso,
                'end': end_date.isoformat() if end_date else range_end_iso,
                'days': range_days,
            }
            results['generated_at'] = datetime.utcnow().isoformat()
            self.latest_backtest_results = results
            # Refresh dashboard with latest backtest insights
            self._generate_live_report()

        return results
    
    def start_live_trading(self):
        """Start live trading mode"""
        if not self.demo_mode:
            logger.error("Live trading with real money not implemented yet - use demo mode")
            return
        
        logger.info("Starting live trading in DEMO MODE")
        self.is_running = True
        
        # Start trading loop in separate thread
        trading_thread = threading.Thread(target=self._trading_loop, daemon=True)
        trading_thread.start()
        
        return trading_thread
    
    def stop_trading(self):
        """Stop live trading"""
        logger.info("Stopping trading bot")
        self.is_running = False
    
    def _trading_loop(self):
        """Main trading loop for live trading"""
        last_update = datetime.now()
        
        while self.is_running:
            try:
                current_time = datetime.now()
                
                # Update data every minute
                if (current_time - last_update).total_seconds() >= 60:
                    
                    if not self.demo_mode:
                        self._refresh_onchain_balances()

                    # Get current price
                    current_price = self.data_handler.get_current_price()
                    if current_price is None:
                        logger.warning("Could not fetch current price")
                        time.sleep(10)
                        continue
                    
                    # Add to historical data
                    self.data_handler.add_price_point(current_price)
                    
                    # Get recent data for analysis
                    recent_data = self.data_handler.get_latest_data(100)
                    if recent_data.empty:
                        logger.warning("No data available for analysis")
                        time.sleep(10)
                        continue
                    
                    # Get trading signal
                    signal, strength = self.strategy_manager.get_signal(recent_data)
                    
                    # Execute trade according to mode
                    if self.demo_mode:
                        self._execute_demo_trade(signal, strength, current_price)
                    else:
                        self._execute_real_trade(signal, strength, current_price)
                    
                    # Update portfolio history
                    self._update_portfolio_state(current_price, signal, strength)
                    
                    # Generate HTML report
                    self._generate_live_report()
                    
                    last_update = current_time
                    
                    logger.info(f"Price: {current_price:.8f} DAI, Signal: {signal}, Strength: {strength:.2f}")
                
                time.sleep(5)  # Check every 5 seconds
                
            except KeyboardInterrupt:
                logger.info("Trading interrupted by user")
                break
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                time.sleep(10)
        
        self.is_running = False
        logger.info("Trading loop stopped")

    def _apply_synced_balances(self, balances: Optional[Dict[str, Decimal]]):
        if not balances:
            return
        self.quote_balance = float(balances.get('dai', Decimal('0')))
        self.asset_balance = float(balances.get('hex', Decimal('0')))
        self.bridge_balance = float(balances.get('pls', Decimal('0')))
        self.bridge_value_quote = float(balances.get('pls_dai', Decimal('0')))

    def _refresh_onchain_balances(self) -> Optional[Dict[str, Decimal]]:
        if not self.trade_executor:
            return None
        try:
            balances = self.trade_executor.sync_balances()
        except Exception as exc:
            logger.error("Unable to refresh on-chain balances: %s", exc)
            return None
        self._apply_synced_balances(balances)
        return balances

    def _execute_real_trade(self, signal: str, strength: float, price: float):
        if not self.trade_executor:
            logger.error("Trade executor unavailable; cannot place live swaps")
            return

        if strength < 0.6:
            return

        balances = self._refresh_onchain_balances()
        if not balances:
            return

        try:
            reserve_swap = self.trade_executor.ensure_pls_reserve()
        except Exception as exc:
            logger.error("Failed to ensure PLS reserve: %s", exc)
            return

        if reserve_swap:
            self._record_reserve_topup(reserve_swap)
            balances = self._refresh_onchain_balances()
            if not balances:
                return

        if signal == 'buy':
            if self.current_position == 'long':
                return

            dai_balance = Decimal(balances.get('dai', Decimal('0')))
            if dai_balance <= Decimal('1'):
                logger.info("DAI balance %.4f insufficient for buy", dai_balance)
                return

            trade_amount = dai_balance

            try:
                swap_result = self.trade_executor.swap_dai_for_hex(trade_amount)
            except Exception as exc:
                logger.error("Swap DAI→HEX failed: %s", exc)
                return

            self.current_position = 'long'
            self.current_position_cost = float(swap_result.amount_in)
            balances = self._refresh_onchain_balances()
            self._record_real_trade('buy', strength, price, swap_result, balances)

        elif signal == 'sell':
            if self.current_position != 'long':
                return

            asset_balance = Decimal(balances.get('hex', Decimal('0')))
            if asset_balance <= Decimal('0'):
                logger.info("%s balance %.4f insufficient for sell", self.config.ASSET_SYMBOL, asset_balance)
                return

            try:
                swap_result = self.trade_executor.swap_hex_for_dai(asset_balance)
            except Exception as exc:
                logger.error("Swap HEX→DAI failed: %s", exc)
                return

            self.current_position = None
            balances = self._refresh_onchain_balances()
            self._record_real_trade('sell', strength, price, swap_result, balances)
            self.current_position_cost = 0.0

    def _record_reserve_topup(self, swap_result):
        self._refresh_onchain_balances()
        record = {
            'timestamp': datetime.utcnow(),
            'type': 'reserve_topup',
            'price': 0.0,
            'asset_amount': 0.0,
            'quote_amount': float(swap_result.amount_in),
            'fee': float(swap_result.fee_dai),
            'signal_strength': 0.0,
            'balance_after': self.balance,
            'native_balance_after': self.pls_balance,
            'tx_hash': swap_result.tx_hash,
            'gas_used': swap_result.gas_used,
            'gas_price_gwei': float(swap_result.metadata.get('gas_price_gwei', Decimal('0'))),
        }
        self.live_trades.append(record)
        if len(self.live_trades) > 500:
            self.live_trades = self.live_trades[-250:]

    def _record_real_trade(
        self,
        trade_type: str,
        strength: float,
        price: float,
        swap_result,
        balances: Optional[Dict[str, Decimal]]
    ):
        self._apply_synced_balances(balances)
        asset_amount = float(swap_result.amount_out if trade_type == 'buy' else swap_result.amount_in)
        quote_amount = float(swap_result.amount_in if trade_type == 'buy' else swap_result.amount_out)
        pnl_pct = None
        if trade_type == 'sell' and self.current_position_cost:
            try:
                pnl_pct = ((swap_result.amount_out - Decimal(self.current_position_cost)) / Decimal(self.current_position_cost)) * 100
            except Exception:
                pnl_pct = None

        record = {
            'timestamp': datetime.utcnow(),
            'type': trade_type,
            'price': price,
            'asset_amount': asset_amount,
            'quote_amount': quote_amount,
            'fee': float(swap_result.fee_dai),
            'signal_strength': strength,
            'balance_after': self.quote_balance,
            'native_balance_after': self.bridge_balance,
            'portfolio_value_after': self.quote_balance + (self.asset_balance * price) + self.bridge_value_quote,
            'tx_hash': swap_result.tx_hash,
            'gas_used': swap_result.gas_used,
            'gas_price_gwei': float(swap_result.metadata.get('gas_price_gwei', Decimal('0'))),
            'fee_pls': float(swap_result.fee_pls),
            'fee_dai': float(swap_result.fee_dai),
        }

        if pnl_pct is not None:
            record['pnl_pct'] = float(pnl_pct)

        self.live_trades.append(record)
        if len(self.live_trades) > 500:
            self.live_trades = self.live_trades[-250:]

    def _execute_demo_trade(self, signal: str, strength: float, price: float):
        """Execute trades in demo mode"""
        if strength < 0.6:  # Minimum signal strength
            return
        
        if signal == 'buy' and self.current_position != 'long':
            # Buy asset with quote currency (routed through bridge on-chain)
            trade_amount = self.quote_balance  # Deploy full balance by default
            if trade_amount > 1.0:  # Minimum trade
                fee = trade_amount * 0.0025  # 0.25% fee
                net_amount = trade_amount - fee
                asset_received = net_amount / price
                
                self.quote_balance -= trade_amount
                self.asset_balance += asset_received
                self.current_position = 'long'
                self.current_position_cost = trade_amount
                trade_record = {
                    'timestamp': datetime.utcnow(),
                    'type': 'buy',
                    'price': price,
                    'asset_amount': asset_received,
                    'quote_amount': trade_amount,
                    'fee': fee,
                    'signal_strength': strength,
                    'balance_after': self.quote_balance,
                    'slippage_pct': 0.0,
                    'portfolio_value_after': self.quote_balance + (self.asset_balance * price),
                }
                self.live_trades.append(trade_record)
                if len(self.live_trades) > 500:
                    self.live_trades = self.live_trades[-250:]
                
                logger.info(f"DEMO BUY: {asset_received:.4f} {self.config.ASSET_SYMBOL} at {price:.8f} {self.config.QUOTE_SYMBOL}")

        elif signal == 'sell' and self.current_position == 'long':
            # Sell asset for quote currency
            if self.asset_balance > 0:
                quote_gross = self.asset_balance * price
                fee = quote_gross * 0.0025
                quote_net = quote_gross - fee
                
                self.quote_balance += quote_net
                asset_sold = self.asset_balance
                self.asset_balance = 0.0
                self.current_position = None
                pnl_pct = 0.0
                if self.current_position_cost > 0:
                    pnl_pct = ((quote_net - self.current_position_cost) / self.current_position_cost) * 100
                trade_record = {
                    'timestamp': datetime.utcnow(),
                    'type': 'sell',
                    'price': price,
                    'asset_amount': asset_sold,
                    'quote_amount': quote_net,
                    'fee': fee,
                    'signal_strength': strength,
                    'balance_after': self.quote_balance,
                    'pnl_pct': pnl_pct,
                    'slippage_pct': 0.0,
                    'portfolio_value_after': self.quote_balance + (self.asset_balance * price),
                }
                self.live_trades.append(trade_record)
                if len(self.live_trades) > 500:
                    self.live_trades = self.live_trades[-250:]
                self.current_position_cost = 0.0

                logger.info(f"DEMO SELL: {asset_sold:.4f} {self.config.ASSET_SYMBOL} at {price:.8f} {self.config.QUOTE_SYMBOL}")
    
    def _update_portfolio_state(self, price: float, signal: str, strength: float):
        """Update portfolio state tracking"""
        asset_value = self.asset_balance * price
        total_value = self.quote_balance + asset_value + self.bridge_value_quote
        
        state = {
            'timestamp': datetime.now(),
            'price': price,
            'quote_balance': self.quote_balance,
            'asset_balance': self.asset_balance,
            'asset_value_quote': asset_value,
            'bridge_balance': self.bridge_balance,
            'native_value_quote': self.pls_value_dai,
            'total_value': total_value,
            'position': self.current_position,
            'signal': signal,
            'signal_strength': strength
        }
        
        self.portfolio_history.append(state)
        
        # Keep only last 1000 points
        if len(self.portfolio_history) > 1000:
            self.portfolio_history = self.portfolio_history[-500:]
    
    def _generate_live_report(self):
        """Generate live trading HTML report"""
        # Get strategy recommendation
        recent_data = self.data_handler.get_latest_data(50)
        if not recent_data.empty:
            strategy = self.strategy_manager.get_active_strategy()
            if strategy and hasattr(strategy, 'get_current_position_recommendation'):
                strategy_data = strategy.get_current_position_recommendation(recent_data)
            else:
                signal, strength = self.strategy_manager.get_signal(recent_data)
                strategy_data = {
                    'recommendation': signal,
                    'signal_strength': strength,
                    'reason': f'Signal from {strategy.name if strategy else "unknown strategy"}'
                }
        else:
            strategy_data = {'recommendation': 'hold', 'signal_strength': 0, 'reason': 'No data available'}
        
        # Generate HTML report
        status_snapshot = self.get_status()
        price_history = self.data_handler.get_latest_data(1000)

        html_file = self.html_generator.generate_live_report(
            portfolio_history=self.portfolio_history[-500:],
            strategy_data=strategy_data,
            status=status_snapshot,
            trades=self.live_trades,
            price_history=price_history,
            backtest_results=self.latest_backtest_results,
        )
        
        # Also save to a fixed filename for easy access
        fixed_filename = os.path.join(self.config.HTML_DIR, "live_trading.html")
        self.latest_dashboard_path = html_file
        try:
            with open(html_file, 'r') as source:
                content = source.read()
            with open(fixed_filename, 'w') as target:
                target.write(content)
            self.latest_dashboard_path = fixed_filename
        except Exception as e:
            logger.warning(f"Could not create fixed filename report: {e}")
    
    def get_status(self) -> Dict:
        """Get current bot status"""
        current_price = self.data_handler.get_current_price() or 0
        asset_value = self.asset_balance * current_price
        total_value = self.quote_balance + asset_value + self.bridge_value_quote
        
        status = {
            'is_running': self.is_running,
            'demo_mode': self.demo_mode,
            'current_price': current_price,
            'quote_balance': self.quote_balance,
            'asset_balance': self.asset_balance,
            'asset_value_quote': asset_value,
            'bridge_balance': self.bridge_balance,
            'bridge_value_quote': self.bridge_value_quote,
            'native_reserve_target_dai': float(self.config.MIN_NATIVE_RESERVE_DAI),
            'total_value': total_value,
            'position': self.current_position,
            'active_strategy': self.strategy_manager.active_strategy,
            'available_strategies': self.strategy_manager.list_strategies(),
            'data_points': len(self.data_handler.price_history),
            'last_price_update': self.data_handler.get_last_price_update(),
            'strategy_metadata': self.strategy_metadata
        }

        history_start, history_end = self.data_handler.get_history_bounds()
        status['history_start'] = history_start.isoformat() if history_start else None
        status['history_end'] = history_end.isoformat() if history_end else None
        if history_start and history_end:
            try:
                span_days = (history_end - history_start).total_seconds() / 86400.0
                status['history_days'] = float(round(span_days, 2))
            except Exception:
                status['history_days'] = None
        else:
            status['history_days'] = None

        return status
    
    def get_portfolio_summary(self) -> str:
        """Get formatted portfolio summary"""
        status = self.get_status()
        
        summary = f"""
PulseChain Trading Bot Status
============================
Mode: {'DEMO' if self.demo_mode else 'LIVE'}
Running: {'Yes' if self.is_running else 'No'}

Portfolio:
  {self.config.QUOTE_SYMBOL} Balance: {status['quote_balance']:.4f}
  {self.config.ASSET_SYMBOL} Balance: {status['asset_balance']:.4f}
  {self.config.ASSET_SYMBOL} Value ({self.config.QUOTE_SYMBOL}): {status['asset_value_quote']:.4f}
  {self.config.BRIDGE_SYMBOL} Balance: {status.get('bridge_balance', 0.0):.4f}
  {self.config.BRIDGE_SYMBOL} Value ({self.config.QUOTE_SYMBOL}): {status.get('bridge_value_quote', 0.0):.4f}
  Total Value: {status['total_value']:.4f}
  
Current Position: {status['position'] or 'None'}
Current Price: {status['current_price']:.8f} {self.config.QUOTE_SYMBOL}
Active Strategy: {status['active_strategy']}
Data Points: {status['data_points']}
        """
        
        return summary.strip()

def main():
    """Main function for CLI interface"""
    parser = argparse.ArgumentParser(description='PulseChain Trading Bot - Multi-Asset DEX Trading')
    parser.add_argument('--backtest', action='store_true', help='Run backtest mode')
    parser.add_argument('--live', action='store_true', help='Run live trading mode')
    parser.add_argument('--days', type=int, default=30, help='Days of data for backtesting')
    parser.add_argument('--strategy', type=str, help='Strategy to use')
    parser.add_argument('--demo', action='store_true', default=True, help='Force demo mode')
    parser.add_argument(
        '--start-date',
        type=str,
        help='UTC start datetime for backtest (e.g. 2024-11-01 or 2024-11-01T00:00)',
    )
    parser.add_argument(
        '--end-date',
        type=str,
        help='UTC end datetime for backtest (defaults to now if omitted)',
    )
    
    args = parser.parse_args()
    
    # Create bot instance
    bot = PulseChainTradingBot(demo_mode=args.demo)
    
    try:
        if args.backtest:
            def parse_datetime_arg(value: Optional[str]) -> Optional[datetime]:
                if not value:
                    return None
                parsed = pd.to_datetime(value, utc=True, errors='coerce')
                if pd.isna(parsed):
                    raise ValueError(f"Could not parse datetime '{value}'. Use ISO format like 2024-11-01T00:00")
                return parsed.to_pydatetime()

            try:
                start_dt = parse_datetime_arg(args.start_date)
                end_dt = parse_datetime_arg(args.end_date)
            except ValueError as exc:
                print(f"❌ {exc}")
                return

            if start_dt and end_dt and end_dt <= start_dt:
                print("❌ End date must be after start date")
                return

            print("🚀 Running PulseChain Trading Bot Backtest...")
            if start_dt:
                display_end = end_dt.isoformat() if end_dt else 'now'
                print(f"📊 Range: {start_dt.isoformat()} → {display_end}")
            else:
                print(f"📊 Testing {args.days} days of data")
            
            results = bot.run_backtest(
                days=args.days,
                strategy_name=args.strategy,
                start_date=start_dt,
                end_date=end_dt,
            )
            
            if 'error' in results:
                print(f"❌ Error: {results['error']}")
                return
            
            # Print results
            print("\n" + "="*50)
            print("BACKTEST RESULTS")
            print("="*50)
            print(f"Strategy: {results.get('strategy_name', 'Unknown')}")
            print(f"Initial Balance: {results.get('initial_balance', 0):.4f} DAI")
            print(f"Final Balance: {results.get('final_balance', 0):.4f} DAI")
            print(f"Total Return: {results.get('total_return_pct', 0):+.2f}%")
            print(f"Total Trades: {results.get('total_trades', 0)}")
            print(f"Win Rate: {results.get('win_rate_pct', 0):.1f}%")
            print(f"Max Drawdown: {results.get('max_drawdown_pct', 0):.2f}%")
            print(f"Sharpe Ratio: {results.get('sharpe_ratio', 0):.2f}")
            
            if 'html_report' in results:
                print(f"\n📊 HTML Report: {results['html_report']}")

            print(f"\n🌐 Dashboard UI: http://{bot.dashboard_host}:{bot.dashboard_port}/dashboard")

        elif args.live:
            print("🔥 Starting PulseChain Trading Bot Live Mode...")
            print("⚠️  DEMO MODE ACTIVE - No real trading")
            print("Press Ctrl+C to stop\n")

            print(f"🌐 Dashboard UI: http://{bot.dashboard_host}:{bot.dashboard_port}/dashboard")

            # Start live trading
            bot.start_live_trading()
            
            # Print status updates
            try:
                while bot.is_running:
                    time.sleep(30)  # Update every 30 seconds
                    print("\n" + "="*40)
                    print(bot.get_portfolio_summary())
                    print("="*40)
                    
            except KeyboardInterrupt:
                print("\n\n🛑 Stopping bot...")
                bot.stop_trading()
                time.sleep(2)
                print("✅ Bot stopped")
        
        else:
            # Show status
            print("🤖 PulseChain Trading Bot")
            print(bot.get_portfolio_summary())
            print("\nUse --backtest to run backtesting or --live for live trading")
            print("Use --help for more options")
            print(f"\n🌐 Dashboard UI: http://{bot.dashboard_host}:{bot.dashboard_port}/dashboard")
    
    except KeyboardInterrupt:
        print("\n\nExiting...")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
