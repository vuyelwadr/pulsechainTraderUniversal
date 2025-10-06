"""
HTML Report Generator for HEX Trading Bot
Creates interactive HTML reports for backtesting and live trading results
"""
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
import logging
import os
from bokeh.plotting import figure, save, output_file
from bokeh.models import HoverTool, CrosshairTool, ColumnDataSource, DatetimeTickFormatter
from bokeh.layouts import column, row
from bokeh.embed import file_html, components
from bokeh.resources import CDN

from bot.config import Config

logger = logging.getLogger(__name__)

class HTMLGenerator:
    """Generates HTML reports for trading results"""
    
    def __init__(self):
        self.config = Config()
        os.makedirs(self.config.HTML_DIR, exist_ok=True)
        
    def generate_backtest_report(self, results: Dict, strategy_name: str = None) -> str:
        """Generate comprehensive HTML backtest report"""
        
        strategy_name = strategy_name or results.get('strategy_name', 'Unknown Strategy')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = os.path.join(self.config.HTML_DIR, f"backtest_{strategy_name}_{timestamp}.html")
        
        # Create the HTML content
        html_content = self._create_backtest_html(results, strategy_name)
        
        # Write to file
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Backtest report generated: {filename}")
        return filename
    
    def generate_live_report(
        self,
        portfolio_history: List[Dict],
        strategy_data: Dict,
        status: Dict,
        trades: Optional[List[Dict]] = None,
        price_history: Optional[pd.DataFrame] = None,
        backtest_results: Optional[Dict] = None,
    ) -> str:
        """Generate the unified live dashboard HTML report."""

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = os.path.join(self.config.HTML_DIR, f"dashboard_{timestamp}.html")

        html_content = self._create_dashboard_html(
            portfolio_history=portfolio_history,
            strategy_data=strategy_data,
            status=status,
            trades=trades or [],
            price_history=price_history,
            backtest_results=backtest_results,
        )

        with open(filename, 'w', encoding='utf-8') as handle:
            handle.write(html_content)

        logger.info("Dashboard report generated: %s", filename)
        return filename

    # ------------------------------------------------------------------
    # Dashboard builders
    # ------------------------------------------------------------------

    def _create_dashboard_html(
        self,
        portfolio_history: List[Dict],
        strategy_data: Dict,
        status: Dict,
        trades: List[Dict],
        price_history: Optional[pd.DataFrame],
        backtest_results: Optional[Dict],
    ) -> str:
        """Compose the full dashboard HTML with live + backtest tabs."""

        price_df = self._prepare_price_frame(price_history)
        portfolio_df = self._prepare_portfolio_frame(portfolio_history)
        trades_df = self._prepare_trades_frame(trades)
        backtest_history_df = self._prepare_portfolio_frame(
            backtest_results.get('portfolio_history', []) if backtest_results else []
        )

        price_script, price_div = self._build_price_chart(price_df, trades_df)
        value_script, value_div = self._build_portfolio_chart(portfolio_df)
        backtest_script, backtest_div = self._build_backtest_chart(backtest_history_df)

        live_metrics_html = self._render_live_metrics(status)
        signal_html = self._render_signal_summary(strategy_data)
        strategy_meta_html = self._render_strategy_metadata(status.get('strategy_metadata'))
        trades_html = self._create_trades_table(trades, title="Session Trades")
        trade_log_html = self._render_trade_log(trades_df)

        backtest_controls_html = self._render_backtest_controls(status, backtest_results)
        backtest_metrics_html = self._render_backtest_metrics(backtest_results)
        backtest_trades_html = ""
        if backtest_results and backtest_results.get('trades'):
            backtest_trades_html = self._create_trades_table(
                backtest_results['trades'],
                title="Backtest Trades",
            )

        bokeh_scripts = "\n".join(
            [segment for segment in (price_script, value_script, backtest_script) if segment]
        )

        resources = CDN.render()

        backtest_tab_html = f"""
        <div class="tab-content" id="tab-backtest">
            {backtest_controls_html}
            {backtest_metrics_html}
            <div class="chart-card full-width">
                {backtest_div}
            </div>
            {backtest_trades_html}
        </div>
        """

        live_tab_html = f"""
        <div class="tab-content active" id="tab-live">
            <div class="live-grid">
                <div class="metrics-grid">
                    {live_metrics_html}
                </div>
                {signal_html}
            </div>
            <div class="chart-grid">
                <div class="chart-card">
                    <h3>HEX/DAI Price (Live)</h3>
                    {price_div}
                </div>
                <div class="chart-card">
                    <h3>Portfolio Total Value</h3>
                    {value_div}
                </div>
            </div>
            {strategy_meta_html}
            {trades_html}
        </div>
        """

        trade_tab_html = f"""
        <div class="tab-content" id="tab-trades">
            {trade_log_html}
        </div>
        """

        backtest_nav_disabled = 'disabled' if not backtest_results else ''
        trades_nav_disabled = 'disabled' if trades_df.empty else ''

        html_template = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>HEX Trading Bot Dashboard</title>
            {resources}
            <style>
                body {{
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    margin: 0 auto;
                    max-width: 1280px;
                    padding: 24px;
                    line-height: 1.6;
                    background: linear-gradient(180deg, #f5f7fb 0%, #ffffff 100%);
                    color: #212529;
                }}

                .dashboard-header {{
                    background: linear-gradient(135deg, #5b8def 0%, #764ba2 100%);
                    color: #ffffff;
                    padding: 28px;
                    border-radius: 16px;
                    box-shadow: 0 12px 24px rgba(118,75,162,0.2);
                    margin-bottom: 24px;
                }}

                .dashboard-header h1 {{
                    margin: 0;
                    font-size: 2.2rem;
                }}

                .dashboard-header p {{
                    margin-top: 12px;
                    opacity: 0.85;
                }}

                .tabs {{
                    display: flex;
                    gap: 12px;
                    margin-bottom: 24px;
                }}

                .tab-button {{
                    background: #ffffff;
                    border-radius: 999px;
                    border: 1px solid #ced4f5;
                    padding: 10px 24px;
                    font-weight: 600;
                    cursor: pointer;
                    transition: all 0.2s ease-in-out;
                    color: #495057;
                }}

                .tab-button.active {{
                    background: linear-gradient(135deg, #5b8def 0%, #764ba2 100%);
                    color: #ffffff;
                    border-color: transparent;
                    box-shadow: 0 10px 20px rgba(118,75,162,0.2);
                }}

                .tab-button.disabled {{
                    opacity: 0.45;
                    cursor: not-allowed;
                }}

                .tab-content {{
                    display: none;
                    animation: fadeIn 0.25s ease-in-out;
                }}

                .tab-content.active {{
                    display: block;
                }}

                .metrics-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
                    gap: 16px;
                }}

                .metric-card {{
                    background: #ffffff;
                    border-radius: 16px;
                    padding: 20px;
                    box-shadow: 0 8px 18px rgba(91, 141, 239, 0.12);
                    border: 1px solid rgba(91, 141, 239, 0.08);
                }}

                .metric-label {{
                    font-size: 0.85rem;
                    text-transform: uppercase;
                    letter-spacing: 0.08em;
                    color: #6c757d;
                }}

                .metric-value {{
                    font-size: 1.7rem;
                    font-weight: 600;
                    margin-top: 8px;
                }}

                .chart-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                    gap: 20px;
                    margin-top: 24px;
                }}

                .chart-card {{
                    background: #ffffff;
                    border-radius: 16px;
                    padding: 18px;
                    box-shadow: 0 8px 20px rgba(0,0,0,0.06);
                    border: 1px solid rgba(0,0,0,0.05);
                }}

                .chart-card h3 {{
                    margin-top: 0;
                    margin-bottom: 12px;
                    color: #495057;
                }}

                .chart-card.full-width {{
                    margin-top: 24px;
                }}

                .live-grid {{
                    display: grid;
                    grid-template-columns: minmax(0, 1fr) minmax(260px, 320px);
                    gap: 20px;
                    align-items: stretch;
                    flex-wrap: wrap;
                }}

                .signal-card {{
                    background: linear-gradient(135deg, #20c997 0%, #38d9a9 100%);
                    border-radius: 16px;
                    padding: 20px;
                    color: #0b3d3a;
                    box-shadow: 0 12px 24px rgba(32, 201, 151, 0.25);
                }}

                .signal-card h3 {{
                    margin-top: 0;
                    color: #0b3d3a;
                }}

                .signal-value {{
                    font-size: 1.8rem;
                    font-weight: 700;
                    margin: 12px 0 4px 0;
                }}

                .signal-meta {{
                    font-size: 0.9rem;
                    opacity: 0.85;
                }}

                .backtest-form {{
                    margin-top: 12px;
                }}

                .form-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
                    gap: 16px;
                }}

                .form-field label {{
                    display: block;
                    font-size: 0.85rem;
                    text-transform: uppercase;
                    letter-spacing: 0.05em;
                    margin-bottom: 6px;
                    color: #495057;
                }}

                .form-field input,
                .form-field select {{
                    width: 100%;
                    padding: 10px 12px;
                    border-radius: 8px;
                    border: 1px solid #ced4da;
                    font-size: 0.95rem;
                    background: #ffffff;
                    box-sizing: border-box;
                }}

                .form-actions {{
                    display: flex;
                    flex-wrap: wrap;
                    gap: 12px;
                    margin-top: 16px;
                }}

                .primary-button {{
                    background: linear-gradient(135deg, #5b8def 0%, #764ba2 100%);
                    color: #ffffff;
                    border: none;
                    padding: 10px 20px;
                    border-radius: 999px;
                    font-weight: 600;
                    cursor: pointer;
                    box-shadow: 0 8px 16px rgba(91,141,239,0.2);
                    transition: transform 0.15s ease-in-out;
                }}

                .primary-button:hover {{
                    transform: translateY(-1px);
                }}

                .secondary-button {{
                    background: transparent;
                    border: 1px solid #ced4da;
                    color: #495057;
                    padding: 10px 18px;
                    border-radius: 999px;
                    font-weight: 500;
                    cursor: pointer;
                }}

                .status-message {{
                    margin-top: 12px;
                    font-size: 0.95rem;
                    color: #495057;
                }}

                .status-message.success {{
                    color: #20a464;
                }}

                .status-message.error {{
                    color: #d64545;
                }}

                .section {{
                    background: #ffffff;
                    border-radius: 16px;
                    padding: 20px;
                    box-shadow: 0 8px 18px rgba(0,0,0,0.05);
                    border: 1px solid rgba(0,0,0,0.05);
                    margin-top: 24px;
                }}

                .section h2, .section h3 {{
                    margin-top: 0;
                    color: #343a40;
                }}

                .table-wrapper {{
                    overflow-x: auto;
                }}

                table.trades-table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin-top: 12px;
                }}

                table.trades-table th,
                table.trades-table td {{
                    padding: 10px 12px;
                    border-bottom: 1px solid rgba(0,0,0,0.06);
                    text-align: left;
                    font-size: 0.95rem;
                }}

                table.trades-table th {{
                    background: #f8f9fc;
                    color: #495057;
                    text-transform: uppercase;
                    letter-spacing: 0.05em;
                    font-size: 0.8rem;
                }}

                table.trades-table tr:hover {{
                    background: rgba(91,141,239,0.08);
                }}

                .positive {{ color: #20a464; }}
                .negative {{ color: #d64545; }}
                .neutral {{ color: #6c757d; }}

                .empty-state {{
                    color: #6c757d;
                    font-style: italic;
                    padding: 12px 0;
                }}

                @keyframes fadeIn {{
                    from {{ opacity: 0; transform: translateY(8px); }}
                    to {{ opacity: 1; transform: translateY(0); }}
                }}

                @media (max-width: 900px) {{
                    .live-grid {{
                        grid-template-columns: 1fr;
                    }}
                }}
            </style>
        </head>
        <body>
            <div class="dashboard-header">
                <h1>HEX Trading Bot Dashboard</h1>
                <p>Real HEX/DAI data feed · Demo execution · Strategy insights</p>
            </div>

            <div class="tabs">
                <button class="tab-button active" data-tab="live">Live Session</button>
                <button class="tab-button {backtest_nav_disabled}" data-tab="backtest">Latest Backtest</button>
                <button class="tab-button {trades_nav_disabled}" data-tab="trades">Trade Log</button>
            </div>

            <section class="tab-wrapper">
                {live_tab_html}
                {backtest_tab_html}
                {trade_tab_html}
            </section>

            {bokeh_scripts}

            <script>
                const tabButtons = Array.from(document.querySelectorAll('.tab-button'));
                const tabContents = Array.from(document.querySelectorAll('.tab-content'));

                tabButtons.forEach(button => {{
                    if (button.classList.contains('disabled')) {{
                        return;
                    }}
                    button.addEventListener('click', () => {{
                        const target = button.dataset.tab;
                        tabButtons.forEach(btn => btn.classList.toggle('active', btn === button));
                        tabContents.forEach(content => {{
                            const isMatch = content.id === `tab-${{target}}`;
                            content.classList.toggle('active', isMatch);
                        }});
                    }});
                }});

                const backtestForm = document.getElementById('backtest-form');
                const backtestStatus = document.getElementById('backtest-status');
                const resetButton = document.getElementById('backtest-reset');
                const fullToggle = document.getElementById('backtest-full');
                const fullButton = document.getElementById('backtest-full-btn');

                function updateStatus(message, variant) {{
                    if (!backtestStatus) return;
                    backtestStatus.textContent = message || '';
                    backtestStatus.className = 'status-message' + (variant ? ' ' + variant : '');
                }}

                if (resetButton && backtestForm) {{
                    resetButton.addEventListener('click', () => {{
                        backtestForm.reset();
                        if (fullToggle) {{
                            fullToggle.value = 'false';
                        }}
                        updateStatus('', null);
                    }});
                }}

                if (fullButton && backtestForm) {{
                    fullButton.addEventListener('click', () => {{
                        if (fullToggle) {{
                            fullToggle.value = 'true';
                        }}
                        backtestForm.requestSubmit(fullButton);
                    }});
                }}

                if (backtestForm) {{
                    backtestForm.addEventListener('submit', async (event) => {{
                        event.preventDefault();
                        const submitter = event.submitter;
                        const isFullDataset = submitter && submitter.id === 'backtest-full-btn';
                        if (fullToggle) {{
                            fullToggle.value = isFullDataset ? 'true' : 'false';
                        }}
                        const formData = new FormData(backtestForm);
                        const payload = {{
                            start_date: formData.get('start_date') || null,
                            end_date: formData.get('end_date') || null,
                            days: formData.get('days') ? Number(formData.get('days')) : null,
                            strategy: formData.get('strategy') || null,
                            full_dataset: isFullDataset,
                        }};

                        updateStatus('Running backtest…', '');
                        try {{
                            const response = await fetch('/api/backtest', {{
                                method: 'POST',
                                headers: {{ 'Content-Type': 'application/json' }},
                                body: JSON.stringify(payload),
                            }});
                            const data = await response.json();
                            if (!response.ok || data.error) {{
                                throw new Error(data.error || 'Backtest failed');
                            }}
                            updateStatus('Backtest complete! Refreshing dashboard…', 'success');
                            setTimeout(() => window.location.reload(), 1500);
                        }} catch (error) {{
                            updateStatus(`Error: ${{error.message}}`, 'error');
                        }}
                    }});
                }}
            </script>
        </body>
        </html>
        """

        return html_template

    # ------------------------------------------------------------------
    # Data preparation helpers
    # ------------------------------------------------------------------

    def _prepare_price_frame(self, price_history: Optional[pd.DataFrame]) -> pd.DataFrame:
        if price_history is None:
            return pd.DataFrame()

        if isinstance(price_history, pd.DataFrame):
            df = price_history.copy()
        else:
            df = pd.DataFrame(price_history)

        if df.empty or 'timestamp' not in df.columns:
            return pd.DataFrame()

        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
        df.dropna(subset=['timestamp'], inplace=True)

        price_col = None
        for candidate in ('price', 'close', 'Price'):
            if candidate in df.columns:
                price_col = candidate
                break

        if price_col is None:
            return pd.DataFrame()

        if price_col != 'price':
            df['price'] = df[price_col]

        df.sort_values('timestamp', inplace=True)
        df = df.tail(1500).reset_index(drop=True)
        return df[['timestamp', 'price']]

    def _prepare_portfolio_frame(self, history: List[Dict]) -> pd.DataFrame:
        df = pd.DataFrame(history)
        if df.empty or 'timestamp' not in df.columns:
            return pd.DataFrame()

        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
        df.dropna(subset=['timestamp'], inplace=True)

        if 'total_value' not in df.columns:
            quote = df.get('quote_balance', pd.Series(dtype=float))
            hex_value = df.get('hex_value_quote', pd.Series(dtype=float))
            df['total_value'] = quote.fillna(0) + hex_value.fillna(0)

        if 'price' not in df.columns and 'close' in df.columns:
            df['price'] = df['close']

        df.sort_values('timestamp', inplace=True)
        df = df.tail(1500).reset_index(drop=True)
        return df

    def _prepare_trades_frame(self, trades: List[Dict]) -> pd.DataFrame:
        df = pd.DataFrame(trades)
        if df.empty or 'timestamp' not in df.columns or 'price' not in df.columns:
            return pd.DataFrame()

        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
        df.dropna(subset=['timestamp'], inplace=True)
        df.sort_values('timestamp', inplace=True)
        return df.tail(500).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Chart builders
    # ------------------------------------------------------------------

    def _build_price_chart(
        self,
        price_df: pd.DataFrame,
        trades_df: pd.DataFrame,
    ) -> (str, str):
        if price_df.empty:
            return "", "<div class=\"empty-state\">Waiting for live price data...</div>"

        source = ColumnDataSource(price_df)
        fig = figure(
            height=320,
            sizing_mode='stretch_width',
            x_axis_type='datetime',
            tools="pan,wheel_zoom,box_zoom,reset,save",
        )
        fig.line('timestamp', 'price', source=source, line_width=2, color="#764ba2")
        fig.title.text = ""
        fig.background_fill_color = "#ffffff"
        fig.xaxis.formatter = DatetimeTickFormatter(
            minutes="%b %d %H:%M",
            hours="%b %d %H:%M",
            days="%b %d",
            months="%b %Y",
        )

        hover = HoverTool(
            tooltips=[
                ("Time", "@timestamp{%F %T UTC}"),
                ("Price", "@price{0.00000000}"),
            ],
            formatters={'@timestamp': 'datetime'},
            mode='vline'
        )
        fig.add_tools(hover, CrosshairTool())

        if not trades_df.empty:
            buys = trades_df[trades_df['type'].str.lower() == 'buy']
            sells = trades_df[trades_df['type'].str.lower() == 'sell']

            if not buys.empty:
                buy_source = ColumnDataSource(buys)
                fig.triangle(
                    'timestamp',
                    'price',
                    source=buy_source,
                    size=10,
                    color="#20a464",
                    legend_label="Buy",
                    alpha=0.9,
                )

            if not sells.empty:
                sell_source = ColumnDataSource(sells)
                fig.inverted_triangle(
                    'timestamp',
                    'price',
                    source=sell_source,
                    size=10,
                    color="#d64545",
                    legend_label="Sell",
                    alpha=0.9,
                )

            fig.legend.location = "top_left"
            fig.legend.click_policy = "hide"

        script, div = components(fig)
        return script, div

    def _build_portfolio_chart(self, portfolio_df: pd.DataFrame) -> (str, str):
        if portfolio_df.empty or 'total_value' not in portfolio_df.columns:
            return "", "<div class=\"empty-state\">Portfolio has not recorded activity yet.</div>"

        source = ColumnDataSource(portfolio_df[['timestamp', 'total_value']])
        fig = figure(
            height=320,
            sizing_mode='stretch_width',
            x_axis_type='datetime',
            tools="pan,wheel_zoom,box_zoom,reset,save",
        )
        fig.line('timestamp', 'total_value', source=source, line_width=2, color="#20c997")
        fig.background_fill_color = "#ffffff"
        fig.xaxis.formatter = DatetimeTickFormatter(
            minutes="%b %d %H:%M",
            hours="%b %d %H:%M",
            days="%b %d",
            months="%b %Y",
        )
        fig.add_tools(
            HoverTool(
                tooltips=[
                    ("Time", "@timestamp{%F %T UTC}"),
                    ("Total Value", "@total_value{0,0.00}"),
                ],
                formatters={'@timestamp': 'datetime'},
                mode='vline'
            ),
            CrosshairTool(),
        )

        script, div = components(fig)
        return script, div

    def _build_backtest_chart(self, backtest_df: pd.DataFrame) -> (str, str):
        if backtest_df.empty or 'total_value' not in backtest_df.columns:
            return "", "<div class=\"empty-state\">Run a backtest to see performance charts.</div>"

        data = backtest_df[['timestamp', 'total_value']].copy()
        data.rename(columns={'total_value': 'equity'}, inplace=True)
        source = ColumnDataSource(data)

        fig = figure(
            height=340,
            sizing_mode='stretch_width',
            x_axis_type='datetime',
            tools="pan,wheel_zoom,box_zoom,reset,save",
        )
        fig.line('timestamp', 'equity', source=source, line_width=2.2, color="#ff922b")
        fig.background_fill_color = "#ffffff"
        fig.xaxis.formatter = DatetimeTickFormatter(
            hours="%b %d %H:%M",
            days="%b %d",
            months="%b %Y",
        )
        fig.add_tools(
            HoverTool(
                tooltips=[
                    ("Time", "@timestamp{%F %T UTC}"),
                    ("Equity", "@equity{0,0.00}"),
                ],
                formatters={'@timestamp': 'datetime'},
                mode='vline'
            ),
            CrosshairTool(),
        )

        script, div = components(fig)
        return script, div

    # ------------------------------------------------------------------
    # HTML section helpers
    # ------------------------------------------------------------------

    def _render_live_metrics(self, status: Dict) -> str:
        if not status:
            return "<p class=\"empty-state\">Status unavailable.</p>"

        def fmt(value, decimals=2):
            try:
                return f"{float(value):,.{decimals}f}"
            except (TypeError, ValueError):
                return "--"

        cards = []
        cards.append(self._metric_card("Total Value", f"{fmt(status.get('total_value'), 2)} DAI"))
        cards.append(self._metric_card("Quote Balance", f"{fmt(status.get('quote_balance'), 2)} DAI"))
        cards.append(self._metric_card("HEX Balance", f"{fmt(status.get('hex_balance'), 4)} HEX"))
        cards.append(self._metric_card("HEX Value", f"{fmt(status.get('hex_value_quote'), 2)} DAI"))
        cards.append(self._metric_card("Current Price", f"{fmt(status.get('current_price'), 6)} DAI"))
        cards.append(self._metric_card("Last Update", self._format_timestamp(status.get('last_price_update'))))
        cards.append(self._metric_card("Active Strategy", status.get('active_strategy', '--')))
        cards.append(self._metric_card("Position", (status.get('position') or 'Flat').title()))
        cards.append(self._metric_card("PLS Balance", f"{fmt(status.get('native_balance'), 4)} PLS"))
        cards.append(self._metric_card("PLS Reserve", f"{fmt(status.get('native_value_quote'), 2)} DAI / {fmt(status.get('native_reserve_target_dai'), 2)} DAI"))

        return "".join(cards)

    def _render_signal_summary(self, strategy_data: Dict) -> str:
        if not strategy_data:
            return "<div class=\"signal-card\"><h3>No Signal</h3><p class=\"signal-meta\">Strategy has not produced a recommendation yet.</p></div>"

        recommendation = strategy_data.get('recommendation', 'hold').upper()
        strength = strategy_data.get('signal_strength')
        reason = strategy_data.get('reason', '')
        strength_display = f"Strength: {strength:.2f}" if isinstance(strength, (int, float)) else "Strength: --"

        return f"""
        <div class="signal-card">
            <h3>Strategy Signal</h3>
            <div class="signal-value">{recommendation}</div>
            <div class="signal-meta">{strength_display}</div>
            <p class="signal-meta">{reason}</p>
        </div>
        """

    def _render_strategy_metadata(self, metadata: Optional[Dict]) -> str:
        if not metadata or 'best_strategy' not in metadata:
            return ""

        best = metadata['best_strategy']
        rows = []
        for label, key in (
            ("Strategy", 'name'),
            ("Timeframe (min)", 'timeframe_minutes'),
            ("Objective", 'objective'),
            ("Total Return %", 'total_return_pct'),
            ("Max Drawdown %", 'max_drawdown_pct'),
            ("Sharpe", 'sharpe_ratio'),
            ("Total Trades", 'total_trades'),
            ("Source", 'source_file'),
            ("Label", 'label'),
        ):
            value = best.get(key)
            if value is None or value == "":
                continue
            if isinstance(value, float):
                if 'return' in key or 'drawdown' in key or 'sharpe' in key:
                    value = f"{value:.2f}"
            rows.append(f"<tr><td>{label}</td><td>{value}</td></tr>")

        if not rows:
            return ""

        return f"""
        <div class="section">
            <h3>Top Strategy Context</h3>
            <p>Loaded automatically from optimizer output. Update `reports/gridTradeAggressive/visualization/01_gridAggressive_top.json` to change.</p>
            <table class="trades-table">
                <tbody>
                    {''.join(rows)}
                </tbody>
            </table>
        </div>
        """

    def _render_backtest_metrics(self, backtest_results: Optional[Dict]) -> str:
        if not backtest_results or backtest_results.get('error'):
            return "<div class=\"section\"><h3>Latest Backtest</h3><p class=\"empty-state\">No completed backtest yet.</p></div>"

        def fmt(value, decimals=2):
            try:
                return f"{float(value):,.{decimals}f}"
            except (TypeError, ValueError):
                return "--"

        metrics = []
        metrics.append(self._metric_card("Strategy", backtest_results.get('strategy_name', backtest_results.get('strategy'))))
        metrics.append(self._metric_card("Initial Balance", f"{fmt(backtest_results.get('initial_balance'), 2)} DAI"))
        metrics.append(self._metric_card("Final Balance", f"{fmt(backtest_results.get('final_balance'), 2)} DAI"))
        metrics.append(self._metric_card("Total Return", f"{fmt(backtest_results.get('total_return_pct'), 2)} %"))
        metrics.append(self._metric_card("Win Rate", f"{fmt(backtest_results.get('win_rate_pct'), 1)} %"))
        metrics.append(self._metric_card("Max Drawdown", f"{fmt(backtest_results.get('max_drawdown_pct'), 2)} %"))
        metrics.append(self._metric_card("Sharpe", fmt(backtest_results.get('sharpe_ratio'), 2)))
        metrics.append(self._metric_card("Trades", fmt(backtest_results.get('total_trades'), 0)))

        range_info = backtest_results.get('requested_range', {}) if isinstance(backtest_results, dict) else {}
        range_str = ""
        if range_info and (range_info.get('start') or range_info.get('end')):
            start_str = self._format_timestamp(range_info.get('start')) if range_info.get('start') else "--"
            end_str = self._format_timestamp(range_info.get('end')) if range_info.get('end') else "--"
            range_str = f"<p class=\"signal-meta\">Requested range: {start_str} → {end_str}</p>"
        if range_info and range_info.get('days'):
            try:
                trailing_days = int(range_info['days'])
            except (TypeError, ValueError):
                trailing_days = None
            if trailing_days:
                supplement = f"<p class=\"signal-meta\">Trailing window: {trailing_days} days</p>"
                range_str = f"{range_str}{supplement}" if range_str else supplement

        report_link = ""
        if backtest_results.get('html_report'):
            report_link = f"<a href=\"{backtest_results['html_report']}\" target=\"_blank\">Open detailed backtest report</a>"

        return f"""
        <div class="section">
            <h3>Latest Backtest</h3>
            {range_str}
            <div class="metrics-grid">
                {''.join(metrics)}
            </div>
            <p>{report_link}</p>
        </div>
        """

    def _render_backtest_controls(self, status: Dict, backtest_results: Optional[Dict]) -> str:
        strategies = (status or {}).get('available_strategies', []) if status else []
        active = (status or {}).get('active_strategy') if status else None
        options_html = ["<option value=\"\">Automatic (active strategy)</option>"]
        for name in strategies:
            selected = " selected" if name == active else ""
            options_html.append(f"<option value=\"{name}\"{selected}>{name}</option>")

        last_range = (backtest_results or {}).get('requested_range', {}) if backtest_results else {}
        start_value = ''
        end_value = ''
        days_value = ''
        data_min_attr = ''
        data_max_attr = ''
        dataset_note = ''
        dataset_start_raw = (status or {}).get('history_start') if status else None
        dataset_end_raw = (status or {}).get('history_end') if status else None
        dataset_span_days = (status or {}).get('history_days') if status else None
        if dataset_start_raw:
            try:
                dt = pd.to_datetime(dataset_start_raw)
                data_min_attr = dt.strftime('%Y-%m-%dT%H:%M')
            except Exception:
                data_min_attr = ''
        if dataset_end_raw:
            try:
                dt = pd.to_datetime(dataset_end_raw)
                data_max_attr = dt.strftime('%Y-%m-%dT%H:%M')
            except Exception:
                data_max_attr = ''
        if dataset_start_raw and dataset_end_raw:
            try:
                start_disp = pd.to_datetime(dataset_start_raw).strftime('%Y-%m-%d %H:%M UTC')
            except Exception:
                start_disp = str(dataset_start_raw)
            try:
                end_disp = pd.to_datetime(dataset_end_raw).strftime('%Y-%m-%d %H:%M UTC')
            except Exception:
                end_disp = str(dataset_end_raw)
            span_text = f" (~{dataset_span_days} days)" if dataset_span_days else ""
            dataset_note = f"Available data range: {start_disp} → {end_disp}{span_text}."

        if last_range:
            if last_range.get('start'):
                try:
                    start_value = pd.to_datetime(last_range['start']).strftime('%Y-%m-%dT%H:%M')
                except Exception:
                    start_value = ''
            if last_range.get('end'):
                try:
                    end_value = pd.to_datetime(last_range['end']).strftime('%Y-%m-%dT%H:%M')
                except Exception:
                    end_value = ''
            if last_range.get('days') is not None:
                try:
                    days_value = str(int(last_range['days']))
                except (TypeError, ValueError):
                    days_value = ''
        else:
            if not start_value and data_min_attr:
                start_value = data_min_attr
            if not end_value and data_max_attr:
                end_value = data_max_attr

        default_days = getattr(self.config, 'BACKTEST_DAYS', 30)

        return f"""
        <div class="section">
            <h3>Run On-Demand Backtest</h3>
            <p class="signal-meta">
                Provide a custom date range or specify a trailing number of days. Leave fields blank to reuse defaults. {dataset_note}
            </p>
            <form id="backtest-form" class="backtest-form">
                <input type="hidden" id="backtest-full" name="full_dataset" value="false">
                <div class="form-grid">
                    <div class="form-field">
                        <label for="backtest-start">Start Date (UTC)</label>
                        <input type="datetime-local" id="backtest-start" name="start_date" value="{start_value}" {'min="' + data_min_attr + '"' if data_min_attr else ''} {'max="' + data_max_attr + '"' if data_max_attr else ''}>
                    </div>
                    <div class="form-field">
                        <label for="backtest-end">End Date (UTC)</label>
                        <input type="datetime-local" id="backtest-end" name="end_date" value="{end_value}" {'min="' + data_min_attr + '"' if data_min_attr else ''} {'max="' + data_max_attr + '"' if data_max_attr else ''}>
                    </div>
                    <div class="form-field">
                        <label for="backtest-days">Look-back Days</label>
                        <input type="number" min="1" max="1095" id="backtest-days" name="days" placeholder="{default_days}" value="{days_value}">
                    </div>
                    <div class="form-field">
                        <label for="backtest-strategy">Strategy</label>
                        <select id="backtest-strategy" name="strategy">
                            {''.join(options_html)}
                        </select>
                    </div>
                </div>
                <div class="form-actions">
                    <button type="submit" class="primary-button" id="backtest-submit">Run Backtest</button>
                    <button type="button" class="secondary-button" id="backtest-full-btn">Full Dataset</button>
                    <button type="button" class="secondary-button" id="backtest-reset">Reset</button>
                </div>
            </form>
            <p id="backtest-status" class="status-message"></p>
        </div>
        """

    def _render_trade_log(self, trades_df: pd.DataFrame) -> str:
        if trades_df is None or trades_df.empty:
            return "<div class=\"section\"><h3>Comprehensive Trade Log</h3><p class=\"empty-state\">No trades recorded for this session.</p></div>"

        display_df = trades_df.copy()
        display_df['timestamp'] = display_df['timestamp'].apply(self._format_timestamp)

        columns = [
            ("Timestamp", 'timestamp', None),
            ("Type", 'type', None),
            ("Price (DAI)", 'price', 8),
            ("HEX Amount", 'hex_amount', 4),
            ("Quote Amount (DAI)", 'quote_amount', 4),
            ("Fee (DAI)", 'fee', 6),
            ("Slippage %", 'slippage_pct', 2),
            ("Signal Strength", 'signal_strength', 2),
            ("P&L %", 'pnl_pct', 2),
            ("Portfolio Value (DAI)", 'portfolio_value_after', 2),
        ]

        def fmt(value, decimals):
            if value in (None, "", "--"):
                return "--"
            try:
                if pd.isna(value):
                    return "--"
            except Exception:
                pass
            try:
                if decimals is None:
                    return str(value)
                return f"{float(value):,.{decimals}f}"
            except (TypeError, ValueError):
                return str(value)

        rows_html = []
        for _, row in display_df.iterrows():
            cells = []
            for label, key, decimals in columns:
                raw_value = row.get(key, "--")
                if key == 'type':
                    cell_value = str(raw_value).upper()
                    css_class = 'positive' if cell_value == 'BUY' else 'negative' if cell_value == 'SELL' else 'neutral'
                    cells.append(f"<td><span class=\"{css_class}\">{cell_value}</span></td>")
                    continue
                cell_value = fmt(raw_value, decimals)
                cells.append(f"<td>{cell_value}</td>")
            rows_html.append(f"<tr>{''.join(cells)}</tr>")

        header_html = ''.join(f"<th>{label}</th>" for label, _, _ in columns)

        return f"""
        <div class="section">
            <h3>Comprehensive Trade Log</h3>
            <p class="signal-meta">Showing {len(display_df)} trades kept in memory (max 500 most recent).</p>
            <div class="table-wrapper">
                <table class="trades-table">
                    <thead>
                        <tr>{header_html}</tr>
                    </thead>
                    <tbody>
                        {''.join(rows_html)}
                    </tbody>
                </table>
            </div>
        </div>
        """

    def _metric_card(self, label: str, value: str) -> str:
        return f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """

    def _format_timestamp(self, value: Optional[object]) -> str:
        if value in (None, ""):
            return "--"
        try:
            ts = pd.to_datetime(value, utc=True, errors='coerce')
        except Exception:
            return str(value)
        if ts is None or pd.isna(ts):
            return str(value)
        return ts.strftime('%Y-%m-%d %H:%M:%S UTC')

    def _create_backtest_html(self, results: Dict, strategy_name: str) -> str:
        """Create HTML content for backtest results"""
        
        # Extract key metrics
        total_return = results.get('total_return_pct', 0)
        win_rate = results.get('win_rate_pct', 0)
        max_drawdown = results.get('max_drawdown_pct', 0)
        sharpe_ratio = results.get('sharpe_ratio', 0)
        total_trades = results.get('total_trades', 0)
        
        # Generate charts if we have portfolio history
        charts_html = ""
        if results.get('portfolio_history'):
            charts_html = self._create_portfolio_charts(results['portfolio_history'])
        
        # Generate trade table
        trades_html = self._create_trades_table(results.get('trades', []), title="Backtest Trade History")
        
        html_template = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>HEX Trading Bot - Backtest Results</title>
            <style>
                body {{ 
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                    background-color: #f8f9fa;
                }}
                
                .header {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 30px;
                    border-radius: 12px;
                    margin-bottom: 30px;
                    text-align: center;
                }}
                
                .header h1 {{ margin: 0; font-size: 2.5em; }}
                .header p {{ margin: 10px 0 0 0; opacity: 0.9; }}
                
                .metrics-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                    gap: 20px;
                    margin-bottom: 30px;
                }}
                
                .metric-card {{
                    background: white;
                    padding: 25px;
                    border-radius: 12px;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                    text-align: center;
                    transition: transform 0.2s;
                }}
                
                .metric-card:hover {{ transform: translateY(-2px); }}
                
                .metric-value {{
                    font-size: 2.5em;
                    font-weight: bold;
                    margin: 10px 0;
                }}
                
                .metric-label {{
                    color: #666;
                    font-size: 0.9em;
                    text-transform: uppercase;
                    letter-spacing: 1px;
                }}
                
                .positive {{ color: #28a745; }}
                .negative {{ color: #dc3545; }}
                .neutral {{ color: #6c757d; }}
                
                .section {{
                    background: white;
                    padding: 30px;
                    border-radius: 12px;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                    margin-bottom: 30px;
                }}
                
                .section h2 {{
                    margin-top: 0;
                    color: #495057;
                    border-bottom: 2px solid #e9ecef;
                    padding-bottom: 10px;
                }}
                
                .trades-table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin-top: 20px;
                }}
                
                .trades-table th, .trades-table td {{
                    padding: 12px;
                    text-align: left;
                    border-bottom: 1px solid #dee2e6;
                }}
                
                .trades-table th {{
                    background-color: #f8f9fa;
                    font-weight: 600;
                    color: #495057;
                }}
                
                .trades-table tr:hover {{
                    background-color: #f8f9fa;
                }}
                
                .timestamp {{
                    color: #6c757d;
                    font-size: 0.9em;
                }}
                
                .chart-container {{
                    margin: 20px 0;
                    text-align: center;
                }}
                
                .footer {{
                    text-align: center;
                    color: #6c757d;
                    margin-top: 40px;
                    padding-top: 20px;
                    border-top: 1px solid #dee2e6;
                }}
                
                .refresh-notice {{
                    background: #e7f3ff;
                    border: 1px solid #b6e0ff;
                    border-radius: 8px;
                    padding: 15px;
                    margin-bottom: 20px;
                    text-align: center;
                }}
            </style>
            <meta http-equiv="refresh" content="30">
        </head>
        <body>
            <div class="header">
                <h1>🚀 HEX Trading Bot</h1>
                <p>Backtest Results - {strategy_name}</p>
                <p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="refresh-notice">
                📊 This report auto-refreshes every 30 seconds
            </div>
            
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-label">Total Return</div>
                    <div class="metric-value {'positive' if total_return > 0 else 'negative' if total_return < 0 else 'neutral'}">
                        {total_return:+.2f}%
                    </div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-label">Win Rate</div>
                    <div class="metric-value {'positive' if win_rate > 50 else 'negative' if win_rate < 50 else 'neutral'}">
                        {win_rate:.1f}%
                    </div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-label">Max Drawdown</div>
                    <div class="metric-value negative">
                        -{max_drawdown:.2f}%
                    </div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-label">Sharpe Ratio</div>
                    <div class="metric-value {'positive' if sharpe_ratio > 1 else 'neutral' if sharpe_ratio > 0 else 'negative'}">
                        {sharpe_ratio:.2f}
                    </div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-label">Total Trades</div>
                    <div class="metric-value neutral">
                        {total_trades}
                    </div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-label">Final Balance</div>
                    <div class="metric-value {'positive' if results.get('final_balance', 0) > results.get('initial_balance', 0) else 'negative'}">
                        {results.get('final_balance', 0):.4f} DAI
                    </div>
                </div>
            </div>
            
            {charts_html}
            
            <div class="section">
                <h2>📊 Detailed Results</h2>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 30px;">
                    <div>
                        <h3>Performance Metrics</h3>
                        <p><strong>Initial Balance:</strong> {results.get('initial_balance', 0):.4f} DAI</p>
                        <p><strong>Final Balance:</strong> {results.get('final_balance', 0):.4f} DAI</p>
                        <p><strong>Total Fees:</strong> {results.get('total_fees', 0):.6f} DAI</p>
                        <p><strong>Volatility:</strong> {results.get('volatility_pct', 0):.2f}%</p>
                        <p><strong>Profit Factor:</strong> {results.get('profit_factor', 0):.2f}</p>
                    </div>
                    
                    <div>
                        <h3>Trading Statistics</h3>
                        <p><strong>Buy Trades:</strong> {results.get('buy_trades', 0)}</p>
                        <p><strong>Sell Trades:</strong> {results.get('sell_trades', 0)}</p>
                        <p><strong>Profitable Trades:</strong> {results.get('profitable_trades', 0)}</p>
                        <p><strong>Losing Trades:</strong> {results.get('losing_trades', 0)}</p>
                        <p><strong>Average Win:</strong> {results.get('avg_win_pct', 0):.2f}%</p>
                        <p><strong>Average Loss:</strong> {results.get('avg_loss_pct', 0):.2f}%</p>
                    </div>
                </div>
            </div>
            
            {trades_html}
            
            <div class="footer">
                <p>📈 HEX Trading Bot | PulseChain Automated Trading</p>
                <p>⚠️ This is for educational purposes only. Past performance does not guarantee future results.</p>
            </div>
        </body>
        </html>
        """
        
        return html_template
    
    def _create_portfolio_charts(self, portfolio_history: List[Dict]) -> str:
        """Create portfolio value charts using simple HTML/CSS (fallback)"""
        if not portfolio_history:
            return ""
        
        # For now, return a simple placeholder
        # In a full implementation, we would generate Bokeh charts here
        return """
        <div class="section">
            <h2>📈 Portfolio Performance</h2>
            <div class="chart-container">
                <p style="color: #6c757d; font-style: italic;">
                    Interactive charts will be displayed here in the full version.
                    Portfolio data contains {len(portfolio_history)} data points.
                </p>
            </div>
        </div>
        """
    
    def _create_trades_table(self, trades: List[Dict], title: str = "Trade History") -> str:
        """Create HTML table for trades"""
        if not trades:
            return f"""
            <div class=\"section\">
                <h3>{title}</h3>
                <p class=\"empty-state\">No trades recorded.</p>
            </div>
            """

        # Limit to last 20 trades for display
        recent_trades = trades[-20:] if len(trades) > 20 else trades

        rows_html = ""
        for trade in recent_trades:
            trade_type = trade.get('type', 'unknown')
            row_class = 'positive' if trade_type == 'buy' else 'negative' if trade_type == 'sell' else 'neutral'
            
            timestamp_str = trade.get('timestamp', '')
            if isinstance(timestamp_str, str):
                timestamp_display = timestamp_str
            else:
                timestamp_display = str(timestamp_str)
            
            pnl_display = ""
            if trade_type == 'sell' and 'pnl_pct' in trade:
                pnl = trade['pnl_pct']
                pnl_class = 'positive' if pnl > 0 else 'negative'
                pnl_display = f"<span class='{pnl_class}'>{pnl:+.2f}%</span>"

            tx_hash = trade.get('tx_hash')
            if tx_hash:
                tx_display = f"{tx_hash[:6]}…{tx_hash[-4:]}"
            else:
                tx_display = "--"
            
            rows_html += f"""
            <tr>
                <td class="timestamp">{timestamp_display}</td>
                <td><span class="{row_class}">{trade_type.upper()}</span></td>
                <td>{trade.get('price', 0):.8f}</td>
                <td>{trade.get('hex_amount', 0):.4f}</td>
                <td>{trade.get('quote_amount', 0):.4f}</td>
                <td>{trade.get('fee', 0):.6f}</td>
                <td>{pnl_display}</td>
                <td>{trade.get('signal_strength', 0):.2f}</td>
                <td>{trade.get('fee_pls', trade.get('fee', 0)):.6f}</td>
                <td>{tx_display}</td>
            </tr>
            """
        
        return f"""
        <div class=\"section\">
            <h3>{title}</h3>
            <p>Showing {'last 20' if len(trades) > 20 else 'all'} trades (Total: {len(trades)})</p>

            <table class=\"trades-table\">
                <thead>
                    <tr>
                        <th>Timestamp</th>
                        <th>Type</th>
                        <th>Price (DAI)</th>
                        <th>HEX Amount</th>
                        <th>Quote Amount (DAI)</th>
                        <th>Fee (DAI)</th>
                        <th>P&L %</th>
                        <th>Signal Strength</th>
                        <th>PLS Fee</th>
                        <th>Tx Hash</th>
                    </tr>
                </thead>
                <tbody>
                    {rows_html}
                </tbody>
            </table>
        </div>
        """
    
    def _create_live_html(self, portfolio_data: List[Dict], strategy_data: Dict) -> str:
        """Create HTML content for live trading"""
        
        current_balance = portfolio_data[-1].get('total_value', 0) if portfolio_data else 0
        current_price = portfolio_data[-1].get('price', 0) if portfolio_data else 0
        position = portfolio_data[-1].get('position', 'none') if portfolio_data else 'none'
        
        recommendation = strategy_data.get('recommendation', 'hold')
        signal_strength = strategy_data.get('signal_strength', 0)
        
        html_template = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>HEX Trading Bot - Live Trading</title>
            <style>
                body {{ 
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                    background-color: #f8f9fa;
                }}
                
                .header {{
                    background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
                    color: white;
                    padding: 30px;
                    border-radius: 12px;
                    margin-bottom: 30px;
                    text-align: center;
                }}
                
                .header h1 {{ margin: 0; font-size: 2.5em; }}
                .header p {{ margin: 10px 0 0 0; opacity: 0.9; }}
                
                .status-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                    gap: 20px;
                    margin-bottom: 30px;
                }}
                
                .status-card {{
                    background: white;
                    padding: 25px;
                    border-radius: 12px;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                    text-align: center;
                }}
                
                .status-value {{
                    font-size: 2em;
                    font-weight: bold;
                    margin: 10px 0;
                }}
                
                .status-label {{
                    color: #666;
                    font-size: 0.9em;
                    text-transform: uppercase;
                    letter-spacing: 1px;
                }}
                
                .live-indicator {{
                    display: inline-block;
                    width: 12px;
                    height: 12px;
                    background-color: #28a745;
                    border-radius: 50%;
                    animation: pulse 2s infinite;
                    margin-right: 8px;
                }}
                
                @keyframes pulse {{
                    0% {{ opacity: 1; }}
                    50% {{ opacity: 0.5; }}
                    100% {{ opacity: 1; }}
                }}
                
                .positive {{ color: #28a745; }}
                .negative {{ color: #dc3545; }}
                .neutral {{ color: #6c757d; }}
                
                .refresh-notice {{
                    background: #e7f3ff;
                    border: 1px solid #b6e0ff;
                    border-radius: 8px;
                    padding: 15px;
                    margin-bottom: 20px;
                    text-align: center;
                }}
            </style>
            <meta http-equiv="refresh" content="10">
        </head>
        <body>
            <div class="header">
                <h1>🔥 HEX Trading Bot</h1>
                <p><span class="live-indicator"></span>Live Trading Mode</p>
                <p class="timestamp">Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="refresh-notice">
                🔄 This page auto-refreshes every 10 seconds | ⚠️ Demo Mode Active
            </div>
            
            <div class="status-grid">
                <div class="status-card">
                    <div class="status-label">Current Price</div>
                    <div class="status-value neutral">
                        {current_price:.8f} DAI
                    </div>
                </div>
                
                <div class="status-card">
                    <div class="status-label">Portfolio Value</div>
                    <div class="status-value positive">
                        {current_balance:.4f} DAI
                    </div>
                </div>
                
                <div class="status-card">
                    <div class="status-label">Current Position</div>
                    <div class="status-value {'positive' if position == 'long' else 'neutral'}">
                        {position.upper() if position else 'NONE'}
                    </div>
                </div>
                
                <div class="status-card">
                    <div class="status-label">Strategy Signal</div>
                    <div class="status-value {'positive' if recommendation == 'buy' else 'negative' if recommendation == 'sell' else 'neutral'}">
                        {recommendation.upper()}
                    </div>
                </div>
                
                <div class="status-card">
                    <div class="status-label">Signal Strength</div>
                    <div class="status-value neutral">
                        {signal_strength:.2f}
                    </div>
                </div>
            </div>
            
            <div style="background: white; padding: 30px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); text-align: center;">
                <h2>🎯 Strategy Status</h2>
                <p style="font-size: 1.2em; margin: 20px 0;">
                    <strong>Recommendation:</strong> 
                    <span class="{'positive' if recommendation == 'buy' else 'negative' if recommendation == 'sell' else 'neutral'}">
                        {recommendation.upper()}
                    </span>
                </p>
                <p style="color: #666;">
                    {strategy_data.get('reason', 'No specific reason provided')}
                </p>
            </div>
            
            <div style="text-align: center; color: #6c757d; margin-top: 40px; padding-top: 20px; border-top: 1px solid #dee2e6;">
                <p>📈 HEX Trading Bot | PulseChain Automated Trading</p>
                <p>⚠️ Demo Mode - No Real Trading</p>
            </div>
        </body>
        </html>
        """
        
        return html_template
