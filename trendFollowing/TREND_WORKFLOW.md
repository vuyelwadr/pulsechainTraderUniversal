# Trend Regime Workflow

This note explains how to regenerate the multi-timeframe trend labels and open the interactive dashboard that visualises them. It assumes you are working inside this repository root (`pulsechainTraderUniversal`).

## 1. Environment Setup

1. Create/activate the project virtual environment (skip if already active):
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
2. Install requirements (Plotly is now included for the dashboard):
   ```bash
   pip install -r requirements.txt
   ```

## 2. Generate Trend Datasets

The generator reads `data/pdai_ohlcv_dai_730day_5m.csv`, then produces per-timeframe regime labels and summaries under `analysis/`.

```bash
python scripts/generate_trend_states.py
```

Outputs written on completion:

- `analysis/trend_states_<tf>.csv` – row-level OHLC data with `trend_state` and `segment_index` for each timeframe (5 min → 2 day).
- `analysis/trend_segments_<tf>.csv` – consolidated regimes, one row per contiguous block with start/end timestamps, net return %, max gain/loss %, duration (bars/hours/days), slope/ADX averages, etc.
- `analysis/trend_states_all_timeframes.csv` / `analysis/trend_segments_all_timeframes.csv` – combined views across every timeframe.

Sanity check: for the daily file the period 2025‑08‑25 → 2025‑10‑07 should appear as a single `DOWNTREND` block with ≈ −43.7 % return.

## 3. Build the Interactive Dashboard

Run the renderer to create an HTML dashboard with price lines and shaded regimes:

```bash
python scripts/make_trend_dashboard.py
```

This writes `analysis/trend_dashboard.html`. Open it in a browser (double-click or `open analysis/trend_dashboard.html`).

Dashboard features:

- Timeframe dropdown covers 5 min → 2 day views.
- Price plotted as a clean line with shaded `UPTREND` / `DOWNTREND` / `RANGE` blocks.
- Summary table (latest ≤ 200 regimes) shows start/end, duration, and net/max gain/loss percentages.
- Range selector + zoom make it practical to inspect any window.

If you notice missing shapes or empty tables for a timeframe, re-run the generator (Step 2) before rebuilding the dashboard.

## 4. Optional: Quick Analytics Examples

Use the combined summary to inspect regimes programmatically. Example: list the strongest daily moves over the past year.

```python
import pandas as pd
summary = pd.read_csv('analysis/trend_segments_1d.csv', parse_dates=['start_time','end_time'])
recent = summary[summary['end_time'] >= summary['end_time'].max() - pd.Timedelta(days=365)]
print(recent[['segment_id','state','start_time','end_time','net_return_pct','duration_days']])
```

## 5. Troubleshooting

- **Different dataset:** point `DATA_SOURCE` in `scripts/generate_trend_states.py` to the new CSV before running step 2.
- **Performance:** both scripts load the entire 730‑day history; ensure enough RAM or trim the CSV first.
- **Validation:** the generator emits warnings if the regression/ADX windows are too short at the dataset start—these are expected for the first few bars and do not affect later segments.

That’s all another agent needs to regenerate the regimes and review them visually before iterating on strategies.
