# Optimizer Overhaul Plan

## 1. Objective
Rewrite the optimization runner so that it:

- Executes parameter searches using the same cost-aware, vectorised evaluation pipeline that powers our fast manual tests.
- Preserves critical workflow features: resumable runs, per-objective/per-period reports, HTML summaries, and ranked outputs.
- Exploits up to ~90 % of available CPU (max parallelism) without bottlenecking on disk I/O.
- Keeps all pricing/cost assumptions aligned with `data/pdai_ohlcv_dai_730day_5m.csv` and `reports/optimizer_top_top_strats_run/swap_cost_cache.json`.

Backwards compatibility is **not** required; focus on a clean, maintainable redesign.

## 2. Current Pain Points

1. **Heavy per-call overhead** – `optimization.runner` spins up the full bot/BacktestEngine pipeline, repeatedly loading CSVs and serialising large artifacts. A single call can take 20–30 minutes.
2. **Poor CPU utilisation** – process pools spend significant time blocking on I/O; effective parallelism is far below hardware limits.
3. **Reporting tightly coupled to execution** – run/resume logic, artifact creation, and evaluation are tangled, making incremental improvements risky.
4. **Limited search space** – optimisation is currently focussed on small parameter tweaks; adding new gates/overlays is cumbersome.

## 3. Functional Requirements

- **Resume capability**: interrupted runs must restart from the last completed call without recomputing finished trials.
- **Comprehensive reporting**:
  - Per-objective and per-period (e.g., 30 d MAR, 90 d CPS, 1 y Utility) tables sorted by total return.
  - Aggregated overall summary (JSON + CSV) and sortable HTML dashboard.
  - Highlight top strategies per objective and globally.
- **Swap-cost fidelity**: always use real swap-cost buckets (round-up + gas) sourced from a repo-managed cache.
- **Central swap-cost cache**: standardise on `data/swap_cost_cache.json`; provide a rate-limited fetcher to refresh it when needed.
- **Max parallelisation**: dynamically size the worker pool (default `num_cpu * 0.9`), ensuring each worker uses the fast in-memory evaluator.
- **Configurable objectives**: support the existing list (`final_balance`, `profit_biased`, `cps_v2_profit_biased`, `mar`, `utility`, `cps`, `cps_v2`) with minimal boilerplate.
- **Artifact hygiene**: produce deterministic, timestamped run directories under `reports/optimizer_top_top_strats_run/`.

## 4. Proposed Architecture

1. **Core Evaluation Engine (`engine/evaluator.py`)**
   - Wrap the vectorised `run_strategy` pipeline (from `scripts/evaluate_vost_strategies`) in a pure-Python function that accepts `(strategy_class, params, trade_size, data, costs)` and returns a metrics dataclass.
   - Load dataset and swap-cost cache once per worker; reuse across calls.
   - Support early-abort hooks (optional) for pruning hopeless candidates.

2. **Worker Pool (`optimizer/execution.py`)**
   - Use `ProcessPoolExecutor` or `concurrent.futures.ThreadPoolExecutor` depending on CPU/NUMA testing (processes preferred).
   - Each worker initialises shared state (data, costs, strategy registry) on first task.
   - Work queue accepts batches of parameter suggestions to minimise IPC overhead.

3. **Orchestrator / Scheduler (`optimizer/scheduler.py`)**
   - Implements optimisation loop (Bayesian or random search) independent of evaluation.
   - Maintains in-memory state (`call_id`, params, objective scores).
   - After every K evaluations, checkpoints to a lightweight SQLite/Parquet/JSONL file.
   - On resume, reloads state and continues from the next pending call.

4. **Persistence Layer**
   - **Trial store**: append-only JSONL or Parquet per run (`trial_id`, params, metrics, timestamps, objective scores).
   - **Checkpoint**: single JSON file with optimiser state, RNG seeds, next call index, and best-so-far summary.
   - **Config manifest**: capture CLI arguments, strategy list, CPU usage, git commit hash.
   - **Swap-cost cache**: expect `data/swap_cost_cache.json`; ensure the loader falls back to the fetch script if missing.

5. **Reporting Pipeline (`reports/generator.py`)**
   - After run completion (or on demand), read the trial store and:
     - Build per-objective CSVs sorted by return percentage.
     - Produce period-specific views (30 d, 90 d, 1 y) by reusing metrics payload.
     - Generate HTML dashboard (e.g., Jinja template) with sortable tables and metric badges.
   - For top-N strategies per objective, optionally regenerate equity/trade plots using the heavy reporters (but only for the shortlist).

6. **CLI Interface (`optimization/runner_cli.py`)**
   - New entrypoint that accepts existing flags plus:
     - `--resume-from <dir>` to reuse checkpoints.
     - `--max-workers <int>` / `--cpu-fraction <float>` (default 0.9).
     - `--report-frequency <int>` (generate intermediate reports every N calls).
   - Persists outputs under `reports/optimizer_top_top_strats_run/<timestamp>/`.

7. **Strategy Registry**
   - Maintain a manifest mapping strategy names to constructors & parameter spaces (JSON or Python module).
   - Facilitate hot-swapping new overlays (e.g., Donchian regime, CSMA risk cap) without editing the optimiser core.

## 5. Task Breakdown

1. **Discovery & Design**
   - Benchmark current runner timing/cpu usage for baseline.
   - Finalise data structures (trial record schema, metrics dataclass).
   - Decide on persistence format (JSONL vs Parquet) and checkpoint cadence.
   - Specify swap-cost fetch workflow (parallel 8-request batches, 1-minute throttle, configurable step/end-price inputs, output to `data/swap_cost_cache.json`).

2. **Core Evaluator Implementation**
   - Extract reusable `run_strategy` logic into `engine/evaluator.py`.
   - Add support for multiple trade sizes and period slicing within a single pass.
   - Unit tests comparing outputs against `scripts/evaluate_vost_strategies`.

3. **Worker Pool & Initialisation**
   - Build worker bootstrap that loads dataset + swap-cost cache once.
   - Implement task dispatch & result collection with graceful shutdown.
   - Stress-test CPU utilisation; tune batching and `chunksize`.

4. **Optimisation Orchestrator**
   - Port/implement Bayesian or evolutionary search that consumes metrics from the pool.
   - Add checkpoint writer/loader and validation (resume from partial run).
   - Provide hooks for early-stopping heuristics (optional).

5. **Persistence & Resume**
   - Define run directory layout (config, trials, checkpoints, logs, reports).
   - Implement resilient file writes (atomic operations, temp files).
   - Integration tests: interrupt run mid-way, resume, verify no duplicated trials.

6. **Reporting Engine**
   - Implement per-objective & per-period aggregations.
   - Generate CSVs + markdown/HTML (sorted by return desc).
   - Produce top-N comparison tables and embed into HTML template.

7. **CLI & User Experience**
   - Build new CLI entrypoint with updated options.
   - Write usage docs / help text, including CPU guidance.
   - Provide quick-start script for running the 1 y stage shown in user command.

8. **Validation & Benchmarking**
   - Compare speed and CPU utilisation vs old runner (target ≥10× faster per call).
   - Cross-check metrics for at least 5 strategies against legacy reports.
   - Run end-to-end stage (e.g., 1 y, 200 calls) to confirm reporting/resume.

9. **Documentation**
   - Update `newstrats/local.md` with overhaul summary.
   - Add README section explaining new runner architecture and requirements.

## 6. Risks & Mitigations

- **Data consistency**: ensure evaluator uses the same cost logic (round-up + gas). Mitigation: unit tests comparing sampled trades with legacy engine.
- **Parallel state bleed**: workers must remain stateless aside from cached data. Mitigation: avoid global mutable objects; pass params explicitly.
- **Resume correctness**: corrupted checkpoints could halt progress. Mitigation: atomic writes, checksum verification, frequent but lightweight checkpoints.
- **Reporting regressions**: new pipeline must match existing ranking/sorting expectations. Mitigation: snapshot current report format and write regression tests.

## 7. Timeline & Milestones

1. **Week 1** – Core evaluator + worker pool prototype (validated against current metrics).
2. **Week 2** – Orchestrator with checkpoint/resume; CLI skeleton.
3. **Week 3** – Reporting engine + HTML/CSV outputs; integration tests.
4. **Week 4** – Optimisation of CPU usage, documentation, final benchmarking, and rollout checklist.

Deliverables at each milestone include benchmarks, sample reports, and updated documentation so the new runner can replace the old pipeline confidently.

## 8. Implementation Snapshot (Oct 8 2025)

- `scripts/fetch_swap_cost_cache.py` adds a reusable, rate-limited fetcher that writes to `data/swap_cost_cache.json`.
- `optimization/engine/evaluator.py` hosts the shared in-memory evaluator + worker initialiser.
- `optimization/orchestrator.py` implements strategy sampling, multiprocessing dispatch, and checkpointing via `optimization/persistence.TrialStore`.
- `optimization/reporting.py` produces per-objective and per-period CSVs plus an HTML dashboard.
- `optimization/runner_cli.py` is the new entrypoint (`python -m optimization.runner_cli …`) supporting resume, CPU fraction control, and consolidated reporting.
