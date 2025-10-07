# Bayesian Optimization Improvements — Summary & Plan

**Repository:** `pulsechainTraderUniversal`

## What changed (high level)

- Walk-forward cross-validation (expanding window, prevents leakage)
- Adaptive acquisition scheduling with xi/kappa annealing when progress stalls
- Dynamic Gaussian Process noise derived from fold variance for robustness
- Batched `gp_minimize` loops that reuse prior evaluations for efficient exploration
- Optional local refinement pass that jitters the top seeds for last-mile gains

## Why

These upgrades bundle the strongest concepts from the optimizer prototypes, aligning with the priorities in `improvements/whatToDo.md`. Expect steadier convergence, better resistance to noisy folds, and modest CPS lift without relying on artificial filters.
