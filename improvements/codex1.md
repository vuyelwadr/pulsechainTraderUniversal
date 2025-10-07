diff --git a/optimization/optimizer_bayes.py b/optimization/optimizer_bayes.py
index b28b3e240536806d897bc5071529d5e71139c909..95c6c9b721496fad57926e31c33757025d523809 100644
--- a/optimization/optimizer_bayes.py
+++ b/optimization/optimizer_bayes.py
@@ -1,119 +1,242 @@
 #!/usr/bin/env python3
 """
 Bayesian Parameter Optimizer for Stage 2 Strategy Optimization
 Intelligently tunes strategy parameters to maximize CPS score using Gaussian Process optimization
 """
 
 import numpy as np
 from typing import Dict, List, Tuple, Any, Optional
 from dataclasses import dataclass
 import json
 import time
-from concurrent.futures import ProcessPoolExecutor
+from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
 import pandas as pd
 
 try:
     from skopt import gp_minimize
     from skopt.space import Real, Integer, Categorical
     from skopt.utils import use_named_args
     from skopt.acquisition import gaussian_ei, gaussian_pi, gaussian_lcb
 except ImportError:
     print("WARNING: scikit-optimize not installed. Install with: pip install scikit-optimize")
     raise
 
 @dataclass
 class OptimizationResult:
     """Container for optimization results"""
     strategy_name: str
     timeframe: str
     best_params: Dict[str, Any]
     best_cps_score: float
     improvement_percent: float
     convergence_iteration: int
     total_iterations: int
     optimization_time: float
     param_history: List[Dict]
     score_history: List[float]
     
 class BayesianOptimizer:
     """
     Enhanced Bayesian Parameter Optimizer for HEX Trading Strategies
     Features:
     - Multi-parameter simultaneous optimization
-    - Acquisition function switching for better exploration
+    - Dynamic acquisition scheduling with adaptive exploration parameters
     - Early stopping on convergence
-    - Cross-validation to prevent overfitting
-    - Parallel evaluation support
+    - Walk-forward cross-validation to prevent time-series leakage
+    - Observation-noise estimation from fold variance
+    - Multi-fidelity screening with lightweight quick evaluations
+    - Parallel evaluation support for validation folds
     """
     
     def __init__(self, 
                  n_calls: int = 100,  # Increased from 50 for 95-98% optimal performance
                  n_initial_points: int = 20,  # More exploration for better global optimum
                  acq_func: str = 'EI',  # Default acquisition when no schedule provided
                  xi: float = 0.01,  # Exploration parameter
                  kappa: float = 1.96,  # Exploration parameter for LCB
                  noise: float = 1e-10,
                  n_jobs: int = 1,
                  random_state: int = 42,
                  convergence_threshold: float = 0.001,
                  convergence_patience: int = 10,
-                 acq_funcs: Optional[List[str]] = None):
+                 acq_funcs: Optional[List[str]] = None,
+                 walk_forward_splits: int = 4,
+                 min_walk_forward_window: int = 100,
+                 multi_fidelity: bool = True,
+                 quick_eval_fraction: float = 0.35,
+                 quick_promotion_quantile: float = 0.4,
+                 min_full_evaluations: int = 6,
+                 batch_acq_size: int = 5,
+                 min_noise: float = 1e-6,
+                 max_noise: float = 0.5,
+                 dynamic_window: int = 6,
+                 parallel_folds: bool = True,
+                 fold_workers: Optional[int] = None):
         """
         Initialize Bayesian Optimizer
         
         Args:
             n_calls: Total number of optimization iterations
             n_initial_points: Number of random initial samples
             acq_func: Acquisition function ('EI', 'PI', 'LCB')
             xi: Exploration parameter for EI and PI
             kappa: Exploration parameter for LCB
             noise: Noise level for GP
             n_jobs: Number of parallel jobs for GP fitting
             random_state: Random seed for reproducibility
             convergence_threshold: Minimum improvement to continue
             convergence_patience: Iterations without improvement before stopping
         """
         self.n_calls = n_calls
         self.n_initial_points = n_initial_points
         self.acq_func = acq_func
         self.xi = xi
         self.kappa = kappa
         self.noise = noise
         self.n_jobs = n_jobs
         self.random_state = random_state
         self.convergence_threshold = convergence_threshold
         self.convergence_patience = convergence_patience
         self.acq_sequence = self._normalize_acq_sequence(acq_funcs)
+        self.walk_forward_splits = max(2, int(walk_forward_splits))
+        self.min_walk_forward_window = max(10, int(min_walk_forward_window))
+        self.multi_fidelity = multi_fidelity
+        self.quick_eval_fraction = min(0.9, max(0.05, quick_eval_fraction))
+        self.quick_promotion_quantile = min(0.95, max(0.0, quick_promotion_quantile))
+        self.min_full_evaluations = max(0, int(min_full_evaluations))
+        self.batch_acq_size = max(1, int(batch_acq_size))
+        self.min_noise = min_noise
+        self.max_noise = max_noise
+        self.dynamic_window = max(3, int(dynamic_window))
+        self.parallel_folds = parallel_folds
+        self.fold_workers = fold_workers
 
         # Track optimization progress
         self.param_history = []
         self.score_history = []
         self.best_score = -np.inf
         self.no_improvement_count = 0
         self.convergence_iteration = None
         self.total_evaluations = 0
+        self.dynamic_noise = noise
+        self.noise_estimates: List[float] = []
+        self.quick_scores: List[float] = []
+        self.current_xi = xi
+        self.current_kappa = kappa
+        self.min_xi = 0.001
+        self.max_xi = 0.25
+        self.min_kappa = 0.5
+        self.max_kappa = 6.0
+
+    # ------------------------------------------------------------------
+    # Helper utilities
+    # ------------------------------------------------------------------
+
+    def _generate_walk_forward_folds(self, data: pd.DataFrame) -> List[pd.DataFrame]:
+        """Generate walk-forward validation folds (future-only evaluation)."""
+        n_samples = len(data)
+        if n_samples <= self.min_walk_forward_window:
+            return [data]
+
+        # Determine fold sizes
+        validation_min = max(5, int(0.1 * n_samples))
+        fold_size = max(validation_min, n_samples // (self.walk_forward_splits + 1))
+        folds: List[pd.DataFrame] = []
+
+        train_end = max(self.min_walk_forward_window, fold_size)
+        while train_end < n_samples:
+            val_end = min(n_samples, train_end + fold_size)
+            if val_end - train_end <= 0:
+                break
+            fold = data.iloc[train_end:val_end]
+            if len(fold) > 0:
+                folds.append(fold)
+            if val_end == n_samples:
+                break
+            train_end = val_end
+
+        if not folds:
+            # Fallback to last portion if we couldn't create multiple folds
+            folds.append(data.iloc[-fold_size:])
+
+        return folds[: self.walk_forward_splits]
+
+    def _update_noise_estimate(self, variance: float) -> None:
+        if not np.isfinite(variance):
+            return
+        self.noise_estimates.append(float(max(0.0, variance)))
+        if len(self.noise_estimates) > 20:
+            self.noise_estimates.pop(0)
+        avg_var = float(np.mean(self.noise_estimates)) if self.noise_estimates else self.noise
+        self.dynamic_noise = float(min(self.max_noise, max(self.min_noise, avg_var)))
+
+    def _should_promote_to_full(self, quick_score: float) -> bool:
+        """Decide whether a quick evaluation warrants full evaluation."""
+        self.quick_scores.append(quick_score)
+        if len(self.quick_scores) <= self.min_full_evaluations:
+            return True
+        threshold = np.quantile(self.quick_scores, self.quick_promotion_quantile)
+        # Always promote if we haven't found any good scores yet
+        if not np.isfinite(threshold):
+            return True
+        # Provide slack so borderline candidates still get evaluated
+        safety_margin = max(self.convergence_threshold * 10, 1e-6)
+        return quick_score + safety_margin >= threshold
+
+    def _select_acquisition(self) -> Tuple[str, float, float]:
+        """Dynamically select acquisition function and parameters."""
+        # Default to EI until we gather enough history
+        default_acq = self.acq_func.upper() if self.acq_func else 'EI'
+        if len(self.score_history) < self.dynamic_window or self.no_improvement_count == 0:
+            return default_acq, self.current_xi, self.current_kappa
+
+        recent_scores = self.score_history[-self.dynamic_window:]
+        recent_improvement = max(recent_scores) - min(recent_scores)
+        stagnating = recent_improvement < self.convergence_threshold * self.dynamic_window
+
+        if stagnating or self.no_improvement_count >= self.convergence_patience // 2:
+            # Encourage exploration when progress stalls
+            new_xi = min(self.max_xi, self.current_xi * 1.5)
+            new_kappa = min(self.max_kappa, self.current_kappa * 1.3)
+            self.current_xi, self.current_kappa = new_xi, new_kappa
+            return 'LCB', new_xi, new_kappa
+
+        # If recent improvements are strong, emphasize exploitation
+        if recent_scores[-1] >= self.best_score - self.convergence_threshold:
+            new_xi = max(self.min_xi, self.current_xi * 0.7)
+            new_kappa = max(self.min_kappa, self.current_kappa * 0.9)
+            self.current_xi, self.current_kappa = new_xi, new_kappa
+            return 'PI', new_xi, new_kappa
+
+        # Otherwise stay on EI but slowly anneal xi toward baseline
+        drift_xi = self.current_xi + (self.xi - self.current_xi) * 0.3
+        drift_kappa = self.current_kappa + (self.kappa - self.current_kappa) * 0.3
+        self.current_xi = float(min(self.max_xi, max(self.min_xi, drift_xi)))
+        self.current_kappa = float(min(self.max_kappa, max(self.min_kappa, drift_kappa)))
+        acq = default_acq if default_acq in {'EI', 'PI', 'LCB'} else 'EI'
+        return acq, self.current_xi, self.current_kappa
 
     @staticmethod
     def _normalize_acq_sequence(acq_funcs: Optional[List[str]]) -> Optional[List[str]]:
         if not acq_funcs:
             return None
         seq = []
         for name in acq_funcs:
             if not name:
                 continue
             label = name.strip().upper()
             if label not in {'EI', 'PI', 'LCB'}:
                 raise ValueError(f"Unsupported acquisition function '{name}'")
             seq.append(label)
         return seq or None
         
     def get_parameter_space(self, strategy_type: str) -> List:
         """
         Define parameter search space based on strategy type
         
         Returns optimized parameter ranges specifically for HEX's volatility
         """
         # Enhanced parameter spaces for HEX's high volatility
         if 'momentum' in strategy_type.lower():
             return [
                 Integer(5, 50, name='fast_period'),     # Fast MA period
diff --git a/optimization/optimizer_bayes.py b/optimization/optimizer_bayes.py
index b28b3e240536806d897bc5071529d5e71139c909..95c6c9b721496fad57926e31c33757025d523809 100644
--- a/optimization/optimizer_bayes.py
+++ b/optimization/optimizer_bayes.py
@@ -164,76 +287,88 @@ class BayesianOptimizer:
                 Real(0.01, 0.3, name='risk_limit'),
             ]
     
     def create_objective(self, strategy, data, backtest_func, cross_validate=True, dimensions: Optional[List]=None):
         """
         Create objective function for optimization
         
         Args:
             strategy: Strategy instance to optimize
             data: Historical price data (REAL HEX data only!)
             backtest_func: Function to run backtest and return CPS score
             cross_validate: Whether to use cross-validation
         """
         def objective(params):
             """Objective function to minimize (negative CPS score)"""
             try:
                 # Convert params list to dict using provided dimensions (names)
                 param_dict = {}
                 space = dimensions if dimensions is not None else self.get_parameter_space(strategy.__class__.__name__)
                 for i, param in enumerate(params):
                     param_dict[space[i].name] = param
                 
                 # Track parameters
                 self.param_history.append(param_dict)
                 
-                if cross_validate:
-                    # Split data into 3 folds for cross-validation
-                    fold_size = len(data) // 3
-                    scores = []
-                    
-                    for i in range(3):
-                        if i == 0:
-                            train_data = data[fold_size:]
-                            val_data = data[:fold_size]
-                        elif i == 1:
-                            train_data = pd.concat([data[:fold_size], data[2*fold_size:]])
-                            val_data = data[fold_size:2*fold_size]
+                final_score = None
+                fold_scores: List[float] = []
+                quick_score: Optional[float] = None
+
+                if self.multi_fidelity and len(data) > self.min_walk_forward_window:
+                    quick_len = max(int(len(data) * self.quick_eval_fraction), self.min_walk_forward_window // 2)
+                    quick_data = data.iloc[-quick_len:]
+                    quick_score = float(backtest_func(strategy, quick_data, param_dict))
+                    if not self._should_promote_to_full(quick_score):
+                        final_score = quick_score
+
+                if final_score is None:
+                    if cross_validate:
+                        # Walk-forward folds: always validate on future segments
+                        folds = self._generate_walk_forward_folds(data)
+                        if self.parallel_folds and len(folds) > 1:
+                            workers = self.fold_workers or min(len(folds), max(1, self.n_jobs))
+                            with ThreadPoolExecutor(max_workers=workers) as executor:
+                                futures = [executor.submit(backtest_func, strategy, fold, param_dict) for fold in folds]
+                                for future in futures:
+                                    score = float(future.result())
+                                    fold_scores.append(score)
                         else:
-                            train_data = data[:2*fold_size]
-                            val_data = data[2*fold_size:]
-                        
-                        # Run backtest on validation fold
-                        score = backtest_func(strategy, val_data, param_dict)
-                        scores.append(score)
-                    
-                    # Average score across folds
-                    final_score = np.mean(scores)
-                else:
-                    # Single evaluation on full dataset
-                    final_score = backtest_func(strategy, data, param_dict)
-                
+                            for fold in folds:
+                                score = float(backtest_func(strategy, fold, param_dict))
+                                fold_scores.append(score)
+
+                        final_score = float(np.mean(fold_scores)) if fold_scores else float(backtest_func(strategy, data, param_dict))
+                    else:
+                        final_score = float(backtest_func(strategy, data, param_dict))
+
+                if quick_score is not None and final_score == quick_score and not fold_scores:
+                    # Track quick-only evaluations for history transparency
+                    fold_scores.append(quick_score)
+
+                variance = np.var(fold_scores, ddof=1) if len(fold_scores) > 1 else 0.0
+                self._update_noise_estimate(float(variance))
+
                 # Track score
                 self.score_history.append(final_score)
                 
                 # Check for convergence
                 if final_score > self.best_score:
                     improvement = final_score - self.best_score
                     if improvement < self.convergence_threshold:
                         self.no_improvement_count += 1
                     else:
                         self.no_improvement_count = 0
                     self.best_score = final_score
                 else:
                     self.no_improvement_count += 1
                 
                 # Early stopping
                 if self.no_improvement_count >= self.convergence_patience:
                     if self.convergence_iteration is None:
                         self.convergence_iteration = len(self.score_history)
                 
                 # Return negative score for minimization
                 return -final_score
                 
             except Exception as e:
                 print(f"Error in objective function: {e}")
                 return 0.0  # Return worst possible score on error
diff --git a/optimization/optimizer_bayes.py b/optimization/optimizer_bayes.py
index b28b3e240536806d897bc5071529d5e71139c909..95c6c9b721496fad57926e31c33757025d523809 100644
--- a/optimization/optimizer_bayes.py
+++ b/optimization/optimizer_bayes.py
@@ -295,87 +430,92 @@ class BayesianOptimizer:
                     return None
                 point.append(val)
             return point
 
         evaluated_x: List[List[Any]] = []
         evaluated_y: List[float] = []
 
         if initial_params:
             for p_dict in initial_params:
                 point = _dict_to_point(p_dict)
                 if point is None:
                     continue
                 y = float(objective(point))
                 evaluated_x.append(point)
                 evaluated_y.append(y)
 
         self.total_evaluations = len(self.score_history)
         target_calls = int(self.n_calls)
         if target_calls < 0:
             target_calls = 0
         remaining_calls = max(0, target_calls - len(evaluated_x))
 
         if not evaluated_x and remaining_calls <= 0:
             raise ValueError("Expected at least one optimization call or a valid initial sample")
 
-        if self.acq_sequence:
-            schedule = list(self.acq_sequence)
-        else:
-            if target_calls < 30:
-                schedule = [self.acq_func.upper() if self.acq_func else 'EI']
-            else:
-                schedule = ['EI', 'PI', 'LCB']
+        sequence_iter = iter(self.acq_sequence) if self.acq_sequence else None
 
-        last_result = None
+        while remaining_calls > 0:
+            if sequence_iter:
+                try:
+                    acq_choice = next(sequence_iter)
+                except StopIteration:
+                    sequence_iter = iter(self.acq_sequence)
+                    acq_choice = next(sequence_iter)
+                acq_func = acq_choice
+                xi = self.current_xi
+                kappa = self.current_kappa
+            else:
+                acq_func, xi, kappa = self._select_acquisition()
 
-        for idx, acq_func in enumerate(schedule):
-            if remaining_calls <= 0:
-                break
-            segments_left = len(schedule) - idx
-            new_calls = remaining_calls if segments_left <= 1 else max(1, remaining_calls // segments_left)
+            new_calls = min(self.batch_acq_size, remaining_calls)
             total_calls = len(evaluated_x) + new_calls
             if total_calls <= len(evaluated_x):
-                continue
-            n_init = max(0, self.n_initial_points - len(evaluated_x))
-            last_result = gp_minimize(
+                break
+
+            n_init = max(0, min(self.n_initial_points, total_calls) - len(evaluated_x))
+
+            result = gp_minimize(
                 func=objective,
                 dimensions=space,
                 n_calls=total_calls,
                 n_initial_points=n_init,
                 acq_func=acq_func,
-                xi=self.xi,
-                kappa=self.kappa,
-                noise=self.noise,
+                xi=xi,
+                kappa=kappa,
+                noise=self.dynamic_noise,
                 n_jobs=self.n_jobs,
                 random_state=self.random_state,
                 x0=evaluated_x if evaluated_x else None,
                 y0=evaluated_y if evaluated_y else None,
             )
-            evaluated_x = list(last_result.x_iters)
-            evaluated_y = list(map(float, last_result.func_vals))
+
+            evaluated_x = list(result.x_iters)
+            evaluated_y = list(map(float, result.func_vals))
             remaining_calls = max(0, target_calls - len(evaluated_x))
             self.total_evaluations = len(self.score_history)
+
             if self.convergence_iteration is not None:
                 break
 
         if not evaluated_x:
             raise RuntimeError("Optimization produced no evaluations")
 
         best_idx = int(np.argmin(evaluated_y))
         best_point = evaluated_x[best_idx]
         best_params = {}
         for dim, value in zip(space, best_point):
             if isinstance(dim, Integer):
                 best_params[dim.name] = int(round(value))
             elif isinstance(dim, Real):
                 best_params[dim.name] = float(value)
             else:
                 best_params[dim.name] = value
 
         best_cps_score = float(-evaluated_y[best_idx])
         improvement = 0.0
         if initial_score:
             try:
                 improvement = ((best_cps_score - initial_score) / initial_score) * 100
             except ZeroDivisionError:
                 improvement = 0.0
 
