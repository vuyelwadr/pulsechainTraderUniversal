 (cd "$(git rev-parse --show-toplevel)" && git apply --3way <<'EOF' 
diff --git a/optimization/optimizer_bayes.py b/optimization/optimizer_bayes.py
index b28b3e240536806d897bc5071529d5e71139c909..b416fc873ccd3d4c0aa90fc0097d0732492c736d 100644
--- a/optimization/optimizer_bayes.py
+++ b/optimization/optimizer_bayes.py
@@ -1,133 +1,270 @@
 #!/usr/bin/env python3
 """
 Bayesian Parameter Optimizer for Stage 2 Strategy Optimization
 Intelligently tunes strategy parameters to maximize CPS score using Gaussian Process optimization
 """
 
+import copy
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
+    - Adaptive acquisition function switching for better exploration
     - Early stopping on convergence
-    - Cross-validation to prevent overfitting
-    - Parallel evaluation support
+    - Walk-forward cross-validation to prevent overfitting
+    - Observation noise modeling to stabilize the GP
+    - Multi-fidelity evaluation pipeline for faster screening
+    - Parallel evaluation support for cross-validation folds
     """
     
-    def __init__(self, 
+    def __init__(self,
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
+                 enable_multi_fidelity: bool = True,
+                 quick_eval_fraction: float = 0.3,
+                 promotion_percentile: float = 0.6,
+                 multi_fidelity_min_points: int = 300,
+                 fold_workers: int = 1,
+                 walk_forward_folds: int = 3,
+                 walk_forward_min_size: int = 120,
+                 walk_forward_warmup: int = 200,
+                 batch_size: int = 5):
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
+        self.batch_size = max(1, int(batch_size))
+
+        # Multi-fidelity configuration
+        self.enable_multi_fidelity = enable_multi_fidelity
+        self.quick_eval_fraction = max(0.0, min(1.0, quick_eval_fraction))
+        self.promotion_percentile = max(0.0, min(1.0, promotion_percentile))
+        self.multi_fidelity_min_points = max(1, int(multi_fidelity_min_points))
+
+        # Validation and parallel evaluation settings
+        self.fold_workers = max(1, int(fold_workers))
+        self.walk_forward_folds = max(1, int(walk_forward_folds))
+        self.walk_forward_min_size = max(10, int(walk_forward_min_size))
+        self.walk_forward_warmup = max(0, int(walk_forward_warmup))
 
         # Track optimization progress
-        self.param_history = []
-        self.score_history = []
+        self.param_history: List[Dict[str, Any]] = []
+        self.score_history: List[float] = []
         self.best_score = -np.inf
         self.no_improvement_count = 0
         self.convergence_iteration = None
         self.total_evaluations = 0
+        self.observation_noises: List[float] = []
+        self.quick_eval_scores: List[float] = []
+        self.promotion_history: List[bool] = []
+        self._acq_index = 0
 
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
+
+    def _clone_strategy(self, strategy):
+        """Attempt to create an independent copy of the strategy for parallel evaluation."""
+        try:
+            return copy.deepcopy(strategy)
+        except Exception:
+            clone_method = getattr(strategy, "clone", None)
+            if callable(clone_method):
+                try:
+                    return clone_method()
+                except Exception:
+                    return None
+        return None
+
+    def _prepare_fold_data(self, data: pd.DataFrame, train_slice: slice, val_slice: slice) -> pd.DataFrame:
+        train_data = data.iloc[train_slice]
+        val_data = data.iloc[val_slice]
+        if train_data.empty:
+            return val_data.copy()
+        warmup = train_data.tail(self.walk_forward_warmup) if self.walk_forward_warmup else train_data.iloc[0:0]
+        combined = pd.concat([warmup, val_data]) if not warmup.empty else val_data
+        if hasattr(combined, 'index'):
+            combined = combined[~combined.index.duplicated(keep='last')]
+        return combined.copy()
+
+    def _generate_walk_forward_splits(self, data: pd.DataFrame) -> List[Tuple[slice, slice]]:
+        total_len = len(data)
+        if total_len < (self.walk_forward_min_size * 2):
+            return []
+
+        fold_count = min(self.walk_forward_folds, max(1, total_len // self.walk_forward_min_size) - 1)
+        if fold_count <= 0:
+            return []
+
+        fold_size = max(self.walk_forward_min_size, total_len // (fold_count + 1))
+        splits: List[Tuple[slice, slice]] = []
+        start = fold_size
+        for _ in range(fold_count):
+            end = start + fold_size
+            if end >= total_len:
+                end = total_len
+            if start >= end:
+                break
+            train_slice = slice(0, start)
+            val_slice = slice(start, end)
+            splits.append((train_slice, val_slice))
+            start = end
+            if end >= total_len:
+                break
+        return splits
+
+    def _record_evaluation(self, score: float, variance: float, promoted: bool = True):
+        self.score_history.append(score)
+        self.observation_noises.append(max(0.0, float(variance)))
+        self.promotion_history.append(promoted)
+        self.total_evaluations += 1
+        if score > self.best_score:
+            improvement = score - self.best_score if self.best_score != -np.inf else np.inf
+            if self.best_score == -np.inf or improvement >= self.convergence_threshold:
+                self.no_improvement_count = 0
+            else:
+                self.no_improvement_count += 1
+            self.best_score = score
+        else:
+            self.no_improvement_count += 1
+
+        if self.no_improvement_count >= self.convergence_patience and self.convergence_iteration is None:
+            self.convergence_iteration = len(self.score_history)
+
+    def _estimate_noise_level(self) -> float:
+        if not self.observation_noises:
+            return self.noise
+        recent_window = max(5, len(self.observation_noises) // 2)
+        recent = self.observation_noises[-recent_window:]
+        dynamic_noise = float(np.mean(recent) + 1e-6)
+        if not np.isfinite(dynamic_noise) or dynamic_noise <= 0:
+            dynamic_noise = self.noise
+        return max(self.noise, dynamic_noise)
+
+    def _compute_promotion_threshold(self) -> Optional[float]:
+        if len(self.quick_eval_scores) < 5:
+            return None
+        percentile = self.promotion_percentile * 100
+        return float(np.percentile(self.quick_eval_scores, percentile))
+
+    def _select_acquisition_settings(self) -> Tuple[str, float, float]:
+        base = self.acq_func.upper() if self.acq_func else 'EI'
+        if self.acq_sequence:
+            base = self.acq_sequence[self._acq_index % len(self.acq_sequence)]
+            self._acq_index += 1
+
+        xi = self.xi
+        kappa = self.kappa
+
+        if len(self.score_history) >= 4:
+            recent = np.array(self.score_history[-4:], dtype=float)
+            improvements = np.diff(recent)
+            if improvements.size > 0:
+                max_improve = float(np.max(improvements))
+                mean_improve = float(np.mean(improvements))
+                if max_improve < self.convergence_threshold and mean_improve <= 0:
+                    return 'LCB', max(xi * 1.5, xi + 0.01), max(kappa, 2.5)
+                if max_improve < self.convergence_threshold * 0.5:
+                    return 'PI', max(xi * 0.5, 1e-4), kappa
+                if max_improve > self.convergence_threshold * 5:
+                    return 'EI', max(xi * 0.3, 1e-4), max(1.0, kappa * 0.8)
+
+        return base, xi, kappa
         
     def get_parameter_space(self, strategy_type: str) -> List:
         """
         Define parameter search space based on strategy type
         
         Returns optimized parameter ranges specifically for HEX's volatility
         """
         # Enhanced parameter spaces for HEX's high volatility
         if 'momentum' in strategy_type.lower():
             return [
                 Integer(5, 50, name='fast_period'),     # Fast MA period
                 Integer(10, 200, name='slow_period'),   # Slow MA period  
                 Real(0.001, 0.1, name='signal_threshold'),  # Entry threshold
                 Real(0.01, 0.2, name='stop_loss'),      # Stop loss %
                 Real(0.02, 0.5, name='take_profit'),    # Take profit %
             ]
         elif 'trend' in strategy_type.lower():
             return [
                 Integer(10, 100, name='trend_period'),
                 Real(0.5, 3.0, name='atr_multiplier'),
                 Real(0.001, 0.05, name='min_trend_strength'),
                 Integer(5, 30, name='confirmation_period'),
                 Real(0.01, 0.15, name='trailing_stop'),
             ]
         elif 'volatility' in strategy_type.lower():
diff --git a/optimization/optimizer_bayes.py b/optimization/optimizer_bayes.py
index b28b3e240536806d897bc5071529d5e71139c909..b416fc873ccd3d4c0aa90fc0097d0732492c736d 100644
--- a/optimization/optimizer_bayes.py
+++ b/optimization/optimizer_bayes.py
@@ -145,238 +282,250 @@ class BayesianOptimizer:
                 Integer(20, 80, name='overbought_level'),
                 Integer(3, 20, name='smoothing_period'),
                 Real(0.001, 0.05, name='divergence_threshold'),
             ]
         elif 'volume' in strategy_type.lower():
             return [
                 Integer(10, 50, name='volume_period'),
                 Real(1.5, 5.0, name='volume_multiplier'),
                 Integer(5, 30, name='price_period'),
                 Real(0.001, 0.05, name='confirmation_threshold'),
                 Real(0.01, 0.1, name='risk_per_trade'),
             ]
         else:
             # Generic parameter space
             return [
                 Integer(5, 50, name='period1'),
                 Integer(10, 100, name='period2'),
                 Real(0.001, 0.1, name='threshold1'),
                 Real(0.01, 0.2, name='threshold2'),
                 Real(0.01, 0.3, name='risk_limit'),
             ]
     
     def create_objective(self, strategy, data, backtest_func, cross_validate=True, dimensions: Optional[List]=None):
         """
         Create objective function for optimization
-        
+
         Args:
             strategy: Strategy instance to optimize
             data: Historical price data (REAL HEX data only!)
             backtest_func: Function to run backtest and return CPS score
             cross_validate: Whether to use cross-validation
         """
+        usable_cross_validation = cross_validate and len(data) >= (self.walk_forward_min_size * 2)
+
         def objective(params):
             """Objective function to minimize (negative CPS score)"""
             try:
                 # Convert params list to dict using provided dimensions (names)
                 param_dict = {}
                 space = dimensions if dimensions is not None else self.get_parameter_space(strategy.__class__.__name__)
                 for i, param in enumerate(params):
                     param_dict[space[i].name] = param
-                
+
                 # Track parameters
                 self.param_history.append(param_dict)
-                
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
+
+                # Optional cheap evaluation before full validation
+                if (self.enable_multi_fidelity and
+                        0 < self.quick_eval_fraction < 1.0 and
+                        len(data) >= self.multi_fidelity_min_points):
+                    quick_len = max(self.walk_forward_min_size, int(len(data) * self.quick_eval_fraction))
+                    quick_subset = data.iloc[-quick_len:]
+                    quick_strategy = self._clone_strategy(strategy) or strategy
+                    quick_score = backtest_func(quick_strategy, quick_subset, param_dict)
+                    threshold = self._compute_promotion_threshold()
+                    if threshold is not None and quick_score < threshold:
+                        self.quick_eval_scores.append(float(quick_score))
+                        self._record_evaluation(float(quick_score), 0.0, promoted=False)
+                        return -float(quick_score)
+                    self.quick_eval_scores.append(float(quick_score))
+
+                final_score = None
+                variance = 0.0
+                promoted = True
+
+                if usable_cross_validation:
+                    splits = self._generate_walk_forward_splits(data)
+                    if splits:
+                        fold_scores: List[float] = []
+                        parallelizable = self.fold_workers > 1 and len(splits) > 1
+                        if parallelizable:
+                            fold_jobs: List[Tuple[Any, pd.DataFrame]] = []
+                            for train_slice, val_slice in splits:
+                                fold_strategy = self._clone_strategy(strategy)
+                                if fold_strategy is None:
+                                    parallelizable = False
+                                    break
+                                fold_data = self._prepare_fold_data(data, train_slice, val_slice)
+                                fold_jobs.append((fold_strategy, fold_data))
+                            if not fold_jobs:
+                                parallelizable = False
+                        if parallelizable:
+                            with ThreadPoolExecutor(max_workers=min(self.fold_workers, len(fold_jobs))) as executor:
+                                futures = [executor.submit(backtest_func, fold_strategy, fold_data, param_dict)
+                                           for fold_strategy, fold_data in fold_jobs]
+                                for future in futures:
+                                    fold_scores.append(float(future.result()))
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
-                # Track score
-                self.score_history.append(final_score)
-                
-                # Check for convergence
-                if final_score > self.best_score:
-                    improvement = final_score - self.best_score
-                    if improvement < self.convergence_threshold:
-                        self.no_improvement_count += 1
-                    else:
-                        self.no_improvement_count = 0
-                    self.best_score = final_score
-                else:
-                    self.no_improvement_count += 1
-                
-                # Early stopping
-                if self.no_improvement_count >= self.convergence_patience:
-                    if self.convergence_iteration is None:
-                        self.convergence_iteration = len(self.score_history)
-                
+                            for train_slice, val_slice in splits:
+                                fold_data = self._prepare_fold_data(data, train_slice, val_slice)
+                                sequential_strategy = self._clone_strategy(strategy) or strategy
+                                fold_score = backtest_func(sequential_strategy, fold_data, param_dict)
+                                fold_scores.append(float(fold_score))
+
+                        if fold_scores:
+                            final_score = float(np.mean(fold_scores))
+                            variance = float(np.var(fold_scores)) if len(fold_scores) > 1 else 0.0
+
+                if final_score is None:
+                    final_strategy = self._clone_strategy(strategy) or strategy
+                    final_score = float(backtest_func(final_strategy, data, param_dict))
+                    variance = 0.0
+
+                self._record_evaluation(final_score, variance, promoted=promoted)
+
                 # Return negative score for minimization
                 return -final_score
-                
+
             except Exception as e:
                 print(f"Error in objective function: {e}")
                 return 0.0  # Return worst possible score on error
         
         return objective
     
     def optimize_strategy(self, 
                          strategy,
                          strategy_type: str,
                          data: pd.DataFrame,
                          backtest_func,
                          initial_score: float = None,
                          dimensions: List = None,
                          initial_params: Optional[List[Dict[str, Any]]] = None) -> OptimizationResult:
         """
         Optimize strategy parameters using Bayesian optimization
         
         Args:
             strategy: Strategy instance to optimize
             strategy_type: Type of strategy for parameter space
             data: Historical price data (REAL HEX data)
             backtest_func: Function to evaluate strategy
             initial_score: Baseline score before optimization
             
         Returns:
             OptimizationResult with best parameters and performance
         """
         start_time = time.time()
         
         # Reset tracking
         self.param_history = []
         self.score_history = []
         self.best_score = -np.inf
         self.no_improvement_count = 0
         self.convergence_iteration = None
+        self.total_evaluations = 0
+        self.observation_noises = []
+        self.quick_eval_scores = []
+        self.promotion_history = []
+        self._acq_index = 0
         
         # Get parameter space (allow explicit override)
         space = dimensions if dimensions is not None else self.get_parameter_space(strategy_type)
 
         # Create objective function with the resolved space so param names match
         objective = self.create_objective(strategy, data, backtest_func, dimensions=space)
 
         def _dict_to_point(p_dict: Dict[str, Any]) -> Optional[List[Any]]:
             point = []
             for dim in space:
                 if dim.name not in p_dict:
                     return None
                 raw = p_dict[dim.name]
                 try:
                     if isinstance(dim, Integer):
                         val = int(round(float(raw)))
                         val = max(dim.low, min(dim.high, val))
                     elif isinstance(dim, Real):
                         val = float(raw)
                         val = min(dim.high, max(dim.low, val))
                     else:
                         val = raw
                 except Exception:
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
-
         last_result = None
 
-        for idx, acq_func in enumerate(schedule):
-            if remaining_calls <= 0:
-                break
-            segments_left = len(schedule) - idx
-            new_calls = remaining_calls if segments_left <= 1 else max(1, remaining_calls // segments_left)
+        while remaining_calls > 0:
+            acq_func, xi, kappa = self._select_acquisition_settings()
+            new_calls = min(self.batch_size, remaining_calls)
             total_calls = len(evaluated_x) + new_calls
             if total_calls <= len(evaluated_x):
-                continue
+                break
             n_init = max(0, self.n_initial_points - len(evaluated_x))
+            noise_level = self._estimate_noise_level()
             last_result = gp_minimize(
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
+                noise=noise_level,
                 n_jobs=self.n_jobs,
                 random_state=self.random_state,
                 x0=evaluated_x if evaluated_x else None,
                 y0=evaluated_y if evaluated_y else None,
             )
             evaluated_x = list(last_result.x_iters)
             evaluated_y = list(map(float, last_result.func_vals))
             remaining_calls = max(0, target_calls - len(evaluated_x))
             self.total_evaluations = len(self.score_history)
-            if self.convergence_iteration is not None:
+            if self.convergence_iteration is not None and len(self.score_history) >= self.convergence_iteration:
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
 
         result = OptimizationResult(
 
EOF
)