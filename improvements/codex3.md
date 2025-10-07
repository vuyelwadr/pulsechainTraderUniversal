 (cd "$(git rev-parse --show-toplevel)" && git apply --3way <<'EOF' 
diff --git a/optimization/optimizer_bayes.py b/optimization/optimizer_bayes.py
index b28b3e240536806d897bc5071529d5e71139c909..bb5e15f171851793efa69f26cb602f304a3590d3 100644
--- a/optimization/optimizer_bayes.py
+++ b/optimization/optimizer_bayes.py
@@ -1,134 +1,309 @@
 #!/usr/bin/env python3
 """
 Bayesian Parameter Optimizer for Stage 2 Strategy Optimization
 Intelligently tunes strategy parameters to maximize CPS score using Gaussian Process optimization
 """
 
 import numpy as np
-from typing import Dict, List, Tuple, Any, Optional
+from typing import Dict, List, Tuple, Any, Optional, Iterable
 from dataclasses import dataclass
 import json
 import time
-from concurrent.futures import ProcessPoolExecutor
+import os
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
     - Acquisition function switching for better exploration
     - Early stopping on convergence
     - Cross-validation to prevent overfitting
     - Parallel evaluation support
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
+                 cv_splits: int = 3,
+                 cv_min_train_fraction: float = 0.4,
+                 cv_validation_fraction: float = 0.2,
+                 walk_forward_warmup: int = 0,
+                 multi_fidelity: Optional[Dict[str, Any]] = None,
+                 evaluation_workers: Optional[int] = None,
+                 batch_size: int = 1,
+                 noise_window: int = 5):
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
+        self.cv_splits = max(1, int(cv_splits))
+        self.cv_min_train_fraction = max(0.0, min(0.9, cv_min_train_fraction))
+        self.cv_validation_fraction = max(0.05, min(0.9, cv_validation_fraction))
+        self.walk_forward_warmup = max(0, int(walk_forward_warmup))
+        self.multi_fidelity_config = multi_fidelity or {}
+        self.multi_fidelity_config.setdefault('enabled', True)
+        self.multi_fidelity_config.setdefault('fraction', 0.35)
+        self.multi_fidelity_config.setdefault('min_points', 200)
+        self.multi_fidelity_config.setdefault('promote_margin', 0.05)
+        self.multi_fidelity_config.setdefault('absolute_threshold', None)
+        self.multi_fidelity_config.setdefault('penalty', 0.85)
+        self.multi_fidelity_config.setdefault('warmup_tail', 0)
+        self.evaluation_workers = (evaluation_workers if evaluation_workers and evaluation_workers > 0
+                                   else None)
+        self.batch_size = max(1, int(batch_size))
+        self.noise_window = max(1, int(noise_window))
+        self.noise_floor = max(float(noise), 1e-12)
+        self.dynamic_improvement_threshold = max(self.convergence_threshold * 5.0, 1e-4)
 
         # Track optimization progress
         self.param_history = []
         self.score_history = []
+        self.quick_score_history: List[float] = []
         self.best_score = -np.inf
         self.no_improvement_count = 0
         self.convergence_iteration = None
         self.total_evaluations = 0
+        self.observation_variances: List[float] = []
+        self.last_noise_level = self.noise_floor
+        self.dynamic_index = 0
+        self.last_acq_func = self.acq_func.upper() if self.acq_func else 'EI'
 
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
-        
+
+    def _generate_walk_forward_folds(self, data: pd.DataFrame) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
+        total_points = len(data)
+        if total_points < 2:
+            return []
+        min_train = max(1, int(total_points * self.cv_min_train_fraction))
+        val_size = max(1, int(total_points * self.cv_validation_fraction))
+        folds: List[Tuple[pd.DataFrame, pd.DataFrame]] = []
+        start_idx = min_train
+        while start_idx + val_size <= total_points and len(folds) < self.cv_splits:
+            train_slice = data.iloc[:start_idx]
+            val_slice = data.iloc[start_idx:start_idx + val_size]
+            if len(val_slice) == 0:
+                break
+            folds.append((train_slice, val_slice))
+            start_idx += val_size
+        if not folds:
+            cutoff = max(1, total_points - val_size)
+            folds.append((data.iloc[:cutoff], data.iloc[cutoff:]))
+        return folds
+
+    def _prepare_fold_data(self, train_data: pd.DataFrame, val_data: pd.DataFrame) -> pd.DataFrame:
+        if self.walk_forward_warmup <= 0 or len(train_data) == 0:
+            return val_data
+        warmup = train_data.iloc[-min(len(train_data), self.walk_forward_warmup):]
+        if len(warmup) == 0:
+            return val_data
+        return pd.concat([warmup, val_data])
+
+    def _evaluate_folds(self,
+                        strategy,
+                        backtest_func,
+                        params: Dict[str, Any],
+                        folds: Iterable[Tuple[pd.DataFrame, pd.DataFrame]]) -> List[float]:
+        def _run_fold(pair: Tuple[pd.DataFrame, pd.DataFrame]) -> float:
+            train_df, val_df = pair
+            eval_df = self._prepare_fold_data(train_df, val_df)
+            return float(backtest_func(strategy, eval_df, params))
+
+        workers = self.evaluation_workers
+        if workers is None:
+            cpu_count = os.cpu_count() or 1
+            workers = min(self.batch_size, cpu_count)
+        workers = max(1, workers)
+
+        if workers == 1:
+            return [_run_fold(fold) for fold in folds]
+
+        with ThreadPoolExecutor(max_workers=workers) as executor:
+            return list(executor.map(_run_fold, folds))
+
+    def _prepare_quick_subset(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
+        cfg = self.multi_fidelity_config
+        if not cfg.get('enabled', True):
+            return None
+        fraction = float(cfg.get('fraction', 0.35))
+        min_points = int(cfg.get('min_points', 200))
+        total_points = len(data)
+        subset_size = max(min_points, int(total_points * fraction))
+        if subset_size <= 0 or subset_size >= total_points:
+            return None
+        return data.iloc[-subset_size:]
+
+    def _should_promote_full_evaluation(self, quick_score: float) -> bool:
+        if not self.multi_fidelity_config.get('enabled', True):
+            return True
+        if not np.isfinite(quick_score):
+            return True
+        if not self.score_history:
+            return True
+        margin = float(self.multi_fidelity_config.get('promote_margin', 0.05))
+        absolute_threshold = self.multi_fidelity_config.get('absolute_threshold')
+        reference = self.best_score if np.isfinite(self.best_score) and self.best_score != -np.inf else max(self.score_history)
+        if not np.isfinite(reference):
+            reference = quick_score
+        cutoff = reference - margin * max(1.0, abs(reference))
+        if absolute_threshold is not None:
+            cutoff = max(cutoff, float(absolute_threshold))
+        return quick_score >= cutoff
+
+    def _update_score_tracking(self, final_score: float) -> None:
+        self.score_history.append(final_score)
+        if self.best_score == -np.inf or not np.isfinite(self.best_score):
+            self.best_score = final_score
+            self.no_improvement_count = 0
+        elif final_score > self.best_score:
+            improvement = final_score - self.best_score
+            if improvement < self.convergence_threshold:
+                self.no_improvement_count += 1
+            else:
+                self.no_improvement_count = 0
+            self.best_score = final_score
+        else:
+            self.no_improvement_count += 1
+        if self.no_improvement_count >= self.convergence_patience and self.convergence_iteration is None:
+            self.convergence_iteration = len(self.score_history)
+
+    def _record_noise(self, scores: Iterable[float]) -> None:
+        clean_scores = [float(s) for s in scores if np.isfinite(s)]
+        if not clean_scores:
+            return
+        if len(clean_scores) > 1:
+            variance = float(np.var(clean_scores, ddof=1))
+        else:
+            variance = 0.0
+        if not np.isfinite(variance) or variance < self.noise_floor:
+            variance = self.noise_floor
+        self.observation_variances.append(variance)
+        if len(self.observation_variances) > self.noise_window * 5:
+            del self.observation_variances[:-self.noise_window * 5]
+
+    def _current_noise(self) -> float:
+        if not self.observation_variances:
+            return self.noise_floor
+        recent = self.observation_variances[-self.noise_window:]
+        return float(max(self.noise_floor, np.median(recent)))
+
+    def _recent_improvement(self, window: int = 5) -> float:
+        if len(self.score_history) < 2:
+            return float('inf')
+        window = max(1, min(window, len(self.score_history) - 1))
+        recent_slice = self.score_history[-window:]
+        prior_slice = self.score_history[:-window]
+        if not prior_slice:
+            prior_best = self.score_history[0]
+        else:
+            prior_best = max(prior_slice)
+        recent_best = max(recent_slice)
+        return recent_best - prior_best
+
+    def _select_acquisition(self) -> str:
+        base = self.last_acq_func if self.last_acq_func else (self.acq_func.upper() if self.acq_func else 'EI')
+        if self.acq_sequence:
+            base = self.acq_sequence[min(self.dynamic_index, len(self.acq_sequence) - 1)]
+        improvement = self._recent_improvement()
+        if not np.isfinite(improvement) or improvement > self.dynamic_improvement_threshold * 3:
+            chosen = 'EI'
+        elif improvement < self.dynamic_improvement_threshold:
+            chosen = 'LCB' if self.no_improvement_count >= max(1, self.convergence_patience // 2) else 'PI'
+        else:
+            chosen = base
+        self.last_acq_func = chosen
+        return chosen
+
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
             return [
diff --git a/optimization/optimizer_bayes.py b/optimization/optimizer_bayes.py
index b28b3e240536806d897bc5071529d5e71139c909..bb5e15f171851793efa69f26cb602f304a3590d3 100644
--- a/optimization/optimizer_bayes.py
+++ b/optimization/optimizer_bayes.py
@@ -164,95 +339,77 @@ class BayesianOptimizer:
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
-                        else:
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
+                final_score = None
+                fold_scores: List[float] = []
+
+                quick_subset = self._prepare_quick_subset(data)
+                if quick_subset is not None:
+                    quick_score = float(backtest_func(strategy, quick_subset, param_dict))
+                    self.quick_score_history.append(quick_score)
+                    self._record_noise([quick_score])
+                    if not self._should_promote_full_evaluation(quick_score):
+                        final_score = quick_score * float(self.multi_fidelity_config.get('penalty', 0.85))
+
+                if final_score is None:
+                    if cross_validate:
+                        folds = self._generate_walk_forward_folds(data)
+                        if folds:
+                            fold_scores = self._evaluate_folds(strategy, backtest_func, param_dict, folds)
+                            if fold_scores:
+                                final_score = float(np.mean(fold_scores))
+                                self._record_noise(fold_scores)
+                    if final_score is None:
+                        # Single evaluation on full dataset
+                        final_score = float(backtest_func(strategy, data, param_dict))
+                        self._record_noise([final_score])
+
                 # Track score
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
+                self._update_score_tracking(final_score)
+
                 # Return negative score for minimization
                 return -final_score
                 
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
diff --git a/optimization/optimizer_bayes.py b/optimization/optimizer_bayes.py
index b28b3e240536806d897bc5071529d5e71139c909..bb5e15f171851793efa69f26cb602f304a3590d3 100644
--- a/optimization/optimizer_bayes.py
+++ b/optimization/optimizer_bayes.py
@@ -295,89 +452,78 @@ class BayesianOptimizer:
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
-        last_result = None
-
-        for idx, acq_func in enumerate(schedule):
-            if remaining_calls <= 0:
-                break
-            segments_left = len(schedule) - idx
-            new_calls = remaining_calls if segments_left <= 1 else max(1, remaining_calls // segments_left)
-            total_calls = len(evaluated_x) + new_calls
+        while remaining_calls > 0 and (self.convergence_iteration is None):
+            step = min(self.batch_size, remaining_calls)
+            total_calls = len(evaluated_x) + step
             if total_calls <= len(evaluated_x):
-                continue
+                break
             n_init = max(0, self.n_initial_points - len(evaluated_x))
-            last_result = gp_minimize(
+            acq_func = self._select_acquisition()
+            noise_level = self._current_noise()
+            result = gp_minimize(
                 func=objective,
                 dimensions=space,
                 n_calls=total_calls,
                 n_initial_points=n_init,
                 acq_func=acq_func,
                 xi=self.xi,
                 kappa=self.kappa,
-                noise=self.noise,
+                noise=noise_level,
                 n_jobs=self.n_jobs,
                 random_state=self.random_state,
                 x0=evaluated_x if evaluated_x else None,
                 y0=evaluated_y if evaluated_y else None,
             )
-            evaluated_x = list(last_result.x_iters)
-            evaluated_y = list(map(float, last_result.func_vals))
+            evaluated_x = list(result.x_iters)
+            evaluated_y = list(map(float, result.func_vals))
             remaining_calls = max(0, target_calls - len(evaluated_x))
             self.total_evaluations = len(self.score_history)
-            if self.convergence_iteration is not None:
-                break
+            self.dynamic_index += 1
+            self.last_noise_level = noise_level
 
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
             strategy_name=strategy.__class__.__name__,
 
EOF
)