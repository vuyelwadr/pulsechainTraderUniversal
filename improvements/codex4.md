 (cd "$(git rev-parse --show-toplevel)" && git apply --3way <<'EOF' 
diff --git a/optimization/optimizer_bayes.py b/optimization/optimizer_bayes.py
index b28b3e240536806d897bc5071529d5e71139c909..74ecd7c1e13fb4fa31ee6453034d00242db22976 100644
--- a/optimization/optimizer_bayes.py
+++ b/optimization/optimizer_bayes.py
@@ -1,133 +1,228 @@
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
+                 dynamic_acq: bool = True,
+                 acq_improvement_window: int = 5,
+                 acq_cooldown: int = 3,
+                 walk_forward_splits: int = 3,
+                 noise_floor: float = 1e-8,
+                 multi_fidelity: bool = True,
+                 coarse_fraction: float = 0.35,
+                 promotion_threshold: float = 0.0,
+                 coarse_penalty: float = 0.5,
+                 parallel_folds: bool = True,
+                 max_fold_workers: Optional[int] = None,
+                 batch_size: int = 1):
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
+        self.dynamic_acq = dynamic_acq
+        self.acq_improvement_window = max(1, acq_improvement_window)
+        self.acq_cooldown = max(1, acq_cooldown)
+        self.walk_forward_splits = max(1, walk_forward_splits)
+        self.noise_floor = max(noise_floor, 1e-12)
+        self.multi_fidelity = multi_fidelity
+        self.coarse_fraction = min(max(coarse_fraction, 0.05), 0.9)
+        self.promotion_threshold = promotion_threshold
+        self.coarse_penalty = max(coarse_penalty, 0.0)
+        self.parallel_folds = parallel_folds
+        self.max_fold_workers = max_fold_workers
+        self.batch_size = max(1, batch_size)
 
         # Track optimization progress
         self.param_history = []
         self.score_history = []
         self.best_score = -np.inf
         self.no_improvement_count = 0
         self.convergence_iteration = None
         self.total_evaluations = 0
+        self.observation_variances: List[float] = []
+        self.current_noise = max(self.noise, self.noise_floor)
+        self._acq_cooldown_counter = 0
 
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
+    def _generate_walk_forward_splits(self, data: pd.DataFrame, n_splits: int) -> Iterable[Tuple[pd.DataFrame, pd.DataFrame]]:
+        """Yield expanding window walk-forward train/validation splits."""
+        total = len(data)
+        if total < 4 or n_splits <= 0:
+            yield data.iloc[:-1], data.iloc[-1:]
+            return
+
+        step = max(1, total // (n_splits + 1))
+        min_train = max(5, step)
+        start_train = 0
+        train_end = min_train
+
+        for split_idx in range(n_splits):
+            val_start = train_end
+            if val_start >= total - 1:
+                break
+            val_end = val_start + step if split_idx < n_splits - 1 else total
+            if val_end > total:
+                val_end = total
+            train_slice = data.iloc[start_train:train_end]
+            val_slice = data.iloc[val_start:val_end]
+            if len(train_slice) < min_train or len(val_slice) == 0:
+                break
+            yield train_slice, val_slice
+            train_end = val_end
+
+    def _recent_improvement(self, window: int) -> float:
+        if len(self.score_history) <= window:
+            return 0.0
+        recent = self.score_history[-(window + 1):]
+        diffs = np.diff(recent)
+        return float(np.mean(diffs)) if len(diffs) else 0.0
+
+    def _choose_acquisition(self, current: str) -> str:
+        current = (current or self.acq_func or 'EI').upper()
+        if not self.dynamic_acq or len(self.score_history) < 2:
+            return current
+
+        if self._acq_cooldown_counter > 0:
+            self._acq_cooldown_counter -= 1
+            return current
+
+        improvement = self._recent_improvement(self.acq_improvement_window)
+        if improvement > self.convergence_threshold * 5:
+            chosen = 'EI'
+        elif improvement > 0:
+            chosen = 'PI' if current == 'EI' else current
+        else:
+            chosen = 'LCB' if current != 'LCB' else 'EI'
+
+        if chosen != current:
+            self._acq_cooldown_counter = self.acq_cooldown
+        return chosen
+
+    def _update_noise(self, observed_variance: float) -> None:
+        if observed_variance is not None and not np.isnan(observed_variance):
+            self.observation_variances.append(float(max(observed_variance, 0.0)))
+        if self.observation_variances:
+            window = self.observation_variances[-(self.acq_improvement_window * 2):]
+            avg_var = float(np.mean(window)) if window else self.noise
+            self.current_noise = max(self.noise_floor, avg_var)
+        else:
+            self.current_noise = max(self.noise_floor, self.noise)
+
+    def _evaluate_backtest(self, strategy, dataset: pd.DataFrame, param_dict: Dict[str, Any], backtest_func) -> float:
+        strat_instance = strategy.__class__()
+        return backtest_func(strat_instance, dataset, param_dict)
         
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
index b28b3e240536806d897bc5071529d5e71139c909..74ecd7c1e13fb4fa31ee6453034d00242db22976 100644
--- a/optimization/optimizer_bayes.py
+++ b/optimization/optimizer_bayes.py
@@ -155,107 +250,123 @@ class BayesianOptimizer:
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
         
         Args:
             strategy: Strategy instance to optimize
             data: Historical price data (REAL HEX data only!)
             backtest_func: Function to run backtest and return CPS score
             cross_validate: Whether to use cross-validation
         """
         def objective(params):
             """Objective function to minimize (negative CPS score)"""
             try:
-                # Convert params list to dict using provided dimensions (names)
-                param_dict = {}
                 space = dimensions if dimensions is not None else self.get_parameter_space(strategy.__class__.__name__)
-                for i, param in enumerate(params):
-                    param_dict[space[i].name] = param
-                
+                param_dict = {dim.name: val for dim, val in zip(space, params)}
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
+                final_score: float
+                variance = 0.0
+
+                promoted = True
+                coarse_score = None
+                if self.multi_fidelity and len(data) > 10:
+                    coarse_len = max(5, int(len(data) * self.coarse_fraction))
+                    if coarse_len < len(data):
+                        coarse_slice = data.iloc[-coarse_len:]
+                    else:
+                        coarse_slice = data
+                    coarse_score = self._evaluate_backtest(strategy, coarse_slice, param_dict, backtest_func)
+                    if coarse_score < self.promotion_threshold:
+                        promoted = False
+                        final_score = coarse_score * (self.coarse_penalty if self.coarse_penalty > 0 else 1.0)
+                    else:
+                        final_score = coarse_score
+
+                if promoted:
+                    if cross_validate:
+                        splits = list(self._generate_walk_forward_splits(data, self.walk_forward_splits))
+                        if not splits:
+                            final_score = self._evaluate_backtest(strategy, data, param_dict, backtest_func)
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
+                            val_slices = [val for _, val in splits]
+                            fold_scores: List[float] = []
+                            if self.parallel_folds and len(val_slices) > 1:
+                                workers = self.max_fold_workers or (self.n_jobs if self.n_jobs and self.n_jobs > 0 else len(val_slices))
+                                workers = max(1, min(workers, len(val_slices)))
+                                if workers > 1:
+                                    with ThreadPoolExecutor(max_workers=workers) as executor:
+                                        futures = [executor.submit(self._evaluate_backtest, strategy, val_slice, param_dict, backtest_func)
+                                                   for val_slice in val_slices]
+                                        fold_scores = [float(f.result()) for f in futures]
+                            if not fold_scores:
+                                fold_scores = [float(self._evaluate_backtest(strategy, val_slice, param_dict, backtest_func))
+                                               for val_slice in val_slices]
+                            final_score = float(np.mean(fold_scores)) if fold_scores else 0.0
+                            variance = float(np.var(fold_scores, ddof=1)) if len(fold_scores) > 1 else 0.0
+                    else:
+                        final_score = self._evaluate_backtest(strategy, data, param_dict, backtest_func)
                 else:
-                    # Single evaluation on full dataset
-                    final_score = backtest_func(strategy, data, param_dict)
-                
-                # Track score
+                    variance = 0.0
+
+                # Track score and adaptive noise
                 self.score_history.append(final_score)
-                
+                self._update_noise(variance)
+
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
-                
+
                 # Early stopping
                 if self.no_improvement_count >= self.convergence_patience:
                     if self.convergence_iteration is None:
                         self.convergence_iteration = len(self.score_history)
-                
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
diff --git a/optimization/optimizer_bayes.py b/optimization/optimizer_bayes.py
index b28b3e240536806d897bc5071529d5e71139c909..74ecd7c1e13fb4fa31ee6453034d00242db22976 100644
--- a/optimization/optimizer_bayes.py
+++ b/optimization/optimizer_bayes.py
@@ -298,75 +409,78 @@ class BayesianOptimizer:
 
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
 
         if self.acq_sequence:
             schedule = list(self.acq_sequence)
         else:
-            if target_calls < 30:
-                schedule = [self.acq_func.upper() if self.acq_func else 'EI']
-            else:
-                schedule = ['EI', 'PI', 'LCB']
+            schedule = [self.acq_func.upper() if self.acq_func else 'EI']
 
+        schedule_idx = 0
         last_result = None
 
-        for idx, acq_func in enumerate(schedule):
-            if remaining_calls <= 0:
-                break
-            segments_left = len(schedule) - idx
-            new_calls = remaining_calls if segments_left <= 1 else max(1, remaining_calls // segments_left)
+        while remaining_calls > 0:
+            base_acq = schedule[schedule_idx % len(schedule)] if schedule else 'EI'
+            adaptive_acq = self._choose_acquisition(base_acq)
+            schedule_idx += 1
+
+            new_calls = min(self.batch_size, remaining_calls)
             total_calls = len(evaluated_x) + new_calls
             if total_calls <= len(evaluated_x):
-                continue
-            n_init = max(0, self.n_initial_points - len(evaluated_x))
+                break
+
+            if evaluated_x:
+                n_init = 0
+            else:
+                n_init = min(self.n_initial_points, total_calls)
             last_result = gp_minimize(
                 func=objective,
                 dimensions=space,
                 n_calls=total_calls,
                 n_initial_points=n_init,
-                acq_func=acq_func,
+                acq_func=adaptive_acq,
                 xi=self.xi,
                 kappa=self.kappa,
-                noise=self.noise,
+                noise=self.current_noise,
                 n_jobs=self.n_jobs,
                 random_state=self.random_state,
                 x0=evaluated_x if evaluated_x else None,
                 y0=evaluated_y if evaluated_y else None,
             )
             evaluated_x = list(last_result.x_iters)
             evaluated_y = list(map(float, last_result.func_vals))
             remaining_calls = max(0, target_calls - len(evaluated_x))
             self.total_evaluations = len(self.score_history)
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
 
EOF
)