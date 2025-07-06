import numpy as np
import time
import warnings
from typing import List, Tuple, Optional, Dict, Any, Callable
from scipy.optimize import brentq
from problems import BaseProblem

# --- 定数定義 ---
EPSILON_L = 100
CA_AMPLIFICATION = 5.0
TAU = 2
ALPHA_L = 0.85
ALPHA_U = 1.20
ALPHA_M_LOCAL = 0.20
ALPHA_M_GLOBAL = 0.05
M_GLOBAL_REBOOT = 100

# --- 数値計算基盤クラス (修正済み) ---
class NumericalUtils:
    @staticmethod
    def ensure_numerical_stability(matrix: np.ndarray) -> np.ndarray:
        matrix = 0.5 * (matrix + matrix.T)
        try:
            eigenvals, eigenvecs = np.linalg.eigh(matrix)
            eigenvals = np.maximum(eigenvals, 1e-14)
            max_eigenval, min_eigenval = np.max(eigenvals), np.min(eigenvals)
            if max_eigenval > 0 and min_eigenval > 0 and max_eigenval / min_eigenval > 1e14:
                eigenvals = np.maximum(eigenvals, max_eigenval / 1e14)
            return eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
        except np.linalg.LinAlgError:
            return np.eye(matrix.shape[0])

    @staticmethod
    def compute_matrix_inverse_sqrt(matrix: np.ndarray) -> np.ndarray:
        try:
            eigenvals, eigenvecs = np.linalg.eigh(matrix)
            eigenvals = np.maximum(eigenvals, 1e-14)
            return eigenvecs @ np.diag(1.0 / np.sqrt(eigenvals)) @ eigenvecs.T
        except np.linalg.LinAlgError:
            return np.eye(matrix.shape[0])

    @staticmethod
    def mirror_boundary_mapping(x: np.ndarray, bounds: np.ndarray) -> np.ndarray:
        x_mapped = x.copy()
        # NaNやinf値を事前にチェック
        if not np.all(np.isfinite(x_mapped)):
            return np.clip(x, bounds[:, 0], bounds[:, 1])

        for d in range(len(x)):
            lb, ub = bounds[d, 0], bounds[d, 1]
            if ub <= lb:
                x_mapped[d] = lb
                continue

            width = ub - lb
            if x_mapped[d] < lb or x_mapped[d] > ub:
                relative_pos = (x_mapped[d] - lb) % (2 * width)
                if relative_pos <= width:
                    x_mapped[d] = lb + relative_pos
                else:
                    x_mapped[d] = ub - (relative_pos - width)
        
        return np.clip(x_mapped, bounds[:, 0], bounds[:, 1])

# --- Firework基底クラス (スケール更新を修正) ---
class Firework:
    # ... (init, _initialize_mean, _initialize_scale, generate_sparks, compute_recombination_weightsは変更なし) ...
    def __init__(self, dimension: int, bounds: np.ndarray, firework_type: str = 'local', 
                 firework_id: int = 0, num_local_fireworks: int = 4):
        self.dimension = dimension
        self.bounds = np.array(bounds)
        self.firework_type = firework_type
        self.firework_id = firework_id
        self.num_local_fireworks = num_local_fireworks
        self.mean = self._initialize_mean()
        self.covariance = np.eye(dimension)
        self.scale = self._initialize_scale()
        self.evolution_path_c = np.zeros(dimension)
        self.evolution_path_sigma = np.zeros(dimension)
        self.learning_rates = self._initialize_learning_rates()
        self.best_fitness = float('inf')
        self.best_solution = None
        self.stagnation_count = 0
        self.recent_fitness_history = []
        self.evaluation_count = 0
        self.last_improvement_iteration = 0
        self.restart_count = 0
        self.min_scale = 1e-12
        self.max_scale = 1e12

    def _initialize_mean(self) -> np.ndarray:
        if self.firework_type == 'global':
            return (self.bounds[:, 0] + self.bounds[:, 1]) / 2
        else:
            return np.random.uniform(self.bounds[:, 0], self.bounds[:, 1])

    def _initialize_scale(self) -> float:
        ub = np.max(self.bounds[:, 1])
        lb = np.min(self.bounds[:, 0])
        expected_norm = np.sqrt(self.dimension)
        if self.firework_type == 'global':
            return (ub - lb) / (2 * expected_norm)
        else:
            global_scale = (ub - lb) / (2 * expected_norm)
            return global_scale / self.num_local_fireworks

    def _initialize_learning_rates(self) -> Dict[str, float]:
        D = self.dimension
        if self.firework_type == 'global':
            return {'cm': 1.0, 'c_mu': 0.25, 'c1': 0.0, 'cc': 0.0, 'c_sigma': 0.0, 'd_sigma': 0.0, 'cr': 0.5, 'cg': 1.0 / self.num_local_fireworks}
        else:
            return {'cm': 1.0, 'c_mu': 0.25, 'c1': 2.0 / ((D + 1.3)**2), 'cc': 4.0 / (D + 4.0), 'c_sigma': 4.0 / (D + 4.0), 'd_sigma': 1.0 + 2.0 * max(0, np.sqrt((D-1)/(D+1)) - 1) * 0.5, 'cr': 0.5}

    def generate_sparks(self, num_sparks: int) -> np.ndarray:
        try:
            sparks = np.random.multivariate_normal(mean=self.mean, cov=self.scale**2 * self.covariance, size=num_sparks)
        except (np.linalg.LinAlgError, ValueError):
            self.covariance = NumericalUtils.ensure_numerical_stability(self.covariance)
            sparks = np.random.multivariate_normal(mean=self.mean, cov=self.scale**2 * self.covariance, size=num_sparks)
        
        sparks = np.apply_along_axis(NumericalUtils.mirror_boundary_mapping, 1, sparks, self.bounds)
        return sparks

    def compute_recombination_weights(self, fitness_values: np.ndarray) -> np.ndarray:
        if self.firework_type == 'local':
            mu = len(fitness_values) // 2
            sorted_indices = np.argsort(fitness_values)
            weights = np.zeros(len(fitness_values))
            for rank, idx in enumerate(sorted_indices[:mu]):
                w = max(0, np.log(mu + 0.5) - np.log(rank + 1))
                weights[idx] = w
            if np.sum(weights) > 0: weights = weights / np.sum(weights)
        else:
            num_select = int(0.95 * len(fitness_values))
            sorted_indices = np.argsort(fitness_values)[:num_select]
            weights = np.zeros(len(fitness_values))
            if num_select > 0: weights[sorted_indices] = 1.0 / num_select
        return weights

    # ★★★ update_parametersメソッドを修正 (スケール更新の安定化) ★★★
    def update_parameters(self, sparks: np.ndarray, fitness_values: np.ndarray) -> None:
        if sparks.shape[0] == 0: return

        weights = self.compute_recombination_weights(fitness_values)
        mu_eff = 1.0 / np.sum(weights**2) if np.sum(weights**2) > 0 else 1.0
        
        weighted_diff = np.zeros(self.dimension)
        if np.sum(weights) > 0:
            weighted_diff = np.dot(weights, sparks - self.mean)

        new_mean = self.mean + self.learning_rates['cm'] * weighted_diff
        
        if (self.firework_type == 'local' and self.learning_rates['d_sigma'] > 0 and self.learning_rates['c_sigma'] > 0):
            try:
                C_inv_sqrt = NumericalUtils.compute_matrix_inverse_sqrt(self.covariance)
                mean_diff_scaled = (new_mean - self.mean) / self.scale
                
                self.evolution_path_sigma = ((1 - self.learning_rates['c_sigma']) * self.evolution_path_sigma + 
                                           np.sqrt(self.learning_rates['c_sigma'] * (2 - self.learning_rates['c_sigma']) * mu_eff) * (C_inv_sqrt @ mean_diff_scaled))
                
                # スケール更新量の計算とクリッピング（発散防止）
                path_norm = np.linalg.norm(self.evolution_path_sigma)
                expected_norm = np.sqrt(self.dimension)
                log_scale_change = (self.learning_rates['c_sigma'] / self.learning_rates['d_sigma']) * (path_norm / expected_norm - 1)
                log_scale_change = np.clip(log_scale_change, -1.0, 1.0) # ★発散を防ぐためのクリッピング
                
                self.scale *= np.exp(log_scale_change)
                self.scale = np.clip(self.scale, self.min_scale, self.max_scale)
            except np.linalg.LinAlgError:
                pass # 数値エラー時はスケール更新をスキップ

        # 共分散行列と平均の更新
        reference_mean = ((1 - self.learning_rates['cr']) * self.mean + self.learning_rates['cr'] * new_mean)
        if self.learning_rates['cc'] > 0:
            self.evolution_path_c = ((1 - self.learning_rates['cc']) * self.evolution_path_c + np.sqrt(self.learning_rates['cc'] * (2 - self.learning_rates['cc']) * mu_eff) * (new_mean - self.mean) / self.scale)
        
        rank_mu_update = np.zeros((self.dimension, self.dimension))
        if np.sum(weights) > 0:
            y = sparks - reference_mean
            rank_mu_update = np.dot(y.T * weights, y)
        
        rank_one_update = np.outer(self.evolution_path_c, self.evolution_path_c)
        self.covariance = ((1 - self.learning_rates['c_mu'] - self.learning_rates['c1']) * self.covariance + self.learning_rates['c_mu'] * rank_mu_update + self.learning_rates['c1'] * rank_one_update)
        self.covariance = NumericalUtils.ensure_numerical_stability(self.covariance)
        
        self.mean = new_mean
        self._update_best_solution(sparks, fitness_values)
        self.evaluation_count += len(sparks)
    
    def _update_best_solution(self, sparks: np.ndarray, fitness_values: np.ndarray) -> None:
        best_idx = np.argmin(fitness_values)
        current_best = fitness_values[best_idx]
        if current_best < self.best_fitness:
            self.best_fitness = current_best
            self.best_solution = sparks[best_idx].copy()
            self.stagnation_count = 0
            self.last_improvement_iteration = self.evaluation_count
        else:
            self.stagnation_count += 1
        self.recent_fitness_history.append(current_best)
        if len(self.recent_fitness_history) > 50: self.recent_fitness_history.pop(0)

    def compute_boundary_radius(self, direction: Optional[np.ndarray] = None) -> float:
        d_B = np.sqrt(self.dimension) + 0.5 * np.sqrt(2 * self.dimension)
        if direction is None:
            avg_eigenval = np.trace(self.covariance) / self.dimension
            return self.scale * np.sqrt(avg_eigenval) * d_B
        else:
            direction_norm = np.linalg.norm(direction)
            if direction_norm == 0: return 0.0
            unit_direction = direction / direction_norm
            radius_squared = unit_direction.T @ self.covariance @ unit_direction
            return self.scale * np.sqrt(radius_squared) * d_B
            
    def check_restart_conditions(self, all_fireworks: List['Firework']) -> Tuple[bool, List[str]]:
        restart_reasons = []
        if len(self.recent_fitness_history) > 1 and np.std(self.recent_fitness_history) <= 1e-5: restart_reasons.append("fitness_converged")
        if self.scale * np.linalg.norm(self.covariance, 2) <= 1e-5: restart_reasons.append("position_converged")
        max_stagnation = 4 * EPSILON_L if self.firework_type == 'global' else EPSILON_L
        if self.stagnation_count >= max_stagnation: restart_reasons.append("not_improving")
        for fw in all_fireworks:
            if fw.firework_id != self.firework_id and fw.best_fitness < self.best_fitness:
                if np.linalg.norm(self.mean - fw.mean) < 1e-5:
                    restart_reasons.append("mean_converged")
                    break
        for fw in all_fireworks:
            if fw.firework_id != self.firework_id and fw.best_fitness < self.best_fitness:
                distance = np.linalg.norm(self.mean - fw.mean)
                fw_radius, self_radius = fw.compute_boundary_radius(), self.compute_boundary_radius()
                if fw_radius > self_radius and distance + self_radius < fw_radius * 1.1:
                    restart_reasons.append("covered_by_better")
                    break
        return len(restart_reasons) > 0, restart_reasons

    def restart(self) -> None:
        self.mean = self._initialize_mean()
        self.covariance = np.eye(self.dimension)
        self.scale = self._initialize_scale()
        self.evolution_path_c = np.zeros(self.dimension)
        self.evolution_path_sigma = np.zeros(self.dimension)
        self.stagnation_count = 0
        self.recent_fitness_history = []
        self.restart_count += 1
        self.best_fitness = float('inf')
        self.best_solution = None

# --- 協調戦略管理クラス (修正済み) ---
class CollaborationManager:
    # ... (前回のコードから変更ありません) ...
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.ca = CA_AMPLIFICATION
        self.min_distance = 1e-12
        self.max_w_value = 10.0
        self.min_w_value = -10.0

    def execute_collaboration(self, fireworks: List) -> None:
        if len(fireworks) < 2: return
        dividing_points = self._compute_all_dividing_points(fireworks)
        for fw in fireworks:
            feature_points = self._select_feature_points(fw, dividing_points, fireworks)
            if feature_points: self._adapt_to_feature_points(fw, feature_points)

    def _compute_all_dividing_points(self, fireworks: List) -> Dict[Tuple[int, int], float]:
        dividing_points = {}
        for i in range(len(fireworks)):
            for j in range(i + 1, len(fireworks)):
                fw1, fw2 = fireworks[i], fireworks[j]
                try:
                    w = self._solve_dividing_equation(fw1, fw2)
                    dividing_points[(fw1.firework_id, fw2.firework_id)] = w
                    dividing_points[(fw2.firework_id, fw1.firework_id)] = w
                except Exception:
                    dividing_points[(fw1.firework_id, fw2.firework_id)] = 0.0
                    dividing_points[(fw2.firework_id, fw1.firework_id)] = 0.0
        return dividing_points

    def _solve_dividing_equation(self, fw1, fw2) -> float:
        distance = np.linalg.norm(fw1.mean - fw2.mean)
        if distance < self.min_distance: return 0.0
        r1, r2 = self._compute_radius_on_line(fw1, fw2), self._compute_radius_on_line(fw2, fw1)
        a1, a2 = self._compute_sensitivity_factors(fw1, fw2)
        
        if fw1.firework_type == 'local' and fw2.firework_type == 'local':
            equation = lambda w: r1 * np.exp(a1 * w) + r2 * np.exp(a2 * w) - distance
        elif fw1.firework_type == 'global':
            equation = lambda w: r1 * np.exp(-a1 * w) - r2 * np.exp(a2 * w) - distance
        else: # fw2 is global
            equation = lambda w: r1 * np.exp(a1 * w) - r2 * np.exp(-a2 * w) - distance
        
        try:
            fa, fb = equation(self.min_w_value), equation(self.max_w_value)
            if np.sign(fa) != np.sign(fb):
                return brentq(equation, self.min_w_value, self.max_w_value, xtol=1e-12, rtol=1e-12)
        except (ValueError, RuntimeError):
            pass 
        return 0.0

    def _compute_radius_on_line(self, fw1, fw2) -> float:
        direction = fw2.mean - fw1.mean
        direction_norm = np.linalg.norm(direction)
        if direction_norm < self.min_distance: return fw1.compute_boundary_radius()
        unit_direction = direction / direction_norm
        return fw1.compute_boundary_radius(unit_direction)

    def _compute_sensitivity_factors(self, fw1, fw2) -> Tuple[float, float]:
        a1, a2 = 1.0, 1.0
        if fw1.best_fitness < fw2.best_fitness: a1, a2 = 0.0, 1.0
        elif fw2.best_fitness < fw1.best_fitness: a1, a2 = 1.0, 0.0
        improvement_threshold = 0.2 * EPSILON_L
        if (fw1.firework_type == 'local' and hasattr(fw1, 'last_improvement_iteration') and fw1.evaluation_count - fw1.last_improvement_iteration < improvement_threshold): a1 = 0.0
        if (fw2.firework_type == 'local' and hasattr(fw2, 'last_improvement_iteration') and fw2.evaluation_count - fw2.last_improvement_iteration < improvement_threshold): a2 = 0.0
        if fw1.firework_type == 'global': a1 *= self.ca
        if fw2.firework_type == 'global': a2 *= self.ca
        return a1, a2

    def _select_feature_points(self, firework, dividing_points: Dict, fireworks: List) -> List[np.ndarray]:
        potential_points = []
        for other_fw in fireworks:
            if other_fw.firework_id == firework.firework_id: continue
            key = (firework.firework_id, other_fw.firework_id)
            if key in dividing_points:
                w = dividing_points[key]
                distance = np.linalg.norm(firework.mean - other_fw.mean)
                if distance > self.min_distance:
                    r, (a_self, _) = self._compute_radius_on_line(firework, other_fw), self._compute_sensitivity_factors(firework, other_fw)
                    r_new = r * np.exp(a_self * w) if firework.firework_type == 'local' else r * np.exp(-a_self * w)
                    direction = (other_fw.mean - firework.mean) / distance
                    potential_points.append(firework.mean + r_new * direction)
        if not potential_points: return []
        distances = [np.linalg.norm(p - firework.mean) for p in potential_points]
        indices = np.argsort(distances)[:min(TAU, len(distances))] if firework.firework_type == 'local' else np.argsort(distances)[-min(TAU, len(distances)):]
        selected_points = [potential_points[i] for i in indices]
        clipped_points = []
        for point in selected_points:
            distance, radius = np.linalg.norm(point - firework.mean), firework.compute_boundary_radius()
            if distance > 1e-9: # ゼロ除算を回避
                if distance < ALPHA_L * radius: point = firework.mean + (point - firework.mean) * (ALPHA_L * radius / distance)
                elif distance > ALPHA_U * radius: point = firework.mean + (point - firework.mean) * (ALPHA_U * radius / distance)
            clipped_points.append(point)
        return clipped_points

    def _adapt_to_feature_points(self, firework, feature_points: List[np.ndarray]) -> None:
        if not feature_points: return
        shift_vector = np.zeros(self.dimension)
        for f_point in feature_points:
            q_point = self._compute_boundary_intersection(firework, f_point)
            shift_vector += (f_point - q_point)
        shift_vector /= len(feature_points)
        max_shift_ratio = ALPHA_M_LOCAL if firework.firework_type == 'local' else ALPHA_M_GLOBAL
        shift_norm = np.linalg.norm(shift_vector)
        if shift_norm > 0:
            max_shift = max_shift_ratio * firework.compute_boundary_radius()
            if shift_norm > max_shift: shift_vector *= max_shift / shift_norm
        new_mean = firework.mean + shift_vector
        covariance_adjustment, boundary_radius = np.zeros((self.dimension, self.dimension)), firework.scale * np.sqrt(self.dimension)
        for f_point in feature_points:
            try:
                C_inv_sqrt = NumericalUtils.compute_matrix_inverse_sqrt(firework.covariance)
                z = C_inv_sqrt @ (f_point - new_mean) / firework.scale
                z_norm_squared = np.dot(z, z)
                if z_norm_squared > 1e-12:
                    lambda_val = 1.0 / (boundary_radius**2) - 1.0 / z_norm_squared
                    adjustment_vector = f_point - new_mean
                    covariance_adjustment += (lambda_val / (firework.scale**2) * np.outer(adjustment_vector, adjustment_vector))
            except Exception: continue
        if len(feature_points) > 0:
            covariance_adjustment /= len(feature_points)
            new_covariance = NumericalUtils.ensure_numerical_stability(firework.covariance + covariance_adjustment)
            firework.mean, firework.covariance = new_mean, new_covariance

    def _compute_boundary_intersection(self, firework, point: np.ndarray) -> np.ndarray:
        direction = point - firework.mean
        direction_norm = np.linalg.norm(direction)
        if direction_norm < self.min_distance: return firework.mean
        unit_direction = direction / direction_norm
        radius = firework.compute_boundary_radius(unit_direction)
        return firework.mean + radius * unit_direction
        
# --- HCFWAメインクラス（アダプター）(デバッグログと反復制限を追加) ---
class HCFWA:
    def __init__(self, 
                 problem: BaseProblem, 
                 num_local_fireworks: int = 4,
                 sparks_per_firework: int = None,
                 max_evaluations: int = 10000000):
        
        self.problem = problem
        self.dimension = problem.dimension
        self.bounds = np.array([problem.lower_bounds, problem.upper_bounds]).T
        self.num_local_fireworks = num_local_fireworks
        
        if sparks_per_firework is None:
            total_sparks = max(20, 4 * self.dimension)
            self.sparks_per_firework = total_sparks // (num_local_fireworks + 1)
        else:
            self.sparks_per_firework = sparks_per_firework

        self.max_evaluations = max_evaluations
        self.collaboration_manager = CollaborationManager(self.dimension)
        self._reset_optimization_state()

    def _initialize_fireworks(self) -> None:
        self.global_firework = Firework(self.dimension, self.bounds, 'global', 0, self.num_local_fireworks)
        self.local_fireworks = [Firework(self.dimension, self.bounds, 'local', i + 1, self.num_local_fireworks) for i in range(self.num_local_fireworks)]
        self.all_fireworks = [self.global_firework] + self.local_fireworks

    def _reset_optimization_state(self) -> None:
        self.best_fitness = float('inf')
        self.best_solution = None
        self.global_evaluation_count = 0
        self.iteration_count = 0
        self.global_stagnation_count = 0
        self.fitness_history = []
        self._initialize_fireworks()

    # ★★★ optimizeメソッドを修正 (デバッグログと反復制限) ★★★
    def optimize(self) -> Tuple[np.ndarray, float, List[float]]:
        objective_function = self.problem.evaluate
        self._reset_optimization_state()

        # 安全のための最大反復回数制限を追加
        iteration_limit = 50000 

        while (self.global_evaluation_count < self.max_evaluations and 
               self.iteration_count < iteration_limit):
            
            # 100反復ごとにデバッグ情報を出力
            if self.iteration_count % 100 == 0:
                scales = [f"FW{fw.firework_id}: {fw.scale:.2e}" for fw in self.all_fireworks]
                print(f"Iter: {self.iteration_count}, Evals: {self.global_evaluation_count}, BestFit: {self.best_fitness:.4e}, Scales: [{', '.join(scales)}]")

            iteration_start_fitness = self.best_fitness
            all_sparks, all_fitness_values, firework_spark_ranges = [], [], []

            for fw in self.all_fireworks:
                sparks, fitness_values = fw.generate_sparks(self.sparks_per_firework), []
                for spark in sparks:
                    if self.global_evaluation_count >= self.max_evaluations: break
                    fitness_values.append(objective_function(spark))
                    self.global_evaluation_count += 1
                
                if not fitness_values: continue
                fitness_values = np.array(fitness_values)
                all_sparks.extend(sparks[:len(fitness_values)])
                all_fitness_values.extend(fitness_values)
                firework_spark_ranges.append((len(all_sparks) - len(fitness_values), len(all_sparks)))
            
            if self.global_evaluation_count >= self.max_evaluations: break

            all_fitness_array = np.array(all_fitness_values)
            if all_fitness_array.size > 0:
                best_idx = np.argmin(all_fitness_array)
                if all_fitness_array[best_idx] < self.best_fitness:
                    self.best_fitness, self.best_solution = all_fitness_array[best_idx], np.array(all_sparks[best_idx])

            for i, fw in enumerate(self.all_fireworks):
                start_idx, end_idx = firework_spark_ranges[i]
                if start_idx < end_idx:
                    fw.update_parameters(np.array(all_sparks[start_idx:end_idx]), all_fitness_array[start_idx:end_idx])
            
            self.collaboration_manager.execute_collaboration(self.all_fireworks)

            for fw in self.all_fireworks:
                should_restart, _ = fw.check_restart_conditions(self.all_fireworks)
                if should_restart: fw.restart()

            self.global_stagnation_count = self.global_stagnation_count + 1 if self.best_fitness >= iteration_start_fitness else 0
            if self.global_stagnation_count >= M_GLOBAL_REBOOT:
                best_solution_backup, best_fitness_backup = (self.best_solution.copy() if self.best_solution is not None else None), self.best_fitness
                for fw in self.all_fireworks: fw.restart()
                self.best_solution, self.best_fitness, self.global_stagnation_count = best_solution_backup, best_fitness_backup, 0
            
            self.fitness_history.append(self.best_fitness)
            self.iteration_count += 1
        
        if self.iteration_count >= iteration_limit:
            print(f"WARNING: Reached iteration limit ({iteration_limit}) before max_evaluations.")
            
        return self.best_solution, self.best_fitness, self.fitness_history