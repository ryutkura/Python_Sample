import numpy as np
import time
import warnings
from typing import List, Tuple, Optional, Dict, Any, Callable

# --- 定数定義（HCFWA論文準拠） ---
EPSILON_V = 1e-5
EPSILON_P = 1e-5
EPSILON_L = 100
CA_AMPLIFICATION = 5.0
TAU = 2
ALPHA_L = 0.85
ALPHA_U = 1.20
ALPHA_M_LOCAL = 0.20
ALPHA_M_GLOBAL = 0.05
M_GLOBAL_REBOOT = 100


class NumericalUtils:
    """数値計算ユーティリティ"""
    MIN_EIGENVALUE = 1e-14
    MAX_CONDITION_NUMBER = 1e14
    EPSILON_ZERO = 1e-12

    @staticmethod
    def ensure_numerical_stability(matrix, min_eigenval=1e-14, max_condition=1e14):
        """共分散行列の数値安定性確保"""
        if not np.all(np.isfinite(matrix)):
            return np.eye(matrix.shape[0])
        matrix = 0.5 * (matrix + matrix.T)
        try:
            eigenvals, eigenvecs = np.linalg.eigh(matrix)
            eigenvals = np.maximum(eigenvals, min_eigenval)
            max_eigenval = np.max(eigenvals)
            if max_eigenval / np.min(eigenvals) > max_condition:
                eigenvals = np.maximum(eigenvals, max_eigenval / max_condition)
            return eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
        except np.linalg.LinAlgError:
            return np.eye(matrix.shape[0])

    @staticmethod
    def compute_matrix_inverse_sqrt(matrix):
        """逆平方根の計算"""
        try:
            eigenvals, eigenvecs = np.linalg.eigh(matrix)
            eigenvals = np.maximum(eigenvals, NumericalUtils.MIN_EIGENVALUE)
            return eigenvecs @ np.diag(1.0 / np.sqrt(eigenvals)) @ eigenvecs.T
        except np.linalg.LinAlgError:
            return np.eye(matrix.shape[0])

    @staticmethod
    def mirror_boundary_mapping(x, bounds):
        """鏡面反射による境界処理"""
        x_mapped = x.copy()
        for d in range(len(x)):
            lb, ub = bounds[d, 0], bounds[d, 1]
            max_iterations = 100
            iteration = 0
            while (x_mapped[d] < lb or x_mapped[d] > ub) and iteration < max_iterations:
                if x_mapped[d] < lb:
                    x_mapped[d] = 2 * lb - x_mapped[d]
                elif x_mapped[d] > ub:
                    x_mapped[d] = 2 * ub - x_mapped[d]
                iteration += 1
            x_mapped[d] = np.clip(x_mapped[d], lb, ub)
        return x_mapped


class Firework:
    """花火クラス"""
    
    def __init__(self, dimension: int, bounds: np.ndarray, firework_type: str = 'local',
                 firework_id: int = 0, num_local_fireworks: int = 4):
        self.dimension = dimension
        self.bounds = np.array(bounds)
        self.firework_type = firework_type
        self.firework_id = firework_id
        self.num_local_fireworks = num_local_fireworks

        # CMA-ESパラメータ
        self.mean = self._initialize_mean()
        self.covariance = np.eye(dimension)
        self.scale = self._initialize_scale()
        
        # 進化パス
        self.evolution_path_c = np.zeros(dimension)
        self.evolution_path_sigma = np.zeros(dimension)
        
        # 学習率
        self.learning_rates = self._initialize_learning_rates()
        
        # 状態管理
        self.best_fitness = float('inf')
        self.best_solution = None
        self.stagnation_count = 0
        self.recent_fitness_history = []
        self.evaluation_count = 0
        self.last_improvement_iteration = 0
        self.restart_count = 0
        
        # スケール制限
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
            return {
                'cm': 1.0, 'c_mu': 0.25, 'c1': 0.0, 'cc': 0.0,
                'c_sigma': 0.0, 'd_sigma': 0.0, 'cr': 1.0,
                'cg': 1.0 / self.num_local_fireworks
            }
        else:
            return {
                'cm': 1.0, 'c_mu': 0.25,
                'c1': 2.0 / ((D + 1.3)**2),
                'cc': 4.0 / (D + 4.0),
                'c_sigma': 4.0 / (D + 4.0),
                'd_sigma': 1.0 + 2.0 * max(0, np.sqrt((D-1)/(D+1)) - 1) * 0.5,
                'cr': 0.5
            }

    def generate_sparks(self, num_sparks: int) -> np.ndarray:
        """火花生成"""
        if num_sparks <= 0:
            return np.array([]).reshape(0, self.dimension)
        
        # 共分散行列の安定化
        self.covariance = NumericalUtils.ensure_numerical_stability(self.covariance)
        
        # スケールの安全性確保
        safe_scale = np.clip(self.scale, self.min_scale, self.max_scale)
        
        try:
            sparks = np.random.multivariate_normal(
                mean=self.mean,
                cov=safe_scale**2 * self.covariance,
                size=num_sparks
            )
        except (np.linalg.LinAlgError, ValueError):
            # フォールバック
            sparks = self.mean + safe_scale * np.random.standard_normal((num_sparks, self.dimension))
        
        # NaN/Inf対策
        sparks = np.where(np.isfinite(sparks), sparks, self.mean)
        
        # 境界処理
        for i in range(num_sparks):
            sparks[i] = NumericalUtils.mirror_boundary_mapping(sparks[i], self.bounds)
        
        return sparks

    def compute_recombination_weights(self, fitness_values: np.ndarray) -> np.ndarray:
        """重み計算"""
        if self.firework_type == 'local':
            mu = max(1, len(fitness_values) // 2)
            sorted_indices = np.argsort(fitness_values)
            weights = np.zeros(len(fitness_values))
            for rank, idx in enumerate(sorted_indices[:mu]):
                w = max(0, np.log(mu + 0.5) - np.log(rank + 1))
                weights[idx] = w
            if np.sum(weights) > 0:
                weights = weights / np.sum(weights)
        else:
            num_select = max(1, int(0.95 * len(fitness_values)))
            sorted_indices = np.argsort(fitness_values)[:num_select]
            weights = np.zeros(len(fitness_values))
            if num_select > 0:
                weights[sorted_indices] = 1.0 / num_select
        return weights

    def update_parameters(self, sparks: np.ndarray, fitness_values: np.ndarray) -> None:
        """パラメータ更新"""
        if len(sparks) == 0:
            return
            
        weights = self.compute_recombination_weights(fitness_values)
        weight_sum_sq = np.sum(weights**2)
        mu_eff = 1.0 / weight_sum_sq if weight_sum_sq > 1e-12 else 1.0

        # 平均更新
        weighted_diff = np.zeros(self.dimension)
        for i, spark in enumerate(sparks):
            if weights[i] > 0:
                weighted_diff += weights[i] * (spark - self.mean)
        
        if not np.all(np.isfinite(weighted_diff)):
            weighted_diff = np.zeros(self.dimension)
        
        new_mean = self.mean + self.learning_rates['cm'] * weighted_diff
        
        if not np.all(np.isfinite(new_mean)):
            new_mean = self.mean.copy()

        # 参照平均
        reference_mean = ((1 - self.learning_rates['cr']) * self.mean +
                          self.learning_rates['cr'] * new_mean)

        # 進化パス更新
        if self.learning_rates['cc'] > 0 and self.scale > 1e-12:
            path_update = (np.sqrt(self.learning_rates['cc'] * (2 - self.learning_rates['cc']) * mu_eff) *
                          (new_mean - self.mean) / self.scale)
            if np.all(np.isfinite(path_update)):
                self.evolution_path_c = ((1 - self.learning_rates['cc']) * self.evolution_path_c + path_update)

        # 共分散行列更新
        rank_mu_update = np.zeros((self.dimension, self.dimension))
        for i, spark in enumerate(sparks):
            if weights[i] > 0:
                y = spark - reference_mean
                if np.all(np.isfinite(y)):
                    rank_mu_update += weights[i] * np.outer(y, y)

        rank_one_update = np.outer(self.evolution_path_c, self.evolution_path_c)

        if np.all(np.isfinite(rank_mu_update)) and np.all(np.isfinite(rank_one_update)):
            self.covariance = (
                (1 - self.learning_rates['c_mu'] - self.learning_rates['c1']) * self.covariance +
                self.learning_rates['c_mu'] * rank_mu_update +
                self.learning_rates['c1'] * rank_one_update
            )
        
        self.covariance = NumericalUtils.ensure_numerical_stability(self.covariance)

        # スケール適応（ローカルのみ）
        if (self.firework_type == 'local' and
            self.learning_rates['d_sigma'] > 0 and
            self.learning_rates['c_sigma'] > 0 and
            self.scale > 1e-12):
            
            try:
                C_inv_sqrt = NumericalUtils.compute_matrix_inverse_sqrt(self.covariance)
                path_sigma_update = (np.sqrt(self.learning_rates['c_sigma'] * (2 - self.learning_rates['c_sigma']) * mu_eff) *
                                    C_inv_sqrt @ (new_mean - self.mean) / self.scale)
                
                if np.all(np.isfinite(path_sigma_update)):
                    self.evolution_path_sigma = ((1 - self.learning_rates['c_sigma']) * self.evolution_path_sigma +
                                               path_sigma_update)

                expected_norm = np.sqrt(self.dimension)
                path_norm = np.linalg.norm(self.evolution_path_sigma)
                
                if np.isfinite(path_norm) and expected_norm > 0:
                    log_scale_change = (self.learning_rates['c_sigma'] / self.learning_rates['d_sigma'] *
                                       (path_norm / expected_norm - 1))
                    log_scale_change = np.clip(log_scale_change, -0.5, 0.5)
                    self.scale *= np.exp(log_scale_change)
            except Exception:
                pass

        self.scale = np.clip(self.scale, self.min_scale, self.max_scale)
        self.mean = new_mean
        self._update_best_solution(sparks, fitness_values)
        self.evaluation_count += len(sparks)

    def _update_best_solution(self, sparks: np.ndarray, fitness_values: np.ndarray) -> None:
        if len(fitness_values) == 0:
            return
        best_idx = np.argmin(fitness_values)
        current_best = fitness_values[best_idx]

        if np.isfinite(current_best) and current_best < self.best_fitness:
            self.best_fitness = current_best
            self.best_solution = sparks[best_idx].copy()
            self.stagnation_count = 0
            self.last_improvement_iteration = self.evaluation_count
        else:
            self.stagnation_count += 1

        self.recent_fitness_history.append(current_best)
        if len(self.recent_fitness_history) > 50:
            self.recent_fitness_history.pop(0)

    def compute_boundary_radius(self, direction: Optional[np.ndarray] = None) -> float:
        d_B = np.sqrt(self.dimension) + 0.5 * np.sqrt(2 * self.dimension)
        if direction is None:
            avg_eigenval = np.trace(self.covariance) / self.dimension
            return self.scale * np.sqrt(max(avg_eigenval, 1e-12)) * d_B
        else:
            dir_norm = np.linalg.norm(direction)
            if dir_norm < 1e-12:
                return self.scale * d_B
            direction = direction / dir_norm
            radius_squared = direction.T @ self.covariance @ direction
            return self.scale * np.sqrt(max(radius_squared, 1e-12)) * d_B

    def check_restart_conditions(self, all_fireworks: List['Firework']) -> Tuple[bool, List[str]]:
        restart_reasons = []

        # 適応度収束
        if len(self.recent_fitness_history) > 10:
            std_val = np.std(self.recent_fitness_history[-10:])
            if np.isfinite(std_val) and std_val <= EPSILON_V:
                restart_reasons.append("fitness_converged")

        # 位置収束
        try:
            eigenvals = np.linalg.eigvalsh(self.covariance)
            position_spread = self.scale * np.sqrt(np.max(eigenvals))
            if np.isfinite(position_spread) and position_spread <= EPSILON_P:
                restart_reasons.append("position_converged")
        except:
            pass

        # 停滞
        max_stagnation = 4 * EPSILON_L if self.firework_type == 'global' else EPSILON_L
        if self.stagnation_count >= max_stagnation:
            restart_reasons.append("not_improving")

        # 平均収束
        for fw in all_fireworks:
            if fw.firework_id != self.firework_id and fw.best_fitness < self.best_fitness:
                dist = np.linalg.norm(self.mean - fw.mean)
                if np.isfinite(dist) and dist < EPSILON_P:
                    restart_reasons.append("mean_converged")
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


class CollaborationManager:
    """協調戦略マネージャー"""
    
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.min_distance = 1e-10

    def execute_collaboration(self, fireworks: List[Firework]) -> None:
        """簡略化された協調処理"""
        if len(fireworks) < 2:
            return
        
        try:
            best_fw = min(fireworks, key=lambda fw: fw.best_fitness)
            
            for fw in fireworks:
                if fw.firework_id == best_fw.firework_id:
                    continue
                
                direction = best_fw.mean - fw.mean
                distance = np.linalg.norm(direction)
                
                if distance > self.min_distance and np.isfinite(distance):
                    adjustment_ratio = 0.03 if fw.firework_type == 'local' else 0.01
                    adjustment = adjustment_ratio * direction / distance * min(distance, fw.scale)
                    if np.all(np.isfinite(adjustment)):
                        fw.mean += adjustment
                        fw.mean = np.clip(fw.mean, fw.bounds[:, 0], fw.bounds[:, 1])
        except Exception:
            pass


class OldHCFWA:
    """
    HCFWA - run_experiment.py 互換版
    
    problem オブジェクトを受け取り、optimize() メソッドで最適化を実行
    """
    
    def __init__(self, problem, num_local_fireworks: int = 4,
                 sparks_per_firework: int = None,
                 max_evaluations: int = 10000000):
        """
        Args:
            problem: BaseProblem互換オブジェクト（dimension, lower_bounds, upper_bounds, evaluate()を持つ）
            num_local_fireworks: ローカル花火の数
            sparks_per_firework: 各花火の火花数（Noneで自動設定）
            max_evaluations: 最大評価回数
        """
        self.problem = problem
        self.dimension = problem.dimension
        self.bounds = np.array([problem.lower_bounds, problem.upper_bounds]).T
        self.num_local_fireworks = num_local_fireworks
        self.max_evaluations = max_evaluations

        # 火花数の設定（PDF版準拠）
        if sparks_per_firework is None:
            total_sparks = max(20, 4 * self.dimension)
            self.sparks_per_firework = total_sparks // (num_local_fireworks + 1)
        else:
            self.sparks_per_firework = sparks_per_firework

        # 花火リスト
        self.all_fireworks = []
        
        # 協調マネージャー
        self.collaboration_manager = CollaborationManager(self.dimension)
        
        # 状態管理
        self.best_fitness = float('inf')
        self.best_solution = None
        self.global_evaluation_count = 0
        self.iteration_count = 0
        self.global_stagnation_count = 0
        self.fitness_history = []

    def _initialize_fireworks(self) -> None:
        """花火の初期化"""
        self.all_fireworks = []
        
        # グローバル花火
        global_fw = Firework(
            dimension=self.dimension,
            bounds=self.bounds,
            firework_type='global',
            firework_id=0,
            num_local_fireworks=self.num_local_fireworks
        )
        self.all_fireworks.append(global_fw)

        # ローカル花火
        for i in range(self.num_local_fireworks):
            local_fw = Firework(
                dimension=self.dimension,
                bounds=self.bounds,
                firework_type='local',
                firework_id=i + 1,
                num_local_fireworks=self.num_local_fireworks
            )
            self.all_fireworks.append(local_fw)

    def optimize(self) -> Tuple[np.ndarray, float, List[float]]:
        """
        最適化実行
        
        Returns:
            (best_solution, best_fitness, fitness_history)
        """
        # 状態初期化
        self.best_fitness = float('inf')
        self.best_solution = None
        self.global_evaluation_count = 0
        self.iteration_count = 0
        self.global_stagnation_count = 0
        self.fitness_history = []
        
        self._initialize_fireworks()

        while self.global_evaluation_count < self.max_evaluations:
            iteration_start_fitness = self.best_fitness

            all_sparks = []
            all_fitness_values = []
            firework_spark_ranges = []

            # 各花火で火花生成・評価
            for fw in self.all_fireworks:
                remaining = self.max_evaluations - self.global_evaluation_count
                if remaining <= 0:
                    break
                
                num_sparks = min(self.sparks_per_firework, remaining)
                sparks = fw.generate_sparks(num_sparks)
                
                if len(sparks) == 0:
                    continue

                # 評価
                fitness_values = []
                for spark in sparks:
                    if self.global_evaluation_count >= self.max_evaluations:
                        break
                    try:
                        fit = self.problem.evaluate(spark)
                        fit = fit if np.isfinite(fit) else 1e30
                    except Exception:
                        fit = 1e30
                    fitness_values.append(fit)
                    self.global_evaluation_count += 1

                if not fitness_values:
                    continue

                fitness_values = np.array(fitness_values)
                sparks = sparks[:len(fitness_values)]

                start_idx = len(all_sparks)
                all_sparks.extend(sparks)
                all_fitness_values.extend(fitness_values)
                firework_spark_ranges.append((fw, start_idx, len(all_sparks)))

            if not all_fitness_values:
                break

            # 全体ベスト更新
            all_fitness_array = np.array(all_fitness_values)
            best_idx = np.argmin(all_fitness_array)
            if all_fitness_array[best_idx] < self.best_fitness:
                self.best_fitness = all_fitness_array[best_idx]
                self.best_solution = np.array(all_sparks[best_idx]).copy()

            # 各花火のパラメータ更新
            for fw, start_idx, end_idx in firework_spark_ranges:
                if start_idx < end_idx:
                    fw_sparks = np.array(all_sparks[start_idx:end_idx])
                    fw_fitness = all_fitness_array[start_idx:end_idx]
                    fw.update_parameters(fw_sparks, fw_fitness)

            # 協調戦略
            try:
                self.collaboration_manager.execute_collaboration(self.all_fireworks)
            except Exception:
                pass

            # リスタートチェック
            for fw in self.all_fireworks:
                try:
                    should_restart, _ = fw.check_restart_conditions(self.all_fireworks)
                    if should_restart:
                        fw.restart()
                except Exception:
                    pass

            # 停滞管理
            if self.best_fitness >= iteration_start_fitness:
                self.global_stagnation_count += 1
            else:
                self.global_stagnation_count = 0

            # グローバルリブート
            if self.global_stagnation_count >= M_GLOBAL_REBOOT:
                best_backup = self.best_solution.copy() if self.best_solution is not None else None
                fit_backup = self.best_fitness
                
                for fw in self.all_fireworks:
                    fw.restart()
                
                self.best_solution = best_backup
                self.best_fitness = fit_backup
                self.global_stagnation_count = 0

            self.fitness_history.append(self.best_fitness)
            self.iteration_count += 1

            # 早期終了
            if self.best_fitness < 1e-14:
                break

        return self.best_solution, self.best_fitness, self.fitness_history


# エイリアス（必要に応じて）
HCFWA = OldHCFWA
