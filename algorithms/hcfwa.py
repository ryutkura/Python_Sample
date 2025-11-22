import numpy as np
import time
import warnings
from typing import List, Tuple, Optional, Dict, Any, Callable
from scipy.optimize import brentq

# BaseProblemの簡易実装（実際の使用時は適切なものに置き換えてください）
class BaseProblem:
    def __init__(self, dimension, lower_bounds, upper_bounds, objective_func):
        self.dimension = dimension
        self.lower_bounds = np.array(lower_bounds)
        self.upper_bounds = np.array(upper_bounds)
        self._objective_func = objective_func
    
    def evaluate(self, x: np.ndarray) -> float:
        try:
            result = float(self._objective_func(x))
            return result if np.isfinite(result) else float('inf')
        except Exception:
            return float('inf')

# --- 定数定義（HCFWA論文準拠 + 安定化調整） ---
EPSILON_V = 1e-5
EPSILON_P = 1e-5  
EPSILON_L = 100
CA_AMPLIFICATION = 5.0
TAU = 2
ALPHA_L = 0.85
ALPHA_U = 1.20
ALPHA_M_LOCAL = 0.20
ALPHA_M_GLOBAL = 0.05
M_GLOBAL_REBOOT = 150  # より保守的に設定

DEBUG = True

# --- 数値計算基盤クラス（堅牢化版） ---
class NumericalUtils:
    MIN_EIGENVALUE = 1e-12
    MAX_CONDITION_NUMBER = 1e12
    EPSILON_ZERO = 1e-12

    @staticmethod
    def ensure_numerical_stability(matrix: np.ndarray) -> np.ndarray:
        """共分散行列の数値安定性確保"""
        if not np.all(np.isfinite(matrix)):
            return np.eye(matrix.shape[0])
        
        matrix = 0.5 * (matrix + matrix.T)  # 対称化
        
        try:
            eigenvals, eigenvecs = np.linalg.eigh(matrix)
            eigenvals = np.maximum(eigenvals, NumericalUtils.MIN_EIGENVALUE)
            
            # 条件数制限
            max_eigenval = np.max(eigenvals)
            if max_eigenval / np.min(eigenvals) > NumericalUtils.MAX_CONDITION_NUMBER:
                eigenvals = np.maximum(eigenvals, max_eigenval / NumericalUtils.MAX_CONDITION_NUMBER)
            
            return eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
        except np.linalg.LinAlgError:
            return np.eye(matrix.shape[0])

    @staticmethod
    def compute_matrix_inverse_sqrt(matrix: np.ndarray) -> np.ndarray:
        """逆平方根の安全な計算"""
        try:
            eigenvals, eigenvecs = np.linalg.eigh(matrix)
            eigenvals = np.maximum(eigenvals, NumericalUtils.MIN_EIGENVALUE)
            return eigenvecs @ np.diag(1.0 / np.sqrt(eigenvals)) @ eigenvecs.T
        except np.linalg.LinAlgError:
            return np.eye(matrix.shape[0])

    @staticmethod
    def mirror_boundary_mapping(x: np.ndarray, bounds: np.ndarray) -> np.ndarray:
        """境界処理（簡略化版）"""
        return np.clip(x, bounds[:, 0], bounds[:, 1])

    @staticmethod
    def compute_expected_chi_norm(dimension: int) -> float:
        """カイ分布期待値の計算"""
        if dimension == 1:
            return np.sqrt(2 / np.pi)
        return np.sqrt(dimension) * (1 - 1/(4*dimension) + 1/(32*dimension**2))

# --- Firework クラス（HCFWA準拠 + 安定化） ---
class Firework:
    def __init__(self, dimension: int, bounds: np.ndarray, firework_type: str, 
                 firework_id: int, num_local_fireworks: int):
        self.dimension = dimension
        self.bounds = bounds
        self.firework_type = firework_type
        self.firework_id = firework_id
        self.num_local_fireworks = num_local_fireworks
        self.rng = np.random.default_rng()
        
        # 状態変数
        self.mean = self._initialize_mean()
        self.covariance = np.eye(dimension)
        self.scale = self._initialize_scale()
        
        # 進化パス
        self.evolution_path_c = np.zeros(dimension)
        self.evolution_path_sigma = np.zeros(dimension)
        
        # 学習率（HCFWA論文準拠）
        self.learning_rates = self._initialize_learning_rates()
        
        # 最適化状態
        self.best_fitness_fw = float('inf')
        self.best_solution_fw = None
        self.stagnation_count = 0
        self.recent_fitness_history = []
        self.evaluation_count_fw = 0
        self.last_improvement_iteration_fw = 0
        self.restart_count = 0
        self.last_restart_eval = 0  # 再起動クールダウン用
        
        # スケール制限
        self.min_scale = 1e-6  # より保守的に
        self.max_scale = 1e6

    def _initialize_mean(self) -> np.ndarray:
        if self.firework_type == 'global':
            return (self.bounds[:, 0] + self.bounds[:, 1]) / 2
        return self.rng.uniform(self.bounds[:, 0], self.bounds[:, 1])

    def _initialize_scale(self) -> float:
        range_size = np.mean(self.bounds[:, 1] - self.bounds[:, 0])
        expected_norm = NumericalUtils.compute_expected_chi_norm(self.dimension)
        base_scale = range_size / (4 * expected_norm)
        
        if self.firework_type == 'global':
            return base_scale
        return base_scale / np.sqrt(max(self.num_local_fireworks, 1))

    def _initialize_learning_rates(self) -> Dict[str, float]:
        """HCFWA論文4.6節準拠の学習率"""
        D = self.dimension
        
        if self.firework_type == 'global':
            return {
                'cm': 1.0, 'c_mu': 0.25, 'c1': 0.0, 'cc': 0.0,
                'c_sigma': 0.0, 'd_sigma': 0.0, 'cr': 1.0,
                'cg': 1.0 / max(self.num_local_fireworks, 1)
            }
        else:
            mu_eff = D / 2
            return {
                'cm': 1.0,
                'c_mu': 0.25,
                'c1': 2.0 / ((D + 1.3)**2 + mu_eff),
                'cc': (4 + mu_eff/D) / (D + 4 + 2*mu_eff/D),
                'c_sigma': (mu_eff + 2) / (D + mu_eff + 5),
                'd_sigma': 1.0 + 2.0 * max(0, np.sqrt((mu_eff - 1)/(D + 1)) - 1) + (mu_eff + 2) / (D + mu_eff + 5),
                'cr': 0.5
            }

    def generate_sparks(self, num_sparks: int) -> np.ndarray:
        """数値安定性を重視した火花生成"""
        if num_sparks <= 0:
            return np.array([]).reshape(0, self.dimension)
        
        # スケールの安全性確保
        safe_scale = np.clip(self.scale, self.min_scale, self.max_scale)
        stable_cov = NumericalUtils.ensure_numerical_stability(self.covariance)
        
        try:
            sparks = self.rng.multivariate_normal(
                mean=self.mean,
                cov=(safe_scale**2) * stable_cov,
                size=num_sparks,
                check_valid='ignore'
            )
        except Exception:
            # フォールバック：独立正規分布
            sparks = (self.mean + safe_scale * 
                     self.rng.standard_normal((num_sparks, self.dimension)))
        
        # NaN/Inf対策
        sparks = np.where(np.isfinite(sparks), sparks, 0.0)
        
        # 境界処理
        return np.apply_along_axis(
            NumericalUtils.mirror_boundary_mapping, 1, sparks, self.bounds
        )

    def update_parameters(self, sparks: np.ndarray, fitness_values: np.ndarray) -> None:
        """HCFWA論文準拠のパラメータ更新"""
        if len(sparks) == 0:
            return
        
        # 重みの計算
        weights = self._compute_recombination_weights(fitness_values)
        if np.sum(weights) == 0:
            return
        
        mu_eff = 1.0 / (np.sum(weights**2) + NumericalUtils.EPSILON_ZERO)
        
        # 平均更新
        weighted_diff = np.sum(weights[:, np.newaxis] * (sparks - self.mean), axis=0)
        new_mean = self.mean + self.learning_rates['cm'] * weighted_diff
        
        # 参照平均
        reference_mean = ((1 - self.learning_rates['cr']) * self.mean + 
                         self.learning_rates['cr'] * new_mean)
        
        # 進化パス更新（ローカル花火のみ）
        if self.learning_rates.get('cc', 0) > 0:
            mean_diff_scaled = (new_mean - self.mean) / (self.scale + NumericalUtils.EPSILON_ZERO)
            self.evolution_path_c = (
                (1 - self.learning_rates['cc']) * self.evolution_path_c +
                np.sqrt(self.learning_rates['cc'] * (2 - self.learning_rates['cc']) * mu_eff) *
                mean_diff_scaled
            )
        
        # 共分散行列更新
        rank_mu_update = np.zeros_like(self.covariance)
        if np.sum(weights) > 0:
            y_diff = sparks - reference_mean
            rank_mu_update = np.dot((y_diff * weights[:, np.newaxis]).T, y_diff)
        
        c1_val = self.learning_rates.get('c1', 0)
        rank_one_update = c1_val * np.outer(self.evolution_path_c, self.evolution_path_c)
        
        self.covariance = (
            (1 - self.learning_rates['c_mu'] - c1_val) * self.covariance +
            self.learning_rates['c_mu'] * (rank_mu_update / (self.scale**2 + NumericalUtils.EPSILON_ZERO)) +
            rank_one_update
        )
        self.covariance = NumericalUtils.ensure_numerical_stability(self.covariance)
        
        # スケール適応（ローカル花火のみ）
        if (self.firework_type == 'local' and 
            self.learning_rates.get('d_sigma', 0) > 0 and 
            self.learning_rates.get('c_sigma', 0) > 0):
            
            try:
                C_inv_sqrt = NumericalUtils.compute_matrix_inverse_sqrt(self.covariance)
                mean_diff_scaled = (new_mean - self.mean) / (self.scale + NumericalUtils.EPSILON_ZERO)
                
                self.evolution_path_sigma = (
                    (1 - self.learning_rates['c_sigma']) * self.evolution_path_sigma +
                    np.sqrt(self.learning_rates['c_sigma'] * (2 - self.learning_rates['c_sigma']) * mu_eff) *
                    (C_inv_sqrt @ mean_diff_scaled)
                )
                
                expected_norm = NumericalUtils.compute_expected_chi_norm(self.dimension)
                path_norm = np.linalg.norm(self.evolution_path_sigma)
                
                log_scale_change = (
                    self.learning_rates['c_sigma'] / (self.learning_rates['d_sigma'] + NumericalUtils.EPSILON_ZERO) *
                    (path_norm / (expected_norm + NumericalUtils.EPSILON_ZERO) - 1)
                )
                
                self.scale *= np.exp(np.clip(log_scale_change, -0.3, 0.3))  # より保守的に
                self.scale = np.clip(self.scale, self.min_scale, self.max_scale)
                
            except Exception:
                pass  # エラー時はスケール更新をスキップ
        
        self.mean = new_mean
        self._update_best_solution(sparks, fitness_values)
        self.evaluation_count_fw += len(sparks)

    def _compute_recombination_weights(self, fitness_values: np.ndarray) -> np.ndarray:
        """HCFWA論文式(16)準拠の重み計算"""
        num_sparks = len(fitness_values)
        weights = np.zeros(num_sparks)
        
        if self.firework_type == 'local':
            mu = max(1, num_sparks // 2)
            sorted_indices = np.argsort(fitness_values)
            
            for rank, idx in enumerate(sorted_indices[:mu]):
                w = max(0, np.log(mu + 0.5) - np.log(rank + 1))
                weights[idx] = w
        else:
            # グローバル花火：上位95%に等重み
            num_select = max(1, int(0.95 * num_sparks))
            sorted_indices = np.argsort(fitness_values)[:num_select]
            weights[sorted_indices] = 1.0 / num_select
        
        sum_weights = np.sum(weights)
        return weights / sum_weights if sum_weights > 0 else weights

    def _update_best_solution(self, sparks: np.ndarray, fitness_values: np.ndarray) -> None:
        """最良解の更新"""
        if len(fitness_values) == 0:
            return
        
        best_idx = np.argmin(fitness_values)
        current_best = fitness_values[best_idx]
        
        if current_best < self.best_fitness_fw:
            self.best_fitness_fw = current_best
            self.best_solution_fw = sparks[best_idx].copy()
            self.stagnation_count = 0
            self.last_improvement_iteration_fw = self.evaluation_count_fw
        else:
            self.stagnation_count += 1
        
        self.recent_fitness_history.append(current_best)
        if len(self.recent_fitness_history) > 30:
            self.recent_fitness_history.pop(0)

    def compute_boundary_radius(self, direction: Optional[np.ndarray] = None) -> float:
        """HCFWA論文式(18)準拠の境界半径計算"""
        expected_norm = NumericalUtils.compute_expected_chi_norm(self.dimension)
        std_norm_approx = 1/np.sqrt(2) if self.dimension > 1 else np.sqrt(1 - 2/np.pi)
        d_B = expected_norm + 0.5 * std_norm_approx
        
        if direction is None:
            avg_eigenval = np.trace(self.covariance) / self.dimension
            return self.scale * np.sqrt(max(avg_eigenval, NumericalUtils.MIN_EIGENVALUE)) * d_B
        else:
            direction_norm = np.linalg.norm(direction)
            if direction_norm < NumericalUtils.EPSILON_ZERO:
                return 0.0
            
            unit_direction = direction / direction_norm
            variance = max(NumericalUtils.MIN_EIGENVALUE, 
                          unit_direction.T @ self.covariance @ unit_direction)
            return self.scale * np.sqrt(variance) * d_B

    def check_restart_conditions(self, all_fireworks: List['Firework']) -> Tuple[bool, List[str]]:
        """再起動条件チェック（クールダウン付き）"""
        # クールダウンチェック
        cooldown = 50  # 50回の評価はクールダウン
        if self.evaluation_count_fw - self.last_restart_eval < cooldown:
            return False, []
        
        restart_reasons = []
        
        # 1. 適応度収束
        if len(self.recent_fitness_history) > 15:
            if np.std(self.recent_fitness_history[-10:]) <= EPSILON_V:
                restart_reasons.append("fitness_converged")
        
        # 2. 位置収束
        try:
            eigenvals = np.linalg.eigvalsh(self.covariance)
            max_std_dev = np.sqrt(np.max(eigenvals))
            if self.scale * max_std_dev <= EPSILON_P:
                restart_reasons.append("position_converged")
        except Exception:
            pass
        
        # 3. 改善なし
        max_stagnation = EPSILON_L * (2 if self.firework_type == 'global' else 1)
        if self.stagnation_count >= max_stagnation:
            restart_reasons.append("not_improving")
        
        # 4. より良い花火との関係（簡略化）
        for fw in all_fireworks:
            if (fw.firework_id != self.firework_id and 
                fw.best_fitness_fw < self.best_fitness_fw * 0.95):
                
                distance = np.linalg.norm(self.mean - fw.mean)
                if distance < EPSILON_P * 5:  # より寛容に
                    restart_reasons.append("mean_converged")
                    break
        
        return len(restart_reasons) > 0, restart_reasons

    def restart(self) -> None:
        """花火の再初期化"""
        # if DEBUG:
        #     print(f"    Restarting FW {self.firework_id} ({self.firework_type})")
        
        self.mean = self._initialize_mean()
        self.covariance = np.eye(self.dimension)
        self.scale = self._initialize_scale()
        self.evolution_path_c.fill(0)
        self.evolution_path_sigma.fill(0)
        self.stagnation_count = 0
        self.recent_fitness_history.clear()
        self.restart_count += 1
        self.last_restart_eval = self.evaluation_count_fw

# --- 簡略化協調マネージャー ---
class CollaborationManager:
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.ca = CA_AMPLIFICATION
        self.min_distance = 1e-6

    def execute_collaboration(self, fireworks: List[Firework]) -> None:
        """簡略化された協調処理"""
        if len(fireworks) < 2:
            return
        
        try:
            # 最良花火を見つける
            best_fw = min(fireworks, key=lambda fw: fw.best_fitness_fw)
            
            for fw in fireworks:
                if fw.firework_id == best_fw.firework_id:
                    continue
                
                # 最良花火への軽微な調整
                direction = best_fw.mean - fw.mean
                distance = np.linalg.norm(direction)
                
                if distance > self.min_distance:
                    adjustment_ratio = 0.03 if fw.firework_type == 'local' else 0.01
                    adjustment = adjustment_ratio * direction / distance * min(distance, fw.scale)
                    fw.mean += adjustment
                    fw.mean = np.clip(fw.mean, fw.bounds[:, 0], fw.bounds[:, 1])
                    
        except Exception as e:
            pass
            # if DEBUG:
            #     warnings.warn(f"Collaboration error: {e}")

# --- HCFWAメインクラス（評価回数保持版） ---
class HCFWA:
    def __init__(self, 
                 problem: BaseProblem, 
                 num_local_fireworks: int = 4,
                 sparks_per_firework: int = None,
                 max_evaluations: int = 100000):
        
        self.problem = problem
        self.dimension = problem.dimension
        self.bounds = np.array([problem.lower_bounds, problem.upper_bounds]).T
        self.num_local_fireworks = num_local_fireworks
        self.max_evaluations = max_evaluations
        
        if sparks_per_firework is None:
            # self.sparks_per_firework = max(5, int(4 + np.floor(3 * np.log(self.dimension))))
            self.sparks_per_firework = max(10, int(6 * self.dimension))
        else:
            self.sparks_per_firework = sparks_per_firework
        
        self._initialize_state()

    def _initialize_state(self):
        """初期状態設定"""
        self.best_fitness = float('inf')
        self.best_solution = None
        self.global_evaluation_count = 0
        self.iteration_count = 0
        self.global_stagnation_count = 0
        self.fitness_history = []
        
        # 花火初期化
        self.fireworks = []
        self.fireworks.append(Firework(self.dimension, self.bounds, 'global', 0, self.num_local_fireworks))
        for i in range(self.num_local_fireworks):
            self.fireworks.append(Firework(self.dimension, self.bounds, 'local', i + 1, self.num_local_fireworks))
        
        self.collaboration_manager = CollaborationManager(self.dimension)
        
        # 初期評価
        self._perform_initial_evaluation()

    def _perform_initial_evaluation(self):
        """初期評価の実行"""
        # if DEBUG:
        #     print("Performing initial evaluation...")
        
        num_initial = max(20, self.dimension * 2)
        initial_solutions = []
        for _ in range(num_initial):
            sol = self.fireworks[0].rng.uniform(self.bounds[:, 0], self.bounds[:, 1])
            initial_solutions.append(sol)
            
        # LHS系統的初期化
        # num_initial = max(100, self.dimension * 15)  # 150個
        # initial_solutions = latin_hypercube_sampling(dimension, bounds, num_initial, rng)
        
        initial_fitnesses = [self.problem.evaluate(sol) for sol in initial_solutions]
        self.global_evaluation_count += num_initial
        
        best_idx = np.argmin(initial_fitnesses)
        self.best_fitness = initial_fitnesses[best_idx]
        self.best_solution = initial_solutions[best_idx].copy()
        
        # 花火に初期最良値を設定
        for fw in self.fireworks:
            fw.best_fitness_fw = self.best_fitness
            fw.best_solution_fw = self.best_solution.copy()
        
        # if DEBUG:
        #     print(f"Initial best fitness: {self.best_fitness:.6e}")

    def _global_reboot(self):
        """グローバルリブート（評価回数保持版）"""
        # if DEBUG:
        #     print(f"  GLOBAL REBOOT at Iter {self.iteration_count}, Evals {self.global_evaluation_count}")
        #     print(f"  Preserving best: {self.best_fitness:.6e}")
        
        # 最良解バックアップ
        best_sol_backup = self.best_solution.copy() if self.best_solution is not None else None
        best_fit_backup = self.best_fitness
        eval_count_backup = self.global_evaluation_count  # 重要：評価回数を保持
        
        # 花火のみリセット
        for fw in self.fireworks:
            fw.restart()
            fw.best_fitness_fw = best_fit_backup
            fw.best_solution_fw = best_sol_backup.copy() if best_sol_backup is not None else None
        
        # 全体状態の復元
        self.best_fitness = best_fit_backup
        self.best_solution = best_sol_backup
        self.global_evaluation_count = eval_count_backup  # 評価回数を保持
        self.global_stagnation_count = 0

    def optimize(self) -> Tuple[np.ndarray, float, List[float]]:
        """最適化実行（進捗監視付き）"""
        start_time = time.time()
        timeout = 300
        
        # 進捗監視用
        max_iters_cap = max(1000, int(self.max_evaluations / max(1, self.sparks_per_firework) / (self.num_local_fireworks + 1)) * 20)
        stuck_iters = 0
        
        while self.global_evaluation_count < self.max_evaluations:
            # タイムアウトチェック
            if time.time() - start_time > timeout:
                # if DEBUG:
                #     print(f"Timeout after {timeout} seconds")
                break
            
            # 進捗監視
            iter_evals_before = self.global_evaluation_count
            iteration_improved = False
            
            # 各花火でスパーク生成・評価
            for fw in self.fireworks:
                remaining_evals = self.max_evaluations - self.global_evaluation_count
                if remaining_evals <= 0:
                    break
                
                num_sparks = min(self.sparks_per_firework, remaining_evals)
                if num_sparks <= 0:
                    continue
                
                sparks = fw.generate_sparks(num_sparks)
                if len(sparks) == 0:
                    continue
                
                # 安全な評価
                fitness_values = []
                for s in sparks:
                    try:
                        f = float(self.problem.evaluate(s))
                        fitness_values.append(f if np.isfinite(f) else float('inf'))
                    except Exception:
                        fitness_values.append(float('inf'))
                
                fitness_values = np.array(fitness_values)
                self.global_evaluation_count += len(sparks)
                
                # 最良解更新
                best_idx = np.argmin(fitness_values)
                if fitness_values[best_idx] < self.best_fitness:
                    self.best_fitness = fitness_values[best_idx]
                    self.best_solution = sparks[best_idx].copy()
                    iteration_improved = True
                    # if DEBUG and self.iteration_count % 10 == 0:
                    #     print(f"    New best: {self.best_fitness:.6e}")
                
                # パラメータ更新
                fw.update_parameters(sparks, fitness_values)
            
            # 進捗がない場合の緊急処置
            if self.global_evaluation_count == iter_evals_before:
                stuck_iters += 1
                # if DEBUG:
                #     print(f"    Stuck iteration {stuck_iters}, forcing evaluation...")
                
                # 緊急サンプリング
                for fw in self.fireworks:
                    if self.global_evaluation_count >= self.max_evaluations:
                        break
                    x = self.fireworks[0].rng.uniform(self.bounds[:, 0], self.bounds[:, 1])
                    try:
                        f = float(self.problem.evaluate(x))
                        f = f if np.isfinite(f) else float('inf')
                    except Exception:
                        f = float('inf')
                    
                    self.global_evaluation_count += 1
                    fw.update_parameters(x[None, :], np.array([f]))
                    
                    if f < self.best_fitness:
                        self.best_fitness = f
                        self.best_solution = x.copy()
                        iteration_improved = True
            else:
                stuck_iters = 0
            
            # 協調処理（頻度制限）
            if self.iteration_count % 3 == 0:
                try:
                    self.collaboration_manager.execute_collaboration(self.fireworks)
                except Exception as e:
                    pass
                    # if DEBUG:
                    #     warnings.warn(f"Collaboration failed: {e}")
            
            # 再起動チェック（頻度制限）
            if self.iteration_count % 5 == 0:
                for fw in self.fireworks:
                    should_restart, reasons = fw.check_restart_conditions(self.fireworks)
                    if should_restart:
                        # if DEBUG:
                        #     print(f"    FW {fw.firework_id} restart: {reasons}")
                        fw.restart()
            
            # 停滞管理
            if iteration_improved:
                self.global_stagnation_count = 0
            else:
                self.global_stagnation_count += 1
            
            # グローバルリブート
            if self.global_stagnation_count >= M_GLOBAL_REBOOT:
                self._global_reboot()
            
            self.fitness_history.append(self.best_fitness)
            self.iteration_count += 1
            
            # 進捗表示
            # if DEBUG and self.iteration_count % 20 == 0:
            #     print(f"Iter: {self.iteration_count:4d} | Evals: {self.global_evaluation_count:6d} | Best: {self.best_fitness:.6e}")
            
            # 安全弁
            if self.iteration_count >= max_iters_cap:
                # if DEBUG:
                #     print("Iteration cap reached. Stopping.")
                break
            
            # 早期終了
            if self.best_fitness < 1e-12:
                # if DEBUG:
                #     print(f"Early termination: {self.best_fitness:.6e}")
                break
        
        # if DEBUG:
        #     print(f"\nOptimization completed:")
        #     print(f"  Iterations: {self.iteration_count}")
        #     print(f"  Evaluations: {self.global_evaluation_count}")
        #     print(f"  Best fitness: {self.best_fitness:.6e}")
        
        return self.best_solution, self.best_fitness, self.fitness_history

# --- テスト用ベンチマーク関数 ---
def sphere_function(x: np.ndarray) -> float:
    return np.sum(x**2)

def rastrigin_function(x: np.ndarray) -> float:
    A = 10
    n = len(x)
    return A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x))

def ackley_function(x: np.ndarray) -> float:
    n = len(x)
    a, b, c = 20, 0.2, 2 * np.pi
    sum_sq_term = -b * np.sqrt(np.sum(x**2) / n)
    cos_term = np.sum(np.cos(c * x))
    return -a * np.exp(sum_sq_term) - np.exp(cos_term / n) + a + np.e

# --- 使用例 ---
if __name__ == "__main__":
    print("=== HCFWA 修正版テスト ===")
    
    # 問題設定
    dimension = 5
    bounds_val = 5.12
    lower_bounds = [-bounds_val] * dimension
    upper_bounds = [bounds_val] * dimension
    
    # 目的関数選択
    objective_func = ackley_function  # または sphere_function, rastrigin_function
    
    # 問題インスタンス作成
    problem = BaseProblem(dimension, lower_bounds, upper_bounds, objective_func)
    
    print(f"Problem: {objective_func.__name__}, Dimension: {dimension}")
    
    # HCFWA実行
    hcfwa = HCFWA(
        problem=problem,
        num_local_fireworks=4,
        max_evaluations=5000 * dimension
    )
    
    print(f"Max evaluations: {hcfwa.max_evaluations}")
    print(f"Sparks per firework: {hcfwa.sparks_per_firework}")
    
    # 最適化実行
    best_sol, best_fit, history = hcfwa.optimize()
    
    # 結果表示
    print("\n=== Results ===")
    print(f"Best solution: {best_sol}")
    print(f"Best fitness: {best_fit:.6e}")
    print(f"Total evaluations: {hcfwa.global_evaluation_count}")
    print(f"Total iterations: {hcfwa.iteration_count}")
    
    # 収束曲線プロット（オプション）
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.semilogy(history, marker='.')
        plt.xlabel('Iterations')
        plt.ylabel('Best Fitness (log scale)')
        plt.title(f'HCFWA Convergence - {objective_func.__name__} (D={dimension})')
        plt.grid(True, alpha=0.3)
        plt.show()
    except ImportError:
        print("matplotlib not available for plotting")
    
    print("\n✓ テスト完了")
