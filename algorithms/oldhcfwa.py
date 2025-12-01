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
<<<<<<< HEAD
    def is_valid_matrix(matrix: np.ndarray) -> bool:
        """行列が数値的に有効かチェック"""
        if matrix is None:
            return False
        if not np.isfinite(matrix).all():
            return False
        return True
    
    @staticmethod
    def safe_matrix_norm(matrix: np.ndarray, ord=2) -> float:
        """安全な行列ノルム計算"""
        if not NumericalUtils.is_valid_matrix(matrix):
            return 1.0
        
        try:
            # スペクトルノルム(ord=2)の代わりにフロベニウスノルムを使用
            if ord == 2:
                # 固有値ベースで最大固有値を取得（SVDより安定）
                try:
                    eigenvals = np.linalg.eigvalsh(matrix)
                    eigenvals = eigenvals[np.isfinite(eigenvals)]
                    if len(eigenvals) > 0:
                        return np.sqrt(np.max(np.abs(eigenvals)))
                    return 1.0
                except np.linalg.LinAlgError:
                    # フォールバック: フロベニウスノルム
                    return np.sqrt(np.sum(matrix**2))
            else:
                return np.linalg.norm(matrix, ord)
        except Exception:
            return 1.0
    
    @staticmethod
    def ensure_numerical_stability(matrix: np.ndarray) -> np.ndarray:
        """共分散行列の数値的安定性を保証（対称化と正定値化）"""
        if not NumericalUtils.is_valid_matrix(matrix):
            return np.eye(matrix.shape[0])
        
        # 対称化
=======
    def ensure_numerical_stability(matrix, min_eigenval=1e-14, max_condition=1e14):
        """共分散行列の数値安定性確保"""
>>>>>>> e6c9d02a0ea4c5521fe0c3ba97a0a54b49d60e14
        matrix = 0.5 * (matrix + matrix.T)
        
        # NaN/Infのチェックと置換
        if not np.isfinite(matrix).all():
            return np.eye(matrix.shape[0])
        
        try:
            eigenvals, eigenvecs = np.linalg.eigh(matrix)
<<<<<<< HEAD
            
            # NaN/Infチェック
            if not np.isfinite(eigenvals).all() or not np.isfinite(eigenvecs).all():
                return np.eye(matrix.shape[0])
            
            # 最小固有値の設定
            min_eigenval = 1e-10
            eigenvals = np.maximum(eigenvals, min_eigenval)
            
            # 条件数のチェックと修正
            max_eigenval = np.max(eigenvals)
            max_condition_number = 1e10
            if max_eigenval > 0 and max_eigenval / np.min(eigenvals) > max_condition_number:
                eigenvals = np.maximum(eigenvals, max_eigenval / max_condition_number)
                
            result = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
            
            # 最終チェック
            if not np.isfinite(result).all():
                return np.eye(matrix.shape[0])
                
            return result
=======
            eigenvals = np.maximum(eigenvals, min_eigenval)
            max_eigenval = np.max(eigenvals)
            if max_eigenval / np.min(eigenvals) > max_condition:
                eigenvals = np.maximum(eigenvals, max_eigenval / max_condition)
            return eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
>>>>>>> e6c9d02a0ea4c5521fe0c3ba97a0a54b49d60e14
        except np.linalg.LinAlgError:
            return np.eye(matrix.shape[0])

    @staticmethod
<<<<<<< HEAD
    def compute_matrix_inverse_sqrt(matrix: np.ndarray) -> np.ndarray:
        """行列の逆平方根を計算 C^(-1/2)"""
        if not NumericalUtils.is_valid_matrix(matrix):
            return np.eye(matrix.shape[0])
        
        try:
            eigenvals, eigenvecs = np.linalg.eigh(matrix)
            
            # NaN/Infチェック
            if not np.isfinite(eigenvals).all() or not np.isfinite(eigenvecs).all():
                return np.eye(matrix.shape[0])
            
            min_eigenval = 1e-10
            eigenvals = np.maximum(eigenvals, min_eigenval)
            
            inv_sqrt_eigenvals = 1.0 / np.sqrt(eigenvals)
            
            # オーバーフロー防止
            max_inv_sqrt = 1e6
            inv_sqrt_eigenvals = np.minimum(inv_sqrt_eigenvals, max_inv_sqrt)
            
            result = eigenvecs @ np.diag(inv_sqrt_eigenvals) @ eigenvecs.T
            
            if not np.isfinite(result).all():
                return np.eye(matrix.shape[0])
                
            return result
=======
    def compute_matrix_inverse_sqrt(matrix):
        """逆平方根の計算"""
        try:
            eigenvals, eigenvecs = np.linalg.eigh(matrix)
            eigenvals = np.maximum(eigenvals, NumericalUtils.MIN_EIGENVALUE)
            return eigenvecs @ np.diag(1.0 / np.sqrt(eigenvals)) @ eigenvecs.T
>>>>>>> e6c9d02a0ea4c5521fe0c3ba97a0a54b49d60e14
        except np.linalg.LinAlgError:
            return np.eye(matrix.shape[0])

    @staticmethod
    def mirror_boundary_mapping(x, bounds):
        """鏡面反射による境界処理"""
        x_mapped = x.copy()
        
        # まずNaN/Infをチェック
        if not np.isfinite(x_mapped).all():
            # 無効な値は範囲の中心に置換
            for d in range(len(x)):
                if not np.isfinite(x_mapped[d]):
                    x_mapped[d] = (bounds[d, 0] + bounds[d, 1]) / 2
        
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
    """花火クラス（PDF版準拠）"""
    
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
        
<<<<<<< HEAD
        # 数値的安定性のための制限
        self.min_scale = 1e-10
        self.max_scale = 1e6
=======
        # スケール制限
        self.min_scale = 1e-12
        self.max_scale = 1e12
>>>>>>> e6c9d02a0ea4c5521fe0c3ba97a0a54b49d60e14

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
<<<<<<< HEAD
        """火花の生成（数値的安定性を強化）"""
        # 共分散行列の安定性確保
        self.covariance = NumericalUtils.ensure_numerical_stability(self.covariance)
        
        # スケールの有効性チェック
        if not np.isfinite(self.scale) or self.scale <= 0:
            self.scale = self._initialize_scale()
        
=======
        """火花生成"""
>>>>>>> e6c9d02a0ea4c5521fe0c3ba97a0a54b49d60e14
        try:
            cov_scaled = self.scale**2 * self.covariance
            # 共分散行列の最終チェック
            cov_scaled = NumericalUtils.ensure_numerical_stability(cov_scaled)
            
            sparks = np.random.multivariate_normal(
<<<<<<< HEAD
                mean=self.mean, 
                cov=cov_scaled, 
=======
                mean=self.mean,
                cov=self.scale**2 * self.covariance,
>>>>>>> e6c9d02a0ea4c5521fe0c3ba97a0a54b49d60e14
                size=num_sparks
            )
        except (np.linalg.LinAlgError, ValueError) as e:
            # フォールバック: 単位共分散で生成
            sparks = np.random.multivariate_normal(
<<<<<<< HEAD
                mean=self.mean, 
                cov=self.scale**2 * np.eye(self.dimension), 
=======
                mean=self.mean,
                cov=self.scale**2 * self.covariance,
>>>>>>> e6c9d02a0ea4c5521fe0c3ba97a0a54b49d60e14
                size=num_sparks
            )
        
        # 境界処理
        for i in range(num_sparks):
            sparks[i] = NumericalUtils.mirror_boundary_mapping(sparks[i], self.bounds)
        
        return sparks

    def compute_recombination_weights(self, fitness_values: np.ndarray) -> np.ndarray:
        """重み計算"""
        if self.firework_type == 'local':
            mu = len(fitness_values) // 2
            sorted_indices = np.argsort(fitness_values)
            weights = np.zeros(len(fitness_values))
            for rank, idx in enumerate(sorted_indices[:mu]):
                w = max(0, np.log(mu + 0.5) - np.log(rank + 1))
                weights[idx] = w
            if np.sum(weights) > 0:
                weights = weights / np.sum(weights)
        else:
            num_select = int(0.95 * len(fitness_values))
            sorted_indices = np.argsort(fitness_values)[:num_select]
            weights = np.zeros(len(fitness_values))
            if num_select > 0:
                weights[sorted_indices] = 1.0 / num_select
        return weights

    def update_parameters(self, sparks: np.ndarray, fitness_values: np.ndarray) -> None:
<<<<<<< HEAD
        """パラメータ更新（数値的安定性を強化）"""
        weights = self.compute_recombination_weights(fitness_values)
        
        # 有効個体数
        weight_sum_sq = np.sum(weights**2)
        mu_eff = 1.0 / weight_sum_sq if weight_sum_sq > 1e-12 else 1.0
        
        # 平均ベクトルの更新
=======
        """パラメータ更新"""
        weights = self.compute_recombination_weights(fitness_values)
        mu_eff = 1.0 / np.sum(weights**2) if np.sum(weights**2) > 0 else 1.0

        # 平均更新
>>>>>>> e6c9d02a0ea4c5521fe0c3ba97a0a54b49d60e14
        weighted_diff = np.zeros(self.dimension)
        for i, spark in enumerate(sparks):
            if weights[i] > 0:
                weighted_diff += weights[i] * (spark - self.mean)
<<<<<<< HEAD
        
        # NaNチェック
        if not np.isfinite(weighted_diff).all():
            weighted_diff = np.zeros(self.dimension)
        
        new_mean = self.mean + self.learning_rates['cm'] * weighted_diff
        
        # 新しい平均の有効性チェック
        if not np.isfinite(new_mean).all():
            new_mean = self.mean.copy()
        
        # 参照平均（リスタート時の位置ずれ防止）
        reference_mean = ((1 - self.learning_rates['cr']) * self.mean + 
                          self.learning_rates['cr'] * new_mean)
        
        # 進化パスの更新 (Localのみ)
        if self.learning_rates['cc'] > 0 and self.scale > 1e-12:
            path_update = (np.sqrt(self.learning_rates['cc'] * (2 - self.learning_rates['cc']) * mu_eff) 
                          * (new_mean - self.mean) / self.scale)
            if np.isfinite(path_update).all():
                self.evolution_path_c = ((1 - self.learning_rates['cc']) * self.evolution_path_c + path_update)
        
        # 共分散行列の更新 (Rank-mu & Rank-one)
=======
        new_mean = self.mean + self.learning_rates['cm'] * weighted_diff

        # 参照平均
        reference_mean = ((1 - self.learning_rates['cr']) * self.mean +
                          self.learning_rates['cr'] * new_mean)

        # 進化パス更新
        if self.learning_rates['cc'] > 0:
            self.evolution_path_c = (
                (1 - self.learning_rates['cc']) * self.evolution_path_c +
                np.sqrt(self.learning_rates['cc'] * (2 - self.learning_rates['cc']) * mu_eff) *
                (new_mean - self.mean) / self.scale
            )

        # 共分散行列更新
>>>>>>> e6c9d02a0ea4c5521fe0c3ba97a0a54b49d60e14
        rank_mu_update = np.zeros((self.dimension, self.dimension))
        for i, spark in enumerate(sparks):
            if weights[i] > 0:
                y = spark - reference_mean
<<<<<<< HEAD
                if np.isfinite(y).all():
                    rank_mu_update += weights[i] * np.outer(y, y)
                
        rank_one_update = np.outer(self.evolution_path_c, self.evolution_path_c)
        
        # 更新の有効性チェック
        if np.isfinite(rank_mu_update).all() and np.isfinite(rank_one_update).all():
            self.covariance = ((1 - self.learning_rates['c_mu'] - self.learning_rates['c1']) * self.covariance +
                              self.learning_rates['c_mu'] * rank_mu_update +
                              self.learning_rates['c1'] * rank_one_update)
        
        self.covariance = NumericalUtils.ensure_numerical_stability(self.covariance)
        
        # ステップサイズの更新 (CSA: Localのみ)
        if (self.firework_type == 'local' and 
            self.learning_rates['d_sigma'] > 0 and 
            self.learning_rates['c_sigma'] > 0 and
            self.scale > 1e-12):
            
            C_inv_sqrt = NumericalUtils.compute_matrix_inverse_sqrt(self.covariance)
            
            path_sigma_update = (np.sqrt(self.learning_rates['c_sigma'] * (2 - self.learning_rates['c_sigma']) * mu_eff) 
                                * C_inv_sqrt @ (new_mean - self.mean) / self.scale)
            
            if np.isfinite(path_sigma_update).all():
                self.evolution_path_sigma = ((1 - self.learning_rates['c_sigma']) * self.evolution_path_sigma + 
                                           path_sigma_update)
            
            expected_norm = np.sqrt(self.dimension)
            path_norm = np.linalg.norm(self.evolution_path_sigma)
            
            if np.isfinite(path_norm) and expected_norm > 0:
                log_scale_change = (self.learning_rates['c_sigma'] / self.learning_rates['d_sigma'] 
                                   * (path_norm / expected_norm - 1))
                # 大きすぎる変化を制限
                log_scale_change = np.clip(log_scale_change, -0.5, 0.5)
                self.scale *= np.exp(log_scale_change)
                
        self.scale = np.clip(self.scale, self.min_scale, self.max_scale)
=======
                rank_mu_update += weights[i] * np.outer(y, y)

        rank_one_update = np.outer(self.evolution_path_c, self.evolution_path_c)

        self.covariance = (
            (1 - self.learning_rates['c_mu'] - self.learning_rates['c1']) * self.covariance +
            self.learning_rates['c_mu'] * rank_mu_update +
            self.learning_rates['c1'] * rank_one_update
        )
        self.covariance = NumericalUtils.ensure_numerical_stability(self.covariance)

        # スケール適応（ローカルのみ）
        if (self.firework_type == 'local' and
            self.learning_rates['d_sigma'] > 0 and
            self.learning_rates['c_sigma'] > 0):
            
            C_inv_sqrt = NumericalUtils.compute_matrix_inverse_sqrt(self.covariance)
            self.evolution_path_sigma = (
                (1 - self.learning_rates['c_sigma']) * self.evolution_path_sigma +
                np.sqrt(self.learning_rates['c_sigma'] * (2 - self.learning_rates['c_sigma']) * mu_eff) *
                C_inv_sqrt @ (new_mean - self.mean) / self.scale
            )

            expected_norm = np.sqrt(self.dimension)
            path_norm = np.linalg.norm(self.evolution_path_sigma)
            log_scale_change = (
                self.learning_rates['c_sigma'] / self.learning_rates['d_sigma'] *
                (path_norm / expected_norm - 1)
            )
            self.scale *= np.exp(log_scale_change)
            self.scale = np.clip(self.scale, self.min_scale, self.max_scale)

>>>>>>> e6c9d02a0ea4c5521fe0c3ba97a0a54b49d60e14
        self.mean = new_mean
        self._update_best_solution(sparks, fitness_values)
        self.evaluation_count += len(sparks)

    def _update_best_solution(self, sparks: np.ndarray, fitness_values: np.ndarray) -> None:
        best_idx = np.argmin(fitness_values)
        current_best = fitness_values[best_idx]
<<<<<<< HEAD
        
        if np.isfinite(current_best) and current_best < self.best_fitness:
=======

        if current_best < self.best_fitness:
>>>>>>> e6c9d02a0ea4c5521fe0c3ba97a0a54b49d60e14
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
<<<<<<< HEAD
            # 平均的な半径（トレースを使用 - SVDより安定）
=======
>>>>>>> e6c9d02a0ea4c5521fe0c3ba97a0a54b49d60e14
            avg_eigenval = np.trace(self.covariance) / self.dimension
            avg_eigenval = max(avg_eigenval, 1e-12)
            return self.scale * np.sqrt(avg_eigenval) * d_B
        else:
<<<<<<< HEAD
            # 特定方向の半径
            dir_norm = np.linalg.norm(direction)
            if dir_norm < 1e-12:
                return self.scale * d_B
            direction = direction / dir_norm
=======
            direction = direction / np.linalg.norm(direction)
>>>>>>> e6c9d02a0ea4c5521fe0c3ba97a0a54b49d60e14
            radius_squared = direction.T @ self.covariance @ direction
            radius_squared = max(radius_squared, 1e-12)
            return self.scale * np.sqrt(radius_squared) * d_B

    def check_restart_conditions(self, all_fireworks: List['Firework']) -> Tuple[bool, List[str]]:
<<<<<<< HEAD
        """リスタート条件の判定（修正版）"""
        restart_reasons = []
        
        # 1. 収束判定
        if len(self.recent_fitness_history) > 1:
            std_fitness = np.std(self.recent_fitness_history)
            if np.isfinite(std_fitness) and std_fitness <= 1e-12:
                restart_reasons.append("fitness_converged")
        
        # 共分散行列のノルムを安全に計算
        cov_norm = NumericalUtils.safe_matrix_norm(self.covariance, 2)
        if self.scale * cov_norm <= 1e-10:
            restart_reasons.append("position_converged")
            
        # 2. 停滞判定
=======
        restart_reasons = []

        # 適応度収束
        if len(self.recent_fitness_history) > 1:
            if np.std(self.recent_fitness_history) <= EPSILON_V:
                restart_reasons.append("fitness_converged")

        # 位置収束
        try:
            position_spread = self.scale * np.sqrt(np.max(np.linalg.eigvalsh(self.covariance)))
            if position_spread <= EPSILON_P:
                restart_reasons.append("position_converged")
        except:
            pass

        # 停滞
>>>>>>> e6c9d02a0ea4c5521fe0c3ba97a0a54b49d60e14
        max_stagnation = 4 * EPSILON_L if self.firework_type == 'global' else EPSILON_L
        if self.stagnation_count >= max_stagnation:
            restart_reasons.append("not_improving")

        # 平均収束
        for fw in all_fireworks:
            if fw.firework_id != self.firework_id and fw.best_fitness < self.best_fitness:
<<<<<<< HEAD
                # 平均位置が近すぎる
                mean_dist = np.linalg.norm(self.mean - fw.mean)
                if np.isfinite(mean_dist) and mean_dist < 1e-8:
                    restart_reasons.append("mean_converged")
                    break
                    
                # 探索範囲が包含されている
                if np.isfinite(mean_dist) and mean_dist > 1e-12:
                    fw_radius = fw.compute_boundary_radius()
                    self_radius = self.compute_boundary_radius()
                    
                    if (np.isfinite(fw_radius) and np.isfinite(self_radius) and
                        fw_radius > self_radius and 
                        mean_dist + self_radius < fw_radius * 1.1):
                        restart_reasons.append("covered_by_better")
                        break
        
=======
                if np.linalg.norm(self.mean - fw.mean) < EPSILON_P:
                    restart_reasons.append("mean_converged")
                    break

>>>>>>> e6c9d02a0ea4c5521fe0c3ba97a0a54b49d60e14
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

    def get_status_info(self) -> Dict[str, Any]:
        return {
            'firework_id': self.firework_id,
            'firework_type': self.firework_type,
            'best_fitness': self.best_fitness,
            'scale': self.scale,
            'stagnation_count': self.stagnation_count,
            'restart_count': self.restart_count
        }


class CollaborationManager:
    """協調戦略マネージャー（簡略版）"""
    
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.ca = CA_AMPLIFICATION
<<<<<<< HEAD
        self.min_distance = 1e-10
        self.max_w_value = 5.0
        self.min_w_value = -5.0
=======
        self.min_distance = 1e-12
>>>>>>> e6c9d02a0ea4c5521fe0c3ba97a0a54b49d60e14

    def execute_collaboration(self, fireworks: List[Firework]) -> None:
        """協調処理（簡略化版）"""
        if len(fireworks) < 2:
            return
<<<<<<< HEAD
            
        try:
            # 全ペアの分割点（境界）を計算
            dividing_points = self._compute_all_dividing_points(fireworks)
            
            for fw in fireworks:
                # 他の花火との境界付近の特徴点を生成・選択
                feature_points = self._select_feature_points(fw, dividing_points, fireworks)
                
                # 特徴点に向けて分布を適応（移動）させる
                if feature_points:
                    self._adapt_to_feature_points(fw, feature_points)
        except Exception as e:
            # 協調戦略でエラーが発生しても続行
            pass

    def _compute_all_dividing_points(self, fireworks: List) -> Dict[Tuple[int, int], float]:
        dividing_points = {}
        for i in range(len(fireworks)):
            for j in range(i + 1, len(fireworks)):
                fw1, fw2 = fireworks[i], fireworks[j]
                try:
                    w = self._solve_dividing_equation(fw1, fw2)
                    if np.isfinite(w):
                        dividing_points[(fw1.firework_id, fw2.firework_id)] = w
                        dividing_points[(fw2.firework_id, fw1.firework_id)] = w
                except Exception:
                    pass
        return dividing_points

    def _solve_dividing_equation(self, fw1, fw2) -> float:
        distance = np.linalg.norm(fw1.mean - fw2.mean)
        if distance < self.min_distance or not np.isfinite(distance):
            return 0.0
            
        r1 = self._compute_radius_on_line(fw1, fw2)
        r2 = self._compute_radius_on_line(fw2, fw1)
        
        if not np.isfinite(r1) or not np.isfinite(r2) or r1 <= 0 or r2 <= 0:
            return 0.0
        
        a1, a2 = self._compute_sensitivity_factors(fw1, fw2)
        
        # 境界方程式: R1(w) + R2(w) = Distance
        if fw1.firework_type == 'local' and fw2.firework_type == 'local':
            def equation(w):
                return r1 * np.exp(a1 * w) + r2 * np.exp(a2 * w) - distance
        else:
            # Local vs Global (Globalは収縮方向)
            if fw1.firework_type == 'global':
                def equation(w):
                    return r1 * np.exp(-a1 * w) - r2 * np.exp(a2 * w) - distance
            else:
                def equation(w):
                    return r1 * np.exp(a1 * w) - r2 * np.exp(-a2 * w) - distance
                    
        try:
            return brentq(equation, self.min_w_value, self.max_w_value, xtol=1e-8, maxiter=50)
        except (ValueError, RuntimeError):
            return 0.0

    def _compute_radius_on_line(self, fw1, fw2) -> float:
        direction = fw2.mean - fw1.mean
        direction_norm = np.linalg.norm(direction)
        if direction_norm < self.min_distance or not np.isfinite(direction_norm):
            return fw1.compute_boundary_radius()
=======
        
        try:
            # 最良花火を見つける
            best_fw = min(fireworks, key=lambda fw: fw.best_fitness)
>>>>>>> e6c9d02a0ea4c5521fe0c3ba97a0a54b49d60e14
            
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
        except Exception:
            pass


class HCFWA:
    """HCFWA メインクラス（PDF版準拠）"""
    
    def __init__(self, dimension: int, bounds: np.ndarray,
                 num_local_fireworks: int = 4,
                 sparks_per_firework: int = None):
        
        self.dimension = dimension
        self.bounds = np.array(bounds)
        self.num_local_fireworks = num_local_fireworks

        # 火花数の設定（PDF版準拠）
        if sparks_per_firework is None:
            total_sparks = max(20, 4 * dimension)
            self.sparks_per_firework = total_sparks // (num_local_fireworks + 1)
        else:
            self.sparks_per_firework = sparks_per_firework

        # 花火の初期化
        self.global_firework = None
        self.local_fireworks = []
        self.all_fireworks = []
        
        # 協調マネージャー
        self.collaboration_manager = CollaborationManager(dimension)
        
        # 状態管理
        self.best_fitness = float('inf')
        self.best_solution = None
        self.global_evaluation_count = 0
        self.iteration_count = 0
        self.global_stagnation_count = 0
        
        # 履歴
        self.fitness_history = []
        self.verbose = True
        self.record_history = True
        
        self._initialize_fireworks()

    def _initialize_fireworks(self) -> None:
        """花火の初期化"""
        self.global_firework = Firework(
            dimension=self.dimension,
            bounds=self.bounds,
            firework_type='global',
            firework_id=0,
            num_local_fireworks=self.num_local_fireworks
        )

        self.local_fireworks = []
        for i in range(self.num_local_fireworks):
            local_fw = Firework(
                dimension=self.dimension,
                bounds=self.bounds,
                firework_type='local',
                firework_id=i + 1,
                num_local_fireworks=self.num_local_fireworks
            )
            self.local_fireworks.append(local_fw)

        self.all_fireworks = [self.global_firework] + self.local_fireworks

        if self.verbose:
            print(f"✓ 花火初期化完了: グローバル×1, ローカル×{self.num_local_fireworks}")
            print(f"  各花火の火花数: {self.sparks_per_firework}")

    def optimize(self,
                 objective_function: Callable[[np.ndarray], float],
                 max_evaluations: int = 10000,
                 target_fitness: Optional[float] = None,
                 max_time: Optional[float] = None) -> Dict[str, Any]:
        """最適化実行（PDF版準拠）"""
        
        start_time = time.time()
        self._reset_optimization_state()

        if self.verbose:
            print(f"=== HCFWA最適化開始 ===")
            print(f"次元: {self.dimension}, 最大評価回数: {max_evaluations}")

        while not self._check_termination(max_evaluations, target_fitness, max_time, start_time):
            improved = self._execute_iteration(objective_function)

            if not improved:
                self.global_stagnation_count += 1
            else:
                self.global_stagnation_count = 0

            # グローバルリブート
            if self.global_stagnation_count >= M_GLOBAL_REBOOT:
                self._global_reboot()

            self.iteration_count += 1

            # 進捗表示
            if self.verbose and self.iteration_count % 10 == 0:
                elapsed = time.time() - start_time
                print(f"反復{self.iteration_count}: 最良適応度={self.best_fitness:.6e}, "
                      f"評価回数={self.global_evaluation_count}, 経過時間={elapsed:.1f}s")

        return self._generate_result(start_time)

    def _execute_iteration(self, objective_function: Callable) -> bool:
        """1イテレーション実行"""
        iteration_start_fitness = self.best_fitness

        all_sparks = []
        all_fitness_values = []
        firework_spark_ranges = []

        # 各花火で火花生成・評価
        for fw in self.all_fireworks:
            sparks = fw.generate_sparks(self.sparks_per_firework)
            
            # 評価
            fitness_values = np.array([objective_function(spark) for spark in sparks])

            all_sparks.extend(sparks)
            all_fitness_values.extend(fitness_values)
            firework_spark_ranges.append((len(all_sparks) - len(sparks), len(all_sparks)))

            self.global_evaluation_count += len(sparks)

        # 全体ベスト更新
        all_fitness_array = np.array(all_fitness_values)
        best_idx = np.argmin(all_fitness_array)
        if all_fitness_array[best_idx] < self.best_fitness:
            self.best_fitness = all_fitness_array[best_idx]
            self.best_solution = np.array(all_sparks[best_idx])

        # 各花火のパラメータ更新
        for i, fw in enumerate(self.all_fireworks):
            start_idx, end_idx = firework_spark_ranges[i]
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
            should_restart, _ = fw.check_restart_conditions(self.all_fireworks)
            if should_restart:
                fw.restart()

        # 履歴記録
        if self.record_history:
            self.fitness_history.append(self.best_fitness)

        return self.best_fitness < iteration_start_fitness

    def _check_termination(self, max_evaluations, target_fitness, max_time, start_time) -> bool:
        """終了条件チェック"""
        if self.global_evaluation_count >= max_evaluations:
            if self.verbose:
                print(f"評価回数上限到達: {self.global_evaluation_count}")
            return True
        
        if target_fitness is not None and self.best_fitness <= target_fitness:
            if self.verbose:
                print(f"目標適応度達成: {self.best_fitness}")
            return True
        
        if max_time is not None and (time.time() - start_time) >= max_time:
            if self.verbose:
                print(f"時間上限到達: {time.time() - start_time:.1f}秒")
            return True
        
        return False

    def _global_reboot(self) -> None:
        """グローバルリブート"""
        if self.verbose:
            print(f"  === グローバルリブート（反復{self.iteration_count}）===")

        best_solution_backup = self.best_solution.copy() if self.best_solution is not None else None
        best_fitness_backup = self.best_fitness

        for fw in self.all_fireworks:
            fw.restart()

        self.best_solution = best_solution_backup
        self.best_fitness = best_fitness_backup
        self.global_stagnation_count = 0

    def _reset_optimization_state(self) -> None:
        """状態リセット"""
        self.best_fitness = float('inf')
        self.best_solution = None
        self.global_evaluation_count = 0
        self.iteration_count = 0
        self.global_stagnation_count = 0
        self.fitness_history = []
        self._initialize_fireworks()

    def _generate_result(self, start_time) -> Dict[str, Any]:
        """結果生成"""
        elapsed_time = time.time() - start_time

        result = {
            'best_fitness': self.best_fitness,
            'best_solution': self.best_solution.copy() if self.best_solution is not None else None,
            'total_evaluations': self.global_evaluation_count,
            'total_iterations': self.iteration_count,
            'elapsed_time': elapsed_time,
            'fitness_history': self.fitness_history.copy()
        }

        if self.verbose:
            print(f"\n=== 最適化完了 ===")
            print(f"最良適応度: {result['best_fitness']:.6e}")
            print(f"総評価回数: {result['total_evaluations']}")
            print(f"総反復回数: {result['total_iterations']}")
            print(f"実行時間: {result['elapsed_time']:.2f}秒")

<<<<<<< HEAD
            # 1. 各花火で火花生成と評価
            for fw in self.all_fireworks:
                sparks = fw.generate_sparks(self.sparks_per_firework)
                fitness_values = []
                
                for spark in sparks:
                    if self.global_evaluation_count < self.max_evaluations:
                        try:
                            fit = self.problem.evaluate(spark)
                            if np.isfinite(fit):
                                fitness_values.append(fit)
                            else:
                                fitness_values.append(1e30)  # 無効な値の代わり
                        except Exception:
                            fitness_values.append(1e30)
                        self.global_evaluation_count += 1
                    else:
                        break
                        
                if not fitness_values:
                    break
                    
                fitness_values = np.array(fitness_values)
                
                # インデックス範囲を記録（後でパラメータ更新に使用）
                start_idx = len(all_sparks)
                all_sparks.extend(sparks[:len(fitness_values)])
                all_fitness_values.extend(fitness_values)
                firework_spark_ranges.append((start_idx, len(all_sparks)))
=======
        return result
>>>>>>> e6c9d02a0ea4c5521fe0c3ba97a0a54b49d60e14


# --- テスト用ベンチマーク関数 ---
def sphere_function(x: np.ndarray) -> float:
    return np.sum(x**2)

<<<<<<< HEAD
            # 3. 各花火のパラメータ更新
            for i, fw in enumerate(self.all_fireworks):
                if i < len(firework_spark_ranges):
                    start_idx, end_idx = firework_spark_ranges[i]
                    if start_idx < end_idx:
                        try:
                            fw.update_parameters(
                                np.array(all_sparks[start_idx:end_idx]), 
                                all_fitness_array[start_idx:end_idx]
                            )
                        except Exception:
                            # 更新に失敗した場合はスキップ
                            pass
            
            # 4. 協調戦略の実行
            try:
                self.collaboration_manager.execute_collaboration(self.all_fireworks)
            except Exception:
                pass

            # 5. リスタート判定
            for fw in self.all_fireworks:
                try:
                    should_restart, _ = fw.check_restart_conditions(self.all_fireworks)
                    if should_restart:
                        fw.restart()
                except Exception:
                    pass

            # 6. グローバル停滞判定（全リスタート）
            if self.best_fitness >= iteration_start_fitness:
                self.global_stagnation_count += 1
            else:
                self.global_stagnation_count = 0
                
            if self.global_stagnation_count >= M_GLOBAL_REBOOT:
                # ベスト解を保持してリセット
                best_solution_backup = (self.best_solution.copy() if self.best_solution is not None else None)
                best_fitness_backup = self.best_fitness
                
                for fw in self.all_fireworks:
                    fw.restart()
                    
                self.best_solution = best_solution_backup
                self.best_fitness = best_fitness_backup
                self.global_stagnation_count = 0
            
            self.fitness_history.append(self.best_fitness)
            self.iteration_count += 1
            
        return self.best_solution, self.best_fitness, self.fitness_history
=======
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
    # print("=== HCFWA テスト（PDF版準拠） ===")

    # 問題設定
    dimension = 5
    bounds_val = 5.12
    bounds = np.array([[-bounds_val, bounds_val]] * dimension)

    # print(f"Problem: Sphere, Dimension: {dimension}")

    # HCFWA実行
    hcfwa = HCFWA(
        dimension=dimension,
        bounds=bounds,
        num_local_fireworks=4
    )

    # 最適化実行
    result = hcfwa.optimize(
        objective_function=sphere_function,
        max_evaluations=5000 * dimension
    )
>>>>>>> e6c9d02a0ea4c5521fe0c3ba97a0a54b49d60e14
