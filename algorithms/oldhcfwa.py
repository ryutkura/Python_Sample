import numpy as np
import time
import warnings
from typing import List, Tuple, Optional, Dict, Any
from scipy.optimize import brentq
from problems import BaseProblem

# 警告の抑制（数値計算上の軽微な警告が出る場合があるため）
warnings.filterwarnings('ignore')

# --- 論文仕様に基づく定数 ---
EPSILON_L = 100       # 改善停滞閾値
CA_AMPLIFICATION = 5.0 # グローバル花火感度増幅係数
TAU = 2               # 特徴点選択数
ALPHA_L = 0.85        # 下限クリッピング係数
ALPHA_U = 1.20        # 上限クリッピング係数
ALPHA_M_LOCAL = 0.20  # ローカル花火平均移動係数
ALPHA_M_GLOBAL = 0.05 # グローバル花火平均移動係数
M_GLOBAL_REBOOT = 100 # グローバル花火再起動閾値

# --- 数値計算ユーティリティ ---
class NumericalUtils:
    @staticmethod
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
        matrix = 0.5 * (matrix + matrix.T)
        
        # NaN/Infのチェックと置換
        if not np.isfinite(matrix).all():
            return np.eye(matrix.shape[0])
        
        try:
            eigenvals, eigenvecs = np.linalg.eigh(matrix)
            
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
        except np.linalg.LinAlgError:
            return np.eye(matrix.shape[0])

    @staticmethod
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
        except np.linalg.LinAlgError:
            return np.eye(matrix.shape[0])

    @staticmethod
    def mirror_boundary_mapping(x: np.ndarray, bounds: np.ndarray) -> np.ndarray:
        """鏡面反射による境界制約処理"""
        x_mapped = x.copy()
        
        # まずNaN/Infをチェック
        if not np.isfinite(x_mapped).all():
            # 無効な値は範囲の中心に置換
            for d in range(len(x)):
                if not np.isfinite(x_mapped[d]):
                    x_mapped[d] = (bounds[d, 0] + bounds[d, 1]) / 2
        
        for d in range(len(x)):
            lb, ub = bounds[d, 0], bounds[d, 1]
            iteration = 0
            # 範囲内に収まるまで反射を繰り返す（無限ループ防止付き）
            while (x_mapped[d] < lb or x_mapped[d] > ub) and iteration < 100:
                if x_mapped[d] < lb:
                    x_mapped[d] = 2 * lb - x_mapped[d]
                elif x_mapped[d] > ub:
                    x_mapped[d] = 2 * ub - x_mapped[d]
                iteration += 1
            # 最終的な安全策としてクリッピング
            x_mapped[d] = np.clip(x_mapped[d], lb, ub)
        return x_mapped

# --- 花火クラス ---
class Firework:
    def __init__(self, dimension: int, bounds: np.ndarray, firework_type: str = 'local', 
                 firework_id: int = 0, num_local_fireworks: int = 4):
        self.dimension = dimension
        self.bounds = np.array(bounds)
        self.firework_type = firework_type
        self.firework_id = firework_id
        self.num_local_fireworks = num_local_fireworks
        
        # CMA-ESパラメータの初期化
        self.mean = self._initialize_mean()
        self.covariance = np.eye(dimension)
        self.scale = self._initialize_scale()
        self.evolution_path_c = np.zeros(dimension)
        self.evolution_path_sigma = np.zeros(dimension)
        
        self.learning_rates = self._initialize_learning_rates()
        
        # 状態管理
        self.best_fitness = float('inf')
        self.best_solution = None
        self.stagnation_count = 0
        self.recent_fitness_history = []
        self.evaluation_count = 0
        self.last_improvement_iteration = 0
        self.restart_count = 0
        
        # 数値的安定性のための制限
        self.min_scale = 1e-10
        self.max_scale = 1e6

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
            # ローカル花火は探索空間を分割して担当するため小さめに設定
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
        """火花の生成（数値的安定性を強化）"""
        # 共分散行列の安定性確保
        self.covariance = NumericalUtils.ensure_numerical_stability(self.covariance)
        
        # スケールの有効性チェック
        if not np.isfinite(self.scale) or self.scale <= 0:
            self.scale = self._initialize_scale()
        
        try:
            cov_scaled = self.scale**2 * self.covariance
            # 共分散行列の最終チェック
            cov_scaled = NumericalUtils.ensure_numerical_stability(cov_scaled)
            
            sparks = np.random.multivariate_normal(
                mean=self.mean, 
                cov=cov_scaled, 
                size=num_sparks
            )
        except (np.linalg.LinAlgError, ValueError) as e:
            # フォールバック: 単位共分散で生成
            sparks = np.random.multivariate_normal(
                mean=self.mean, 
                cov=self.scale**2 * np.eye(self.dimension), 
                size=num_sparks
            )
        
        # 境界処理
        for i in range(num_sparks):
            sparks[i] = NumericalUtils.mirror_boundary_mapping(sparks[i], self.bounds)
            
        return sparks

    def compute_recombination_weights(self, fitness_values: np.ndarray) -> np.ndarray:
        if self.firework_type == 'local':
            mu = len(fitness_values) // 2
            sorted_indices = np.argsort(fitness_values)
            weights = np.zeros(len(fitness_values))
            
            # 対数重み付け
            for rank, idx in enumerate(sorted_indices[:mu]):
                w = max(0, np.log(mu + 0.5) - np.log(rank + 1))
                weights[idx] = w
                
            if np.sum(weights) > 0:
                weights = weights / np.sum(weights)
        else:
            # グローバル花火は上位95%を一様重み付け
            num_select = int(0.95 * len(fitness_values))
            sorted_indices = np.argsort(fitness_values)[:num_select]
            weights = np.zeros(len(fitness_values))
            if num_select > 0:
                weights[sorted_indices] = 1.0 / num_select
                
        return weights

    def update_parameters(self, sparks: np.ndarray, fitness_values: np.ndarray) -> None:
        """パラメータ更新（数値的安定性を強化）"""
        weights = self.compute_recombination_weights(fitness_values)
        
        # 有効個体数
        weight_sum_sq = np.sum(weights**2)
        mu_eff = 1.0 / weight_sum_sq if weight_sum_sq > 1e-12 else 1.0
        
        # 平均ベクトルの更新
        weighted_diff = np.zeros(self.dimension)
        for i, spark in enumerate(sparks):
            if weights[i] > 0:
                weighted_diff += weights[i] * (spark - self.mean)
        
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
        rank_mu_update = np.zeros((self.dimension, self.dimension))
        for i, spark in enumerate(sparks):
            if weights[i] > 0:
                y = spark - reference_mean
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
        self.mean = new_mean
        
        # ベスト解の更新
        self._update_best_solution(sparks, fitness_values)
        self.evaluation_count += len(sparks)

    def _update_best_solution(self, sparks: np.ndarray, fitness_values: np.ndarray) -> None:
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
        """境界半径（探索範囲の広がり）を計算"""
        d_B = np.sqrt(self.dimension) + 0.5 * np.sqrt(2 * self.dimension)
        
        if direction is None:
            # 平均的な半径（トレースを使用 - SVDより安定）
            avg_eigenval = np.trace(self.covariance) / self.dimension
            avg_eigenval = max(avg_eigenval, 1e-12)
            return self.scale * np.sqrt(avg_eigenval) * d_B
        else:
            # 特定方向の半径
            dir_norm = np.linalg.norm(direction)
            if dir_norm < 1e-12:
                return self.scale * d_B
            direction = direction / dir_norm
            radius_squared = direction.T @ self.covariance @ direction
            radius_squared = max(radius_squared, 1e-12)
            return self.scale * np.sqrt(radius_squared) * d_B
            
    def check_restart_conditions(self, all_fireworks: List['Firework']) -> Tuple[bool, List[str]]:
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
        max_stagnation = 4 * EPSILON_L if self.firework_type == 'global' else EPSILON_L
        if self.stagnation_count >= max_stagnation:
            restart_reasons.append("not_improving")
            
        # 3. 包含判定（より良い花火にカバーされているか）
        for fw in all_fireworks:
            if fw.firework_id != self.firework_id and fw.best_fitness < self.best_fitness:
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
        
        return len(restart_reasons) > 0, restart_reasons

    def restart(self) -> None:
        """パラメータの再初期化"""
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

# --- 協調戦略管理クラス ---
class CollaborationManager:
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.ca = CA_AMPLIFICATION
        self.min_distance = 1e-10
        self.max_w_value = 5.0
        self.min_w_value = -5.0

    def execute_collaboration(self, fireworks: List) -> None:
        if len(fireworks) < 2:
            return
            
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
            
        unit_direction = direction / direction_norm
        return fw1.compute_boundary_radius(unit_direction)

    def _compute_sensitivity_factors(self, fw1, fw2) -> Tuple[float, float]:
        a1, a2 = 1.0, 1.0
        
        # 性能が良い方は感度を下げる（現状維持傾向）
        if fw1.best_fitness < fw2.best_fitness:
            a1, a2 = 0.0, 1.0
        elif fw2.best_fitness < fw1.best_fitness:
            a1, a2 = 1.0, 0.0
            
        # 停滞している場合は感度を下げる
        improvement_threshold = 0.2 * EPSILON_L
        if (fw1.firework_type == 'local' and hasattr(fw1, 'last_improvement_iteration') and 
            fw1.evaluation_count - fw1.last_improvement_iteration < improvement_threshold):
            a1 = 0.0
        if (fw2.firework_type == 'local' and hasattr(fw2, 'last_improvement_iteration') and 
            fw2.evaluation_count - fw2.last_improvement_iteration < improvement_threshold):
            a2 = 0.0
            
        # グローバル花火は感度を増幅
        if fw1.firework_type == 'global': a1 *= self.ca
        if fw2.firework_type == 'global': a2 *= self.ca
        
        return a1, a2

    def _select_feature_points(self, firework, dividing_points: Dict, fireworks: List) -> List[np.ndarray]:
        potential_points = []
        
        for other_fw in fireworks:
            if other_fw.firework_id == firework.firework_id:
                continue
                
            key = (firework.firework_id, other_fw.firework_id)
            if key in dividing_points:
                w = dividing_points[key]
                distance = np.linalg.norm(firework.mean - other_fw.mean)
                
                if distance > self.min_distance and np.isfinite(distance):
                    r = self._compute_radius_on_line(firework, other_fw)
                    if not np.isfinite(r) or r <= 0:
                        continue
                        
                    a_self, _ = self._compute_sensitivity_factors(firework, other_fw)
                    
                    # 境界上の点を計算
                    if firework.firework_type == 'local':
                        r_new = r * np.exp(a_self * w)
                    else:
                        r_new = r * np.exp(-a_self * w)
                    
                    if not np.isfinite(r_new) or r_new <= 0:
                        continue
                        
                    direction = (other_fw.mean - firework.mean) / distance
                    point = firework.mean + r_new * direction
                    
                    if np.isfinite(point).all():
                        potential_points.append(point)
        
        if not potential_points:
            return []
            
        # 特徴点の選択（Localは近い点、Globalは遠い点を優先）
        distances = [np.linalg.norm(p - firework.mean) for p in potential_points]
        valid_indices = [i for i, d in enumerate(distances) if np.isfinite(d)]
        
        if not valid_indices:
            return []
        
        if firework.firework_type == 'local':
            sorted_indices = sorted(valid_indices, key=lambda i: distances[i])[:min(TAU, len(valid_indices))]
        else:
            sorted_indices = sorted(valid_indices, key=lambda i: distances[i], reverse=True)[:min(TAU, len(valid_indices))]
            
        selected_points = [potential_points[i] for i in sorted_indices]
        
        # クリッピング処理
        clipped_points = []
        for point in selected_points:
            distance = np.linalg.norm(point - firework.mean)
            if distance < 1e-12:
                continue
                
            radius = firework.compute_boundary_radius()
            
            if distance < ALPHA_L * radius:
                clipped_point = firework.mean + (point - firework.mean) * (ALPHA_L * radius / distance)
            elif distance > ALPHA_U * radius:
                clipped_point = firework.mean + (point - firework.mean) * (ALPHA_U * radius / distance)
            else:
                clipped_point = point
                
            if np.isfinite(clipped_point).all():
                clipped_points.append(clipped_point)
            
        return clipped_points

    def _adapt_to_feature_points(self, firework, feature_points: List[np.ndarray]) -> None:
        if not feature_points:
            return
            
        # 1. 平均ベクトルの移動
        shift_vector = np.zeros(self.dimension)
        valid_count = 0
        
        for f_point in feature_points:
            # 境界との交点
            q_point = self._compute_boundary_intersection(firework, f_point)
            diff = f_point - q_point
            if np.isfinite(diff).all():
                shift_vector += diff
                valid_count += 1
        
        if valid_count == 0:
            return
            
        shift_vector /= valid_count
        
        # 移動量の制限
        max_shift_ratio = ALPHA_M_LOCAL if firework.firework_type == 'local' else ALPHA_M_GLOBAL
        shift_norm = np.linalg.norm(shift_vector)
        
        if shift_norm > 1e-12 and np.isfinite(shift_norm):
            radius_estimate = firework.compute_boundary_radius()
            max_shift = max_shift_ratio * radius_estimate
            if shift_norm > max_shift:
                shift_vector *= max_shift / shift_norm
                
        new_mean = firework.mean + shift_vector
        
        if not np.isfinite(new_mean).all():
            return
        
        # 2. 共分散行列の適応（形状変化）
        covariance_adjustment = np.zeros((self.dimension, self.dimension))
        boundary_radius = firework.scale * np.sqrt(self.dimension)
        adjustment_count = 0
        
        for f_point in feature_points:
            try:
                # マハラノビス距離に基づく調整
                C_inv_sqrt = NumericalUtils.compute_matrix_inverse_sqrt(firework.covariance)
                z = C_inv_sqrt @ (f_point - new_mean) / firework.scale
                z_norm_squared = np.dot(z, z)
                
                if z_norm_squared > 1e-10 and np.isfinite(z_norm_squared):
                    # 境界に合わせるための伸縮
                    lambda_val = 1.0 / (boundary_radius**2) - 1.0 / z_norm_squared
                    
                    # 大きすぎる調整を制限
                    lambda_val = np.clip(lambda_val, -1.0, 1.0)
                    
                    adjustment_vector = f_point - new_mean
                    adj = (lambda_val / (firework.scale**2) * np.outer(adjustment_vector, adjustment_vector))
                    
                    if np.isfinite(adj).all():
                        covariance_adjustment += adj
                        adjustment_count += 1
            except Exception:
                continue
                
        if adjustment_count > 0:
            covariance_adjustment /= adjustment_count
            new_covariance = firework.covariance + covariance_adjustment
            new_covariance = NumericalUtils.ensure_numerical_stability(new_covariance)
            
            if NumericalUtils.is_valid_matrix(new_covariance):
                firework.covariance = new_covariance
                
        firework.mean = new_mean

    def _compute_boundary_intersection(self, firework, point: np.ndarray) -> np.ndarray:
        direction = point - firework.mean
        direction_norm = np.linalg.norm(direction)
        if direction_norm < self.min_distance or not np.isfinite(direction_norm):
            return firework.mean.copy()
            
        unit_direction = direction / direction_norm
        radius = firework.compute_boundary_radius(unit_direction)
        
        if not np.isfinite(radius):
            return firework.mean.copy()
            
        return firework.mean + radius * unit_direction

# --- HCFWAメインクラス ---
class OldHCFWA:
    """
    Hierarchical Cooperative Fireworks Algorithm (HCFWA)
    CMA-ESベースの花火と協調戦略を組み合わせたアルゴリズム
    """
    def __init__(self, 
                 problem: BaseProblem, 
                 num_local_fireworks: int = 4,
                 sparks_per_firework: int = None,
                 max_evaluations: int = 10000000):
        
        self.problem = problem
        self.dimension = problem.dimension
        self.bounds = np.array([problem.lower_bounds, problem.upper_bounds]).T
        self.num_local_fireworks = num_local_fireworks
        
        # 火花数の設定
        if sparks_per_firework is None:
            total_sparks = max(20, 4 * self.dimension)
            self.sparks_per_firework = total_sparks // (num_local_fireworks + 1)
        else:
            self.sparks_per_firework = sparks_per_firework

        self.max_evaluations = max_evaluations
        self.collaboration_manager = CollaborationManager(self.dimension)
        
        # 状態初期化
        self.best_fitness = float('inf')
        self.best_solution = None
        self.global_evaluation_count = 0
        self.iteration_count = 0
        self.global_stagnation_count = 0
        self.fitness_history = []
        
        self._initialize_fireworks()

    def _initialize_fireworks(self) -> None:
        # グローバル花火1個 + ローカル花火N個
        self.global_firework = Firework(self.dimension, self.bounds, 'global', 
                                      0, self.num_local_fireworks)
        self.local_fireworks = [
            Firework(self.dimension, self.bounds, 'local', i + 1, self.num_local_fireworks) 
            for i in range(self.num_local_fireworks)
        ]
        self.all_fireworks = [self.global_firework] + self.local_fireworks

    def optimize(self) -> Tuple[np.ndarray, float, List[float]]:
        # 初期化
        self.best_fitness = float('inf')
        self.best_solution = None
        self.global_evaluation_count = 0
        self.iteration_count = 0
        self.fitness_history = []
        self._initialize_fireworks()

        while self.global_evaluation_count < self.max_evaluations:
            iteration_start_fitness = self.best_fitness
            all_sparks = []
            all_fitness_values = []
            firework_spark_ranges = []

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

            if self.global_evaluation_count >= self.max_evaluations and not all_fitness_values:
                break

            # 2. 全体ベストの更新
            all_fitness_array = np.array(all_fitness_values)
            if all_fitness_array.size > 0:
                best_idx = np.argmin(all_fitness_array)
                if all_fitness_array[best_idx] < self.best_fitness:
                    self.best_fitness = all_fitness_array[best_idx]
                    self.best_solution = all_sparks[best_idx].copy()

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
