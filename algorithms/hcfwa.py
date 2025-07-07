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

# --- 数値計算基盤クラス (改良版) ---
class NumericalUtils:
    @staticmethod
    def ensure_numerical_stability(matrix: np.ndarray, min_eigenval: float = 1e-10) -> np.ndarray:
        """共分散行列の数値的安定性を保証"""
        # 対称化
        matrix = 0.5 * (matrix + matrix.T)
        
        try:
            # 固有値分解
            eigenvals, eigenvecs = np.linalg.eigh(matrix)
            
            # 固有値の修正
            eigenvals = np.maximum(eigenvals, min_eigenval)
            
            # 条件数の制限
            max_eigenval = np.max(eigenvals)
            min_eigenval_allowed = max_eigenval / 1e10  # 条件数を1e10以下に制限
            eigenvals = np.maximum(eigenvals, min_eigenval_allowed)
            
            # 再構成
            return eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
            
        except np.linalg.LinAlgError:
            # エラー時は単位行列に小さな値を加えた行列を返す
            return np.eye(matrix.shape[0]) * min_eigenval
    
    @staticmethod
    def compute_matrix_inverse_sqrt(matrix: np.ndarray, regularization: float = 1e-10) -> np.ndarray:
        """行列の逆平方根を安定的に計算"""
        try:
            eigenvals, eigenvecs = np.linalg.eigh(matrix)
            # 正則化
            eigenvals = np.maximum(eigenvals, regularization)
            # 逆平方根の計算
            inv_sqrt_eigenvals = 1.0 / np.sqrt(eigenvals)
            return eigenvecs @ np.diag(inv_sqrt_eigenvals) @ eigenvecs.T
        except np.linalg.LinAlgError:
            # エラー時は正則化された単位行列の逆平方根を返す
            return np.eye(matrix.shape[0]) / np.sqrt(regularization)
    
    @staticmethod
    def safe_norm(matrix: np.ndarray, ord=2) -> float:
        """行列ノルムの安全な計算"""
        try:
            # Frobenius normを使用（より安定）
            if ord == 2:
                return np.sqrt(np.sum(matrix**2))
            else:
                return np.linalg.norm(matrix, ord=ord)
        except:
            return 1.0
    
    @staticmethod
    def mirror_boundary_mapping(x: np.ndarray, bounds: np.ndarray) -> np.ndarray:
        """境界制約の処理"""
        x_mapped = x.copy()
        
        # NaNやInfのチェック
        if not np.all(np.isfinite(x_mapped)):
            # 無効な値は境界の中心にリセット
            invalid_mask = ~np.isfinite(x_mapped)
            x_mapped[invalid_mask] = (bounds[invalid_mask, 0] + bounds[invalid_mask, 1]) / 2
        
        # 境界内への射影
        for d in range(len(x)):
            lb, ub = bounds[d, 0], bounds[d, 1]
            if ub <= lb:
                x_mapped[d] = lb
                continue
            
            # 周期的境界条件
            width = ub - lb
            if x_mapped[d] < lb or x_mapped[d] > ub:
                # 境界内に収まるまで折り返し
                relative_pos = (x_mapped[d] - lb) % (2 * width)
                if relative_pos <= width:
                    x_mapped[d] = lb + relative_pos
                else:
                    x_mapped[d] = ub - (relative_pos - width)
        
        # 最終的なクリップ
        return np.clip(x_mapped, bounds[:, 0], bounds[:, 1])

# --- Firework基底クラス (数値安定性を強化) ---
class Firework:
    def __init__(self, dimension: int, bounds: np.ndarray, firework_type: str = 'local',
                 firework_id: int = 0, num_local_fireworks: int = 4):
        self.dimension = dimension
        self.bounds = np.array(bounds)
        self.firework_type = firework_type
        self.firework_id = firework_id
        self.num_local_fireworks = num_local_fireworks
        
        # 初期化
        self.mean = self._initialize_mean()
        self.covariance = np.eye(dimension)
        self.scale = self._initialize_scale()
        
        # 進化パス
        self.evolution_path_c = np.zeros(dimension)
        self.evolution_path_sigma = np.zeros(dimension)
        
        # 学習率
        self.learning_rates = self._initialize_learning_rates()
        
        # 最適化状態
        self.best_fitness = float('inf')
        self.best_solution = None
        self.stagnation_count = 0
        self.recent_fitness_history = []
        self.evaluation_count = 0
        self.last_improvement_iteration = 0
        self.restart_count = 0
        
        # スケールの制限（より保守的に）
        self.min_scale = 1e-8
        self.max_scale = 1e8
    
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
                'cm': 1.0,
                'c_mu': 0.25,
                'c1': 0.0,
                'cc': 0.0,
                'c_sigma': 0.0,
                'd_sigma': 0.0,
                'cr': 1.0,  # 論文通りに修正
                'cg': 1.0 / self.num_local_fireworks
            }
        else:
            return {
                'cm': 1.0,
                'c_mu': 0.25,
                'c1': 2.0 / ((D + 1.3)**2),
                'cc': 4.0 / (D + 4.0),
                'c_sigma': 4.0 / (D + 4.0),
                'd_sigma': 1.0 + 2.0 * max(0, np.sqrt((D-1)/(D+1)) - 1) * 0.5,
                'cr': 0.5
            }
    
    def generate_sparks(self, num_sparks: int) -> np.ndarray:
        """火花の生成（数値的に安定）"""
        # 共分散行列の安定化
        stable_cov = NumericalUtils.ensure_numerical_stability(self.covariance)
        
        # スケールの制限
        safe_scale = np.clip(self.scale, self.min_scale, self.max_scale)
        
        try:
            # 共分散行列のスケーリング
            scaled_cov = safe_scale**2 * stable_cov
            
            # サンプリング
            sparks = np.random.multivariate_normal(
                mean=self.mean, 
                cov=scaled_cov, 
                size=num_sparks
            )
        except (np.linalg.LinAlgError, ValueError) as e:
            # エラー時は等方的なガウス分布からサンプリング
            warnings.warn(f"Multivariate normal sampling failed: {e}. Using isotropic Gaussian.")
            sparks = self.mean + safe_scale * np.random.randn(num_sparks, self.dimension)
        
        # 境界処理
        for i in range(num_sparks):
            sparks[i] = NumericalUtils.mirror_boundary_mapping(sparks[i], self.bounds)
        
        return sparks
    
    def compute_recombination_weights(self, fitness_values: np.ndarray) -> np.ndarray:
        """重みの計算"""
        if len(fitness_values) == 0:
            return np.array([])
        
        if self.firework_type == 'local':
            # 局所花火：上位半分を選択
            mu = max(1, len(fitness_values) // 2)
            sorted_indices = np.argsort(fitness_values)
            weights = np.zeros(len(fitness_values))
            
            for rank, idx in enumerate(sorted_indices[:mu]):
                w = max(0, np.log(mu + 0.5) - np.log(rank + 1))
                weights[idx] = w
            
            # 正規化
            weight_sum = np.sum(weights)
            if weight_sum > 0:
                weights = weights / weight_sum
        else:
            # 大域花火：下位5%を除外
            num_select = max(1, int(0.95 * len(fitness_values)))
            sorted_indices = np.argsort(fitness_values)[:num_select]
            weights = np.zeros(len(fitness_values))
            if num_select > 0:
                weights[sorted_indices] = 1.0 / num_select
        
        return weights
    
    def update_parameters(self, sparks: np.ndarray, fitness_values: np.ndarray) -> None:
        """パラメータ更新（数値的に安定）"""
        if sparks.shape[0] == 0:
            return
        
        # 古いパラメータを保存（大域花火の学習率適用用）
        old_mean = self.mean.copy()
        old_covariance = self.covariance.copy()
        old_scale = self.scale
        
        # 重みの計算
        weights = self.compute_recombination_weights(fitness_values)
        mu_eff = 1.0 / (np.sum(weights**2) + 1e-10)
        
        # 平均の更新
        weighted_diff = np.zeros(self.dimension)
        if np.sum(weights) > 0:
            weighted_diff = np.dot(weights, sparks - self.mean)
        
        new_mean = self.mean + self.learning_rates['cm'] * weighted_diff
        
        # 参照平均
        reference_mean = ((1 - self.learning_rates['cr']) * self.mean + 
                         self.learning_rates['cr'] * new_mean)
        
        # 進化パスの更新（rank-1更新用）
        if self.learning_rates['cc'] > 0:
            self.evolution_path_c = ((1 - self.learning_rates['cc']) * self.evolution_path_c +
                                   np.sqrt(self.learning_rates['cc'] * (2 - self.learning_rates['cc']) * mu_eff) *
                                   (new_mean - self.mean) / (self.scale + 1e-10))
        
        # 共分散行列の更新
        rank_mu_update = np.zeros((self.dimension, self.dimension))
        if np.sum(weights) > 0:
            y = sparks - reference_mean
            rank_mu_update = np.dot(y.T * weights, y)
        
        rank_one_update = np.outer(self.evolution_path_c, self.evolution_path_c)
        
        new_covariance = ((1 - self.learning_rates['c_mu'] - self.learning_rates['c1']) * self.covariance +
                         self.learning_rates['c_mu'] * rank_mu_update +
                         self.learning_rates['c1'] * rank_one_update)
        
        # 共分散行列の安定化
        new_covariance = NumericalUtils.ensure_numerical_stability(new_covariance)
        
        # スケールの更新（局所花火のみ）
        new_scale = self.scale
        if (self.firework_type == 'local' and 
            self.learning_rates['d_sigma'] > 0 and 
            self.learning_rates['c_sigma'] > 0):
            
            try:
                C_inv_sqrt = NumericalUtils.compute_matrix_inverse_sqrt(self.covariance)
                mean_diff_scaled = (new_mean - self.mean) / (self.scale + 1e-10)
                
                self.evolution_path_sigma = ((1 - self.learning_rates['c_sigma']) * self.evolution_path_sigma +
                                           np.sqrt(self.learning_rates['c_sigma'] * (2 - self.learning_rates['c_sigma']) * mu_eff) *
                                           (C_inv_sqrt @ mean_diff_scaled))
                
                # スケール更新
                path_norm = np.linalg.norm(self.evolution_path_sigma)
                expected_norm = np.sqrt(self.dimension)
                
                log_scale_change = (self.learning_rates['c_sigma'] / self.learning_rates['d_sigma']) * (path_norm / expected_norm - 1)
                log_scale_change = np.clip(log_scale_change, -0.5, 0.5)  # より保守的な制限
                
                new_scale = self.scale * np.exp(log_scale_change)
                new_scale = np.clip(new_scale, self.min_scale, self.max_scale)
                
            except (np.linalg.LinAlgError, RuntimeWarning):
                # エラー時はスケール更新をスキップ
                pass
        
        # 大域花火の場合は学習率を適用（論文通り）
        if self.firework_type == 'global' and 'cg' in self.learning_rates:
            cg = self.learning_rates['cg']
            self.mean = cg * new_mean + (1 - cg) * old_mean
            self.covariance = cg * new_covariance + (1 - cg) * old_covariance
            self.scale = cg * new_scale + (1 - cg) * old_scale
        else:
            self.mean = new_mean
            self.covariance = new_covariance
            self.scale = new_scale
        
        # 最良解の更新
        self._update_best_solution(sparks, fitness_values)
        self.evaluation_count += len(sparks)
    
    def _update_best_solution(self, sparks: np.ndarray, fitness_values: np.ndarray) -> None:
        """最良解の更新"""
        if len(fitness_values) == 0:
            return
        
        best_idx = np.argmin(fitness_values)
        current_best = fitness_values[best_idx]
        
        if current_best < self.best_fitness:
            self.best_fitness = current_best
            self.best_solution = sparks[best_idx].copy()
            self.stagnation_count = 0
            self.last_improvement_iteration = self.evaluation_count
        else:
            self.stagnation_count += 1
        
        # 履歴の更新
        self.recent_fitness_history.append(current_best)
        if len(self.recent_fitness_history) > 50:
            self.recent_fitness_history.pop(0)
    
    def compute_boundary_radius(self, direction: Optional[np.ndarray] = None) -> float:
        """境界半径の計算"""
        d_B = np.sqrt(self.dimension) + 0.5 * np.sqrt(2 * self.dimension)
        
        if direction is None:
            # 平均的な半径
            avg_eigenval = np.trace(self.covariance) / self.dimension
            return self.scale * np.sqrt(max(avg_eigenval, 1e-10)) * d_B
        else:
            # 特定方向の半径
            direction_norm = np.linalg.norm(direction)
            if direction_norm < 1e-10:
                return 0.0
            
            unit_direction = direction / direction_norm
            radius_squared = unit_direction.T @ self.covariance @ unit_direction
            return self.scale * np.sqrt(max(radius_squared, 1e-10)) * d_B
    
    def check_restart_conditions(self, all_fireworks: List['Firework']) -> Tuple[bool, List[str]]:
        """再起動条件のチェック"""
        restart_reasons = []
        
        # 適応度が収束
        if len(self.recent_fitness_history) > 10:
            if np.std(self.recent_fitness_history[-10:]) <= 1e-10:
                restart_reasons.append("fitness_converged")
        
        # 位置が収束（安全な計算）
        try:
            cov_norm = NumericalUtils.safe_norm(self.covariance)
            if self.scale * cov_norm <= 1e-5:
                restart_reasons.append("position_converged")
        except:
            # エラー時は収束したとみなす
            restart_reasons.append("position_converged")
        
        # 改善なし
        max_stagnation = 4 * EPSILON_L if self.firework_type == 'global' else EPSILON_L
        if self.stagnation_count >= max_stagnation:
            restart_reasons.append("not_improving")
        
        # 他の花火との関係
        for fw in all_fireworks:
            if fw.firework_id != self.firework_id and fw.best_fitness < self.best_fitness:
                # 平均位置が収束
                if np.linalg.norm(self.mean - fw.mean) < 1e-5:
                    restart_reasons.append("mean_converged")
                    break
                
                # より良い花火に覆われている
                try:
                    distance = np.linalg.norm(self.mean - fw.mean)
                    fw_radius = fw.compute_boundary_radius()
                    self_radius = self.compute_boundary_radius()
                    
                    if fw_radius > self_radius and distance + self_radius < fw_radius * 1.1:
                        restart_reasons.append("covered_by_better")
                        break
                except:
                    # エラー時はスキップ
                    pass
        
        return len(restart_reasons) > 0, restart_reasons
    
    def restart(self) -> None:
        """花火の再初期化"""
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

# --- 協調戦略管理クラス（数値的に安定） ---
class CollaborationManager:
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.ca = CA_AMPLIFICATION
        self.min_distance = 1e-8
        self.max_w_value = 5.0  # より保守的な値
        self.min_w_value = -5.0
    
    def execute_collaboration(self, fireworks: List[Firework]) -> None:
        """協調の実行"""
        if len(fireworks) < 2:
            return
        
        try:
            dividing_points = self._compute_all_dividing_points(fireworks)
            
            for fw in fireworks:
                feature_points = self._select_feature_points(fw, dividing_points, fireworks)
                if feature_points:
                    self._adapt_to_feature_points(fw, feature_points)
        except Exception as e:
            # 協調でエラーが発生しても最適化を継続
            warnings.warn(f"Collaboration error: {e}")
    
    def _compute_all_dividing_points(self, fireworks: List[Firework]) -> Dict[Tuple[int, int], float]:
        """全ての分割点を計算"""
        dividing_points = {}
        
        for i in range(len(fireworks)):
            for j in range(i + 1, len(fireworks)):
                fw1, fw2 = fireworks[i], fireworks[j]
                
                try:
                    w = self._solve_dividing_equation(fw1, fw2)
                    dividing_points[(fw1.firework_id, fw2.firework_id)] = w
                    dividing_points[(fw2.firework_id, fw1.firework_id)] = w
                except Exception:
                    # エラー時は0.0を使用
                    dividing_points[(fw1.firework_id, fw2.firework_id)] = 0.0
                    dividing_points[(fw2.firework_id, fw1.firework_id)] = 0.0
        
        return dividing_points
    
    def _solve_dividing_equation(self, fw1: Firework, fw2: Firework) -> float:
        """分割方程式を解く"""
        distance = np.linalg.norm(fw1.mean - fw2.mean)
        if distance < self.min_distance:
            return 0.0
        
        # 半径の計算
        r1 = self._compute_radius_on_line(fw1, fw2)
        r2 = self._compute_radius_on_line(fw2, fw1)
        
        # 感度係数
        a1, a2 = self._compute_sensitivity_factors(fw1, fw2)
        
        # 方程式の定義
        if fw1.firework_type == 'local' and fw2.firework_type == 'local':
            def equation(w):
                return r1 * np.exp(np.clip(a1 * w, -10, 10)) + r2 * np.exp(np.clip(a2 * w, -10, 10)) - distance
        elif fw1.firework_type == 'global':
            def equation(w):
                return r1 * np.exp(np.clip(-a1 * w, -10, 10)) - r2 * np.exp(np.clip(a2 * w, -10, 10)) - distance
        else:  # fw2 is global
            def equation(w):
                return r1 * np.exp(np.clip(a1 * w, -10, 10)) - r2 * np.exp(np.clip(-a2 * w, -10, 10)) - distance
        
        # 解の探索
        try:
            # 端点での値を確認
            fa = equation(self.min_w_value)
            fb = equation(self.max_w_value)
            
            if np.sign(fa) != np.sign(fb):
                return brentq(equation, self.min_w_value, self.max_w_value, xtol=1e-8, rtol=1e-8)
            else:
                # 解が存在しない場合は0を返す
                return 0.0
        except (ValueError, RuntimeError):
            return 0.0
    
    def _compute_radius_on_line(self, fw1: Firework, fw2: Firework) -> float:
        """線上での半径を計算"""
        direction = fw2.mean - fw1.mean
        direction_norm = np.linalg.norm(direction)
        
        if direction_norm < self.min_distance:
            return fw1.compute_boundary_radius()
        
        unit_direction = direction / direction_norm
        return fw1.compute_boundary_radius(unit_direction)
    
    def _compute_sensitivity_factors(self, fw1: Firework, fw2: Firework) -> Tuple[float, float]:
        """感度係数を計算"""
        a1, a2 = 1.0, 1.0
        
        # 適応度に基づく調整
        if fw1.best_fitness < fw2.best_fitness * 0.99:  # 1%の余裕を持たせる
            a1, a2 = 0.0, 1.0
        elif fw2.best_fitness < fw1.best_fitness * 0.99:
            a1, a2 = 1.0, 0.0
        
        # 最近改善した花火は保護
        improvement_threshold = 0.2 * EPSILON_L
        if (fw1.firework_type == 'local' and 
            fw1.evaluation_count - fw1.last_improvement_iteration < improvement_threshold):
            a1 = 0.0
        
        if (fw2.firework_type == 'local' and 
            fw2.evaluation_count - fw2.last_improvement_iteration < improvement_threshold):
            a2 = 0.0
        
        # 大域花火の感度を増幅
        if fw1.firework_type == 'global':
            a1 *= self.ca
        if fw2.firework_type == 'global':
            a2 *= self.ca
        
        return a1, a2
    
    def _select_feature_points(self, firework: Firework, dividing_points: Dict, 
                              fireworks: List[Firework]) -> List[np.ndarray]:
        """特徴点を選択"""
        potential_points = []
        
        for other_fw in fireworks:
            if other_fw.firework_id == firework.firework_id:
                continue
            
            key = (firework.firework_id, other_fw.firework_id)
            if key not in dividing_points:
                continue
            
            w = dividing_points[key]
            distance = np.linalg.norm(firework.mean - other_fw.mean)
            
            if distance > self.min_distance:
                r = self._compute_radius_on_line(firework, other_fw)
                a_self, _ = self._compute_sensitivity_factors(firework, other_fw)
                
                # 新しい半径（オーバーフロー防止）
                if firework.firework_type == 'local':
                    r_new = r * np.exp(np.clip(a_self * w, -10, 10))
                else:
                    r_new = r * np.exp(np.clip(-a_self * w, -10, 10))
                
                direction = (other_fw.mean - firework.mean) / distance
                potential_points.append(firework.mean + r_new * direction)
        
        if not potential_points:
            return []
        
        # 距離に基づいて選択
        distances = [np.linalg.norm(p - firework.mean) for p in potential_points]
        
        if firework.firework_type == 'local':
            # 局所花火：最も近いものを選択
            indices = np.argsort(distances)[:min(TAU, len(distances))]
        else:
            # 大域花火：最も遠いものを選択
            indices = np.argsort(distances)[-min(TAU, len(distances)):]
        
        selected_points = [potential_points[i] for i in indices]
        
        # クリッピング
        clipped_points = []
        for point in selected_points:
            distance = np.linalg.norm(point - firework.mean)
            radius = firework.compute_boundary_radius()
            
            if distance > 1e-9:  # ゼロ除算を回避
                if distance < ALPHA_L * radius:
                    point = firework.mean + (point - firework.mean) * (ALPHA_L * radius / distance)
                elif distance > ALPHA_U * radius:
                    point = firework.mean + (point - firework.mean) * (ALPHA_U * radius / distance)
            
            clipped_points.append(point)
        
        return clipped_points
    
    def _adapt_to_feature_points(self, firework: Firework, feature_points: List[np.ndarray]) -> None:
        """特徴点への適応"""
        if not feature_points:
            return
        
        # 平均シフトの計算
        shift_vector = np.zeros(self.dimension)
        for f_point in feature_points:
            q_point = self._compute_boundary_intersection(firework, f_point)
            shift_vector += (f_point - q_point)
        
        shift_vector /= len(feature_points)
        
        # シフトの制限
        max_shift_ratio = ALPHA_M_LOCAL if firework.firework_type == 'local' else ALPHA_M_GLOBAL
        shift_norm = np.linalg.norm(shift_vector)
        
        if shift_norm > 0:
            max_shift = max_shift_ratio * firework.compute_boundary_radius()
            if shift_norm > max_shift:
                shift_vector *= max_shift / shift_norm
        
        # 新しい平均位置
        new_mean = firework.mean + shift_vector
        
        # 共分散行列の調整
        covariance_adjustment = np.zeros((self.dimension, self.dimension))
        boundary_radius = firework.scale * np.sqrt(self.dimension)
        
        for f_point in feature_points:
            try:
                # 安定的な計算
                C_inv_sqrt = NumericalUtils.compute_matrix_inverse_sqrt(firework.covariance)
                z = C_inv_sqrt @ (f_point - new_mean) / (firework.scale + 1e-10)
                z_norm_squared = np.dot(z, z)
                
                if z_norm_squared > 1e-8:
                    lambda_val = 1.0 / (boundary_radius**2 + 1e-10) - 1.0 / (z_norm_squared + 1e-10)
                    
                    # オーバーフロー防止
                    lambda_val = np.clip(lambda_val, -1e6, 1e6)
                    
                    adjustment_vector = f_point - new_mean
                    scale_squared = firework.scale**2 + 1e-10
                    
                    covariance_adjustment += (lambda_val / scale_squared) * np.outer(adjustment_vector, adjustment_vector)
            except Exception:
                # エラー時はスキップ
                continue
        
        # 共分散行列の更新
        if len(feature_points) > 0:
            covariance_adjustment /= len(feature_points)
            new_covariance = firework.covariance + covariance_adjustment
            new_covariance = NumericalUtils.ensure_numerical_stability(new_covariance)
            
            # 更新
            firework.mean = new_mean
            firework.covariance = new_covariance
    
    def _compute_boundary_intersection(self, firework: Firework, point: np.ndarray) -> np.ndarray:
        """境界との交点を計算"""
        direction = point - firework.mean
        direction_norm = np.linalg.norm(direction)
        
        if direction_norm < self.min_distance:
            return firework.mean
        
        unit_direction = direction / direction_norm
        radius = firework.compute_boundary_radius(unit_direction)
        
        return firework.mean + radius * unit_direction

# --- HCFWAメインクラス ---
class HCFWA:
    """階層協調花火アルゴリズム"""
    
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
        
        self._reset_optimization_state()
    
    def _initialize_fireworks(self) -> None:
        """花火の初期化"""
        self.global_firework = Firework(
            self.dimension, self.bounds, 'global', 0, self.num_local_fireworks
        )
        
        self.local_fireworks = [
            Firework(self.dimension, self.bounds, 'local', i + 1, self.num_local_fireworks)
            for i in range(self.num_local_fireworks)
        ]
        
        self.all_fireworks = [self.global_firework] + self.local_fireworks
    
    def _reset_optimization_state(self) -> None:
        """最適化状態のリセット"""
        self.best_fitness = float('inf')
        self.best_solution = None
        self.global_evaluation_count = 0
        self.iteration_count = 0
        self.global_stagnation_count = 0
        self.fitness_history = []
        self._initialize_fireworks()
    
    def optimize(self) -> Tuple[np.ndarray, float, List[float]]:
        """最適化の実行"""
        objective_function = self.problem.evaluate
        self._reset_optimization_state()
        
        # 安全のための最大反復回数
        iteration_limit = 50000
        
        while (self.global_evaluation_count < self.max_evaluations and 
               self.iteration_count < iteration_limit):
            
            # デバッグ情報の出力
            # if self.iteration_count % 100 == 0:
            #     scales = [f"FW{fw.firework_id}: {fw.scale:.2e}" for fw in self.all_fireworks]
            #     print(f"Iter: {self.iteration_count}, Evals: {self.global_evaluation_count}, "
            #           f"BestFit: {self.best_fitness:.4e}, Scales: [{', '.join(scales)}]")
            
            iteration_start_fitness = self.best_fitness
            
            # 全ての火花を生成・評価
            all_sparks = []
            all_fitness_values = []
            firework_spark_ranges = []
            
            for fw in self.all_fireworks:
                try:
                    # 火花生成
                    sparks = fw.generate_sparks(self.sparks_per_firework)
                    fitness_values = []
                    
                    # 評価
                    for spark in sparks:
                        if self.global_evaluation_count >= self.max_evaluations:
                            break
                        
                        try:
                            fitness = objective_function(spark)
                            if np.isfinite(fitness):
                                fitness_values.append(fitness)
                            else:
                                fitness_values.append(float('inf'))
                        except Exception:
                            fitness_values.append(float('inf'))
                        
                        self.global_evaluation_count += 1
                    
                    if not fitness_values:
                        continue
                    
                    fitness_values = np.array(fitness_values)
                    all_sparks.extend(sparks[:len(fitness_values)])
                    all_fitness_values.extend(fitness_values)
                    firework_spark_ranges.append((len(all_sparks) - len(fitness_values), len(all_sparks)))
                    
                except Exception as e:
                    warnings.warn(f"Error in spark generation/evaluation for FW{fw.firework_id}: {e}")
                    continue
            
            if self.global_evaluation_count >= self.max_evaluations:
                break
            
            # 最良解の更新
            if all_fitness_values:
                all_fitness_array = np.array(all_fitness_values)
                best_idx = np.argmin(all_fitness_array)
                
                if all_fitness_array[best_idx] < self.best_fitness:
                    self.best_fitness = all_fitness_array[best_idx]
                    self.best_solution = np.array(all_sparks[best_idx])
            
            # パラメータ更新
            for i, fw in enumerate(self.all_fireworks):
                if i < len(firework_spark_ranges):
                    start_idx, end_idx = firework_spark_ranges[i]
                    if start_idx < end_idx:
                        try:
                            fw.update_parameters(
                                np.array(all_sparks[start_idx:end_idx]),
                                all_fitness_array[start_idx:end_idx]
                            )
                        except Exception as e:
                            warnings.warn(f"Error in parameter update for FW{fw.firework_id}: {e}")
            
            # 協調
            try:
                self.collaboration_manager.execute_collaboration(self.all_fireworks)
            except Exception as e:
                warnings.warn(f"Error in collaboration: {e}")
            
            # 再起動チェック
            for fw in self.all_fireworks:
                try:
                    should_restart, reasons = fw.check_restart_conditions(self.all_fireworks)
                    if should_restart:
                        fw.restart()
                except Exception as e:
                    warnings.warn(f"Error in restart check for FW{fw.firework_id}: {e}")
            
            # 大域的な停滞チェック
            if self.best_fitness >= iteration_start_fitness:
                self.global_stagnation_count += 1
            else:
                self.global_stagnation_count = 0
            
            # 全体再起動
            if self.global_stagnation_count >= M_GLOBAL_REBOOT:
                # 最良解を保存
                best_solution_backup = self.best_solution.copy() if self.best_solution is not None else None
                best_fitness_backup = self.best_fitness
                
                # 全花火を再起動
                for fw in self.all_fireworks:
                    fw.restart()
                
                # 最良解を復元
                self.best_solution = best_solution_backup
                self.best_fitness = best_fitness_backup
                self.global_stagnation_count = 0
            
            # 履歴の更新
            self.fitness_history.append(self.best_fitness)
            self.iteration_count += 1
        
        # if self.iteration_count >= iteration_limit:
        #     print(f"WARNING: Reached iteration limit ({iteration_limit}) before max_evaluations.")
        
        return self.best_solution, self.best_fitness, self.fitness_history