import numpy as np
import time
import warnings
from typing import List, Tuple, Optional, Dict, Any, Callable
from scipy.optimize import brentq
from problems import BaseProblem

# --- 定数定義 ---
EPSILON_L = 100
EPSILON_V = 1e-5  # 適応度収束閾値を追加
EPSILON_P = 1e-5  # 位置収束閾値を追加
CA_AMPLIFICATION = 5.0
TAU = 2
ALPHA_L = 0.85
ALPHA_U = 1.20
ALPHA_M_LOCAL = 0.20
ALPHA_M_GLOBAL = 0.05
M_GLOBAL_REBOOT = 100

# デバッグ用フラグ
DEBUG = False  # Trueにすると詳細ログが出力される

# --- 数値計算基盤クラス (より堅牢な実装) ---
class NumericalUtils:
    @staticmethod
    def ensure_numerical_stability(matrix: np.ndarray, epsilon=1e-14) -> np.ndarray:
        if not np.all(np.isfinite(matrix)): return np.eye(matrix.shape[0])
        matrix = 0.5 * (matrix + matrix.T)
        try:
            eigenvals, eigenvecs = np.linalg.eigh(matrix)
            eigenvals[eigenvals < epsilon] = epsilon
            max_eigenval, min_eigenval = np.max(eigenvals), np.min(eigenvals)
            if min_eigenval > 0 and max_eigenval / min_eigenval > 1e14:
                eigenvals = np.maximum(eigenvals, max_eigenval / 1e14)
            return eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
        except np.linalg.LinAlgError: return np.eye(matrix.shape[0])

    @staticmethod
    def compute_matrix_inverse_sqrt(matrix: np.ndarray, epsilon=1e-14) -> np.ndarray:
        try:
            eigenvals, eigenvecs = np.linalg.eigh(matrix)
            eigenvals[eigenvals < epsilon] = epsilon
            return eigenvecs @ np.diag(1.0 / np.sqrt(eigenvals)) @ eigenvecs.T
        except np.linalg.LinAlgError: return np.eye(matrix.shape[0])

    @staticmethod
    def safe_norm(matrix: np.ndarray, ord=2) -> float:
        """行列のノルムを安全に計算"""
        try:
            return np.linalg.norm(matrix, ord)
        except Exception:
            return np.inf  # 数値異常時は発散扱い

    @staticmethod
    def mirror_boundary_mapping(x: np.ndarray, bounds: np.ndarray) -> np.ndarray:
        x_mapped = x.copy()
        if not np.all(np.isfinite(x_mapped)): return np.clip(x, bounds[:, 0], bounds[:, 1])
        for d in range(len(x)):
            lb, ub = bounds[d, 0], bounds[d, 1]
            if ub <= lb: continue
            width = ub - lb
            if not (lb <= x_mapped[d] <= ub):
                val, rem = divmod(x_mapped[d] - lb, 2 * width)
                x_mapped[d] = lb + rem if val % 2 == 0 else ub - rem
        return np.clip(x_mapped, bounds[:, 0], bounds[:, 1])

# --- Firework基底クラス (数値安定性を強化) ---
class Firework:
    def __init__(self, dimension: int, bounds: np.ndarray, firework_type: str, firework_id: int, num_local_fireworks: int):
        self.dimension = dimension
        self.bounds = bounds
        self.firework_type = firework_type
        self.firework_id = firework_id
        self.num_local_fireworks = num_local_fireworks
        self.rng = np.random.default_rng()
        self._reset_state()
        
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
        
    def _reset_state(self):
        self.mean = self.rng.uniform(self.bounds[:, 0], self.bounds[:, 1]) if self.firework_type == 'local' else (self.bounds[:, 0] + self.bounds[:, 1]) / 2
        self.covariance = np.eye(self.dimension)
        ub, lb = np.max(self.bounds[:, 1]), np.min(self.bounds[:, 0])
        expected_norm = np.sqrt(self.dimension) or 1
        global_scale = (ub - lb) / (2 * expected_norm)
        self.scale = global_scale / self.num_local_fireworks if self.firework_type == 'local' else global_scale
        self.evolution_path_c = np.zeros(self.dimension)
        self.evolution_path_sigma = np.zeros(self.dimension)
        D = self.dimension
        mu = (4 * D // 2)
        mu_eff = mu
        self.learning_rates = {
            'cm': 1.0, 'c_mu': 0.25,
            'c1': 2.0 / ((D + 1.3)**2 + mu_eff),
            'cc': (4.0 + mu_eff / D) / (D + 4.0 + 2 * mu_eff / D),
            'c_sigma': (mu_eff + 2.0) / (D + mu_eff + 5.0),
            'd_sigma': 1.0 + 2.0 * max(0, np.sqrt((mu_eff - 1.0) / (D + 1.0)) - 1.0) + (mu_eff + 2.0) / (D + mu_eff + 5.0),
            'cr': 0.5
        } if self.firework_type == 'local' else {'cm': 1.0, 'c_mu': 0.25, 'c1': 0.0, 'cc': 0.0, 'c_sigma': 0.0, 'd_sigma': 0.0, 'cr': 0.5, 'cg': 1.0 / (self.num_local_fireworks or 1)}
        self.best_fitness = float('inf')
        self.stagnation_count = 0
    
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
        safe_scale = np.clip(np.nan_to_num(self.scale), 1e-8, 1e8)
        stable_cov = NumericalUtils.ensure_numerical_stability(self.covariance)
        try:
            sparks = self.rng.multivariate_normal(mean=self.mean, cov=(safe_scale**2) * stable_cov, size=num_sparks, check_valid='warn', tol=1e-8)
        except (np.linalg.LinAlgError, ValueError):
            sparks = self.mean + safe_scale * self.rng.standard_normal(size=(num_sparks, self.dimension))
        return np.apply_along_axis(NumericalUtils.mirror_boundary_mapping, 1, sparks, self.bounds)
    
    
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
        
        # パラメータの更新（グローバル花火の場合はブレンドしない）
        if self.firework_type == 'global':
            # 論文通りの実装：ブレンドなし
            self.mean = new_mean
            self.covariance = new_covariance
            self.scale = new_scale
        else:
            # ローカル花火はそのまま更新
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
        
        # 適応度が収束（EPSILON_Vを使用）
        if len(self.recent_fitness_history) > 10:
            if np.std(self.recent_fitness_history[-10:]) <= EPSILON_V:
                restart_reasons.append("fitness_converged")
        
        # 位置が収束（EPSILON_Pを使用）
        try:
            cov_norm = NumericalUtils.safe_norm(self.covariance)
            if self.scale * cov_norm <= EPSILON_P:
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
        
        # 計算量を制限
        max_other_fireworks = min(len(fireworks) - 1, 10)  # 最大10個の花火と比較
        other_fireworks = [fw for fw in fireworks if fw.firework_id != firework.firework_id]
    
        if len(other_fireworks) > max_other_fireworks:
            # ランダムにサンプリング
            indices = np.random.choice(len(other_fireworks), max_other_fireworks, replace=False)
            other_fireworks = [other_fireworks[i] for i in indices]
        
        for other_fw in other_fireworks:
            key = (firework.firework_id, other_fw.firework_id)
            if key not in dividing_points:
                continue
            
            w = dividing_points[key]
            
            # wが異常値の場合はスキップ
            if not np.isfinite(w) or abs(w) > 100:
                continue
                
            distance = np.linalg.norm(firework.mean - other_fw.mean)
            
            if distance > self.min_distance:
                r = self._compute_radius_on_line(firework, other_fw)
                a_self, _ = self._compute_sensitivity_factors(firework, other_fw)
                
                # 新しい半径（オーバーフロー防止）
                try:
                    if firework.firework_type == 'local':
                        exp_arg = np.clip(a_self * w, -10, 10)
                        r_new = r * np.exp(exp_arg)
                    else:
                        exp_arg = np.clip(-a_self * w, -10, 10)
                        r_new = r * np.exp(exp_arg)
                        
                    # 半径が異常値の場合はスキップ
                    if not np.isfinite(r_new) or r_new <= 0 or r_new > 1e10:
                        continue
                        
                except (OverflowError, FloatingPointError):
                    continue
                
                direction = (other_fw.mean - firework.mean) / distance
                potential_points.append(firework.mean + r_new * direction)
        
        if not potential_points:
            return []
        
        # 最大でTAU個まで
        if len(potential_points) > TAU * 2:
            # ランダムサンプリング
            indices = np.random.choice(len(potential_points), TAU * 2, replace=False)
            potential_points = [potential_points[i] for i in indices]
        
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
                 max_evaluations: int = 100000):
        
        self.problem = problem
        self.dimension = problem.dimension
        self.bounds = np.array([problem.lower_bounds, problem.upper_bounds]).T
        self.num_local_fireworks = num_local_fireworks
        self.max_evaluations = max_evaluations

        if sparks_per_firework is None:
            total_sparks = max(20, int(4 + np.floor(3 * np.log(self.dimension))))
            self.sparks_per_firework = total_sparks
        else:
            self.sparks_per_firework = sparks_per_firework
        
        self._reset_state()

    def _reset_state(self):
        """状態をリセット"""
        self.best_fitness = float('inf')
        self.best_solution = None
        self.global_evaluation_count = 0
        self.iteration_count = 0
        self.global_stagnation_count = 0
        self.fitness_history = []
        
        # 花火の初期化
        self.fireworks = []
        # グローバル花火（ID=0）
        self.fireworks.append(Firework(self.dimension, self.bounds, 'global', 0, self.num_local_fireworks))
        # ローカル花火（ID=1,2,3...）
        for i in range(self.num_local_fireworks):
            self.fireworks.append(Firework(self.dimension, self.bounds, 'local', i + 1, self.num_local_fireworks))
        
        # 協調マネージャーの初期化
        self.collaboration_manager = CollaborationManager(self.dimension)
    
    def optimize(self) -> Tuple[np.ndarray, float, List[float]]:
        """最適化を実行"""
        self._reset_state()
        objective_function = self.problem.evaluate
        
        # タイムアウト設定（5分）
        start_time = time.time()
        timeout = 300  # 5分
        
        while self.global_evaluation_count < self.max_evaluations:
            # タイムアウトチェック
            if time.time() - start_time > timeout:
                warnings.warn(f"Optimization timed out after {timeout} seconds")
                break
                
            # デバッグ情報
            if DEBUG and self.iteration_count % 10 == 0:
                print(f"Iteration: {self.iteration_count}, Evaluations: {self.global_evaluation_count}, Best: {self.best_fitness:.6e}")
                
            iteration_start_fitness = self.best_fitness
            all_sparks_data = []

            # スパーク生成と評価
            for fw in self.fireworks:
                if self.global_evaluation_count >= self.max_evaluations: 
                    break
                    
                # 残り評価回数を考慮
                remaining_evals = self.max_evaluations - self.global_evaluation_count
                num_sparks = min(self.sparks_per_firework, remaining_evals)
                
                if num_sparks <= 0:
                    break
                    
                sparks = fw.generate_sparks(num_sparks)
                fitness_values = np.array([objective_function(s) for s in sparks])
                self.global_evaluation_count += len(sparks)
                all_sparks_data.append({'sparks': sparks, 'fitness': fitness_values, 'firework': fw})

            # 全体最良解の更新
            for data in all_sparks_data:
                if data['fitness'].size > 0:
                    best_idx = np.argmin(data['fitness'])
                    if data['fitness'][best_idx] < self.best_fitness:
                        self.best_fitness = data['fitness'][best_idx]
                        self.best_solution = data['sparks'][best_idx]

            # 各花火のパラメータ更新
            for data in all_sparks_data:
                data['firework'].update_parameters(data['sparks'], data['fitness'])

            # 協調処理（エラーハンドリング強化）
            try:
                # 協調処理の実行時間を制限
                collab_start = time.time()
                self.collaboration_manager.execute_collaboration(self.fireworks)
                
                if time.time() - collab_start > 1.0:  # 1秒以上かかったら警告
                    if DEBUG:
                        print(f"Warning: Collaboration took {time.time() - collab_start:.2f} seconds")
                        
            except Exception as e:
                warnings.warn(f"Collaboration failed: {e}")
                
            # 再起動チェック
            for fw in self.fireworks:
                should_restart, reasons = fw.check_restart_conditions(self.fireworks)
                if should_restart: 
                    if DEBUG:
                        print(f"Firework {fw.firework_id} restarting: {reasons}")
                    fw.restart()
            
            # 全体停滞チェック
            if self.best_fitness >= iteration_start_fitness * 0.9999:  # わずかな改善も認める
                self.global_stagnation_count += 1
            else: 
                self.global_stagnation_count = 0
            
            # 全体リブートの頻度を下げる
            if self.global_stagnation_count >= M_GLOBAL_REBOOT * 2:  # より厳しい条件に
                if DEBUG:
                    print(f"Global reboot at iteration {self.iteration_count}")
                self._reset_state()
                continue  # 次のイテレーションへ

            self.fitness_history.append(self.best_fitness)
            self.iteration_count += 1
            
            # 早期終了条件
            if self.best_fitness < 1e-10:  # 十分良い解が見つかった
                if DEBUG:
                    print(f"Early termination: found good solution {self.best_fitness:.6e}")
                break

        return self.best_solution, self.best_fitness, self.fitness_history