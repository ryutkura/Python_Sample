import numpy as np
import time
from typing import List, Tuple, Optional, Dict, Any

# --- 定数定義 ---
EPSILON_V = 1e-5
EPSILON_P = 1e-5
EPSILON_L = 100
M_GLOBAL_REBOOT = 150

# SaFWA由来の定数
SAFWA_M_MIN = 2       # 最小火花数
SAFWA_M_MAX = 40      # 最大火花数
SAFWA_A_MIN = 0.01    # 最小振幅係数
SAFWA_A_MAX = 0.5     # 最大振幅係数


class NumericalUtils:
    """数値計算ユーティリティ"""
    MIN_EIGENVALUE = 1e-12
    MAX_CONDITION_NUMBER = 1e12
    EPSILON_ZERO = 1e-12

    @staticmethod
    def ensure_numerical_stability(matrix: np.ndarray) -> np.ndarray:
        if not np.all(np.isfinite(matrix)):
            return np.eye(matrix.shape[0])
        matrix = 0.5 * (matrix + matrix.T)
        try:
            eigenvals, eigenvecs = np.linalg.eigh(matrix)
            eigenvals = np.maximum(eigenvals, NumericalUtils.MIN_EIGENVALUE)
            max_eigenval = np.max(eigenvals)
            if max_eigenval / np.min(eigenvals) > NumericalUtils.MAX_CONDITION_NUMBER:
                eigenvals = np.maximum(eigenvals, max_eigenval / NumericalUtils.MAX_CONDITION_NUMBER)
            return eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
        except np.linalg.LinAlgError:
            return np.eye(matrix.shape[0])

    @staticmethod
    def compute_matrix_inverse_sqrt(matrix: np.ndarray) -> np.ndarray:
        try:
            eigenvals, eigenvecs = np.linalg.eigh(matrix)
            eigenvals = np.maximum(eigenvals, NumericalUtils.MIN_EIGENVALUE)
            return eigenvecs @ np.diag(1.0 / np.sqrt(eigenvals)) @ eigenvecs.T
        except np.linalg.LinAlgError:
            return np.eye(matrix.shape[0])

    @staticmethod
    def compute_expected_chi_norm(dimension: int) -> float:
        if dimension == 1:
            return np.sqrt(2 / np.pi)
        return np.sqrt(dimension) * (1 - 1/(4*dimension) + 1/(32*dimension**2))


class HybridFirework:
    """
    ハイブリッド花火クラス
    - HCFWA: CMA-ES風の共分散適応、階層的協調
    - SaFWA: 自己適応的火花数・振幅、ガウシアン突然変異
    """
    
    def __init__(self, dimension: int, bounds: np.ndarray, firework_type: str,
                 firework_id: int, num_local_fireworks: int):
        self.dimension = dimension
        self.bounds = bounds
        self.firework_type = firework_type
        self.firework_id = firework_id
        self.num_local_fireworks = num_local_fireworks
        self.rng = np.random.default_rng()
        
        # === HCFWA由来: CMA-ES風パラメータ ===
        self.mean = self._initialize_mean()
        self.covariance = np.eye(dimension)
        self.scale = self._initialize_scale()
        self.evolution_path_c = np.zeros(dimension)
        self.evolution_path_sigma = np.zeros(dimension)
        self.learning_rates = self._initialize_learning_rates()
        
        # === SaFWA由来: 自己適応パラメータ ===
        self.amplitude = self._initialize_amplitude()  # 爆発振幅
        self.num_sparks_adaptive = self._initialize_spark_count()  # 適応的火花数
        self.mutation_rate = 0.1  # ガウシアン突然変異率
        self.success_history = []  # 成功履歴（適応用）
        
        # === 共通状態 ===
        self.best_fitness_fw = float('inf')
        self.best_solution_fw = None
        self.stagnation_count = 0
        self.recent_fitness_history = []
        self.evaluation_count_fw = 0
        self.last_improvement_eval = 0
        self.restart_count = 0
        self.last_restart_eval = 0
        
        # スケール制限
        self.min_scale = 1e-8
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

    def _initialize_amplitude(self) -> float:
        """SaFWA: 初期振幅"""
        range_size = np.mean(self.bounds[:, 1] - self.bounds[:, 0])
        return range_size * 0.2  # 探索範囲の20%

    def _initialize_spark_count(self) -> int:
        """SaFWA: 初期火花数"""
        return max(SAFWA_M_MIN, min(SAFWA_M_MAX, 5 + self.dimension))

    def _initialize_learning_rates(self) -> Dict[str, float]:
        D = self.dimension
        if self.firework_type == 'global':
            return {
                'cm': 1.0, 'c_mu': 0.25, 'c1': 0.0, 'cc': 0.0,
                'c_sigma': 0.0, 'd_sigma': 0.0, 'cr': 1.0,
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

    def generate_sparks(self, base_num_sparks: int) -> np.ndarray:
        """
        ハイブリッド火花生成
        - HCFWA: 共分散行列を使った多変量正規分布
        - SaFWA: 適応的火花数 + ガウシアン突然変異
        """
        # SaFWA: 適応的火花数の決定
        num_sparks = self._compute_adaptive_spark_count(base_num_sparks)
        
        if num_sparks <= 0:
            return np.array([]).reshape(0, self.dimension)
        
        sparks = []
        
        # === Part 1: HCFWA式火花（共分散ベース）===
        num_cma_sparks = max(1, num_sparks // 2)
        safe_scale = np.clip(self.scale, self.min_scale, self.max_scale)
        stable_cov = NumericalUtils.ensure_numerical_stability(self.covariance)
        
        try:
            cma_sparks = self.rng.multivariate_normal(
                mean=self.mean,
                cov=(safe_scale**2) * stable_cov,
                size=num_cma_sparks
            )
        except Exception:
            cma_sparks = self.mean + safe_scale * self.rng.standard_normal((num_cma_sparks, self.dimension))
        
        sparks.extend(cma_sparks)
        
        # === Part 2: SaFWA式火花（振幅ベース + 突然変異）===
        num_safwa_sparks = num_sparks - num_cma_sparks
        if num_safwa_sparks > 0:
            safwa_sparks = self._generate_safwa_sparks(num_safwa_sparks)
            sparks.extend(safwa_sparks)
        
        sparks = np.array(sparks)
        
        # === Part 3: ガウシアン突然変異（SaFWA由来）===
        sparks = self._apply_gaussian_mutation(sparks)
        
        # NaN/Inf対策と境界処理
        sparks = np.where(np.isfinite(sparks), sparks, self.mean)
        sparks = np.clip(sparks, self.bounds[:, 0], self.bounds[:, 1])
        
        return sparks

    def _compute_adaptive_spark_count(self, base_num: int) -> int:
        """SaFWA: 成功率に基づく適応的火花数"""
        if len(self.success_history) < 5:
            return base_num
        
        # 最近の成功率を計算
        recent_success_rate = np.mean(self.success_history[-10:])
        
        # 成功率が高い→火花数減少（収束促進）
        # 成功率が低い→火花数増加（探索拡大）
        if recent_success_rate > 0.3:
            adaptive_num = int(base_num * 0.8)
        elif recent_success_rate < 0.1:
            adaptive_num = int(base_num * 1.5)
        else:
            adaptive_num = base_num
        
        return max(SAFWA_M_MIN, min(SAFWA_M_MAX, adaptive_num))

    def _generate_safwa_sparks(self, num_sparks: int) -> List[np.ndarray]:
        """SaFWA式の火花生成（振幅ベース）"""
        sparks = []
        for _ in range(num_sparks):
            # ランダムな次元数を選択
            num_dims = self.rng.integers(1, self.dimension + 1)
            dims = self.rng.choice(self.dimension, num_dims, replace=False)
            
            spark = self.mean.copy()
            for d in dims:
                # 適応的振幅を使用
                displacement = self.amplitude * self.rng.uniform(-1, 1)
                spark[d] += displacement
            
            sparks.append(spark)
        
        return sparks

    def _apply_gaussian_mutation(self, sparks: np.ndarray) -> np.ndarray:
        """SaFWA: ガウシアン突然変異"""
        mutated_sparks = sparks.copy()
        
        for i in range(len(sparks)):
            if self.rng.random() < self.mutation_rate:
                # ガウシアン突然変異を適用
                mutation_scale = self.scale * 0.1
                mutation = self.rng.standard_normal(self.dimension) * mutation_scale
                mutated_sparks[i] += mutation
        
        return mutated_sparks

    def update_parameters(self, sparks: np.ndarray, fitness_values: np.ndarray) -> None:
        """
        ハイブリッドパラメータ更新
        - HCFWA: 共分散行列・進化パスの更新
        - SaFWA: 振幅・火花数の自己適応
        """
        if len(sparks) == 0:
            return
        
        # === HCFWA部分: 重み計算と平均更新 ===
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
            self.learning_rates.get('d_sigma', 0) > 0):
            self._update_step_size(new_mean, mu_eff)
        
        self.mean = new_mean
        
        # === SaFWA部分: 自己適応 ===
        improvement = self._update_best_solution(sparks, fitness_values)
        self._adapt_safwa_parameters(improvement, fitness_values)
        
        self.evaluation_count_fw += len(sparks)

    def _update_step_size(self, new_mean: np.ndarray, mu_eff: float) -> None:
        """HCFWA: ステップサイズ適応（CSA）"""
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
            
            self.scale *= np.exp(np.clip(log_scale_change, -0.3, 0.3))
            self.scale = np.clip(self.scale, self.min_scale, self.max_scale)
        except Exception:
            pass

    def _compute_recombination_weights(self, fitness_values: np.ndarray) -> np.ndarray:
        """重み計算"""
        num_sparks = len(fitness_values)
        weights = np.zeros(num_sparks)
        
        if self.firework_type == 'local':
            mu = max(1, num_sparks // 2)
            sorted_indices = np.argsort(fitness_values)
            for rank, idx in enumerate(sorted_indices[:mu]):
                w = max(0, np.log(mu + 0.5) - np.log(rank + 1))
                weights[idx] = w
        else:
            num_select = max(1, int(0.95 * num_sparks))
            sorted_indices = np.argsort(fitness_values)[:num_select]
            weights[sorted_indices] = 1.0 / num_select
        
        sum_weights = np.sum(weights)
        return weights / sum_weights if sum_weights > 0 else weights

    def _update_best_solution(self, sparks: np.ndarray, fitness_values: np.ndarray) -> bool:
        """最良解更新、改善があったかを返す"""
        if len(fitness_values) == 0:
            return False
        
        best_idx = np.argmin(fitness_values)
        current_best = fitness_values[best_idx]
        improved = False
        
        if current_best < self.best_fitness_fw:
            self.best_fitness_fw = current_best
            self.best_solution_fw = sparks[best_idx].copy()
            self.stagnation_count = 0
            self.last_improvement_eval = self.evaluation_count_fw
            improved = True
        else:
            self.stagnation_count += 1
        
        self.recent_fitness_history.append(current_best)
        if len(self.recent_fitness_history) > 30:
            self.recent_fitness_history.pop(0)
        
        return improved

    def _adapt_safwa_parameters(self, improved: bool, fitness_values: np.ndarray) -> None:
        """SaFWA: 自己適応パラメータの更新"""
        # 成功履歴を更新
        self.success_history.append(1.0 if improved else 0.0)
        if len(self.success_history) > 20:
            self.success_history.pop(0)
        
        # 振幅の適応
        if len(self.success_history) >= 5:
            recent_success_rate = np.mean(self.success_history[-5:])
            
            if recent_success_rate > 0.2:
                # 成功率高い → 振幅縮小（収束）
                self.amplitude *= 0.95
            elif recent_success_rate < 0.05:
                # 成功率低い → 振幅拡大（探索）
                self.amplitude *= 1.1
            
            # 振幅の制限
            range_size = np.mean(self.bounds[:, 1] - self.bounds[:, 0])
            self.amplitude = np.clip(
                self.amplitude,
                range_size * SAFWA_A_MIN,
                range_size * SAFWA_A_MAX
            )
        
        # 突然変異率の適応
        if self.stagnation_count > 20:
            self.mutation_rate = min(0.3, self.mutation_rate * 1.1)
        elif improved:
            self.mutation_rate = max(0.05, self.mutation_rate * 0.95)

    def check_restart_conditions(self, all_fireworks: List['HybridFirework']) -> Tuple[bool, List[str]]:
        """再起動条件チェック"""
        cooldown = 50
        if self.evaluation_count_fw - self.last_restart_eval < cooldown:
            return False, []
        
        restart_reasons = []
        
        if len(self.recent_fitness_history) > 15:
            if np.std(self.recent_fitness_history[-10:]) <= EPSILON_V:
                restart_reasons.append("fitness_converged")
        
        try:
            eigenvals = np.linalg.eigvalsh(self.covariance)
            max_std_dev = np.sqrt(np.max(eigenvals))
            if self.scale * max_std_dev <= EPSILON_P:
                restart_reasons.append("position_converged")
        except Exception:
            pass
        
        max_stagnation = EPSILON_L * (2 if self.firework_type == 'global' else 1)
        if self.stagnation_count >= max_stagnation:
            restart_reasons.append("not_improving")
        
        return len(restart_reasons) > 0, restart_reasons

    def restart(self) -> None:
        """再初期化"""
        self.mean = self._initialize_mean()
        self.covariance = np.eye(self.dimension)
        self.scale = self._initialize_scale()
        self.evolution_path_c.fill(0)
        self.evolution_path_sigma.fill(0)
        
        # SaFWAパラメータもリセット
        self.amplitude = self._initialize_amplitude()
        self.mutation_rate = 0.1
        self.success_history.clear()
        
        self.stagnation_count = 0
        self.recent_fitness_history.clear()
        self.restart_count += 1
        self.last_restart_eval = self.evaluation_count_fw


class AdaptiveCollaborationManager:
    """
    適応的協調マネージャー
    - HCFWA: 階層的協調（グローバル↔ローカル）
    - SaFWA: エリート情報の共有
    """
    
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.collaboration_history = []

    def execute_collaboration(self, fireworks: List[HybridFirework]) -> None:
        """適応的協調処理"""
        if len(fireworks) < 2:
            return
        
        try:
            # エリート花火を特定
            sorted_fws = sorted(fireworks, key=lambda fw: fw.best_fitness_fw)
            elite_fw = sorted_fws[0]
            
            # === 協調戦略1: エリートへの誘導 ===
            for fw in fireworks:
                if fw.firework_id == elite_fw.firework_id:
                    continue
                
                direction = elite_fw.mean - fw.mean
                distance = np.linalg.norm(direction)
                
                if distance > 1e-10:
                    # 適応度差に基づく誘導強度
                    fitness_ratio = fw.best_fitness_fw / (elite_fw.best_fitness_fw + 1e-10)
                    guidance_strength = min(0.1, 0.02 * np.log1p(fitness_ratio))
                    
                    if fw.firework_type == 'local':
                        guidance_strength *= 1.5  # ローカルはより積極的に
                    
                    adjustment = guidance_strength * direction
                    fw.mean += adjustment
                    fw.mean = np.clip(fw.mean, fw.bounds[:, 0], fw.bounds[:, 1])
            
            # === 協調戦略2: 共分散情報の共有（ローカル間）===
            local_fws = [fw for fw in fireworks if fw.firework_type == 'local']
            if len(local_fws) >= 2:
                self._share_covariance_info(local_fws, elite_fw)
            
            # === 協調戦略3: グローバル花火の更新 ===
            global_fws = [fw for fw in fireworks if fw.firework_type == 'global']
            if global_fws and local_fws:
                self._update_global_from_locals(global_fws[0], local_fws)
                
        except Exception:
            pass

    def _share_covariance_info(self, local_fws: List[HybridFirework], elite_fw: HybridFirework) -> None:
        """ローカル花火間での共分散情報共有"""
        # エリートの共分散方向を他の花火に少し伝える
        for fw in local_fws:
            if fw.firework_id == elite_fw.firework_id:
                continue
            
            # 共分散の混合（非常に控えめに）
            mix_ratio = 0.05
            fw.covariance = (1 - mix_ratio) * fw.covariance + mix_ratio * elite_fw.covariance
            fw.covariance = NumericalUtils.ensure_numerical_stability(fw.covariance)

    def _update_global_from_locals(self, global_fw: HybridFirework, local_fws: List[HybridFirework]) -> None:
        """ローカル花火の情報でグローバル花火を更新"""
        # ローカル花火の重心を計算
        weights = []
        positions = []
        
        for fw in local_fws:
            if fw.best_fitness_fw < float('inf'):
                # 適応度が良いほど重みを大きく
                w = 1.0 / (fw.best_fitness_fw + 1e-10)
                weights.append(w)
                positions.append(fw.mean)
        
        if weights:
            weights = np.array(weights)
            weights /= np.sum(weights)
            positions = np.array(positions)
            
            weighted_center = np.sum(weights[:, None] * positions, axis=0)
            
            # グローバル花火を重心方向に少し移動
            direction = weighted_center - global_fw.mean
            global_fw.mean += 0.1 * direction
            global_fw.mean = np.clip(global_fw.mean, global_fw.bounds[:, 0], global_fw.bounds[:, 1])


class SaHCFWA:
    """
    SaHCFWA: Self-adaptive Hierarchical Collaborative Fireworks Algorithm
    
    HCFWAとSaFWAのハイブリッド:
    - HCFWA由来: 階層的構造、CMA-ES風共分散適応、協調戦略
    - SaFWA由来: 自己適応的火花数・振幅、ガウシアン突然変異
    
    新規性:
    1. 適応的な探索・活用バランス
    2. 階層間での情報共有強化
    3. 自己適応メカニズムの統合
    """
    
    def __init__(self, problem, num_local_fireworks: int = 4,
                 sparks_per_firework: int = None,
                 max_evaluations: int = None):
        
        self.problem = problem
        self.dimension = problem.dimension
        self.bounds = np.array([problem.lower_bounds, problem.upper_bounds]).T
        self.num_local_fireworks = num_local_fireworks
        
        # 最大評価回数
        if max_evaluations is None:
            self.max_evaluations = getattr(problem, 'max_evaluations', 10000 * self.dimension)
        else:
            self.max_evaluations = max_evaluations
        
        # 火花数（適応的に変更される基準値）
        if sparks_per_firework is None:
            self.base_sparks_per_firework = max(10, 4 + int(3 * np.log(self.dimension)))
        else:
            self.base_sparks_per_firework = sparks_per_firework
        
        self._initialize_state()

    def _initialize_state(self):
        """初期状態設定"""
        self.best_fitness = float('inf')
        self.best_solution = None
        self.global_evaluation_count = 0
        self.iteration_count = 0
        self.global_stagnation_count = 0
        self.fitness_history = []
        
        # ハイブリッド花火の初期化
        self.fireworks = []
        self.fireworks.append(HybridFirework(
            self.dimension, self.bounds, 'global', 0, self.num_local_fireworks))
        for i in range(self.num_local_fireworks):
            self.fireworks.append(HybridFirework(
                self.dimension, self.bounds, 'local', i + 1, self.num_local_fireworks))
        
        # 適応的協調マネージャー
        self.collaboration_manager = AdaptiveCollaborationManager(self.dimension)
        
        # 初期評価
        self._perform_initial_evaluation()

    def _perform_initial_evaluation(self):
        """初期評価"""
        num_initial = max(20, self.dimension * 2)
        rng = self.fireworks[0].rng
        
        initial_solutions = [rng.uniform(self.bounds[:, 0], self.bounds[:, 1]) 
                           for _ in range(num_initial)]
        initial_fitnesses = [self.problem.evaluate(sol) for sol in initial_solutions]
        self.global_evaluation_count += num_initial
        
        best_idx = np.argmin(initial_fitnesses)
        self.best_fitness = initial_fitnesses[best_idx]
        self.best_solution = initial_solutions[best_idx].copy()
        
        # 花火に初期最良値を設定
        for fw in self.fireworks:
            fw.best_fitness_fw = self.best_fitness
            fw.best_solution_fw = self.best_solution.copy()

    def _global_reboot(self):
        """グローバルリブート"""
        best_sol_backup = self.best_solution.copy() if self.best_solution is not None else None
        best_fit_backup = self.best_fitness
        
        for fw in self.fireworks:
            fw.restart()
            fw.best_fitness_fw = best_fit_backup
            fw.best_solution_fw = best_sol_backup.copy() if best_sol_backup is not None else None
        
        self.best_fitness = best_fit_backup
        self.best_solution = best_sol_backup
        self.global_stagnation_count = 0

    def optimize(self) -> Tuple[np.ndarray, float, List[float]]:
        """最適化実行"""
        start_time = time.time()
        timeout = 600  # 10分
        
        while self.global_evaluation_count < self.max_evaluations:
            if time.time() - start_time > timeout:
                break
            
            iteration_improved = False
            
            # 各花火で火花生成・評価
            for fw in self.fireworks:
                remaining = self.max_evaluations - self.global_evaluation_count
                if remaining <= 0:
                    break
                
                # 適応的火花数
                num_sparks = min(self.base_sparks_per_firework, remaining)
                sparks = fw.generate_sparks(num_sparks)
                
                if len(sparks) == 0:
                    continue
                
                # 評価
                fitness_values = []
                for s in sparks:
                    try:
                        f = float(self.problem.evaluate(s))
                        fitness_values.append(f if np.isfinite(f) else float('inf'))
                    except Exception:
                        fitness_values.append(float('inf'))
                
                fitness_values = np.array(fitness_values)
                self.global_evaluation_count += len(sparks)
                
                # グローバルベスト更新
                best_idx = np.argmin(fitness_values)
                if fitness_values[best_idx] < self.best_fitness:
                    self.best_fitness = fitness_values[best_idx]
                    self.best_solution = sparks[best_idx].copy()
                    iteration_improved = True
                
                # パラメータ更新
                fw.update_parameters(sparks, fitness_values)
            
            # 協調処理
            if self.iteration_count % 2 == 0:
                self.collaboration_manager.execute_collaboration(self.fireworks)
            
            # リスタートチェック
            if self.iteration_count % 5 == 0:
                for fw in self.fireworks:
                    should_restart, _ = fw.check_restart_conditions(self.fireworks)
                    if should_restart:
                        fw.restart()
            
            # 停滞管理
            if iteration_improved:
                self.global_stagnation_count = 0
            else:
                self.global_stagnation_count += 1
            
            if self.global_stagnation_count >= M_GLOBAL_REBOOT:
                self._global_reboot()
            
            self.fitness_history.append(self.best_fitness)
            self.iteration_count += 1
            
            # 早期終了
            if self.best_fitness < 1e-14:
                break
        
        return self.best_solution, self.best_fitness, self.fitness_history


# エイリアス（run_experiment.py互換）
HCFWA = SaHCFWA
