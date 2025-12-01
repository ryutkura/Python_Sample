import numpy as np
import time
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum
from dataclasses import dataclass, field

# --- 定数定義 ---
EPSILON_V = 1e-5
EPSILON_P = 1e-5
EPSILON_L = 100
M_GLOBAL_REBOOT = 150

# SaFWA定数（論文より）
SAFWA_M = 90          # 総火花数パラメータ m
SAFWA_A_HAT = 2.0     # 最大振幅 Â
SAFWA_A = 0.04        # 下限係数 a
SAFWA_B = 0.8         # 上限係数 b
SAFWA_M_HAT = 8       # ガウシアン突然変異火花数 m̂
SAFWA_LP = 10         # 学習周期 LP
SAFWA_F = 0.5         # DE変異スケール因子 F


class CSGS(Enum):
    """4つの候補解生成戦略（論文Eq.6-9）"""
    DE_RAND_1 = 1           # CSGS1
    DE_RAND_2 = 2           # CSGS2
    DE_BEST_2 = 3           # CSGS3
    DE_CURRENT_TO_BEST_2 = 4  # CSGS4


@dataclass
class StrategyPool:
    """自己適応戦略プール（論文Section III-A）"""
    strategies: List[CSGS] = field(default_factory=lambda: list(CSGS))
    probabilities: np.ndarray = field(default=None)
    
    # 成功/失敗カウント
    strat_flag_success: Dict[CSGS, int] = field(default_factory=dict)
    strat_flag_failure: Dict[CSGS, int] = field(default_factory=dict)
    total_flag_success: Dict[CSGS, List[int]] = field(default_factory=dict)
    total_flag_failure: Dict[CSGS, List[int]] = field(default_factory=dict)
    
    current_iteration: int = 0
    last_update_iteration: int = 0
    
    def __post_init__(self):
        n = len(self.strategies)
        if self.probabilities is None:
            self.probabilities = np.ones(n) / n  # 均等初期化
        
        for s in self.strategies:
            self.strat_flag_success[s] = 0
            self.strat_flag_failure[s] = 0
            self.total_flag_success[s] = []
            self.total_flag_failure[s] = []
    
    def select_strategy(self, rng: np.random.Generator) -> CSGS:
        """ルーレット選択で戦略を選ぶ"""
        idx = rng.choice(len(self.strategies), p=self.probabilities)
        return self.strategies[idx]
    
    def record_result(self, strategy: CSGS, success: bool):
        """成功/失敗を記録"""
        if success:
            self.strat_flag_success[strategy] += 1
        else:
            self.strat_flag_failure[strategy] += 1
    
    def end_iteration(self):
        """イテレーション終了時の処理"""
        # straFlagS/Fをtotalに蓄積
        for s in self.strategies:
            self.total_flag_success[s].append(self.strat_flag_success[s])
            self.total_flag_failure[s].append(self.strat_flag_failure[s])
            # リセット
            self.strat_flag_success[s] = 0
            self.strat_flag_failure[s] = 0
        
        self.current_iteration += 1
        
        # LP周期ごとに確率更新
        if (self.current_iteration - self.last_update_iteration) >= SAFWA_LP:
            self._update_probabilities()
            self.last_update_iteration = self.current_iteration
    
    def _update_probabilities(self):
        """確率更新（論文Eq.4-5）"""
        epsilon = 1e-10
        new_probs = []
        
        for s in self.strategies:
            # 直近LP期間の合計
            recent_success = sum(self.total_flag_success[s][-SAFWA_LP:])
            recent_failure = sum(self.total_flag_failure[s][-SAFWA_LP:])
            
            if recent_success > 0:
                # Eq.(4) 通常ケース
                p_prime = recent_success / (recent_success + recent_failure + epsilon)
            else:
                # Eq.(4) 成功ゼロのケース
                p_prime = epsilon / (recent_failure + epsilon)
            
            new_probs.append(p_prime)
        
        # Eq.(5) 正規化
        total = sum(new_probs)
        self.probabilities = np.array(new_probs) / total if total > 0 else np.ones(len(self.strategies)) / len(self.strategies)
        
        # 履歴クリア
        for s in self.strategies:
            self.total_flag_success[s] = []
            self.total_flag_failure[s] = []


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
    - SaFWA: 自己適応的CSGS選択、標準FWA火花生成
    """
    
    def __init__(self, dimension: int, bounds: np.ndarray, firework_type: str,
                 firework_id: int, num_local_fireworks: int,
                 strategy_pool: StrategyPool):
        self.dimension = dimension
        self.bounds = bounds
        self.firework_type = firework_type
        self.firework_id = firework_id
        self.num_local_fireworks = num_local_fireworks
        self.strategy_pool = strategy_pool
        self.rng = np.random.default_rng()
        
        # === HCFWA由来: CMA-ES風パラメータ ===
        self.mean = self._initialize_mean()
        self.covariance = np.eye(dimension)
        self.scale = self._initialize_scale()
        self.evolution_path_c = np.zeros(dimension)
        self.evolution_path_sigma = np.zeros(dimension)
        self.learning_rates = self._initialize_learning_rates()
        
        # === 共通状態 ===
        self.fitness = float('inf')  # 現在の適応度
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

    def compute_spark_count(self, all_fireworks: List['HybridFirework']) -> int:
        """
        火花数計算（論文Eq.1）
        s_i = m * (y_max - f(x_i) + ξ) / Σ(y_max - f(x_j) + ξ)
        """
        xi = 1e-10  # ゼロ除算回避
        
        fitnesses = [fw.fitness for fw in all_fireworks]
        y_max = max(fitnesses)
        
        numerator = y_max - self.fitness + xi
        denominator = sum(y_max - f + xi for f in fitnesses)
        
        s_i = int(SAFWA_M * numerator / denominator)
        
        # 上下限クリップ（論文のa, bパラメータ）
        s_min = int(SAFWA_A * SAFWA_M)
        s_max = int(SAFWA_B * SAFWA_M)
        
        return max(s_min, min(s_max, s_i))

    def compute_amplitude(self, all_fireworks: List['HybridFirework']) -> float:
        """
        爆発振幅計算（論文Eq.2）
        A_i = Â * (f(x_i) - y_min + ξ) / Σ(f(x_j) - y_min + ξ)
        """
        xi = 1e-10
        
        fitnesses = [fw.fitness for fw in all_fireworks]
        y_min = min(fitnesses)
        
        numerator = self.fitness - y_min + xi
        denominator = sum(f - y_min + xi for f in fitnesses)
        
        range_size = np.mean(self.bounds[:, 1] - self.bounds[:, 0])
        A_i = SAFWA_A_HAT * range_size * numerator / denominator
        
        return A_i

    def generate_sparks(self, num_sparks: int, amplitude: float,
                       all_fireworks: List['HybridFirework']) -> Tuple[np.ndarray, List[CSGS]]:
        """
        ハイブリッド火花生成
        - 標準FWA爆発火花
        - CSGS（DE変異）火花
        - ガウシアン突然変異火花
        """
        if num_sparks <= 0:
            return np.array([]).reshape(0, self.dimension), []
        
        sparks = []
        strategies_used = []
        
        # === Part 1: 標準FWA爆発火花 ===
        num_explosion = max(1, num_sparks // 3)
        for _ in range(num_explosion):
            spark = self.mean.copy()
            # ランダムな次元数を選択
            num_dims = self.rng.integers(1, self.dimension + 1)
            dims = self.rng.choice(self.dimension, num_dims, replace=False)
            
            for d in dims:
                displacement = amplitude * self.rng.uniform(-1, 1)
                spark[d] += displacement
            
            sparks.append(spark)
            strategies_used.append(None)  # 標準FWA火花
        
        # === Part 2: CSGS火花（DE変異、論文のメイン貢献）===
        num_csgs = max(1, num_sparks // 3)
        population = np.array([fw.mean for fw in all_fireworks])
        best_idx = np.argmin([fw.fitness for fw in all_fireworks])
        
        for _ in range(num_csgs):
            strategy = self.strategy_pool.select_strategy(self.rng)
            strategies_used.append(strategy)
            
            spark = self._apply_csgs(strategy, population, best_idx)
            sparks.append(spark)
        
        # === Part 3: CMA-ES風火花（HCFWA由来）===
        num_cma = num_sparks - num_explosion - num_csgs
        if num_cma > 0:
            safe_scale = np.clip(self.scale, self.min_scale, self.max_scale)
            stable_cov = NumericalUtils.ensure_numerical_stability(self.covariance)
            
            try:
                cma_sparks = self.rng.multivariate_normal(
                    mean=self.mean,
                    cov=(safe_scale**2) * stable_cov,
                    size=num_cma
                )
            except Exception:
                cma_sparks = self.mean + safe_scale * self.rng.standard_normal((num_cma, self.dimension))
            
            for s in cma_sparks:
                sparks.append(s)
                strategies_used.append(None)
        
        sparks = np.array(sparks)
        
        # NaN/Inf対策と境界処理
        sparks = np.where(np.isfinite(sparks), sparks, self.mean)
        sparks = np.clip(sparks, self.bounds[:, 0], self.bounds[:, 1])
        
        return sparks, strategies_used

    def _apply_csgs(self, strategy: CSGS, population: np.ndarray, best_idx: int) -> np.ndarray:
        """
        CSGS適用（論文Eq.6-9）
        """
        n = len(population)
        F = SAFWA_F
        
        # 自身以外からランダムに個体を選択
        available = [i for i in range(n) if i != self.firework_id]
        
        if strategy == CSGS.DE_RAND_1:
            # Eq.(6): V = X_r1 + F*(X_r2 - X_r3)
            if len(available) < 3:
                return self.mean + self.rng.standard_normal(self.dimension) * 0.1
            r1, r2, r3 = self.rng.choice(available, 3, replace=False)
            v = population[r1] + F * (population[r2] - population[r3])
            
        elif strategy == CSGS.DE_RAND_2:
            # Eq.(7): V = X_r1 + F*(X_r2 - X_r3) + F*(X_r4 - X_r5)
            if len(available) < 5:
                return self.mean + self.rng.standard_normal(self.dimension) * 0.1
            r1, r2, r3, r4, r5 = self.rng.choice(available, 5, replace=False)
            v = population[r1] + F * (population[r2] - population[r3]) + F * (population[r4] - population[r5])
            
        elif strategy == CSGS.DE_BEST_2:
            # Eq.(8): V = X_best + F*(X_r1 - X_r2) + F*(X_r3 - X_r4)
            if len(available) < 4:
                return self.mean + self.rng.standard_normal(self.dimension) * 0.1
            r1, r2, r3, r4 = self.rng.choice(available, 4, replace=False)
            v = population[best_idx] + F * (population[r1] - population[r2]) + F * (population[r3] - population[r4])
            
        elif strategy == CSGS.DE_CURRENT_TO_BEST_2:
            # Eq.(9): V = X_i + F*(X_best - X_i) + F*(X_r1 - X_r2) + F*(X_r3 - X_r4)
            if len(available) < 4:
                return self.mean + self.rng.standard_normal(self.dimension) * 0.1
            r1, r2, r3, r4 = self.rng.choice(available, 4, replace=False)
            v = self.mean + F * (population[best_idx] - self.mean) + F * (population[r1] - population[r2]) + F * (population[r3] - population[r4])
        else:
            v = self.mean.copy()
        
        return v

    def generate_gaussian_sparks(self) -> np.ndarray:
        """
        ガウシアン突然変異火花生成（論文Algorithm 1, line 8）
        """
        sparks = []
        for _ in range(SAFWA_M_HAT):
            spark = self.mean.copy()
            # ランダムな次元を選択
            num_dims = self.rng.integers(1, self.dimension + 1)
            dims = self.rng.choice(self.dimension, num_dims, replace=False)
            
            # ガウシアン係数
            g = self.rng.standard_normal()
            
            for d in dims:
                spark[d] *= (1 + g)
            
            sparks.append(spark)
        
        sparks = np.array(sparks)
        sparks = np.clip(sparks, self.bounds[:, 0], self.bounds[:, 1])
        
        return sparks

    def update_parameters(self, sparks: np.ndarray, fitness_values: np.ndarray,
                         strategies_used: List[Optional[CSGS]]) -> None:
        """
        パラメータ更新
        - HCFWA: 共分散行列・進化パス
        - SaFWA: 戦略成功/失敗の記録
        """
        if len(sparks) == 0:
            return
        
        # === SaFWA部分: 戦略の成功/失敗を記録 ===
        for i, strategy in enumerate(strategies_used):
            if strategy is not None:
                # 新しい火花が現在より良ければ成功
                success = fitness_values[i] < self.fitness
                self.strategy_pool.record_result(strategy, success)
        
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
        
        # 最良解更新
        self._update_best_solution(sparks, fitness_values)
        
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
        """最良解更新"""
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
        
        # 現在の適応度も更新（火花数・振幅計算用）
        self.fitness = min(self.fitness, current_best)
        
        self.recent_fitness_history.append(current_best)
        if len(self.recent_fitness_history) > 30:
            self.recent_fitness_history.pop(0)
        
        return improved

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
        self.fitness = float('inf')
        
        self.stagnation_count = 0
        self.recent_fitness_history.clear()
        self.restart_count += 1
        self.last_restart_eval = self.evaluation_count_fw


class AdaptiveCollaborationManager:
    """適応的協調マネージャー"""
    
    def __init__(self, dimension: int):
        self.dimension = dimension

    def execute_collaboration(self, fireworks: List[HybridFirework]) -> None:
        """適応的協調処理"""
        if len(fireworks) < 2:
            return
        
        try:
            sorted_fws = sorted(fireworks, key=lambda fw: fw.best_fitness_fw)
            elite_fw = sorted_fws[0]
            
            for fw in fireworks:
                if fw.firework_id == elite_fw.firework_id:
                    continue
                
                direction = elite_fw.mean - fw.mean
                distance = np.linalg.norm(direction)
                
                if distance > 1e-10:
                    fitness_ratio = fw.best_fitness_fw / (elite_fw.best_fitness_fw + 1e-10)
                    guidance_strength = min(0.1, 0.02 * np.log1p(fitness_ratio))
                    
                    if fw.firework_type == 'local':
                        guidance_strength *= 1.5
                    
                    adjustment = guidance_strength * direction
                    fw.mean += adjustment
                    fw.mean = np.clip(fw.mean, fw.bounds[:, 0], fw.bounds[:, 1])
                    
        except Exception:
            pass


class SaHCFWA:
    """
    SaHCFWA: Self-adaptive Hierarchical Collaborative Fireworks Algorithm
    
    HCFWAとSaFWAの真のハイブリッド:
    - HCFWA: 階層的構造、CMA-ES風共分散適応
    - SaFWA: 自己適応的CSGS選択（DE変異戦略）、標準FWA火花生成
    """
    
    def __init__(self, problem, num_local_fireworks: int = 4,
                 sparks_per_firework: int = None,
                 max_evaluations: int = None):
        
        self.problem = problem
        self.dimension = problem.dimension
        self.bounds = np.array([problem.lower_bounds, problem.upper_bounds]).T
        self.num_local_fireworks = num_local_fireworks
        
        if max_evaluations is None:
            self.max_evaluations = getattr(problem, 'max_evaluations', 10000 * self.dimension)
        else:
            self.max_evaluations = max_evaluations
        
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
        
        # 共有戦略プール（SaFWAの核心）
        self.strategy_pool = StrategyPool()
        
        # ハイブリッド花火の初期化
        self.fireworks = []
        self.fireworks.append(HybridFirework(
            self.dimension, self.bounds, 'global', 0,
            self.num_local_fireworks, self.strategy_pool))
        for i in range(self.num_local_fireworks):
            self.fireworks.append(HybridFirework(
                self.dimension, self.bounds, 'local', i + 1,
                self.num_local_fireworks, self.strategy_pool))
        
        self.collaboration_manager = AdaptiveCollaborationManager(self.dimension)
        
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
        
        for fw in self.fireworks:
            fw.best_fitness_fw = self.best_fitness
            fw.best_solution_fw = self.best_solution.copy()
            fw.fitness = self.best_fitness

    def _global_reboot(self):
        """グローバルリブート"""
        best_sol_backup = self.best_solution.copy() if self.best_solution is not None else None
        best_fit_backup = self.best_fitness
        
        for fw in self.fireworks:
            fw.restart()
            fw.best_fitness_fw = best_fit_backup
            fw.best_solution_fw = best_sol_backup.copy() if best_sol_backup is not None else None
            fw.fitness = best_fit_backup
        
        self.best_fitness = best_fit_backup
        self.best_solution = best_sol_backup
        self.global_stagnation_count = 0

    def optimize(self) -> Tuple[np.ndarray, float, List[float]]:
        """最適化実行"""
        start_time = time.time()
        timeout = 600
        
        while self.global_evaluation_count < self.max_evaluations:
            if time.time() - start_time > timeout:
                break
            
            iteration_improved = False
            
            for fw in self.fireworks:
                remaining = self.max_evaluations - self.global_evaluation_count
                if remaining <= 0:
                    break
                
                # === SaFWA: 適応的火花数・振幅計算 ===
                num_sparks = fw.compute_spark_count(self.fireworks)
                amplitude = fw.compute_amplitude(self.fireworks)
                
                num_sparks = min(num_sparks, remaining)
                
                # === ハイブリッド火花生成 ===
                sparks, strategies_used = fw.generate_sparks(num_sparks, amplitude, self.fireworks)
                
                # === ガウシアン突然変異火花 ===
                if remaining > len(sparks):
                    gaussian_sparks = fw.generate_gaussian_sparks()
                    gaussian_sparks = gaussian_sparks[:min(len(gaussian_sparks), remaining - len(sparks))]
                    if len(gaussian_sparks) > 0:
                        sparks = np.vstack([sparks, gaussian_sparks]) if len(sparks) > 0 else gaussian_sparks
                        strategies_used.extend([None] * len(gaussian_sparks))
                
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
                fw.update_parameters(sparks, fitness_values, strategies_used)
            
            # === SaFWA: イテレーション終了時の戦略確率更新 ===
            self.strategy_pool.end_iteration()
            
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
            
            if self.best_fitness < 1e-14:
                break
        
        return self.best_solution, self.best_fitness, self.fitness_history


# エイリアス
HCFWA = SaHCFWA
