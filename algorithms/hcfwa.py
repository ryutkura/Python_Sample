import numpy as np
from typing import List, Tuple, Optional, Dict, Any

# --- 定数 ---
EPSILON_L = 100
M_GLOBAL_REBOOT = 100


class HCFWA:
    """修正版 HCFWA - 高速かつ正確"""
    
    def __init__(self, problem, num_local_fireworks: int = 4,
                 sparks_per_firework: int = None,
                 max_evaluations: int = None):
        
        self.problem = problem
        self.dimension = problem.dimension
        self.bounds = np.array([problem.lower_bounds, problem.upper_bounds]).T
        self.num_local_fireworks = num_local_fireworks
        
        # max_evaluations
        if max_evaluations is None:
            self.max_evaluations = getattr(problem, 'max_evaluations', 10000 * self.dimension)
        else:
            self.max_evaluations = max_evaluations

        # 火花数: CMA-ES標準に近い値
        if sparks_per_firework is None:
            self.sparks_per_firework = 4 + int(3 * np.log(self.dimension))
        else:
            self.sparks_per_firework = sparks_per_firework

        # 状態
        self.all_fireworks = []
        self.best_fitness = float('inf')
        self.best_solution = None
        self.global_evaluation_count = 0
        self.fitness_history = []

    def _initialize_fireworks(self):
        """花火の初期化"""
        self.all_fireworks = []
        
        # グローバル花火（中心から開始）
        self.all_fireworks.append(_Firework(
            self.dimension, self.bounds, 'global', 0, self.num_local_fireworks))
        
        # ローカル花火（ランダム位置から開始）
        for i in range(self.num_local_fireworks):
            self.all_fireworks.append(_Firework(
                self.dimension, self.bounds, 'local', i + 1, self.num_local_fireworks))

    def optimize(self) -> Tuple[np.ndarray, float, List[float]]:
        """最適化実行"""
        # 初期化
        self.best_fitness = float('inf')
        self.best_solution = None
        self.global_evaluation_count = 0
        self.fitness_history = []
        stagnation_count = 0
        
        self._initialize_fireworks()
        
        evaluate = self.problem.evaluate

        while self.global_evaluation_count < self.max_evaluations:
            prev_best = self.best_fitness
            
            for fw in self.all_fireworks:
                remaining = self.max_evaluations - self.global_evaluation_count
                if remaining <= 0:
                    break
                
                n_sparks = min(self.sparks_per_firework, remaining)
                sparks = fw.generate_sparks(n_sparks)
                
                if len(sparks) == 0:
                    continue
                
                # 評価
                fitness_values = np.array([evaluate(s) for s in sparks])
                self.global_evaluation_count += len(sparks)
                
                # グローバルベスト更新
                best_idx = np.argmin(fitness_values)
                if fitness_values[best_idx] < self.best_fitness:
                    self.best_fitness = fitness_values[best_idx]
                    self.best_solution = sparks[best_idx].copy()
                
                # 花火のパラメータ更新
                fw.update(sparks, fitness_values)
            
            # 協調: 最良花火へ向かう
            self._collaborate()
            
            # リスタートチェック
            for fw in self.all_fireworks:
                if fw.should_restart():
                    fw.restart()
            
            # グローバル停滞
            if self.best_fitness < prev_best:
                stagnation_count = 0
            else:
                stagnation_count += 1
            
            if stagnation_count >= M_GLOBAL_REBOOT:
                self._global_restart()
                stagnation_count = 0
            
            self.fitness_history.append(self.best_fitness)
            
            if self.best_fitness < 1e-14:
                break

        return self.best_solution, self.best_fitness, self.fitness_history

    def _collaborate(self):
        """簡易協調"""
        if len(self.all_fireworks) < 2:
            return
        
        best_fw = min(self.all_fireworks, key=lambda f: f.best_fitness)
        
        for fw in self.all_fireworks:
            if fw.firework_id == best_fw.firework_id:
                continue
            if not np.isfinite(best_fw.best_fitness):
                continue
            
            direction = best_fw.mean - fw.mean
            dist = np.linalg.norm(direction)
            if dist > 1e-12:
                step = 0.05 * fw.sigma * direction / dist
                fw.mean = fw.mean + step
                fw.mean = np.clip(fw.mean, self.bounds[:, 0], self.bounds[:, 1])

    def _global_restart(self):
        """グローバルリスタート"""
        backup_sol = self.best_solution.copy() if self.best_solution is not None else None
        backup_fit = self.best_fitness
        
        for fw in self.all_fireworks:
            fw.restart()
        
        self.best_solution = backup_sol
        self.best_fitness = backup_fit


class _Firework:
    """内部花火クラス（CMA-ES風）"""
    
    def __init__(self, dimension: int, bounds: np.ndarray, 
                 firework_type: str, firework_id: int, num_local: int):
        self.dimension = dimension
        self.bounds = bounds
        self.firework_type = firework_type
        self.firework_id = firework_id
        self.num_local = num_local
        
        # CMA-ESパラメータ
        self.mean = self._init_mean()
        self.sigma = self._init_sigma()
        self.C = np.eye(dimension)  # 共分散行列
        self.p_sigma = np.zeros(dimension)  # 進化パス
        self.p_c = np.zeros(dimension)
        
        # 学習率（CMA-ES標準）
        self._init_learning_rates()
        
        # 状態
        self.best_fitness = float('inf')
        self.best_solution = None
        self.stagnation = 0
        self.generation = 0

    def _init_mean(self):
        if self.firework_type == 'global':
            return (self.bounds[:, 0] + self.bounds[:, 1]) / 2
        return np.random.uniform(self.bounds[:, 0], self.bounds[:, 1])

    def _init_sigma(self):
        range_size = np.mean(self.bounds[:, 1] - self.bounds[:, 0])
        if self.firework_type == 'global':
            return range_size / 4
        return range_size / (4 * max(1, self.num_local))

    def _init_learning_rates(self):
        D = self.dimension
        
        # CMA-ES標準パラメータ
        self.mu = max(1, int(self.dimension / 2))
        
        # 重み
        weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        weights = weights / np.sum(weights)
        self.weights = weights
        self.mu_eff = 1.0 / np.sum(weights ** 2)
        
        # 学習率
        self.c_sigma = (self.mu_eff + 2) / (D + self.mu_eff + 5)
        self.d_sigma = 1 + 2 * max(0, np.sqrt((self.mu_eff - 1) / (D + 1)) - 1) + self.c_sigma
        self.c_c = (4 + self.mu_eff / D) / (D + 4 + 2 * self.mu_eff / D)
        self.c_1 = 2 / ((D + 1.3) ** 2 + self.mu_eff)
        self.c_mu = min(1 - self.c_1, 2 * (self.mu_eff - 2 + 1 / self.mu_eff) / ((D + 2) ** 2 + self.mu_eff))
        
        # 期待値
        self.chi_n = np.sqrt(D) * (1 - 1 / (4 * D) + 1 / (21 * D ** 2))

    def generate_sparks(self, n: int) -> np.ndarray:
        """火花生成"""
        if n <= 0:
            return np.zeros((0, self.dimension))
        
        # 共分散の平方根を計算
        try:
            # C = B * D^2 * B^T の形で分解
            D_sq, B = np.linalg.eigh(self.C)
            D_sq = np.maximum(D_sq, 1e-20)
            D_diag = np.sqrt(D_sq)
            
            # サンプリング: x = m + sigma * B * D * z
            z = np.random.randn(n, self.dimension)
            sparks = self.mean + self.sigma * (z @ np.diag(D_diag) @ B.T)
        except:
            sparks = self.mean + self.sigma * np.random.randn(n, self.dimension)
        
        # 境界処理
        sparks = np.clip(sparks, self.bounds[:, 0], self.bounds[:, 1])
        return sparks

    def update(self, sparks: np.ndarray, fitness: np.ndarray):
        """CMA-ES風の更新"""
        if len(sparks) == 0:
            return
        
        self.generation += 1
        
        # ソートしてμ個選択
        sorted_idx = np.argsort(fitness)
        mu = min(len(self.weights), len(sorted_idx))
        
        # 選択された個体
        selected = sparks[sorted_idx[:mu]]
        selected_fitness = fitness[sorted_idx[:mu]]
        
        # 重み（個体数に合わせて調整）
        if mu < len(self.weights):
            weights = self.weights[:mu]
            weights = weights / np.sum(weights)
        else:
            weights = self.weights
        
        # 古い平均
        old_mean = self.mean.copy()
        
        # 新しい平均
        self.mean = np.sum(weights[:, None] * selected, axis=0)
        
        # 境界チェック
        self.mean = np.clip(self.mean, self.bounds[:, 0], self.bounds[:, 1])
        
        # ローカル花火のみ共分散とステップサイズを更新
        if self.firework_type == 'local':
            self._update_evolution_paths(old_mean, weights, selected)
            self._update_covariance(old_mean, weights, selected)
            self._update_sigma()
        
        # ベスト更新
        if selected_fitness[0] < self.best_fitness:
            self.best_fitness = selected_fitness[0]
            self.best_solution = selected[0].copy()
            self.stagnation = 0
        else:
            self.stagnation += 1

    def _update_evolution_paths(self, old_mean, weights, selected):
        """進化パスの更新"""
        # C^(-1/2)の計算
        try:
            D_sq, B = np.linalg.eigh(self.C)
            D_sq = np.maximum(D_sq, 1e-20)
            C_inv_sqrt = B @ np.diag(1.0 / np.sqrt(D_sq)) @ B.T
        except:
            C_inv_sqrt = np.eye(self.dimension)
        
        # p_sigma更新
        y = (self.mean - old_mean) / self.sigma
        self.p_sigma = (1 - self.c_sigma) * self.p_sigma + \
                       np.sqrt(self.c_sigma * (2 - self.c_sigma) * self.mu_eff) * C_inv_sqrt @ y
        
        # p_c更新
        h_sig = np.linalg.norm(self.p_sigma) / np.sqrt(1 - (1 - self.c_sigma) ** (2 * self.generation)) < \
                (1.4 + 2 / (self.dimension + 1)) * self.chi_n
        
        self.p_c = (1 - self.c_c) * self.p_c + \
                   h_sig * np.sqrt(self.c_c * (2 - self.c_c) * self.mu_eff) * y

    def _update_covariance(self, old_mean, weights, selected):
        """共分散行列の更新"""
        # Rank-one更新
        rank_one = np.outer(self.p_c, self.p_c)
        
        # Rank-mu更新
        y_k = (selected - old_mean) / self.sigma
        mu = len(weights)
        rank_mu = sum(weights[i] * np.outer(y_k[i], y_k[i]) for i in range(mu))
        
        # 更新
        self.C = (1 - self.c_1 - self.c_mu) * self.C + \
                 self.c_1 * rank_one + \
                 self.c_mu * rank_mu
        
        # 対称化と正定値化
        self.C = (self.C + self.C.T) / 2
        try:
            D_sq, B = np.linalg.eigh(self.C)
            D_sq = np.maximum(D_sq, 1e-20)
            D_sq = np.minimum(D_sq, 1e20)
            self.C = B @ np.diag(D_sq) @ B.T
        except:
            self.C = np.eye(self.dimension)

    def _update_sigma(self):
        """ステップサイズの更新（CSA）"""
        self.sigma = self.sigma * np.exp(
            (self.c_sigma / self.d_sigma) * (np.linalg.norm(self.p_sigma) / self.chi_n - 1)
        )
        # 制限
        self.sigma = np.clip(self.sigma, 1e-20, 1e10)

    def should_restart(self) -> bool:
        """リスタート条件"""
        if self.stagnation >= EPSILON_L:
            return True
        if self.sigma < 1e-12:
            return True
        return False

    def restart(self):
        """リスタート"""
        self.mean = self._init_mean()
        self.sigma = self._init_sigma()
        self.C = np.eye(self.dimension)
        self.p_sigma = np.zeros(self.dimension)
        self.p_c = np.zeros(self.dimension)
        self.stagnation = 0
        self.best_fitness = float('inf')
        self.best_solution = None
        self.generation = 0

