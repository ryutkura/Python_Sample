import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
from problems import BaseProblem

@dataclass
class Firework:
    """花火（個体）を表すデータクラス"""
    position: np.ndarray
    fitness: float
    amplitude: float = 0.0
    num_sparks: int = 0
    strategy_index: int = -1  # 生成された戦略のID

class SaFWA:
    """
    Self-Adaptive Fireworks Algorithm (SaFWA) のPython実装。
    Java版のロジック（適応的戦略選択、学習期間など）を移植しています。
    """
    def __init__(self, 
                 problem: BaseProblem, 
                 population_size: int = 5, 
                 max_evaluations: int = 300000,
                 learning_period: int = 10):
        
        self.problem = problem
        self.dim = problem.dimension
        self.bounds = np.array([problem.lower_bounds, problem.upper_bounds]).T
        
        # パラメータ
        self.population_size = population_size
        self.max_evaluations = max_evaluations
        self.total_sparks = 50  # m
        self.learning_period = learning_period  # LP
        
        # 爆発パラメータ
        self.max_amplitude = 40.0 * (self.problem.upper_bounds[0] - self.problem.lower_bounds[0]) / 100.0
        self.min_amplitude = 1e-5
        
        # 自己適応メカニズム用パラメータ
        self.strategy_num = 5  # 5つの戦略を使用
        self.strategy_probs = np.ones(self.strategy_num) / self.strategy_num
        self.strategy_success_count = np.zeros(self.strategy_num)
        self.strategy_failure_count = np.zeros(self.strategy_num)
        
        self.rng = np.random.default_rng()
        self.evaluations = 0

    def optimize(self) -> Tuple[np.ndarray, float, List[float]]:
        # 初期化
        fireworks = []
        best_solution = None
        best_fitness = float('inf')
        fitness_history = []
        
        # 初期個体の生成
        for _ in range(self.population_size):
            pos = self.rng.uniform(self.problem.lower_bounds, self.problem.upper_bounds)
            fit = self.problem.evaluate(pos)
            self.evaluations += 1
            fireworks.append(Firework(position=pos, fitness=fit))
            
            if fit < best_fitness:
                best_fitness = fit
                best_solution = pos.copy()

        current_generation = 0

        # メインループ
        while self.evaluations < self.max_evaluations:
            fitness_history.append(best_fitness)
            
            # 学習期間ごとの確率更新
            if current_generation > 0 and current_generation % self.learning_period == 0:
                self._update_probabilities()

            # 振幅と火花数の基本計算（EFWAベース）
            self._calculate_base_amplitude_and_sparks(fireworks)
            
            all_candidates = fireworks[:]  # 親も次世代の候補に含める
            
            # 各花火に対して戦略を選択・適用して火花を生成
            for i, fw in enumerate(fireworks):
                # 戦略の選択（ルーレット選択）
                strategy_idx = self._select_strategy()
                
                # 選択された戦略で火花を生成
                sparks = self._apply_strategy(fw, strategy_idx, best_solution)
                
                # 評価と成功判定
                for spark_pos in sparks:
                    if self.evaluations >= self.max_evaluations:
                        break
                        
                    # 境界制約処理
                    spark_pos = np.clip(spark_pos, self.problem.lower_bounds, self.problem.upper_bounds)
                    
                    spark_fit = self.problem.evaluate(spark_pos)
                    self.evaluations += 1
                    
                    # 成功/失敗のカウント
                    if spark_fit < fw.fitness:
                        self.strategy_success_count[strategy_idx] += 1
                    else:
                        self.strategy_failure_count[strategy_idx] += 1
                        
                    # ベスト更新
                    if spark_fit < best_fitness:
                        best_fitness = spark_fit
                        best_solution = spark_pos.copy()
                        
                    all_candidates.append(Firework(position=spark_pos, fitness=spark_fit, strategy_index=strategy_idx))

            # 次世代選択（距離ベース）
            fireworks = self._select_next_generation(all_candidates)
            current_generation += 1

        return best_solution, best_fitness, fitness_history

    def _select_strategy(self) -> int:
        """確率に基づいて戦略を選択する"""
        r = self.rng.random()
        cumulative_prob = 0.0
        for i, prob in enumerate(self.strategy_probs):
            cumulative_prob += prob
            if r <= cumulative_prob:
                return i
        return self.strategy_num - 1

    def _update_probabilities(self):
        """学習期間終了時に戦略選択確率を更新する"""
        epsilon = 1e-10
        total_success = np.sum(self.strategy_success_count)
        total_failure = np.sum(self.strategy_failure_count)
        
        # 全く成功も失敗もしていない（稀なケース）はスキップ
        if total_success + total_failure == 0:
            return

        # 各戦略のスコア計算 (成功率のようなもの)
        scores = np.zeros(self.strategy_num)
        for i in range(self.strategy_num):
            s = self.strategy_success_count[i]
            f = self.strategy_failure_count[i]
            if s + f > 0:
                # 成功数が多いほど、失敗数が少ないほど高スコアになる簡易式
                scores[i] = (s + epsilon) / (s + f + epsilon)
            else:
                scores[i] = epsilon

        # 確率の更新（スコアに比例させる）
        total_score = np.sum(scores)
        if total_score > 0:
            self.strategy_probs = scores / total_score
            # 最低確率を保証（探索の維持）
            min_prob = 0.05
            self.strategy_probs = self.strategy_probs * (1.0 - self.strategy_num * min_prob) + min_prob
        
        # カウンタのリセット
        self.strategy_success_count = np.zeros(self.strategy_num)
        self.strategy_failure_count = np.zeros(self.strategy_num)

    def _apply_strategy(self, fw: Firework, strategy_idx: int, best_pos: np.ndarray) -> List[np.ndarray]:
        """
        選択された戦略に基づいて火花を生成する。
        Java版の意図を汲み取り、振幅や生成方法を変える5つの戦略を定義。
        """
        sparks = []
        num_sparks = max(1, fw.num_sparks)
        
        # 戦略定義
        # 0: 標準爆発 (Standard)
        # 1: 振幅増大 (Exploration)
        # 2: 振幅縮小 (Exploitation)
        # 3: ガウス変異 (Gaussian)
        # 4: ランダム探索 (Random)
        
        amplitude = fw.amplitude
        if strategy_idx == 1:
            amplitude *= 1.5  # 広範囲探索
        elif strategy_idx == 2:
            amplitude *= 0.5  # 局所探索

        if strategy_idx in [0, 1, 2]:
            # 通常の爆発
            for _ in range(num_sparks):
                displacement = amplitude * self.rng.uniform(-1, 1, size=self.dim)
                new_pos = fw.position + displacement
                sparks.append(new_pos)
                
        elif strategy_idx == 3:
            # ガウス変異（現在位置とベスト位置の間を探索）
            for _ in range(num_sparks):
                gaussian_coeff = self.rng.standard_normal(size=self.dim)
                new_pos = fw.position + (best_pos - fw.position) * gaussian_coeff
                sparks.append(new_pos)
                
        elif strategy_idx == 4:
            # ランダム探索（選ばれた次元のみ大きく動かす）
            for _ in range(num_sparks):
                new_pos = fw.position.copy()
                # ランダムに選んだ次元を再初期化に近い形で動かす
                mask = self.rng.random(size=self.dim) < 0.2  # 20%の次元
                if np.any(mask):
                    new_pos[mask] = self.rng.uniform(
                        self.problem.lower_bounds[mask], 
                        self.problem.upper_bounds[mask]
                    )
                sparks.append(new_pos)
                
        return sparks

    def _calculate_base_amplitude_and_sparks(self, fireworks: List[Firework]):
        """EFWA準拠の基本振幅と火花数の計算"""
        fits = np.array([fw.fitness for fw in fireworks])
        min_fit = np.min(fits)
        max_fit = np.max(fits)
        sum_fit_diff = np.sum(max_fit - fits)
        epsilon = 1e-10

        for fw in fireworks:
            # 火花数（良い個体ほど多い）
            if sum_fit_diff < epsilon:
                fw.num_sparks = self.total_sparks // self.population_size
            else:
                ratio = (max_fit - fw.fitness) / (sum_fit_diff + epsilon)
                fw.num_sparks = int(self.total_sparks * ratio)
            
            # 制約（最低数と最大数）
            fw.num_sparks = int(np.clip(fw.num_sparks, 2, self.total_sparks * 0.8))

            # 振幅（良い個体ほど小さい＝集中探索）
            # 最良個体は非常に小さく、悪い個体は大きく
            fw.amplitude = self.max_amplitude * (fw.fitness - min_fit + epsilon) / (np.sum(fits - min_fit) + epsilon)
            fw.amplitude = max(fw.amplitude, self.min_amplitude)

    def _select_next_generation(self, candidates: List[Firework]) -> List[Firework]:
        """
        次世代の選択。
        1. エリート保存（最良個体は無条件で残す）
        2. 残りは距離ベース（混雑していない場所の個体を優先）で選択
        """
        # フィットネスでソート
        candidates.sort(key=lambda x: x.fitness)
        
        # エリート選択
        next_gen = [candidates[0]]
        candidates.pop(0)
        
        if len(next_gen) >= self.population_size:
            return next_gen

        # 距離ベース選択（計算コスト削減のため、簡易的なユークリッド距離の総和を使用）
        # 残りの候補から N-1 個を選ぶ
        
        # 位置情報の配列化
        positions = np.array([fw.position for fw in candidates])
        n_candidates = len(candidates)
        
        # 全候補間の距離を計算するのは重いので、ダウンサンプリングするか、
        # シンプルに「他の全個体との距離の和」が大きい（孤立している）ものを選ぶ
        
        distances = np.zeros(n_candidates)
        for i in range(n_candidates):
            # 自分以外の全個体との距離の和
            dist_sum = np.sum(np.linalg.norm(positions - positions[i], axis=1))
            distances[i] = dist_sum
            
        # 距離の和を確率に変換（距離が大きいほど選ばれやすく）
        total_dist = np.sum(distances)
        if total_dist == 0:
            probs = np.ones(n_candidates) / n_candidates
        else:
            probs = distances / total_dist
            
        # 確率に基づいて選択
        num_needed = self.population_size - 1
        if num_needed > 0 and n_candidates > 0:
            selected_indices = self.rng.choice(n_candidates, size=num_needed, replace=False, p=probs)
            for idx in selected_indices:
                next_gen.append(candidates[idx])
        
        # まだ足りない場合（候補が少なすぎる場合）はランダム生成で埋める
        while len(next_gen) < self.population_size:
            pos = self.rng.uniform(self.problem.lower_bounds, self.problem.upper_bounds)
            fit = self.problem.evaluate(pos)
            self.evaluations += 1
            next_gen.append(Firework(pos, fit))

        return next_gen