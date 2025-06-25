import numpy as np
from dataclasses import dataclass, field
from problems import BaseProblem

@dataclass
class Firework:
    """花火（個体）を表すデータクラス。"""
    position: np.ndarray
    fitness: float
    amplitude: float = 0.0
    num_sparks: int = 0

class EFWA:
    """
    Enhanced Fireworks Algorithm (EFWA) の実装。
   
    """
    def __init__(self, problem: BaseProblem, population_size: int = 5, max_evaluations: int = 300000):
        self.problem = problem
        self.population_size = population_size
        self.max_evaluations = max_evaluations
        
        # Java版のデフォルトパラメータ
        self.explosion_amplitude = 40.0
        self.max_explosion_sparks = 50
        self.bounding_coeff_a = 0.04
        self.bounding_coeff_b = 0.8
        self.gaussian_sparks = 5
        self.use_non_linear_decrease = True
        
        # 最小爆発振幅のパラメータを問題の探索範囲に応じて設定
        avg_range = np.mean(self.problem.upper_bounds - self.problem.lower_bounds)
        self.initial_min_amplitude = 0.8 * avg_range
        self.final_min_amplitude = 0.001 * avg_range
        
        self.rng = np.random.default_rng()

    def set_parameters(self, **kwargs):
        """パラメータを一括で設定する。"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def optimize(self) -> tuple[np.ndarray, float, list[float]]:
        dim = self.problem.dimension
        lb = self.problem.lower_bounds
        ub = self.problem.upper_bounds
        
        current_evaluations = 0
        best_position = np.zeros(dim)
        best_fitness = float('inf')
        fitness_history = []

        # 1. 初期化
        fireworks = []
        for _ in range(self.population_size):
            pos = self.rng.uniform(lb, ub, size=dim)
            fit = self.problem.evaluate(pos)
            current_evaluations += 1
            fireworks.append(Firework(position=pos, fitness=fit))
            if fit < best_fitness:
                best_fitness = fit
                best_position = pos.copy()

        # 2. メインループ
        while current_evaluations < self.max_evaluations:
            fitness_history.append(best_fitness)
            
            self._calculate_amplitude_and_sparks(fireworks, current_evaluations)
            
            all_sparks = fireworks[:] # 親個体も候補に含める

            # 3. 爆発火花とガウス火花の生成
            # 爆発火花
            for fw in fireworks:
                for _ in range(fw.num_sparks):
                    if current_evaluations >= self.max_evaluations: break
                    spark_pos = self._generate_explosion_spark(fw)
                    spark_fit = self.problem.evaluate(spark_pos)
                    current_evaluations += 1
                    all_sparks.append(Firework(position=spark_pos, fitness=spark_fit))
                    if spark_fit < best_fitness:
                        best_fitness = spark_fit
                        best_position = spark_pos.copy()

            # ガウス火花
            for _ in range(self.gaussian_sparks):
                if current_evaluations >= self.max_evaluations: break
                parent_fw = self.rng.choice(fireworks)
                spark_pos = self._generate_gaussian_spark(parent_fw, best_position)
                spark_fit = self.problem.evaluate(spark_pos)
                current_evaluations += 1
                all_sparks.append(Firework(position=spark_pos, fitness=spark_fit))
                if spark_fit < best_fitness:
                    best_fitness = spark_fit
                    best_position = spark_pos.copy()

            # 4. 次世代選択
            fireworks = self._select_next_generation(all_sparks)

        return best_position, best_fitness, fitness_history

    def _calculate_amplitude_and_sparks(self, fireworks: list[Firework], evals: int):
        # ... (詳細はJava版のロジックをPythonで実装)
        fitness_values = np.array([fw.fitness for fw in fireworks])
        min_fit, max_fit = np.min(fitness_values), np.max(fitness_values)
        sum_fit = np.sum(fitness_values)
        epsilon = 1e-10

        min_amplitude = self._calculate_minimal_amplitude(evals)

        for fw in fireworks:
            # Amplitude calculation
            fw.amplitude = self.explosion_amplitude * (fw.fitness - min_fit + epsilon) / (sum_fit - min_fit * self.population_size + epsilon)
            
            # Spark count calculation
            spark_val = (max_fit - fw.fitness + epsilon) / (self.population_size * max_fit - sum_fit + epsilon)
            fw.num_sparks = int(np.round(self.max_explosion_sparks * spark_val))
            
            # Bounding
            lower_bound_sparks = int(np.round(self.bounding_coeff_a * self.max_explosion_sparks))
            upper_bound_sparks = int(np.round(self.bounding_coeff_b * self.max_explosion_sparks))
            fw.num_sparks = np.clip(fw.num_sparks, lower_bound_sparks, upper_bound_sparks)

    def _calculate_minimal_amplitude(self, evals: int) -> float:
        # ... (詳細はJava版のロジックをPythonで実装)
        ratio = evals / self.max_evaluations
        if self.use_non_linear_decrease:
            # The Java code has a slight error in its non-linear formula. A more standard one is used here.
            # return self.initial_min_amplitude * (self.final_min_amplitude / self.initial_min_amplitude) ** ratio
            return self.initial_min_amplitude - (self.initial_min_amplitude - self.final_min_amplitude) * np.sqrt(ratio * (2 - ratio))
        else:
            return self.initial_min_amplitude - (self.initial_min_amplitude - self.final_min_amplitude) * ratio

    def _generate_explosion_spark(self, fw: Firework) -> np.ndarray:
        # ... (詳細はJava版のロジックをPythonで実装)
        dim = self.problem.dimension
        spark_pos = fw.position.copy()
        # 50%の確率で次元を選択
        dims_to_change = self.rng.random(size=dim) < 0.5
        
        displacements = fw.amplitude * (self.problem.upper_bounds - self.problem.lower_bounds) * self.rng.uniform(-1, 1, size=dim)
        spark_pos[dims_to_change] += displacements[dims_to_change]
        
        # 境界チェックとマッピング
        out_of_bounds = (spark_pos < self.problem.lower_bounds) | (spark_pos > self.problem.upper_bounds)
        spark_pos[out_of_bounds] = self.rng.uniform(self.problem.lower_bounds[out_of_bounds], self.problem.upper_bounds[out_of_bounds])
        
        return spark_pos

    def _generate_gaussian_spark(self, fw: Firework, best_pos: np.ndarray) -> np.ndarray:
        # ... (詳細はJava版のロジックをPythonで実装)
        dim = self.problem.dimension
        spark_pos = fw.position.copy()
        dims_to_change = self.rng.random(size=dim) < 0.5
        
        e = self.rng.standard_normal()
        spark_pos[dims_to_change] += (best_pos[dims_to_change] - spark_pos[dims_to_change]) * e

        # 境界チェックとマッピング
        out_of_bounds = (spark_pos < self.problem.lower_bounds) | (spark_pos > self.problem.upper_bounds)
        spark_pos[out_of_bounds] = self.rng.uniform(self.problem.lower_bounds[out_of_bounds], self.problem.upper_bounds[out_of_bounds])

        return spark_pos

    def _select_next_generation(self, all_sparks: list[Firework]) -> list[Firework]:
        # ... (詳細はJava版のロジックをPythonで実装)
        all_sparks.sort(key=lambda s: s.fitness)
        
        # エリート選択
        next_generation = [all_sparks[0]]
        
        # 残りをランダムに選択
        candidates = all_sparks[1:]
        if len(candidates) > 0:
            num_to_select = min(self.population_size - 1, len(candidates))
            selected_indices = self.rng.choice(len(candidates), size=num_to_select, replace=False)
            for i in selected_indices:
                next_generation.append(candidates[i])
        
        return next_generation