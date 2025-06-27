import numpy as np
from typing import Tuple, List, Callable
from problems import BaseProblem # ★我々の問題クラスをインポート

class DynFWA:
    """
    Dynamic Search Fireworks Algorithm (dynFWA)
    提供されたコードのロジックはそのままに、システムに適応させました。
   
    """
    def __init__(self, 
                 problem: BaseProblem, # ★引数をproblemオブジェクトに変更
                 n_fireworks: int = 5,
                 n_sparks: int = 150,
                 amplitude_init: float = None,
                 amplitude_final: float = 1e-4,
                 ca: float = 1.2,
                 cr: float = 0.9,
                 max_evaluations: int = 300000):
        
        self.problem = problem # ★problemオブジェクトを保持
        # ★boundsをproblemオブジェクトから取得するように変更
        self.bounds = np.array([self.problem.lower_bounds, self.problem.upper_bounds]).T
        
        self.n_fireworks = n_fireworks
        self.n_sparks = n_sparks
        self.amplitude_final = amplitude_final
        self.ca = ca
        self.cr = cr
        self.max_evaluations = max_evaluations
        self.evaluations = 0
        
        if amplitude_init is None:
            self.amplitude_init = np.max(self.problem.upper_bounds - self.problem.lower_bounds)
        else:
            self.amplitude_init = amplitude_init
            
        self.cf_amplitude = np.max(self.problem.upper_bounds - self.problem.lower_bounds)
        self.rng = np.random.default_rng()

    # --- ここから下のメソッド群は、提供されたコードのロジックを一切変更していません ---

    def _calculate_sparks_number(self, fitness_values: np.ndarray) -> np.ndarray:
        epsilon = np.finfo(float).eps
        y_max = np.max(fitness_values)
        numerator = y_max - fitness_values + epsilon
        denominator = np.sum(y_max - fitness_values) + epsilon
        sparks = self.n_sparks * numerator / denominator
        sparks = np.round(sparks).astype(int)
        sparks = np.maximum(sparks, 1)
        if np.sum(sparks) > self.n_sparks:
            sparks = np.floor(sparks * self.n_sparks / np.sum(sparks)).astype(int)
            sparks = np.maximum(sparks, 1)
            while np.sum(sparks) > self.n_sparks:
                max_idx = np.argmax(sparks)
                if sparks[max_idx] > 1: sparks[max_idx] -= 1
                else: break
        deficit = self.n_sparks - np.sum(sparks)
        if deficit > 0:
            for idx in np.argsort(fitness_values)[:deficit]:
                sparks[idx] += 1
        return sparks

    def _calculate_amplitudes(self, fitness_values: np.ndarray, cf_index: int) -> np.ndarray:
        epsilon = np.finfo(float).eps
        y_min = np.min(fitness_values)
        A_hat = np.max(self.problem.upper_bounds - self.problem.lower_bounds)
        numerator = fitness_values - y_min + epsilon
        denominator = np.sum(fitness_values - y_min) + epsilon
        amplitudes = A_hat * numerator / denominator
        amplitudes[cf_index] = self.cf_amplitude
        return amplitudes

    def _generate_explosion_sparks(self, firework: np.ndarray, amplitude: float, n_sparks: int) -> np.ndarray:
        sparks = []
        d = len(firework)
        for _ in range(n_sparks):
            spark = firework.copy()
            z = self.rng.random(d) < 0.5
            if not z.any(): z[self.rng.integers(d)] = True
            
            for k in range(d):
                if z[k]:
                    offset = amplitude * self.rng.uniform(-1, 1)
                    spark[k] += offset
                    if spark[k] < self.bounds[k, 0] or spark[k] > self.bounds[k, 1]:
                        spark[k] = self.bounds[k, 0] + self.rng.random() * (self.bounds[k, 1] - self.bounds[k, 0])
            sparks.append(spark)
        return np.array(sparks) if sparks else np.array([])

    def _update_cf_amplitude(self, cf_fitness: float, best_spark_fitness: float):
        if best_spark_fitness < cf_fitness: self.cf_amplitude *= self.ca
        else: self.cf_amplitude *= self.cr
        max_amplitude = np.max(self.problem.upper_bounds - self.problem.lower_bounds)
        self.cf_amplitude = min(self.cf_amplitude, max_amplitude)
        self.cf_amplitude = max(self.cf_amplitude, self.amplitude_final)

    def _selection(self, candidates: np.ndarray, fitness_values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        best_idx = np.argmin(fitness_values)
        selected_indices = [best_idx]
        remaining_indices = list(range(len(candidates)))
        remaining_indices.remove(best_idx)
        if len(remaining_indices) >= self.n_fireworks - 1:
            choices = self.rng.choice(remaining_indices, self.n_fireworks - 1, replace=False)
            selected_indices.extend(choices)
        else:
            selected_indices.extend(remaining_indices)
        return candidates[selected_indices], fitness_values[selected_indices]

    # ★★★ optimizeメソッドをシステムに適応 ★★★
    def optimize(self) -> Tuple[np.ndarray, float, List[float]]:
        # 目的関数を self.problem.evaluate に変更
        objective_func = self.problem.evaluate
        
        # 初期化
        fireworks = self.rng.uniform(self.problem.lower_bounds, self.problem.upper_bounds, size=(self.n_fireworks, self.problem.dimension))
        fitness_values = np.array([objective_func(fw) for fw in fireworks])
        self.evaluations = self.n_fireworks
        
        cf_index = np.argmin(fitness_values)
        best_position = fireworks[cf_index].copy()
        best_fitness = fitness_values[cf_index]
        
        fitness_history = [best_fitness]
        
        # メインループ
        while self.evaluations < self.max_evaluations:
            n_sparks_array = self._calculate_sparks_number(fitness_values)
            amplitudes = self._calculate_amplitudes(fitness_values, cf_index)
            
            all_candidates_list = [fireworks]
            all_fitness_list = [fitness_values]
            
            for i in range(self.n_fireworks):
                explosion_sparks = self._generate_explosion_sparks(fireworks[i], amplitudes[i], n_sparks_array[i])
                if explosion_sparks.size > 0:
                    spark_fitness = np.array([objective_func(spark) for spark in explosion_sparks])
                    self.evaluations += len(explosion_sparks)
                    all_candidates_list.append(explosion_sparks)
                    all_fitness_list.append(spark_fitness)
                if self.evaluations >= self.max_evaluations: break
                    
            all_candidates = np.vstack(all_candidates_list)
            all_fitness = np.hstack(all_fitness_list)
            
            best_spark_fitness = np.min(all_fitness)
            self._update_cf_amplitude(fitness_values[cf_index], best_spark_fitness)
            
            fireworks, fitness_values = self._selection(all_candidates, all_fitness)
            
            cf_index = np.argmin(fitness_values)
            if fitness_values[cf_index] < best_fitness:
                best_position = fireworks[cf_index].copy()
                best_fitness = fitness_values[cf_index]
                
            fitness_history.append(best_fitness)
            
        # ★戻り値の型を既存のシステムに合わせる
        return best_position, best_fitness, fitness_history