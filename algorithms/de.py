import numpy as np
from typing import List, Tuple
from problems import BaseProblem

class DE:
    """
    Differential Evolution (DE) implementation based on Storn & Price (1995).
    Strategy: DE/rand/1/bin (Classic DE)
    Reference: Java implementation (DE_Claude.java)
    """
    def __init__(self, 
                 problem: BaseProblem, 
                 population_size: int = 20,  # Java版のデフォルト: 20
                 f_weight: float = 0.8,      # Scaling factor (F): 0.8
                 cr_rate: float = 0.95,      # Crossover probability (CR): 0.95
                 max_evaluations: int = 300000):
        
        self.problem = problem
        self.dim = problem.dimension
        self.bounds = np.array([problem.lower_bounds, problem.upper_bounds]).T
        
        # パラメータ設定
        self.pop_size = population_size
        self.F = f_weight
        self.CR = cr_rate
        self.max_evaluations = max_evaluations
        
        self.rng = np.random.default_rng()
        self.evaluations = 0

    def optimize(self) -> Tuple[np.ndarray, float, List[float]]:
        # --- 1. 初期化 (Initialization) ---
        population = self.rng.uniform(
            self.problem.lower_bounds, 
            self.problem.upper_bounds, 
            (self.pop_size, self.dim)
        )
        
        # 初期評価
        fitness = np.array([self.problem.evaluate(ind) for ind in population])
        self.evaluations += self.pop_size
        
        # ベスト解の記録
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx].copy()
        best_fitness = fitness[best_idx]
        fitness_history = [best_fitness]

        # --- 2. メインループ ---
        while self.evaluations < self.max_evaluations:
            
            # 次世代の個体群（今の世代をコピーして開始）
            new_population = population.copy()
            new_fitness = fitness.copy()
            
            for i in range(self.pop_size):
                if self.evaluations >= self.max_evaluations:
                    break

                # --- Mutation (DE/rand/1) ---
                # 自分(i)以外の3つの異なる個体(r1, r2, r3)を選ぶ
                candidates = [idx for idx in range(self.pop_size) if idx != i]
                r1, r2, r3 = self.rng.choice(candidates, 3, replace=False)
                
                x_r1 = population[r1]
                x_r2 = population[r2]
                x_r3 = population[r3]
                
                # 変異ベクトル v = x_r1 + F * (x_r2 - x_r3)
                mutant = x_r1 + self.F * (x_r2 - x_r3)
                
                # --- Crossover (Binomial) ---
                # 少なくとも1つの次元は必ず更新する (j_rand)
                j_rand = self.rng.integers(0, self.dim)
                
                trial = population[i].copy()
                
                # 各次元について CR以下 または j_rand なら変異ベクトルの値を採用
                cross_mask = self.rng.random(self.dim) < self.CR
                cross_mask[j_rand] = True
                
                trial[cross_mask] = mutant[cross_mask]
                
                # --- Boundary Handling ---
                # はみ出した場合はクリッピング（Java版では明記されていませんが一般的な処理）
                trial = np.clip(trial, self.problem.lower_bounds, self.problem.upper_bounds)
                
                # --- Selection ---
                trial_fitness = self.problem.evaluate(trial)
                self.evaluations += 1
                
                # ターゲット個体(親)より良ければ(または同じなら)更新
                if trial_fitness <= fitness[i]:
                    new_population[i] = trial
                    new_fitness[i] = trial_fitness
                    
                    # 全体ベストの更新
                    if trial_fitness < best_fitness:
                        best_fitness = trial_fitness
                        best_solution = trial.copy()

            # 世代交代
            population = new_population
            fitness = new_fitness
            fitness_history.append(best_fitness)

        return best_solution, best_fitness, fitness_history