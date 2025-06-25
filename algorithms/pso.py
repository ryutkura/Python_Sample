import numpy as np
from dataclasses import dataclass
# problemsパッケージからBaseProblemクラスをインポートします
from problems import BaseProblem

@dataclass
class Particle:
    """粒子を表すデータクラス。"""
    position: np.ndarray
    velocity: np.ndarray
    fitness: float
    best_position: np.ndarray
    best_fitness: float

class PSO:
    """
    Particle Swarm Optimization (PSO) の実装。
   
    """
    def __init__(self, problem: BaseProblem, swarm_size: int = 30, max_iterations: int = 10000):
        self.problem = problem
        self.swarm_size = swarm_size
        self.max_iterations = max_iterations
        # Java版のデフォルトパラメータ
        self.inertia_weight = 0.729
        self.c1 = 1.49445
        self.c2 = 1.49445
        self.rng = np.random.default_rng()

    def set_parameters(self, inertia_weight: float, c1: float, c2: float):
        self.inertia_weight = inertia_weight
        self.c1 = c1
        self.c2 = c2

    def optimize(self) -> tuple[np.ndarray, float]:
        dim = self.problem.dimension
        lb = self.problem.lower_bounds
        ub = self.problem.upper_bounds
        
        global_best_position = np.zeros(dim)
        global_best_fitness = float('inf')
        
        # ★追加：評価値の履歴を保存するリスト
        fitness_history = []
        
        particles = []
        for _ in range(self.swarm_size):
            pos = self.rng.uniform(lb, ub, size=dim)
            vel = self.rng.uniform(-0.1 * (ub - lb), 0.1 * (ub - lb), size=dim)
            fit = self.problem.evaluate(pos)
            
            particles.append(Particle(pos, vel, fit, pos.copy(), fit))
            
            if fit < global_best_fitness:
                global_best_fitness = fit
                global_best_position = pos.copy()

        for _ in range(self.max_iterations):
            for p in particles:
                r1, r2 = self.rng.random(size=dim), self.rng.random(size=dim)
                cognitive_vel = self.c1 * r1 * (p.best_position - p.position)
                social_vel = self.c2 * r2 * (global_best_position - p.position)
                p.velocity = self.inertia_weight * p.velocity + cognitive_vel + social_vel
                p.position += p.velocity
                p.position = np.clip(p.position, lb, ub)
                p.fitness = self.problem.evaluate(p.position)
                
                if p.fitness < p.best_fitness:
                    p.best_fitness = p.fitness
                    p.best_position = p.position.copy()
                    if p.fitness < global_best_fitness:
                        global_best_fitness = p.fitness
                        global_best_position = p.position.copy()
            # ★追加：毎反復終了時のグローバルベスト評価値を履歴に追加
            fitness_history.append(global_best_fitness)
        
        # ★変更：fitness_historyも一緒に返す
        return global_best_position, global_best_fitness, fitness_history