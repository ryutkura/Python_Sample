import numpy as np
from typing import List, Tuple, Literal
from problems import BaseProblem


class DE:
    """
    Differential Evolution (DE) implementation based on Storn & Price (1995).
    
    This implementation faithfully reproduces the original paper's schemes:
    - DE1: DE/rand/1 with exponential crossover (Equations 12-15)
    - DE2: DE/current-to-best/1 with exponential crossover (Equation 16)
    
    Reference: Storn, R. and Price, K. (1995). "Differential Evolution - 
    A simple and efficient adaptive scheme for global optimization over 
    continuous spaces." TR-95-012.
    """
    
    def __init__(self, 
                 problem: BaseProblem, 
                 population_size: int = 20,
                 f_weight: float = 0.8,
                 cr_rate: float = 0.9,
                 max_evaluations: int = 300000,
                 scheme: Literal["DE1", "DE2"] = "DE1",
                 lambda_param: float = 0.5,
                 crossover_type: Literal["exponential", "binomial"] = "exponential",
                 boundary_handling: Literal["clip", "random_reinit", "reflection"] = "random_reinit"):
        """
        Parameters
        ----------
        problem : BaseProblem
            The optimization problem to solve.
        population_size : int
            Number of individuals in the population (NP in the paper).
        f_weight : float
            Scaling factor F for differential mutation. Typically in [0.4, 1.0].
        cr_rate : float
            Crossover probability CR. For exponential crossover, this controls
            the expected length L of the crossover segment.
        max_evaluations : int
            Maximum number of function evaluations.
        scheme : {"DE1", "DE2"}
            DE1: rand/1 mutation (Equation 12)
            DE2: current-to-best/1 mutation (Equation 16)
        lambda_param : float
            Lambda parameter for DE2 scheme. Controls greediness toward best.
        crossover_type : {"exponential", "binomial"}
            exponential: Original paper's continuous segment crossover (Equation 15)
            binomial: Standard dimension-wise independent crossover
        boundary_handling : {"clip", "random_reinit", "reflection"}
            How to handle solutions that exceed bounds.
        """
        self.problem = problem
        self.dim = problem.dimension
        self.lower_bounds = np.array(problem.lower_bounds)
        self.upper_bounds = np.array(problem.upper_bounds)
        
        self.pop_size = population_size
        self.F = f_weight
        self.CR = cr_rate
        self.max_evaluations = max_evaluations
        self.scheme = scheme
        self.lambda_param = lambda_param
        self.crossover_type = crossover_type
        self.boundary_handling = boundary_handling
        
        self.rng = np.random.default_rng()
        self.evaluations = 0

    def _exponential_crossover(self, target: np.ndarray, mutant: np.ndarray) -> np.ndarray:
        """
        Exponential (continuous segment) crossover as described in Equation (15).
        
        A segment of length L starting at random position n is copied from the
        mutant vector to the trial vector. The length L is determined by 
        repeatedly drawing random numbers until one exceeds CR.
        
        This ensures at least one dimension is always taken from the mutant.
        """
        trial = target.copy()
        
        # n: random starting position (Equation 15)
        n = self.rng.integers(0, self.dim)
        
        # L: segment length, determined by CR probability
        # L is drawn from [1, D] with probability Pr(L=k) = CR^(k-1) * (1-CR) for k < D
        # and Pr(L=D) = CR^(D-1)
        L = 1
        while self.rng.random() < self.CR and L < self.dim:
            L += 1
        
        # Copy L consecutive dimensions from mutant, wrapping around (modulo D)
        for j in range(L):
            idx = (n + j) % self.dim
            trial[idx] = mutant[idx]
        
        return trial

    def _binomial_crossover(self, target: np.ndarray, mutant: np.ndarray) -> np.ndarray:
        """
        Binomial crossover (standard DE/rand/1/bin).
        
        Each dimension is independently selected from mutant with probability CR.
        At least one dimension (j_rand) is guaranteed to come from mutant.
        """
        trial = target.copy()
        
        j_rand = self.rng.integers(0, self.dim)
        cross_mask = self.rng.random(self.dim) < self.CR
        cross_mask[j_rand] = True
        
        trial[cross_mask] = mutant[cross_mask]
        return trial

    def _crossover(self, target: np.ndarray, mutant: np.ndarray) -> np.ndarray:
        """Apply the configured crossover operator."""
        if self.crossover_type == "exponential":
            return self._exponential_crossover(target, mutant)
        else:
            return self._binomial_crossover(target, mutant)

    def _handle_boundaries(self, vector: np.ndarray) -> np.ndarray:
        """
        Handle boundary constraint violations.
        
        The original paper doesn't specify a boundary handling method explicitly,
        but mentions incorporating constraints into the objective function.
        We provide several common approaches.
        """
        if self.boundary_handling == "clip":
            # Simple clipping to bounds
            return np.clip(vector, self.lower_bounds, self.upper_bounds)
        
        elif self.boundary_handling == "random_reinit":
            # Reinitialize violating dimensions randomly within bounds
            result = vector.copy()
            for j in range(self.dim):
                if result[j] < self.lower_bounds[j] or result[j] > self.upper_bounds[j]:
                    result[j] = self.rng.uniform(self.lower_bounds[j], self.upper_bounds[j])
            return result
        
        elif self.boundary_handling == "reflection":
            # Reflect back from boundaries
            result = vector.copy()
            for j in range(self.dim):
                lb, ub = self.lower_bounds[j], self.upper_bounds[j]
                width = ub - lb
                while result[j] < lb or result[j] > ub:
                    if result[j] < lb:
                        result[j] = 2 * lb - result[j]
                    if result[j] > ub:
                        result[j] = 2 * ub - result[j]
                    # Safety check for extreme cases
                    if abs(result[j] - lb) > 2 * width or abs(result[j] - ub) > 2 * width:
                        result[j] = self.rng.uniform(lb, ub)
                        break
            return result
        
        return vector

    def _mutate_de1(self, population: np.ndarray, i: int) -> np.ndarray:
        """
        DE1 mutation: DE/rand/1 (Equation 12)
        
        v = x_r1 + F * (x_r2 - x_r3)
        
        where r1, r2, r3 are mutually distinct and different from i.
        """
        # Select three distinct individuals, all different from i
        candidates = [idx for idx in range(self.pop_size) if idx != i]
        r1, r2, r3 = self.rng.choice(candidates, 3, replace=False)
        
        # Equation (12): v = x_r1 + F * (x_r2 - x_r3)
        mutant = population[r1] + self.F * (population[r2] - population[r3])
        
        return mutant

    def _mutate_de2(self, population: np.ndarray, i: int, best_idx: int) -> np.ndarray:
        """
        DE2 mutation: DE/current-to-best/1 (Equation 16)
        
        v = x_i + λ * (x_best - x_i) + F * (x_r2 - x_r3)
        
        This scheme is more greedy as it incorporates the current best solution.
        """
        # Select two distinct individuals, different from i and best
        candidates = [idx for idx in range(self.pop_size) if idx != i and idx != best_idx]
        r2, r3 = self.rng.choice(candidates, 2, replace=False)
        
        # Equation (16)
        x_i = population[i]
        x_best = population[best_idx]
        
        mutant = (x_i + 
                  self.lambda_param * (x_best - x_i) + 
                  self.F * (population[r2] - population[r3]))
        
        return mutant

    def optimize(self) -> Tuple[np.ndarray, float, List[float]]:
        """
        Run the DE optimization.
        
        Returns
        -------
        best_solution : np.ndarray
            The best solution found.
        best_fitness : float
            The fitness value of the best solution.
        fitness_history : List[float]
            History of best fitness values (one per generation).
        """
        # --- 1. Initialization ---
        # Generate initial population uniformly at random (as per paper)
        population = self.rng.uniform(
            self.lower_bounds, 
            self.upper_bounds, 
            (self.pop_size, self.dim)
        )
        
        # Evaluate initial population
        fitness = np.array([self.problem.evaluate(ind) for ind in population])
        self.evaluations = self.pop_size
        
        # Track best solution
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx].copy()
        best_fitness = fitness[best_idx]
        fitness_history = [best_fitness]

        # --- 2. Main Evolution Loop ---
        while self.evaluations < self.max_evaluations:
            
            # Prepare next generation (synchronous update as per paper)
            new_population = population.copy()
            new_fitness = fitness.copy()
            
            for i in range(self.pop_size):
                if self.evaluations >= self.max_evaluations:
                    break

                # --- Mutation ---
                if self.scheme == "DE1":
                    mutant = self._mutate_de1(population, i)
                else:  # DE2
                    mutant = self._mutate_de2(population, i, best_idx)
                
                # --- Crossover ---
                trial = self._crossover(population[i], mutant)
                
                # --- Boundary Handling ---
                trial = self._handle_boundaries(trial)
                
                # --- Selection (Greedy) ---
                # As per paper: accept if trial is better or equal
                trial_fitness = self.problem.evaluate(trial)
                self.evaluations += 1
                
                if trial_fitness <= fitness[i]:
                    new_population[i] = trial
                    new_fitness[i] = trial_fitness
                    
                    # Update global best
                    if trial_fitness < best_fitness:
                        best_fitness = trial_fitness
                        best_solution = trial.copy()

            # Generation replacement (synchronous)
            population = new_population
            fitness = new_fitness
            
            # Update best index for next generation (needed for DE2)
            best_idx = np.argmin(fitness)
            
            fitness_history.append(best_fitness)

        return best_solution, best_fitness, fitness_history


# Convenience classes for specific DE variants
class DE1(DE):
    """DE1 scheme with exponential crossover (original paper's primary scheme)."""
    def __init__(self, problem: BaseProblem, **kwargs):
        kwargs.setdefault("scheme", "DE1")
        kwargs.setdefault("crossover_type", "exponential")
        super().__init__(problem, **kwargs)


class DE2(DE):
    """DE2 scheme with exponential crossover (paper's greedy variant)."""
    def __init__(self, problem: BaseProblem, **kwargs):
        kwargs.setdefault("scheme", "DE2")
        kwargs.setdefault("crossover_type", "exponential")
        super().__init__(problem, **kwargs)


class DE_rand_1_bin(DE):
    """Classic DE/rand/1/bin (commonly used variant, not original paper's DE1)."""
    def __init__(self, problem: BaseProblem, **kwargs):
        kwargs.setdefault("scheme", "DE1")
        kwargs.setdefault("crossover_type", "binomial")
        super().__init__(problem, **kwargs)
