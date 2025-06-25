import numpy as np
from .base_problem import BaseProblem

class SphereFunction(BaseProblem):
    """
    Sphereベンチマーク関数。
    f(x) = Σ(x_i - shift_i)^2
    """
    def __init__(self, dimension: int, shift_value: float = 0.0):
        # Sphere関数の探索範囲は[-100, 100]
        super().__init__(dimension, -100.0, 100.0, shift_value)

    def evaluate(self, position: np.ndarray) -> float:
        """
        Sphere関数の値を評価します。
        Javaのforループの代わりに、NumPyのベクトル演算で簡潔に記述できます。
       
        """
        shifted_pos = position - self._shift
        return np.sum(shifted_pos ** 2)