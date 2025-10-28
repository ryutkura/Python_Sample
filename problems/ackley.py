import numpy as np
from .base_problem import BaseProblem

class AckleyFunction(BaseProblem):
    """
    Ackleyベンチマーク関数。
    """
    def __init__(self, dimension: int, shift_value: float = 0.0):
        # Ackley関数の標準的な探索範囲は[-32.768, 32.768]
        super().__init__(dimension, -32.768, 32.768, shift_value)
        # Ackley関数に特有のパラメータ
        self.a = 20
        self.b = 0.2
        self.c = 2 * np.pi

    def evaluate(self, position: np.ndarray) -> float:
        """
        Ackley関数の値を評価します。
        """
        shifted_pos = position - self._shift
        d = self.dimension
        
        sum1 = np.sum(shifted_pos**2)
        sum2 = np.sum(np.cos(self.c * shifted_pos))
        
        term1 = -self.a * np.exp(-self.b * np.sqrt(sum1 / d))
        term2 = -np.exp(sum2 / d)
        
        return term1 + term2 + self.a + np.exp(1)