import numpy as np
from .base_problem import BaseProblem

class RosenbrockFunction(BaseProblem):
    """
    Rosenbrockベンチマーク関数。
    f(x) = Σ [100 * (x_{i+1} - x_i^2)^2 + (x_i - 1)^2]
    """
    def __init__(self, dimension: int, shift_value: float = 0.0):
        # Rosenbrock関数の探索範囲は[-100, 100]
        super().__init__(dimension, -100.0, 100.0, shift_value)

    def evaluate(self, position: np.ndarray) -> float:
        """
        Rosenbrock関数の値を評価します。
        forループの代わりにNumPyの配列スライスを使うことで、高速に計算します。
       
        """
        # positionからシフト値を引く
        x = position - self._shift
        
        # xの全要素から最後の要素を除いた配列 (x_i に相当)
        x_i = x[:-1]
        # xの全要素から最初の要素を除いた配列 (x_{i+1} に相当)
        x_i1 = x[1:]
        
        # Rosenbrock関数の式をベクトル演算で計算
        term1 = 100 * (x_i1 - x_i**2)**2
        term2 = (x_i - 1)**2
        
        # 各項の合計を返す
        return np.sum(term1 + term2)