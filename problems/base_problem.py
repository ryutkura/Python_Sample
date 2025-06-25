from abc import ABC, abstractmethod
import numpy as np

class BaseProblem(ABC):
    """
    全てのベンチマーク問題の基底クラス（抽象クラス）。
    JavaのOptimizationProblemインターフェースとBenchmarkFunctionクラスの役割を兼ねます。
   
    """
    def __init__(self, dimension: int, lower_bounds: float, upper_bounds: float, shift_value: float = 0.0):
        self._dimension = dimension
        self._lower_bounds = np.full(dimension, lower_bounds)
        self._upper_bounds = np.full(dimension, upper_bounds)
        self._shift = np.full(dimension, shift_value)

    @abstractmethod
    def evaluate(self, position: np.ndarray) -> float:
        """
        与えられた位置(position)の評価値を計算する抽象メソッド。
        このクラスを継承するクラスは、必ずこのメソッドを実装する必要があります。
        """
        pass

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def lower_bounds(self) -> np.ndarray:
        return self._lower_bounds

    @property
    def upper_bounds(self) -> np.ndarray:
        return self._upper_bounds