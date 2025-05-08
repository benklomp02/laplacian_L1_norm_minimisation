from abc import ABC, abstractmethod
from typing import Set
import numpy as np
import networkx as nx
import random


class Objective(ABC):
    @abstractmethod
    def __init__(self, n: int, weights: np.ndarray):
        self._n = n
        self._weights = weights

    @abstractmethod
    def compute(self, I: Set[int], J: Set[int]) -> float:
        raise NotImplementedError()


class Objective0(Objective):
    """The L1 norm approximation from the script."""

    def __init__(self, n: int, weights: np.ndarray):
        super().__init__(n, weights)

    def _W(self, I: int, J: int):
        return sum(self._weights[i][j] for i in I for j in J)

    def compute(self, I: Set[int], J: Set[int]) -> float:
        denom = len(I) * len(J)
        wt = self._W(I, J) / denom
        return wt


class Objective1(Objective):
    """An own implementation of the L1 norm approximation."""

    def __init__(self, n: int, weights: np.ndarray):
        super().__init__(n, weights)
        G = nx.from_numpy_array(weights)

    def _W(self, I: int, J: int):
        return sum(min(self._weights[i][j] for j in J) for i in I)

    def compute(self, I: Set[int], J: Set[int]) -> float:
        denom = len(I) + len(J)
        _wt = self._W(I, J)
        # Focus on a single vertex only using out degree
        wt = _wt / denom
        return wt


class Objective2(Objective):
    """Negative L1 norm objective of the Laplacian."""

    def __init__(self, n: int, weights: np.ndarray):
        super().__init__(n, weights)
        self._L = nx.laplacian_matrix(nx.from_numpy_array(self._weights)).toarray()

    def _build_u(self, I: Set[int], J: Set[int]) -> np.ndarray:
        a, b = np.zeros(self._n), np.zeros(self._n)
        a[list(I)] = 1
        b[list(J)] = 1
        t = 1 / np.sqrt(len(I) * len(J) * (len(I) + len(J)))
        return -t * len(J) * a + t * len(I) * b

    def compute(self, I: Set[int], J: Set[int]) -> float:
        return -np.linalg.norm(self._L @ self._build_u(I, J), ord=1)


class Objective3(Objective):
    """L1 norm objective of the Laplacian."""

    def __init__(self, n: int, weights: np.ndarray):
        super().__init__(n, weights)
        self._L = nx.laplacian_matrix(nx.from_numpy_array(self._weights)).toarray()

    def _build_u(self, I: Set[int], J: Set[int]) -> np.ndarray:
        a, b = np.zeros(self._n), np.zeros(self._n)
        a[list(I)] = 1
        b[list(J)] = 1
        t = 1 / np.sqrt(len(I) * len(J) * (len(I) + len(J)))
        return -t * len(J) * a + t * len(I) * b

    def compute(self, I: Set[int], J: Set[int]) -> float:
        return np.linalg.norm(self._L @ self._build_u(I, J), ord=1)


class RandomObjective(Objective):
    """A completely random objective function."""

    def __init__(self, n: int, weights: np.ndarray):
        super().__init__(n, weights)

    def compute(self, I: Set[int], J: Set[int]) -> float:
        return random.random()
