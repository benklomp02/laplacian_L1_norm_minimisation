from statistics import mean
import numpy as np
import networkx as nx
from typing import Callable, Protocol

from src.utils.objectives import Objective
from src.utils.basis_vector import generate_random_x


class ErrorComputation(Protocol):
    """
    Protocol for different basis error evaluation strategies.
    """

    def compute(self, n: int, basis: np.ndarray, weights: np.ndarray) -> float: ...


class RelativeProjectionNormGain:
    """
    Measures the relative gain in projection norm compared to Laplacian eigenbasis.
    Uses random signals and averages projection energy.
    """

    def compute(self, n: int, basis: np.ndarray, weights: np.ndarray) -> float:
        lap = ErrorCalculator._compute_laplacian_matrix(n, weights)
        return ErrorCalculator._compare_projection_norms(n, lap, basis)


class GraphSmoothnessError:
    """
    Measures average normalized smoothness x^T L x / ||x||^2 over all basis vectors.
    Result is normalized by n.
    """

    def compute(self, n: int, basis: np.ndarray, weights: np.ndarray) -> float:
        L = ErrorCalculator._compute_laplacian_matrix(n, weights)
        return np.mean([x.T @ L @ x / (np.linalg.norm(x) ** 2) for x in basis.T])


class DiagonalizationError:
    """
    Measures how close B^T L B is to diagonal, normalized by Frobenius norm of L.
    """

    def compute(self, n: int, basis: np.ndarray, weights: np.ndarray) -> float:
        L = ErrorCalculator._compute_laplacian_matrix(n, weights)
        BLB = basis.T @ L @ basis
        diag = np.diag(np.diag(BLB))
        return np.linalg.norm(BLB - diag, "fro") / np.linalg.norm(L, "fro")


class L1Variation:
    """
    Measures the total L1 variation of the basis.
    """

    def compute(self, n: int, basis: np.ndarray, weights: np.ndarray) -> float:
        L = ErrorCalculator._compute_laplacian_matrix(n, weights)
        return mean(
            np.linalg.norm(L @ generate_random_x(basis), ord=1) / np.linalg.norm(x, ord=1)
            for x in basis.T
        )


class ErrorCalculator:
    def __init__(self, error_strategy: ErrorComputation = RelativeProjectionNormGain()):
        self.error_strategy = error_strategy

    @staticmethod
    def _is_orthonormal(basis: np.ndarray) -> bool:
        norms = np.linalg.norm(basis, axis=0)
        inner = basis.T @ basis
        return np.allclose(norms, 1) and np.allclose(
            inner, np.eye(basis.shape[1]), atol=1e-10
        )

    @staticmethod
    def _generate_signal(n: int, type: str = "white") -> np.ndarray:
        if type == "white":
            return np.random.randn(n)
        elif type == "lowpass":
            signal = np.random.randn(n)
            signal.sort()
            return signal
        else:
            raise ValueError(f"Unknown signal type: {type}")

    @staticmethod
    def _compute_laplacian_matrix(n: int, weights: np.ndarray) -> np.ndarray:
        G = nx.from_numpy_array(weights)
        return nx.laplacian_matrix(G).toarray()

    @staticmethod
    def _projection_norm(X: np.ndarray, n: int, trials: int = 100) -> float:
        return mean(
            np.linalg.norm(X @ ErrorCalculator._generate_signal(n))
            for _ in range(trials)
        )

    @staticmethod
    def _compare_projection_norms(n: int, A: np.ndarray, B: np.ndarray) -> float:
        e_A = ErrorCalculator._projection_norm(A, n)
        e_B = ErrorCalculator._projection_norm(B, n)
        return (e_A - e_B) / e_A

    def compute_avg_error(
        self,
        n: int,
        compute_basis: Callable[[int, Objective], np.ndarray],
        objective: Objective,
        weights: np.ndarray,
    ) -> float:
        basis = compute_basis(n, objective)
        if not self._is_orthonormal(basis):
            raise ValueError("The basis is not orthonormal.")
        return self.error_strategy.compute(n, basis, weights)
