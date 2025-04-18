import numpy as np

from src.utils.objectives import Objective


def _compute(n: int, objective: Objective) -> np.ndarray:
    """A greedy algorithm for a general objective function."""
    basis = []
    tau = [{i} for i in range(n)]
    for _ in range(n - 1):
        i, j = max(
            ((i, j) for i in range(len(tau)) for j in range(len(tau)) if i != j),
            key=lambda comb: objective.compute(tau[comb[0]], tau[comb[1]]),
        )
        a, b = np.zeros(n), np.zeros(n)
        a[list(tau[i])] = 1
        b[list(tau[j])] = 1
        t = 1 / np.sqrt(len(tau[i]) * len(tau[j]) * (len(tau[i]) + len(tau[j])))
        u = -t * len(tau[j]) * a + t * len(tau[i]) * b
        basis.append(u)
        tau[i] |= tau[j]
        tau.pop(j)
    u1 = np.full(n, 1 / np.sqrt(n))
    basis.append(u1)
    return np.column_stack(basis[::-1])


compute_basis = _compute
