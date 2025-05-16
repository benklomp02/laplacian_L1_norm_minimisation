import numpy as np


def _compute(basis: np.ndarray, p=0.5) -> np.ndarray:
    """Generates a random vector in the span of the given orthonormal basis.

    Args:
        basis (np.ndarray): An orthonormal basis.
        p (float): Probability of a coefficient being 0. Default is 0.5.

    Returns:
        np.ndaray: A random vector in the span of the basis.
    """
    nrows, _ = basis.shape
    # Generate random coefficients where p is the probability of being 0
    coefficients = np.random.choice([0, 1], size=(nrows,), p=[p, 1 - p])
    # Generate random coefficients
    coefficients = np.random.randn(nrows) * coefficients
    # Normalize coefficients
    coefficients /= np.linalg.norm(coefficients)
    # Generate random vector in the span of the basis
    x = basis @ coefficients
    return x


generate_random_x = _compute
