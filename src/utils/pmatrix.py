import numpy as np


from typing import Generator


def _compute(n, m) -> Generator[np.ndarray, None, None]:
    assert n >= m >= 2
    M = np.zeros((n, m))

    def f(i, free, toBeUsed):
        if i == n:
            yield M
        else:
            for j in range(m):
                if (toBeUsed >> j) & 0x1:
                    M[i, j] = 1
                    yield from f(i + 1, free ^ (1 << j), toBeUsed ^ (1 << j))
                    M[i, j] = 0
            if n - i > toBeUsed.bit_count():
                for j in range(m):
                    if (free >> j) & 0x1:
                        M[i, j] = 1
                        yield from f(i + 1, free ^ (1 << j), toBeUsed)
                        M[i, j] = 0

    yield from f(0, 0, (1 << m) - 1)


generate_all_partition_matrices = _compute


def _compute_random(n, m) -> np.ndarray:
    assert n >= m >= 2
    x = np.linspace(0, m - 1, m)
    y = np.random.choice(x, n - m)
    z = np.concatenate((x, y))
    np.random.shuffle(z)
    M = np.zeros((n, m))
    for i in range(n):
        M[i, int(z[i])] = 1
    return M


create_random_partition_matrix = _compute_random
