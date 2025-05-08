import numpy as np


from typing import Generator


def _compute(n: int, m: int) -> Generator[np.ndarray, None, None]:
    pm = np.array(n, m)

    def _rec(i, fmask, gmask):
        if i == n:
            return pm.copy()
        for j in range(m):
            if (gmask >> j) & 0x1:
                pm[i][j] = 1
                yield from _rec(i + 1, fmask ^ (1 << j), gmask ^ (1 << j))
                pm[i][j] = 0
        if gmask.bit_count() < n - i:
            # already used values can be used always
            for j in range(m):
                if (fmask >> j) & 0x1:
                    pm[i][j] = 1
                    yield from _rec(i + 1, fmask ^ (1 << j), gmask ^ (1 << j))
                    pm[i][j] = 0

    yield from _rec(0, 0, (1 << m) - 1)


generate_all_partition_matrices = _compute
