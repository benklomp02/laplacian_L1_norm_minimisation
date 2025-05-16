import numpy as np


def f_original(weights: np.ndarray, M: np.ndarray, a: np.ndarray) -> float:
    s = M @ a
    D = weights.sum(axis=1)
    diff = s * D - weights @ s
    return np.abs(diff).sum()


def f_approx_triangle_inequality(weights, M, a):
    Wm = M.T @ weights @ M
    W_sym = Wm + Wm.T
    np.fill_diagonal(W_sym, 0)
    lower = np.tril(W_sym, -1).sum(axis=1)
    upper = np.triu(W_sym, +1).sum(axis=1)
    f_vec = lower - upper
    return float(f_vec @ a)


def f_approx_sign_by_median(weights, M, a):
    m = M.shape[1]
    Wm = M.T @ weights @ M
    t = m // 2  # Take the median of a
    row_sum = Wm.sum(axis=1)
    sp = a[t + 1 :].dot(row_sum[t + 1 :]) - (Wm[t + 1 :, :] @ a).sum()
    sm = a[:t].dot(row_sum[:t]) - (Wm[:t, :] @ a).sum()
    return sp - sm


def f_approx_sign_by_mean(weights, M, a):
    Wm = M.T @ weights @ M
    t = np.abs(a - a.mean()).argmin()  # The the closest value to the mean of a
    row_sum = Wm.sum(axis=1)
    sp = a[:t].dot(row_sum[:t]) - (Wm[:t, :] @ a).sum()
    sm = a[t + 1 :].dot(row_sum[t + 1 :]) - (Wm[t + 1 :, :] @ a).sum()
    return sm - sp


def f_approx_sign_by_weighted(weights, M, a):
    M_inv = np.argmax(M, axis=1)
    W_out = weights.sum(axis=1)
    A = a[M_inv] * W_out
    B = weights.dot(a[M_inv])
    beta = np.where(A > B, 1, -1)
    cluster_size = M.sum(axis=0)
    alpha = np.where(M.T.dot(beta) > cluster_size / 2, 1, -1)
    Wm = M.T @ weights @ M
    diff = a[:, None] - a[None, :]
    return float((alpha[:, None] * diff * Wm).sum())
