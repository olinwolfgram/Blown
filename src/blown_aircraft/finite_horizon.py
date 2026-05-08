from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FiniteHorizonLQRResult:
    k_seq: np.ndarray
    p_seq: np.ndarray
    a_seq: np.ndarray
    b_seq: np.ndarray
    q_mat: np.ndarray
    r_mat: np.ndarray
    qf_mat: np.ndarray
    state_indices: tuple[int, ...]
    input_indices: tuple[int, ...]
    horizon_steps: int


def _select_subsystem(
    a_mat: np.ndarray,
    b_mat: np.ndarray,
    state_indices: tuple[int, ...] | list[int] | None,
    input_indices: tuple[int, ...] | list[int] | None,
) -> tuple[np.ndarray, np.ndarray, tuple[int, ...], tuple[int, ...]]:
    nx, nu = a_mat.shape[-2], b_mat.shape[-1]
    if state_indices is None:
        state_idx = tuple(range(nx))
    else:
        state_idx = tuple(int(i) for i in state_indices)
    if input_indices is None:
        input_idx = tuple(range(nu))
    else:
        input_idx = tuple(int(i) for i in input_indices)

    if a_mat.ndim == 2:
        a_sub = a_mat[np.ix_(state_idx, state_idx)]
    else:
        a_sub = a_mat[:, state_idx][:, :, state_idx]

    if b_mat.ndim == 2:
        b_sub = b_mat[np.ix_(state_idx, input_idx)]
    else:
        b_sub = b_mat[:, state_idx][:, :, input_idx]

    return a_sub, b_sub, state_idx, input_idx


def _expand_time_series(mat: np.ndarray, horizon_steps: int) -> np.ndarray:
    if mat.ndim == 2:
        return np.repeat(mat[None, :, :], horizon_steps, axis=0)
    if mat.ndim == 3:
        if mat.shape[0] != horizon_steps:
            raise ValueError(f"Expected first dimension {horizon_steps}, got {mat.shape[0]}")
        return mat
    raise ValueError("Expected a 2-D or 3-D array")


def solve_finite_horizon_lqr(
    a_mat: np.ndarray,
    b_mat: np.ndarray,
    q_mat: np.ndarray,
    r_mat: np.ndarray,
    qf_mat: np.ndarray,
    horizon_steps: int,
    *,
    state_indices: tuple[int, ...] | list[int] | None = None,
    input_indices: tuple[int, ...] | list[int] | None = None,
) -> FiniteHorizonLQRResult:
    """Solve the discrete finite-horizon LQR problem by backward Riccati recursion.

    The dynamics are
        x_{k+1} = A_k x_k + B_k u_k

    and the cost is
        sum_{k=0}^{N-1} (x_k^T Q x_k + u_k^T R u_k) + x_N^T Q_f x_N

    `a_mat` and `b_mat` may be either constant 2-D matrices or time-varying
    3-D arrays with leading dimension `horizon_steps`.
    """

    a_sub, b_sub, state_idx, input_idx = _select_subsystem(a_mat, b_mat, state_indices, input_indices)
    a_seq = _expand_time_series(np.asarray(a_sub, dtype=float), horizon_steps)
    b_seq = _expand_time_series(np.asarray(b_sub, dtype=float), horizon_steps)
    q_use = np.asarray(q_mat, dtype=float)
    r_use = np.asarray(r_mat, dtype=float)
    qf_use = np.asarray(qf_mat, dtype=float)

    nx = a_seq.shape[1]
    nu = b_seq.shape[2]
    p_seq = np.zeros((horizon_steps + 1, nx, nx), dtype=float)
    k_seq = np.zeros((horizon_steps, nu, nx), dtype=float)
    p_seq[-1] = qf_use

    for k in range(horizon_steps - 1, -1, -1):
        ak = a_seq[k]
        bk = b_seq[k]
        pk1 = p_seq[k + 1]
        s_mat = r_use + bk.T @ pk1 @ bk
        k_gain = np.linalg.solve(s_mat, bk.T @ pk1 @ ak)
        p_mat = q_use + ak.T @ pk1 @ (ak - bk @ k_gain)
        k_seq[k] = k_gain
        p_seq[k] = 0.5 * (p_mat + p_mat.T)

    return FiniteHorizonLQRResult(
        k_seq=k_seq,
        p_seq=p_seq,
        a_seq=a_seq,
        b_seq=b_seq,
        q_mat=q_use,
        r_mat=r_use,
        qf_mat=qf_use,
        state_indices=state_idx,
        input_indices=input_idx,
        horizon_steps=horizon_steps,
    )
