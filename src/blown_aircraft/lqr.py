from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_continuous_are, solve_discrete_are


@dataclass(frozen=True)
class LQRResult:
    k_gain: np.ndarray
    p_matrix: np.ndarray
    closed_loop_matrix: np.ndarray
    eigenvalues: np.ndarray
    state_indices: tuple[int, ...]
    input_indices: tuple[int, ...]
    discrete_time: bool


def _select_subsystem(
    a_mat: np.ndarray,
    b_mat: np.ndarray,
    state_indices: tuple[int, ...] | list[int] | None,
    input_indices: tuple[int, ...] | list[int] | None,
) -> tuple[np.ndarray, np.ndarray, tuple[int, ...], tuple[int, ...]]:
    nx, nu = a_mat.shape[0], b_mat.shape[1]
    if state_indices is None:
        state_idx = tuple(range(nx))
    else:
        state_idx = tuple(int(i) for i in state_indices)
    if input_indices is None:
        input_idx = tuple(range(nu))
    else:
        input_idx = tuple(int(i) for i in input_indices)

    a_sub = a_mat[np.ix_(state_idx, state_idx)]
    b_sub = b_mat[np.ix_(state_idx, input_idx)]
    return a_sub, b_sub, state_idx, input_idx


def design_lqr(
    a_mat: np.ndarray,
    b_mat: np.ndarray,
    q_mat: np.ndarray,
    r_mat: np.ndarray,
    *,
    state_indices: tuple[int, ...] | list[int] | None = None,
    input_indices: tuple[int, ...] | list[int] | None = None,
    discrete_time: bool = False,
) -> LQRResult:
    """Solve a continuous- or discrete-time algebraic Riccati equation.

    For the current project stage, this uses SciPy's packaged Riccati solvers
    so the plant and control architecture can be validated quickly. Later, the
    same interface can be preserved while replacing the solver backend with
    textbook algorithms for class comparisons.
    """

    a_sub, b_sub, state_idx, input_idx = _select_subsystem(a_mat, b_mat, state_indices, input_indices)
    q_use = np.asarray(q_mat, dtype=float)
    r_use = np.asarray(r_mat, dtype=float)

    if discrete_time:
        p_mat = solve_discrete_are(a_sub, b_sub, q_use, r_use)
        gain = np.linalg.solve(r_use + b_sub.T @ p_mat @ b_sub, b_sub.T @ p_mat @ a_sub)
        closed_loop = a_sub - b_sub @ gain
    else:
        p_mat = solve_continuous_are(a_sub, b_sub, q_use, r_use)
        gain = np.linalg.solve(r_use, b_sub.T @ p_mat)
        closed_loop = a_sub - b_sub @ gain

    eigvals = np.linalg.eigvals(closed_loop)
    return LQRResult(
        k_gain=gain,
        p_matrix=p_mat,
        closed_loop_matrix=closed_loop,
        eigenvalues=eigvals,
        state_indices=state_idx,
        input_indices=input_idx,
        discrete_time=discrete_time,
    )
