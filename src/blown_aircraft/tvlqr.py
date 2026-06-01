from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .finite_horizon import FiniteHorizonLQRResult, solve_finite_horizon_lqr
from .scp import affine_linearize_trajectory


@dataclass(frozen=True)
class TVLQRPlan:
    lqr: FiniteHorizonLQRResult
    a_seq: np.ndarray
    b_seq: np.ndarray
    c_seq: np.ndarray
    x_nominal: np.ndarray
    u_nominal: np.ndarray


def design_tvlqr(
    dynamics,
    x_nominal: np.ndarray,
    u_nominal: np.ndarray,
    q_mat: np.ndarray,
    r_mat: np.ndarray,
    qf_mat: np.ndarray,
    *,
    dynamics_jacobian=None,
    state_indices: tuple[int, ...] | list[int] | None = None,
    input_indices: tuple[int, ...] | list[int] | None = None,
    fd_eps_x: float = 1e-5,
    fd_eps_u: float = 1e-5,
) -> TVLQRPlan:
    x_nom = np.asarray(x_nominal, dtype=float)
    u_nom = np.asarray(u_nominal, dtype=float)
    horizon_steps = u_nom.shape[0]

    a_seq, b_seq, c_seq = affine_linearize_trajectory(
        dynamics,
        x_nom,
        u_nom,
        dynamics_jacobian=dynamics_jacobian,
        fd_eps_x=fd_eps_x,
        fd_eps_u=fd_eps_u,
    )
    lqr = solve_finite_horizon_lqr(
        a_seq,
        b_seq,
        q_mat,
        r_mat,
        qf_mat,
        horizon_steps,
        state_indices=state_indices,
        input_indices=input_indices,
    )
    return TVLQRPlan(
        lqr=lqr,
        a_seq=a_seq,
        b_seq=b_seq,
        c_seq=c_seq,
        x_nominal=x_nom,
        u_nominal=u_nom,
    )


def rollout_tvlqr(
    dynamics,
    x0: np.ndarray,
    plan: TVLQRPlan,
    *,
    u_lower: np.ndarray | None = None,
    u_upper: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    x0 = np.asarray(x0, dtype=float)
    x_nom = np.asarray(plan.x_nominal, dtype=float)
    u_nom = np.asarray(plan.u_nominal, dtype=float)
    horizon_steps = u_nom.shape[0]
    nx = x_nom.shape[1]
    nu = u_nom.shape[1]

    x_hist = np.zeros((horizon_steps + 1, nx), dtype=float)
    u_hist = np.zeros((horizon_steps, nu), dtype=float)
    x_hist[0] = x0

    state_idx = np.asarray(plan.lqr.state_indices, dtype=int)
    input_idx = np.asarray(plan.lqr.input_indices, dtype=int)
    u_lower_arr = None if u_lower is None else np.asarray(u_lower, dtype=float)
    u_upper_arr = None if u_upper is None else np.asarray(u_upper, dtype=float)

    for k in range(horizon_steps):
        uk = u_nom[k].copy()
        dx_sub = x_hist[k, state_idx] - x_nom[k, state_idx]
        du_sub = -plan.lqr.k_seq[k] @ dx_sub
        uk[input_idx] += du_sub
        if u_lower_arr is not None:
            uk = np.maximum(uk, u_lower_arr)
        if u_upper_arr is not None:
            uk = np.minimum(uk, u_upper_arr)
        u_hist[k] = uk
        x_hist[k + 1] = np.asarray(dynamics(x_hist[k], uk), dtype=float)

    return x_hist, u_hist
