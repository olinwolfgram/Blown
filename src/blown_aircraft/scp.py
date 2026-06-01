from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cvx
import numpy as np


@dataclass(frozen=True)
class SCPResult:
    x_seq: np.ndarray
    u_seq: np.ndarray
    cost_history: np.ndarray
    converged: bool
    iterations: int
    trust_region_state: np.ndarray
    trust_region_input: np.ndarray
    termination_reason: str


def _expand_trust_region(value: float | np.ndarray, size: int) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return np.full(size, float(arr), dtype=float)
    if arr.shape != (size,):
        raise ValueError(f"Expected trust-region shape ({size},), got {arr.shape}")
    return arr


def _as_array_or_repeat(value: np.ndarray, horizon_len: int) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 1:
        return np.repeat(arr[None, :], horizon_len, axis=0)
    if arr.shape[0] != horizon_len:
        raise ValueError(f"Expected leading dimension {horizon_len}, got {arr.shape[0]}")
    return arr


def _as_array_or_default(value: np.ndarray | None, horizon_len: int, width: int) -> np.ndarray:
    if value is None:
        return np.zeros((horizon_len, width), dtype=float)
    return _as_array_or_repeat(value, horizon_len)


def _as_matrix_sequence(value: np.ndarray, horizon_len: int, matrix_size: int) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 2:
        if arr.shape != (matrix_size, matrix_size):
            raise ValueError(f"Expected matrix shape ({matrix_size}, {matrix_size}), got {arr.shape}")
        return np.repeat(arr[None, :, :], horizon_len, axis=0)
    if arr.shape != (horizon_len, matrix_size, matrix_size):
        raise ValueError(
            f"Expected matrix sequence shape ({horizon_len}, {matrix_size}, {matrix_size}), got {arr.shape}"
        )
    return arr


def _as_matrix_sequence_or_zero(value: np.ndarray | None, horizon_len: int, matrix_size: int) -> np.ndarray:
    if value is None:
        return np.zeros((horizon_len, matrix_size, matrix_size), dtype=float)
    return _as_matrix_sequence(value, horizon_len, matrix_size)


def finite_difference_jacobians(
    dynamics,
    x: np.ndarray,
    u: np.ndarray,
    eps_x: float = 1e-5,
    eps_u: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    u = np.asarray(u, dtype=float)
    nx = x.size
    nu = u.size
    a_mat = np.zeros((nx, nx), dtype=float)
    b_mat = np.zeros((nx, nu), dtype=float)

    for i in range(nx):
        dx = np.zeros(nx, dtype=float)
        dx[i] = eps_x
        a_mat[:, i] = (dynamics(x + dx, u) - dynamics(x - dx, u)) / (2.0 * eps_x)

    for j in range(nu):
        du = np.zeros(nu, dtype=float)
        du[j] = eps_u
        b_mat[:, j] = (dynamics(x, u + du) - dynamics(x, u - du)) / (2.0 * eps_u)

    return a_mat, b_mat


def rollout_dynamics(
    dynamics,
    x0: np.ndarray,
    u_seq: np.ndarray,
    *,
    u_lower: np.ndarray | None = None,
    u_upper: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    u_use = np.asarray(u_seq, dtype=float).copy()
    if u_lower is not None:
        u_use = np.maximum(u_use, np.asarray(u_lower, dtype=float))
    if u_upper is not None:
        u_use = np.minimum(u_use, np.asarray(u_upper, dtype=float))

    horizon_steps = u_use.shape[0]
    nx = np.asarray(x0, dtype=float).size
    x_seq = np.zeros((horizon_steps + 1, nx), dtype=float)
    x_seq[0] = np.asarray(x0, dtype=float)
    for k in range(horizon_steps):
        x_seq[k + 1] = np.asarray(dynamics(x_seq[k], u_use[k]), dtype=float)
    return x_seq, u_use


def affine_linearize_trajectory(
    dynamics,
    x_seq: np.ndarray,
    u_seq: np.ndarray,
    *,
    dynamics_jacobian=None,
    fd_eps_x: float = 1e-5,
    fd_eps_u: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    horizon_steps = u_seq.shape[0]
    nx = x_seq.shape[1]
    nu = u_seq.shape[1]
    a_seq = np.zeros((horizon_steps, nx, nx), dtype=float)
    b_seq = np.zeros((horizon_steps, nx, nu), dtype=float)
    c_seq = np.zeros((horizon_steps, nx), dtype=float)

    for k in range(horizon_steps):
        xk = x_seq[k]
        uk = u_seq[k]
        if dynamics_jacobian is None:
            ak, bk = finite_difference_jacobians(dynamics, xk, uk, eps_x=fd_eps_x, eps_u=fd_eps_u)
        else:
            ak, bk = dynamics_jacobian(xk, uk)
        fk = np.asarray(dynamics(xk, uk), dtype=float)
        ck = fk - ak @ xk - bk @ uk
        a_seq[k] = ak
        b_seq[k] = bk
        c_seq[k] = ck

    return a_seq, b_seq, c_seq


def quadratic_tracking_cost(
    x_seq: np.ndarray,
    u_seq: np.ndarray,
    x_ref_seq: np.ndarray,
    u_ref_seq: np.ndarray,
    q_mat: np.ndarray,
    r_mat: np.ndarray,
    qf_mat: np.ndarray,
    x_terminal_ref: np.ndarray | None = None,
    rd_mat: np.ndarray | None = None,
) -> float:
    q_seq = _as_matrix_sequence(q_mat, u_seq.shape[0], x_seq.shape[1])
    r_seq = _as_matrix_sequence(r_mat, u_seq.shape[0], u_seq.shape[1])
    qf_use = np.asarray(qf_mat, dtype=float)
    cost = 0.0
    for k in range(u_seq.shape[0]):
        dx = x_seq[k] - x_ref_seq[k]
        du = u_seq[k] - u_ref_seq[k]
        cost += 0.5 * float(dx @ q_seq[k] @ dx + du @ r_seq[k] @ du)
        if rd_mat is not None:
            if k == 0:
                delta_u = u_seq[k] - u_ref_seq[k]
            else:
                delta_u = u_seq[k] - u_seq[k - 1]
            cost += 0.5 * float(delta_u @ rd_mat @ delta_u)
    x_terminal_use = x_ref_seq[-1] if x_terminal_ref is None else np.asarray(x_terminal_ref, dtype=float)
    dx_final = x_seq[-1] - x_terminal_use
    cost += 0.5 * float(dx_final @ qf_use @ dx_final)
    return cost


def solve_scp(
    dynamics,
    x0: np.ndarray,
    u_init: np.ndarray,
    x_ref: np.ndarray | None,
    u_ref: np.ndarray | None,
    q_mat: np.ndarray | None,
    r_mat: np.ndarray | None,
    qf_mat: np.ndarray | None,
    *,
    x_terminal_ref: np.ndarray | None = None,
    terminal_state_indices: tuple[int, ...] | list[int] | None = None,
    terminal_state_tolerance: float | np.ndarray | None = None,
    rd_mat: np.ndarray | None = None,
    u_lower: np.ndarray | None = None,
    u_upper: np.ndarray | None = None,
    x_lower: np.ndarray | None = None,
    x_upper: np.ndarray | None = None,
    trust_region_state: float | np.ndarray = 1.0,
    trust_region_input: float | np.ndarray = 1.0,
    min_trust_region_state: float | np.ndarray | None = None,
    min_trust_region_input: float | np.ndarray | None = None,
    max_trust_region_state: float | np.ndarray | None = None,
    max_trust_region_input: float | np.ndarray | None = None,
    trust_region_shrink: float = 0.5,
    trust_region_expand: float = 1.25,
    max_iter: int = 25,
    tol: float = 1e-3,
    solver: str = "OSQP",
    verbose: bool = False,
    dynamics_jacobian=None,
    fd_eps_x: float = 1e-5,
    fd_eps_u: float = 1e-5,
) -> SCPResult:
    u_seq = np.asarray(u_init, dtype=float).copy()
    x0 = np.asarray(x0, dtype=float)
    horizon_steps, nu = u_seq.shape
    nx = x0.size

    x_ref_seq = _as_array_or_default(x_ref, horizon_steps + 1, nx)
    u_ref_seq = _as_array_or_default(u_ref, horizon_steps, nu)
    q_use = _as_matrix_sequence_or_zero(q_mat, horizon_steps, nx)
    r_use = _as_matrix_sequence_or_zero(r_mat, horizon_steps, nu)
    qf_use = np.zeros((nx, nx), dtype=float) if qf_mat is None else np.asarray(qf_mat, dtype=float)
    x_terminal_use = x_ref_seq[-1] if x_terminal_ref is None else np.asarray(x_terminal_ref, dtype=float)
    terminal_idx = None if terminal_state_indices is None else np.asarray(tuple(int(i) for i in terminal_state_indices), dtype=int)
    if terminal_state_tolerance is None:
        terminal_tol = None
    else:
        if terminal_idx is None:
            raise ValueError("terminal_state_tolerance requires terminal_state_indices")
        terminal_tol = _expand_trust_region(terminal_state_tolerance, terminal_idx.size)
    rd_use = None if rd_mat is None else np.asarray(rd_mat, dtype=float)
    rho_x = _expand_trust_region(trust_region_state, nx)
    rho_u = _expand_trust_region(trust_region_input, nu)
    rho_x_min = (
        np.maximum(1.0e-8, 0.1 * rho_x)
        if min_trust_region_state is None
        else _expand_trust_region(min_trust_region_state, nx)
    )
    rho_u_min = (
        np.maximum(1.0e-8, 0.1 * rho_u)
        if min_trust_region_input is None
        else _expand_trust_region(min_trust_region_input, nu)
    )
    rho_x_max = rho_x.copy() if max_trust_region_state is None else _expand_trust_region(max_trust_region_state, nx)
    rho_u_max = rho_u.copy() if max_trust_region_input is None else _expand_trust_region(max_trust_region_input, nu)

    x_seq, u_seq = rollout_dynamics(dynamics, x0, u_seq, u_lower=u_lower, u_upper=u_upper)
    cost_history = [
        quadratic_tracking_cost(x_seq, u_seq, x_ref_seq, u_ref_seq, q_use, r_use, qf_use, x_terminal_use, rd_use)
    ]
    if not np.isfinite(cost_history[-1]):
        raise RuntimeError("Initial SCP rollout produced a non-finite cost. Check the initial guess and constraints.")
    converged = False
    termination_reason = "max_iterations_reached"

    u_lower_arr = None if u_lower is None else np.asarray(u_lower, dtype=float)
    u_upper_arr = None if u_upper is None else np.asarray(u_upper, dtype=float)
    x_lower_arr = None if x_lower is None else np.asarray(x_lower, dtype=float)
    x_upper_arr = None if x_upper is None else np.asarray(x_upper, dtype=float)

    for iteration in range(max_iter):
        if verbose:
            print(
                f"[SCP] iteration {iteration + 1}/{max_iter} | "
                f"cost={cost_history[-1]:.6f} | "
                f"rho_x={rho_x} | rho_u={rho_u}"
            )
        accepted = False
        attempt = 0
        while not accepted:
            attempt += 1
            a_seq, b_seq, c_seq = affine_linearize_trajectory(
                dynamics,
                x_seq,
                u_seq,
                dynamics_jacobian=dynamics_jacobian,
                fd_eps_x=fd_eps_x,
                fd_eps_u=fd_eps_u,
            )

            x_cvx = cvx.Variable((horizon_steps + 1, nx))
            u_cvx = cvx.Variable((horizon_steps, nu))
            objective = 0.0
            constraints = [x_cvx[0] == x0]

            for k in range(horizon_steps):
                objective += cvx.quad_form(x_cvx[k] - x_ref_seq[k], q_use[k])
                objective += cvx.quad_form(u_cvx[k] - u_ref_seq[k], r_use[k])
                if rd_use is not None:
                    if k == 0:
                        objective += cvx.quad_form(u_cvx[k] - u_ref_seq[k], rd_use)
                    else:
                        objective += cvx.quad_form(u_cvx[k] - u_cvx[k - 1], rd_use)

                constraints.append(x_cvx[k + 1] == a_seq[k] @ x_cvx[k] + b_seq[k] @ u_cvx[k] + c_seq[k])
                constraints.append(cvx.abs(x_cvx[k] - x_seq[k]) <= rho_x)
                constraints.append(cvx.abs(u_cvx[k] - u_seq[k]) <= rho_u)

                if u_lower_arr is not None:
                    constraints.append(u_cvx[k] >= u_lower_arr)
                if u_upper_arr is not None:
                    constraints.append(u_cvx[k] <= u_upper_arr)
                if x_lower_arr is not None:
                    constraints.append(x_cvx[k] >= x_lower_arr)
                if x_upper_arr is not None:
                    constraints.append(x_cvx[k] <= x_upper_arr)

            if terminal_idx is None or terminal_idx.size == 0:
                constraints.append(cvx.abs(x_cvx[horizon_steps] - x_seq[horizon_steps]) <= rho_x)
            else:
                terminal_mask = np.ones(nx, dtype=bool)
                terminal_mask[terminal_idx] = False
                if np.any(terminal_mask):
                    constraints.append(
                        cvx.abs(x_cvx[horizon_steps, terminal_mask] - x_seq[horizon_steps, terminal_mask])
                        <= rho_x[terminal_mask]
                    )
            if x_lower_arr is not None:
                constraints.append(x_cvx[horizon_steps] >= x_lower_arr)
            if x_upper_arr is not None:
                constraints.append(x_cvx[horizon_steps] <= x_upper_arr)
            if terminal_idx is not None:
                if terminal_tol is None:
                    constraints.append(x_cvx[horizon_steps, terminal_idx] == x_terminal_use[terminal_idx])
                else:
                    constraints.append(cvx.abs(x_cvx[horizon_steps, terminal_idx] - x_terminal_use[terminal_idx]) <= terminal_tol)
            objective += cvx.quad_form(x_cvx[horizon_steps] - x_terminal_use, qf_use)

            prob = cvx.Problem(cvx.Minimize(objective), constraints)
            prob.solve(solver=solver, warm_start=True, verbose=False)
            if prob.status not in {"optimal", "optimal_inaccurate"}:
                can_shrink = np.any(rho_x > rho_x_min * (1.0 + 1.0e-9)) or np.any(rho_u > rho_u_min * (1.0 + 1.0e-9))
                if not can_shrink:
                    termination_reason = f"subproblem_{prob.status}"
                    accepted = True
                    if verbose:
                        print(
                            f"[SCP]   terminating with last feasible iterate after subproblem status={prob.status}"
                        )
                    break
                rho_x = np.maximum(rho_x_min, trust_region_shrink * rho_x)
                rho_u = np.maximum(rho_u_min, trust_region_shrink * rho_u)
                if verbose:
                    print(
                        f"[SCP]   rejected subproblem status={prob.status}; "
                        f"shrinking trust region to rho_x={rho_x}, rho_u={rho_u}"
                    )
                continue

            u_candidate = np.asarray(u_cvx.value, dtype=float)
            x_rollout, u_rollout = rollout_dynamics(dynamics, x0, u_candidate, u_lower=u_lower_arr, u_upper=u_upper_arr)
            rollout_finite = np.all(np.isfinite(x_rollout)) and np.all(np.isfinite(u_rollout))
            new_cost = (
                quadratic_tracking_cost(x_rollout, u_rollout, x_ref_seq, u_ref_seq, q_use, r_use, qf_use, x_terminal_use, rd_use)
                if rollout_finite
                else float("inf")
            )
            improvement = cost_history[-1] - new_cost
            cost_finite = np.isfinite(new_cost)

            if verbose:
                cost_text = f"{new_cost:.6f}" if cost_finite else "non-finite"
                improvement_text = f"{improvement:.6f}" if np.isfinite(improvement) else "non-finite"
                print(
                    f"[SCP]   attempt {attempt} | candidate_cost={cost_text} | "
                    f"improvement={improvement_text}"
                )

            if rollout_finite and cost_finite and improvement >= 0.0:
                cost_history.append(new_cost)
                x_seq = x_rollout
                u_seq = u_rollout
                accepted = True
                if improvement > 10.0 * tol:
                    rho_x = np.minimum(rho_x_max, trust_region_expand * rho_x)
                    rho_u = np.minimum(rho_u_max, trust_region_expand * rho_u)
                if improvement < tol:
                    converged = True
                    termination_reason = "converged_tol"
                break

            can_shrink = np.any(rho_x > rho_x_min * (1.0 + 1.0e-9)) or np.any(rho_u > rho_u_min * (1.0 + 1.0e-9))
            if not can_shrink:
                reason = (
                    "non-finite nonlinear rollout"
                    if not rollout_finite
                    else "non-finite candidate cost"
                    if not cost_finite
                    else "no improving candidate found"
                )
                termination_reason = "minimum_trust_region_after_" + reason.replace(" ", "_").replace("-", "_")
                accepted = True
                if verbose:
                    print(
                        "[SCP]   terminating with last feasible iterate because "
                        f"{reason} persisted at the minimum trust region"
                    )
                break
            rho_x = np.maximum(rho_x_min, trust_region_shrink * rho_x)
            rho_u = np.maximum(rho_u_min, trust_region_shrink * rho_u)
            if verbose:
                print(
                    f"[SCP]   rejecting candidate; shrinking trust region to "
                    f"rho_x={rho_x}, rho_u={rho_u}"
                )

        if converged or termination_reason != "max_iterations_reached":
            break

    return SCPResult(
        x_seq=x_seq,
        u_seq=u_seq,
        cost_history=np.asarray(cost_history, dtype=float),
        converged=converged,
        iterations=len(cost_history) - 1,
        trust_region_state=rho_x,
        trust_region_input=rho_u,
        termination_reason=termination_reason,
    )
