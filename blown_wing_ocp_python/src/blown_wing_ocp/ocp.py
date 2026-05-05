from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds, NonlinearConstraint, minimize

from .dynamics import longitudinal_state_derivative
from .types import OCPResult, TrimResult, VehicleParameters


@dataclass(frozen=True)
class LongitudinalOCPConfig:
    horizon_s: float = 6.0
    nodes: int = 21
    target_dx_m: float = 45.0
    target_dh_m: float = 8.0
    target_speed_mps: float = 10.5
    target_theta_rad: float = math.radians(2.0)
    q_track: tuple[float, ...] = (0.02, 0.06, 1.0, 1.0, 4.0, 2.0)
    r_track: tuple[float, ...] = (1e-7, 0.5)
    r_rate: tuple[float, ...] = (5e-8, 0.2)
    terminal_weight: tuple[float, ...] = (0.1, 6.0, 4.0, 4.0, 20.0, 8.0)
    alpha_limit_rad: float = math.radians(14.0)


def build_reference(trim: TrimResult, cfg: LongitudinalOCPConfig) -> tuple[np.ndarray, np.ndarray]:
    t = np.linspace(0.0, cfg.horizon_s, cfg.nodes)
    x_ref = np.zeros((6, cfg.nodes), dtype=float)
    x_ref[0, :] = trim.state[0] + np.linspace(0.0, cfg.target_dx_m, cfg.nodes)
    x_ref[1, :] = trim.state[1] + np.linspace(0.0, cfg.target_dh_m, cfg.nodes)
    x_ref[2, :] = np.linspace(trim.state[2], cfg.target_speed_mps, cfg.nodes)
    x_ref[3, :] = 0.0
    x_ref[4, :] = np.linspace(trim.theta_rad, cfg.target_theta_rad, cfg.nodes)
    x_ref[5, :] = 0.0
    return t, x_ref


def solve_longitudinal_ocp(
    vehicle: VehicleParameters,
    trim: TrimResult,
    cfg: LongitudinalOCPConfig | None = None,
) -> OCPResult:
    if cfg is None:
        cfg = LongitudinalOCPConfig()

    nx = 6
    nu = 2
    n = cfg.nodes
    dt = cfg.horizon_s / (n - 1)
    t_s, x_ref = build_reference(trim, cfg)
    u_ref = np.tile(trim.control.reshape(-1, 1), (1, n - 1))

    x0_guess = x_ref.copy()
    x0_guess[:, 0] = trim.state
    u0_guess = u_ref.copy()
    z0 = np.concatenate([x0_guess.reshape(-1, order="F"), u0_guess.reshape(-1, order="F")])

    x_lb = np.tile(
        np.array(
            [
                -10.0,
                -2.0,
                4.0,
                -4.0,
                math.radians(-10.0),
                math.radians(-60.0),
            ],
            dtype=float,
        ),
        n,
    )
    x_ub = np.tile(
        np.array(
            [
                cfg.target_dx_m + 20.0,
                cfg.target_dh_m + 20.0,
                20.0,
                4.0,
                math.radians(15.0),
                math.radians(60.0),
            ],
            dtype=float,
        ),
        n,
    )
    u_lb = np.tile(
        np.array([vehicle.propulsion.rpm_grid.min(), -vehicle.elevator_limit_rad], dtype=float),
        n - 1,
    )
    u_ub = np.tile(
        np.array([vehicle.propulsion.rpm_grid.max(), vehicle.elevator_limit_rad], dtype=float),
        n - 1,
    )
    bounds = Bounds(np.concatenate([x_lb, u_lb]), np.concatenate([x_ub, u_ub]))

    q_mat = np.diag(np.asarray(cfg.q_track, dtype=float))
    qf_mat = np.diag(np.asarray(cfg.terminal_weight, dtype=float))
    r_mat = np.diag(np.asarray(cfg.r_track, dtype=float))
    rr_mat = np.diag(np.asarray(cfg.r_rate, dtype=float))

    def unpack(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x_flat = z[: nx * n]
        u_flat = z[nx * n :]
        x_hist = x_flat.reshape((nx, n), order="F")
        u_hist = u_flat.reshape((nu, n - 1), order="F")
        return x_hist, u_hist

    progress = {"iter": 0, "last_cost": float("nan")}

    def objective(z: np.ndarray) -> float:
        x_hist, u_hist = unpack(z)
        cost = 0.0
        for k in range(n - 1):
            dx = x_hist[:, k] - x_ref[:, k]
            du = u_hist[:, k] - trim.control
            cost += float(dx @ q_mat @ dx + du @ r_mat @ du)
            if k > 0:
                dru = (u_hist[:, k] - u_hist[:, k - 1]) / dt
                cost += float(dru @ rr_mat @ dru)

            alpha = math.atan2(x_hist[3, k], max(x_hist[2, k], 1e-6))
            if abs(alpha) > cfg.alpha_limit_rad:
                cost += 1e4 * (abs(alpha) - cfg.alpha_limit_rad) ** 2

        dx_terminal = x_hist[:, -1] - x_ref[:, -1]
        cost += float(dx_terminal @ qf_mat @ dx_terminal)
        progress["last_cost"] = cost
        return cost

    def dynamics_residual(z: np.ndarray) -> np.ndarray:
        x_hist, u_hist = unpack(z)
        residuals = [x_hist[:, 0] - trim.state]
        for k in range(n - 1):
            f_k = longitudinal_state_derivative(x_hist[:, k], u_hist[:, k], vehicle)
            residuals.append(x_hist[:, k + 1] - x_hist[:, k] - dt * f_k)
        return np.concatenate(residuals)

    def alpha_residual(z: np.ndarray) -> np.ndarray:
        x_hist, _ = unpack(z)
        alphas = np.arctan2(x_hist[3, :], np.maximum(x_hist[2, :], 1e-6))
        return np.abs(alphas) - cfg.alpha_limit_rad

    eq_constraint = NonlinearConstraint(dynamics_residual, 0.0, 0.0)
    ineq_constraint = NonlinearConstraint(alpha_residual, -np.inf, 0.0)

    print(
        f"Starting longitudinal OCP: nodes={n}, horizon={cfg.horizon_s:.1f} s, "
        f"decision_vars={z0.size}, target_dx={cfg.target_dx_m:.1f} m, target_dh={cfg.target_dh_m:.1f} m"
    )

    def callback(_xk: np.ndarray) -> None:
        progress["iter"] += 1
        print(f"  iter {progress['iter']:03d} | objective {progress['last_cost']:.6g}")

    result = minimize(
        objective,
        x0=z0,
        method="SLSQP",
        bounds=bounds,
        constraints=[eq_constraint, ineq_constraint],
        callback=callback,
        options={"maxiter": 80, "ftol": 1e-5, "disp": True},
    )

    x_hist, u_hist = unpack(result.x)
    return OCPResult(
        t_s=t_s,
        x_hist=x_hist,
        u_hist=u_hist,
        objective=float(result.fun),
        success=bool(result.success),
        message=str(result.message),
        solver_output={"nit": result.nit, "status": result.status},
    )
