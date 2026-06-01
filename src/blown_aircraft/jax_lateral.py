from __future__ import annotations

import numpy as np

from .lateral import lateral_state_derivative


def _finite_difference_jacobians(
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


def build_jax_lateral_dynamics(vehicle, dt: float, *, w_trim_mps: float = 0.0, theta_trim_rad: float = 0.0):
    """Discrete-time planar lateral dynamics interface.

    The lateral model now uses the full planar state [x, y, u, v, phi, psi, p, r]
    with dynamic forward speed. The underlying aerodynamics stack is NumPy-based,
    so the Jacobian callback currently uses finite differences while preserving the
    same interface expected by the SCP/LQR demos.
    """

    def rk4_step(x_lat: np.ndarray, u_lat: np.ndarray) -> np.ndarray:
        k1 = lateral_state_derivative(x_lat, u_lat, vehicle, w_trim_mps=w_trim_mps, theta_trim_rad=theta_trim_rad)
        k2 = lateral_state_derivative(x_lat + 0.5 * dt * k1, u_lat, vehicle, w_trim_mps=w_trim_mps, theta_trim_rad=theta_trim_rad)
        k3 = lateral_state_derivative(x_lat + 0.5 * dt * k2, u_lat, vehicle, w_trim_mps=w_trim_mps, theta_trim_rad=theta_trim_rad)
        k4 = lateral_state_derivative(x_lat + dt * k3, u_lat, vehicle, w_trim_mps=w_trim_mps, theta_trim_rad=theta_trim_rad)
        return x_lat + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def dynamics_np(x_lat: np.ndarray, u_lat: np.ndarray) -> np.ndarray:
        return np.asarray(rk4_step(np.asarray(x_lat, dtype=float), np.asarray(u_lat, dtype=float)), dtype=float)

    def jacobian_np(x_lat: np.ndarray, u_lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return _finite_difference_jacobians(dynamics_np, x_lat, u_lat)

    return dynamics_np, jacobian_np
