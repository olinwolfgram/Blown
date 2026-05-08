from __future__ import annotations

import numpy as np


def linearize(model, x0: np.ndarray, u0: np.ndarray, dt: float | None = None, eps_x: float = 1e-6, eps_u: float = 1e-6) -> dict[str, np.ndarray]:
    x0 = np.asarray(x0, dtype=float)
    u0 = np.asarray(u0, dtype=float)
    nx = x0.size
    nu = u0.size
    a_mat = np.zeros((nx, nx), dtype=float)
    b_mat = np.zeros((nx, nu), dtype=float)
    f0 = np.asarray(model(x0, u0), dtype=float)

    for i in range(nx):
        dx = np.zeros(nx, dtype=float)
        dx[i] = eps_x
        a_mat[:, i] = (model(x0 + dx, u0) - model(x0 - dx, u0)) / (2.0 * eps_x)

    for j in range(nu):
        du = np.zeros(nu, dtype=float)
        du[j] = eps_u
        b_mat[:, j] = (model(x0, u0 + du) - model(x0, u0 - du)) / (2.0 * eps_u)

    result = {"A": a_mat, "B": b_mat, "f0": f0}
    if dt is not None:
        result["Ad"] = np.eye(nx) + a_mat * dt
        result["Bd"] = b_mat * dt
    return result
