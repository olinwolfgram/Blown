from __future__ import annotations

import math

import numpy as np
from scipy.optimize import minimize

from .geometry import Vehicle
from .longitudinal import longitudinal_state_derivative


def solve_longitudinal_trim(vehicle: Vehicle, speed_mps: float, flight_path_angle_rad: float = 0.0) -> dict:
    """Solve a steady longitudinal operating point.

    This is the right first optimization layer for getting to trim.
    It is much lighter than SCP because trim is a static force/moment balance problem.
    """

    def make_state(alpha: float) -> np.ndarray:
        theta = alpha + flight_path_angle_rad
        u = speed_mps * math.cos(alpha)
        w = speed_mps * math.sin(alpha)
        return np.array([0.0, 0.0, u, w, theta, 0.0], dtype=float)

    def objective(z: np.ndarray) -> float:
        alpha, rpm, de, df = z
        x = make_state(alpha)
        u_ctrl = np.array([rpm, de, df], dtype=float)
        dx = longitudinal_state_derivative(x, u_ctrl, vehicle)
        resid = np.array([dx[2], dx[3], dx[5]], dtype=float)
        return float(resid @ np.diag([1.0, 1.0, 10.0]) @ resid)

    lim = vehicle.control_limits_rad
    bounds = [
        (math.radians(-10.0), math.radians(15.0)),
        (3200.0, 12440.0),
        (-lim["elevator"], lim["elevator"]),
        (0.0, lim["flap"]),
    ]
    guess = np.array([math.radians(2.0), 9500.0, 0.0, 0.0], dtype=float)
    result = minimize(objective, guess, method="SLSQP", bounds=bounds, options={"maxiter": 200, "disp": False})
    alpha, rpm, de, df = result.x
    state = make_state(alpha)
    control = np.array([rpm, de, df], dtype=float)
    residual = longitudinal_state_derivative(state, control, vehicle)[2:]
    return {
        "success": bool(result.success),
        "message": str(result.message),
        "alpha_rad": float(alpha),
        "theta_rad": float(alpha + flight_path_angle_rad),
        "state": state,
        "control": control,
        "residual": residual,
    }
