from __future__ import annotations

import math

import numpy as np
from scipy.optimize import minimize

from .dynamics import longitudinal_state_derivative
from .types import TrimResult, VehicleParameters


def longitudinal_state_from_speed_alpha(speed_mps: float, alpha_rad: float, theta_rad: float) -> np.ndarray:
    u = speed_mps * math.cos(alpha_rad)
    w = speed_mps * math.sin(alpha_rad)
    return np.array([0.0, 0.0, u, w, theta_rad, 0.0], dtype=float)


def solve_longitudinal_trim(
    vehicle: VehicleParameters,
    speed_mps: float = 10.0,
    flight_path_angle_rad: float = 0.0,
    rpm_guess: float = 9800.0,
    elevator_guess_rad: float = 0.0,
) -> TrimResult:
    """Solve a steady longitudinal trim point using a small nonlinear program."""

    def objective(z: np.ndarray) -> float:
        alpha, rpm, de = z
        theta = alpha + flight_path_angle_rad
        state = longitudinal_state_from_speed_alpha(speed_mps, alpha, theta)
        control = np.array([rpm, de], dtype=float)
        dx = longitudinal_state_derivative(state, control, vehicle)
        resid = np.array([dx[2], dx[3], dx[5]], dtype=float)
        return float(
            resid @ np.diag([1.0, 1.0, 10.0]) @ resid
            + 1e-8 * (rpm - rpm_guess) ** 2
            + 1e-3 * de**2
        )

    bounds = [
        (math.radians(-8.0), math.radians(15.0)),
        (vehicle.propulsion.rpm_grid.min(), vehicle.propulsion.rpm_grid.max()),
        (-vehicle.elevator_limit_rad, vehicle.elevator_limit_rad),
    ]
    x0 = np.array([math.radians(2.0), rpm_guess, elevator_guess_rad], dtype=float)
    result = minimize(objective, x0=x0, bounds=bounds, method="SLSQP", options={"disp": False, "maxiter": 200})

    alpha, rpm, de = result.x
    theta = alpha + flight_path_angle_rad
    state = longitudinal_state_from_speed_alpha(speed_mps, alpha, theta)
    control = np.array([rpm, de], dtype=float)
    dx = longitudinal_state_derivative(state, control, vehicle)
    residual = np.array([dx[2], dx[3], dx[5]], dtype=float)

    return TrimResult(
        airspeed_mps=speed_mps,
        alpha_rad=float(alpha),
        theta_rad=float(theta),
        rpm=float(rpm),
        elevator_rad=float(de),
        state=state,
        control=control,
        residual=residual,
        success=bool(result.success),
        message=str(result.message),
    )
