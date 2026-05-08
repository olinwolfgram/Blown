from __future__ import annotations

import math

import numpy as np

from .geometry import Vehicle
from .propulsion import collective_rpm_to_full_control
from .rigid_body_ac import total_forces_and_moments


def longitudinal_state_derivative(x_lon: np.ndarray, u_lon: np.ndarray, vehicle: Vehicle) -> np.ndarray:
    """Reduced longitudinal dynamics.

    State:
    [x, h, u, w, theta, q]

    Control:
    [rpm_collective, delta_e, delta_f]
    """

    x_fwd, h, u, w, theta, q = np.asarray(x_lon, dtype=float)
    rpm_collective, delta_e, delta_f = np.asarray(u_lon, dtype=float)

    full_state = np.array([x_fwd, 0.0, -h, u, 0.0, w, 0.0, theta, 0.0, 0.0, q, 0.0], dtype=float)
    full_control = collective_rpm_to_full_control(
        rpm_collective,
        np.array([delta_e, 0.0, 0.0, delta_f], dtype=float),
        vehicle,
    )

    force, moment, _ = total_forces_and_moments(full_state, full_control, vehicle)
    u_dot = force[0] / vehicle.mass_kg - q * w
    w_dot = force[2] / vehicle.mass_kg + q * u
    q_dot = moment[1] / vehicle.inertia[1, 1]
    theta_dot = q
    x_dot = u * math.cos(theta) + w * math.sin(theta)
    h_dot = u * math.sin(theta) - w * math.cos(theta)
    return np.array([x_dot, h_dot, u_dot, w_dot, theta_dot, q_dot], dtype=float)
