from __future__ import annotations

import math

import numpy as np

from .geometry import Vehicle
from .propulsion import split_rpm_to_full_control
from .rigid_body_ac import total_forces_and_moments


def lateral_state_derivative(
    x_lat: np.ndarray,
    u_lat: np.ndarray,
    vehicle: Vehicle,
    *,
    w_trim_mps: float = 0.0,
    theta_trim_rad: float = 0.0,
) -> np.ndarray:
    """Planar lateral-directional dynamics with dynamic forward speed.

    State:
    [x, y, u, v, phi, psi, p, r]

    Control:
    [rpm_left, rpm_right, delta_a, delta_r]
    """

    x_pos, y_pos, u, v, phi, psi, p, r = np.asarray(x_lat, dtype=float)
    rpm_left, rpm_right, delta_a, delta_r = np.asarray(u_lat, dtype=float)

    full_state = np.array([x_pos, y_pos, 0.0, u, v, w_trim_mps, phi, theta_trim_rad, psi, p, 0.0, r], dtype=float)
    full_control = split_rpm_to_full_control(
        rpm_left,
        rpm_right,
        np.array([0.0, delta_a, delta_r, 0.0], dtype=float),
        vehicle,
    )

    force, moment, _ = total_forces_and_moments(full_state, full_control, vehicle)
    x_dot = u * math.cos(psi) - v * math.sin(psi)
    y_dot = u * math.sin(psi) + v * math.cos(psi)
    u_dot = force[0] / vehicle.mass_kg + r * v
    v_dot = force[1] / vehicle.mass_kg - r * u
    phi_dot = p
    psi_dot = r
    p_dot = moment[0] / vehicle.inertia[0, 0]
    r_dot = moment[2] / vehicle.inertia[2, 2]
    return np.array([x_dot, y_dot, u_dot, v_dot, phi_dot, psi_dot, p_dot, r_dot], dtype=float)
