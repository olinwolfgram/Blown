from __future__ import annotations

import math

import numpy as np

from .geometry import Vehicle
from .propulsion import split_rpm_to_full_control
from .rigid_body_ac import total_forces_and_moments


def lateral_state_derivative(x_lat: np.ndarray, u_lat: np.ndarray, vehicle: Vehicle) -> np.ndarray:
    """Reduced lateral-directional dynamics.

    State:
    [y, v, phi, psi, p, r]

    Control:
    [rpm_left, rpm_right, delta_a, delta_r]
    """

    y_pos, v, phi, psi, p, r = np.asarray(x_lat, dtype=float)
    rpm_left, rpm_right, delta_a, delta_r = np.asarray(u_lat, dtype=float)

    u_trim = 10.0
    full_state = np.array([0.0, y_pos, 0.0, u_trim, v, 0.0, phi, 0.0, psi, p, 0.0, r], dtype=float)
    full_control = split_rpm_to_full_control(
        rpm_left,
        rpm_right,
        np.array([0.0, delta_a, delta_r, 0.0], dtype=float),
        vehicle,
    )

    force, moment, _ = total_forces_and_moments(full_state, full_control, vehicle)
    y_dot = u_trim * math.sin(psi) + v * math.cos(psi)
    v_dot = force[1] / vehicle.mass_kg - r * u_trim
    phi_dot = p
    psi_dot = r
    p_dot = moment[0] / vehicle.inertia[0, 0]
    r_dot = moment[2] / vehicle.inertia[2, 2]
    return np.array([y_dot, v_dot, phi_dot, psi_dot, p_dot, r_dot], dtype=float)
