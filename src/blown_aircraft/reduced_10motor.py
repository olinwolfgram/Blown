from __future__ import annotations

import numpy as np

from .geometry import Vehicle
from .rigid_body_ac import total_forces_and_moments


def longitudinal_state_derivative_10motor(
    x_lon: np.ndarray,
    u_lon: np.ndarray,
    vehicle: Vehicle,
    *,
    flap_trim_rad: float,
) -> np.ndarray:
    """Reduced longitudinal dynamics with individual motor RPM inputs.

    State:
    [x, h, u, w, theta, q]

    Control:
    [rpm_1, ..., rpm_10, delta_e]
    """

    x_fwd, h, u, w, theta, q = np.asarray(x_lon, dtype=float)
    u_lon = np.asarray(u_lon, dtype=float)
    n_props = int(vehicle.propulsion["n_props"])
    rpm_vec = u_lon[:n_props]
    delta_e = float(u_lon[n_props])

    full_state = np.array([x_fwd, 0.0, -h, u, 0.0, w, 0.0, theta, 0.0, 0.0, q, 0.0], dtype=float)
    full_control = np.concatenate(
        [rpm_vec, np.array([delta_e, 0.0, 0.0, flap_trim_rad], dtype=float)],
        dtype=float,
    )

    force, moment, _ = total_forces_and_moments(full_state, full_control, vehicle)
    u_dot = force[0] / vehicle.mass_kg - q * w
    w_dot = force[2] / vehicle.mass_kg + q * u
    q_dot = moment[1] / vehicle.inertia[1, 1]
    theta_dot = q
    x_dot = u * np.cos(theta) + w * np.sin(theta)
    h_dot = u * np.sin(theta) - w * np.cos(theta)
    return np.array([x_dot, h_dot, u_dot, w_dot, theta_dot, q_dot], dtype=float)


def lateral_state_derivative_10motor(
    x_lat: np.ndarray,
    u_lat: np.ndarray,
    vehicle: Vehicle,
    *,
    w_trim_mps: float,
    theta_trim_rad: float,
    elevator_trim_rad: float,
    flap_trim_rad: float,
) -> np.ndarray:
    """Planar lateral dynamics with individual motor RPM inputs.

    State:
    [x, y, u, v, phi, psi, p, r]

    Control:
    [rpm_1, ..., rpm_10, delta_a, delta_r]
    """

    x_pos, y_pos, u, v, phi, psi, p, r = np.asarray(x_lat, dtype=float)
    u_lat = np.asarray(u_lat, dtype=float)
    n_props = int(vehicle.propulsion["n_props"])
    rpm_vec = u_lat[:n_props]
    delta_a = float(u_lat[n_props])
    delta_r = float(u_lat[n_props + 1])

    full_state = np.array([x_pos, y_pos, 0.0, u, v, w_trim_mps, phi, theta_trim_rad, psi, p, 0.0, r], dtype=float)
    full_control = np.concatenate(
        [rpm_vec, np.array([elevator_trim_rad, delta_a, delta_r, flap_trim_rad], dtype=float)],
        dtype=float,
    )

    force, moment, _ = total_forces_and_moments(full_state, full_control, vehicle)
    x_dot = u * np.cos(psi) - v * np.sin(psi)
    y_dot = u * np.sin(psi) + v * np.cos(psi)
    u_dot = force[0] / vehicle.mass_kg + r * v
    v_dot = force[1] / vehicle.mass_kg - r * u
    phi_dot = p
    psi_dot = r
    p_dot = moment[0] / vehicle.inertia[0, 0]
    r_dot = moment[2] / vehicle.inertia[2, 2]
    return np.array([x_dot, y_dot, u_dot, v_dot, phi_dot, psi_dot, p_dot, r_dot], dtype=float)

