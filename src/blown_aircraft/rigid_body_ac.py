from __future__ import annotations

import math

import numpy as np

from .aerodynamics import aerodynamic_forces_and_moments
from .geometry import Vehicle
from .propulsion import propulsion_forces_and_moments


def gravity_force_body(phi: float, theta: float, vehicle: Vehicle) -> np.ndarray:
    mg = vehicle.mass_kg * vehicle.gravity_mps2
    return np.array(
        [
            -mg * math.sin(theta),
            mg * math.sin(phi) * math.cos(theta),
            mg * math.cos(phi) * math.cos(theta),
        ],
        dtype=float,
    )


def body_to_ned_matrix(phi: float, theta: float, psi: float) -> np.ndarray:
    cphi, sphi = math.cos(phi), math.sin(phi)
    cth, sth = math.cos(theta), math.sin(theta)
    cps, sps = math.cos(psi), math.sin(psi)
    return np.array(
        [
            [cth * cps, sphi * sth * cps - cphi * sps, cphi * sth * cps + sphi * sps],
            [cth * sps, sphi * sth * sps + cphi * cps, cphi * sth * sps - sphi * cps],
            [-sth, sphi * cth, cphi * cth],
        ],
        dtype=float,
    )


def euler_rate_matrix(phi: float, theta: float) -> np.ndarray:
    tan_theta = math.tan(theta)
    sec_theta = 1.0 / max(math.cos(theta), 1e-6)
    return np.array(
        [
            [1.0, math.sin(phi) * tan_theta, math.cos(phi) * tan_theta],
            [0.0, math.cos(phi), -math.sin(phi)],
            [0.0, math.sin(phi) * sec_theta, math.cos(phi) * sec_theta],
        ],
        dtype=float,
    )


def total_forces_and_moments(
    state: np.ndarray,
    control: np.ndarray,
    vehicle: Vehicle,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Sum aero, propulsion, and gravity.

    State ordering:
    [pn, pe, pd, u, v, w, phi, theta, psi, p, q, r]

    Control ordering:
    [rpm_1, ..., rpm_10, delta_e, delta_a, delta_r, delta_f]
    """

    phi = float(state[6])
    theta = float(state[7])

    faero, maero, aero_diag = aerodynamic_forces_and_moments(state, control, vehicle)
    fprop, mprop, prop_diag = propulsion_forces_and_moments(control, vehicle)
    fgrav = gravity_force_body(phi, theta, vehicle)

    total_force = faero + fprop + fgrav
    total_moment = maero + mprop
    return total_force, total_moment, {"aero": aero_diag, "propulsion": prop_diag, "gravity_force_body": fgrav}


def full_state_derivative(state: np.ndarray, control: np.ndarray, vehicle: Vehicle) -> np.ndarray:
    pn, pe, pd, u, v, w, phi, theta, psi, p, q, r = np.asarray(state, dtype=float)
    force, moment, _ = total_forces_and_moments(state, control, vehicle)

    vel_body = np.array([u, v, w], dtype=float)
    omega = np.array([p, q, r], dtype=float)

    pos_dot = body_to_ned_matrix(phi, theta, psi) @ vel_body
    vel_dot = force / vehicle.mass_kg - np.cross(omega, vel_body)
    euler_dot = euler_rate_matrix(phi, theta) @ omega
    omega_dot = np.linalg.solve(vehicle.inertia, moment - np.cross(omega, vehicle.inertia @ omega))

    return np.concatenate([pos_dot, vel_dot, euler_dot, omega_dot])
