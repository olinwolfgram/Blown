from __future__ import annotations

import math

import numpy as np

from .aero import aero_forces_and_moments_body
from .propulsion import total_thrust_n
from .types import VehicleParameters


def gravity_body(phi: float, theta: float, vehicle: VehicleParameters) -> np.ndarray:
    """Return the body-axis gravity force vector.

    Reference:
    - Nelson, Flight Stability and Automatic Control
    """

    mg = vehicle.mass_kg * vehicle.gravity_mps2
    return np.array(
        [
            -mg * math.sin(theta),
            mg * math.sin(phi) * math.cos(theta),
            mg * math.cos(phi) * math.cos(theta),
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


def full_state_derivative(state: np.ndarray, control: np.ndarray, vehicle: VehicleParameters) -> np.ndarray:
    """Return 6DOF rigid-body state derivatives.

    State ordering:
    [pn, pe, pd, u, v, w, phi, theta, psi, p, q, r]

    Control ordering:
    [rpm, elevator, aileron, rudder, flap]
    """

    pn, pe, pd, u, v, w, phi, theta, psi, p, q, r = state
    rpm = control[0]

    faero, maero, coeffs = aero_forces_and_moments_body(u, v, w, p, q, r, control, vehicle)
    fgrav = gravity_body(phi, theta, vehicle)
    fprop = np.array([total_thrust_n(rpm, vehicle), 0.0, 0.0], dtype=float)

    forces = faero + fgrav + fprop
    moments = maero

    omega = np.array([p, q, r], dtype=float)
    vel = np.array([u, v, w], dtype=float)
    i_mat = np.array(
        [
            [vehicle.ixx_kgm2, 0.0, -vehicle.ixz_kgm2],
            [0.0, vehicle.iyy_kgm2, 0.0],
            [-vehicle.ixz_kgm2, 0.0, vehicle.izz_kgm2],
        ],
        dtype=float,
    )

    pos_dot = body_to_ned_matrix(phi, theta, psi) @ vel
    vel_dot = forces / vehicle.mass_kg - np.cross(omega, vel)
    omega_dot = np.linalg.solve(i_mat, moments - np.cross(omega, i_mat @ omega))
    euler_dot = euler_rate_matrix(phi, theta) @ omega

    return np.concatenate([pos_dot, vel_dot, euler_dot, omega_dot])


def longitudinal_state_derivative(state: np.ndarray, control: np.ndarray, vehicle: VehicleParameters) -> np.ndarray:
    """Return reduced longitudinal dynamics.

    State ordering:
    [x_fwd, h, u, w, theta, q]

    Control ordering:
    [rpm, elevator]
    """

    x_fwd, h, u, w, theta, q = state
    rpm, de = control
    full_control = np.array([rpm, de, 0.0, 0.0, 0.0], dtype=float)

    faero, maero, _ = aero_forces_and_moments_body(u, 0.0, w, 0.0, q, 0.0, full_control, vehicle)
    fgrav = gravity_body(0.0, theta, vehicle)
    fprop = np.array([total_thrust_n(rpm, vehicle), 0.0, 0.0], dtype=float)
    forces = faero + fgrav + fprop

    u_dot = forces[0] / vehicle.mass_kg - q * w
    w_dot = forces[2] / vehicle.mass_kg + q * u
    q_dot = maero[1] / vehicle.iyy_kgm2
    theta_dot = q
    x_dot = u * math.cos(theta) + w * math.sin(theta)
    h_dot = -( -u * math.sin(theta) + w * math.cos(theta) )

    return np.array([x_dot, h_dot, u_dot, w_dot, theta_dot, q_dot], dtype=float)
