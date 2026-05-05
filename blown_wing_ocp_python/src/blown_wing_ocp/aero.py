from __future__ import annotations

import math

import numpy as np

from .propulsion import total_disk_area_m2, total_thrust_n
from .types import VehicleParameters


def dynamic_pressure(rho_kgpm3: float, airspeed_mps: float) -> float:
    """Return dynamic pressure q = 0.5 rho V^2.

    Reference:
    - Anderson, Fundamentals of Aerodynamics, 5th ed.
    """

    return 0.5 * rho_kgpm3 * airspeed_mps**2


def airspeed_alpha_beta(u: float, v: float, w: float) -> tuple[float, float, float]:
    v_total = float(np.sqrt(u**2 + v**2 + w**2))
    alpha = float(np.arctan2(w, max(u, 1e-6)))
    beta = float(np.arcsin(np.clip(v / max(v_total, 1e-6), -1.0, 1.0)))
    return v_total, alpha, beta


def blown_wing_increment(
    airspeed_mps: float,
    alpha_rad: float,
    rpm: float,
    flap_rad: float,
    vehicle: VehicleParameters,
) -> tuple[float, float, float, float]:
    """Return blown-wing increments and wake ratio.

    The wake model is a compact surrogate:
    - actuator-disk-inspired induced velocity scaling,
    - exponential streamwise decay from prop disk to wing,
    - flap-augmented lift and pitch effect.

    References for motivation:
    - Agrawal et al. (2019), blown flap wind-tunnel testing
    - Long (2021 thesis), discussion of slipstream/flap interaction
    """

    thrust_total = total_thrust_n(rpm, vehicle)
    disk_area = total_disk_area_m2(vehicle)
    rho = vehicle.rho_kgpm3
    if thrust_total <= 0.0 or disk_area <= 1e-9:
        return 0.0, 0.0, 0.0, 1.0

    u_axial = max(airspeed_mps * math.cos(alpha_rad), 0.5)
    v_induced = math.sqrt(max(thrust_total, 0.0) / (2.0 * rho * disk_area))
    v_wake = u_axial + 2.0 * v_induced
    wake_ratio = max((v_wake / max(u_axial, 1e-3)) ** 2 - 1.0, 0.0)

    eta_span = vehicle.blown["blown_span_fraction"]
    decay = math.exp(
        -vehicle.blown["prop_to_wing_leading_edge_m"] / max(vehicle.blown["wake_decay_length_m"], 1e-6)
    )
    eta = eta_span * decay
    flap_pos = max(flap_rad, 0.0)

    dcl = eta * vehicle.blown["lift_gain"] * wake_ratio * (1.0 + vehicle.blown["flap_lift_gain"] * flap_pos)
    dcd = eta * vehicle.blown["drag_gain"] * wake_ratio + vehicle.blown["flap_drag_gain"] * flap_pos**2
    dcm = eta * (
        vehicle.blown["pitch_gain"] * wake_ratio
        + vehicle.blown["flap_pitch_gain"] * flap_pos * wake_ratio
    )
    return dcl, dcd, dcm, wake_ratio


def aerodynamic_coefficients(
    u: float,
    v: float,
    w: float,
    p: float,
    q: float,
    r: float,
    control: np.ndarray,
    vehicle: VehicleParameters,
) -> dict[str, float]:
    """Return low-order aerodynamic coefficients.

    References:
    - Nelson, Flight Stability and Automatic Control
    - Anderson, Fundamentals of Aerodynamics
    """

    rpm, de, da, dr, df = control
    airspeed_mps, alpha_rad, beta_rad = airspeed_alpha_beta(u, v, w)
    qhat = q * vehicle.chord_m / max(2.0 * airspeed_mps, 1e-6)

    aero = vehicle.aero
    dcl, dcd, dcm, wake_ratio = blown_wing_increment(airspeed_mps, alpha_rad, rpm, df, vehicle)

    cl = (
        aero["cl0"]
        + aero["cl_alpha_per_rad"] * (alpha_rad + vehicle.wing_incidence_rad)
        + aero["cl_q"] * qhat
        + aero["cl_de_per_rad"] * de
        + dcl
    )
    k_induced = 1.0 / (math.pi * vehicle.oswald_e * vehicle.aspect_ratio)
    cd = aero["cd0"] + k_induced * cl**2 + dcd
    cy = aero["cy_beta_per_rad"] * beta_rad
    cm = (
        aero["cm0"]
        + aero["cm_alpha_per_rad"] * (alpha_rad + vehicle.wing_incidence_rad)
        + aero["cm_q"] * qhat
        + aero["cm_de_per_rad"] * de
        + dcm
    )
    cl_roll = aero["cl_beta_per_rad"] * beta_rad + aero["cl_p"] * (p * vehicle.span_m / max(2.0 * airspeed_mps, 1e-6))
    cn_yaw = aero["cn_beta_per_rad"] * beta_rad + aero["cn_r"] * (r * vehicle.span_m / max(2.0 * airspeed_mps, 1e-6))

    return {
        "V": airspeed_mps,
        "alpha": alpha_rad,
        "beta": beta_rad,
        "qhat": qhat,
        "CL": cl,
        "CD": cd,
        "CY": cy,
        "Cl": cl_roll,
        "Cm": cm,
        "Cn": cn_yaw,
        "wake_ratio": wake_ratio,
    }


def aero_forces_and_moments_body(
    u: float,
    v: float,
    w: float,
    p: float,
    q: float,
    r: float,
    control: np.ndarray,
    vehicle: VehicleParameters,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    coeffs = aerodynamic_coefficients(u, v, w, p, q, r, control, vehicle)
    qbar = dynamic_pressure(vehicle.rho_kgpm3, coeffs["V"])

    drag_n = qbar * vehicle.area_m2 * coeffs["CD"]
    lift_n = qbar * vehicle.area_m2 * coeffs["CL"]
    side_n = qbar * vehicle.area_m2 * coeffs["CY"]

    alpha = coeffs["alpha"]
    beta = coeffs["beta"]

    fx = -drag_n * math.cos(alpha) * math.cos(beta) + lift_n * math.sin(alpha)
    fy = side_n
    fz = -drag_n * math.sin(alpha) - lift_n * math.cos(alpha)

    l_moment = qbar * vehicle.area_m2 * vehicle.span_m * coeffs["Cl"]
    m_moment = qbar * vehicle.area_m2 * vehicle.chord_m * coeffs["Cm"]
    n_moment = qbar * vehicle.area_m2 * vehicle.span_m * coeffs["Cn"]

    return np.array([fx, fy, fz], dtype=float), np.array([l_moment, m_moment, n_moment], dtype=float), coeffs
