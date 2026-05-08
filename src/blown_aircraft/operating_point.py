from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .geometry import Vehicle
from .lateral import lateral_state_derivative
from .linearize import linearize
from .longitudinal import longitudinal_state_derivative
from .propulsion import collective_rpm_to_full_control
from .trim import solve_longitudinal_trim


@dataclass(frozen=True)
class CruiseOperatingPoint:
    speed_mps: float
    flight_path_angle_rad: float
    longitudinal_state: np.ndarray
    longitudinal_control: np.ndarray
    lateral_state: np.ndarray
    lateral_control: np.ndarray
    full_state: np.ndarray
    full_control: np.ndarray
    longitudinal_state_derivative: np.ndarray
    lateral_state_derivative: np.ndarray
    longitudinal_trim_residual: np.ndarray
    lateral_trim_residual: np.ndarray


def build_symmetric_cruise_operating_point(
    vehicle: Vehicle,
    speed_mps: float = 10.0,
    flight_path_angle_rad: float = 0.0,
    flap_rad: float = 0.0,
) -> CruiseOperatingPoint:
    """Build a symmetric level-flight cruise operating point.

    Assumptions:
    - zero roll and yaw attitude,
    - zero angular rates,
    - zero sideslip,
    - equal left/right propeller speed,
    - zero aileron and rudder,
    - level or specified flight-path angle.

    For the first control-design pass this is the standard cruise trim point
    about which we linearize both longitudinal and lateral models.
    """

    trim = solve_longitudinal_trim(vehicle, speed_mps=speed_mps, flight_path_angle_rad=flight_path_angle_rad)
    x_lon = trim["state"].copy()
    u_lon = trim["control"].copy()
    u_lon[2] = flap_rad

    x_fwd, h, u, w, theta, q = x_lon
    rpm_collective, delta_e, delta_f = u_lon
    n_props = int(vehicle.propulsion["n_props"])

    full_state = np.array(
        [x_fwd, 0.0, -h, u, 0.0, w, 0.0, theta, 0.0, 0.0, q, 0.0],
        dtype=float,
    )
    full_control = collective_rpm_to_full_control(
        rpm_collective,
        np.array([delta_e, 0.0, 0.0, delta_f], dtype=float),
        vehicle,
    )

    x_lat = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
    u_lat = np.array([rpm_collective, rpm_collective, 0.0, 0.0], dtype=float)

    lon_dx = longitudinal_state_derivative(x_lon, u_lon, vehicle)
    lat_dx = lateral_state_derivative(x_lat, u_lat, vehicle)
    lon_trim_resid = np.array([lon_dx[2], lon_dx[3], lon_dx[5]], dtype=float)
    lat_trim_resid = np.array([lat_dx[1], lat_dx[4], lat_dx[5]], dtype=float)

    return CruiseOperatingPoint(
        speed_mps=speed_mps,
        flight_path_angle_rad=flight_path_angle_rad,
        longitudinal_state=x_lon,
        longitudinal_control=u_lon,
        lateral_state=x_lat,
        lateral_control=u_lat,
        full_state=full_state,
        full_control=full_control,
        longitudinal_state_derivative=lon_dx,
        lateral_state_derivative=lat_dx,
        longitudinal_trim_residual=lon_trim_resid,
        lateral_trim_residual=lat_trim_resid,
    )


def linearize_about_cruise(
    vehicle: Vehicle,
    speed_mps: float = 10.0,
    flight_path_angle_rad: float = 0.0,
    flap_rad: float = 0.0,
    dt: float | None = None,
) -> dict[str, object]:
    op = build_symmetric_cruise_operating_point(vehicle, speed_mps, flight_path_angle_rad, flap_rad)
    lon_lin = linearize(
        lambda x, u: longitudinal_state_derivative(x, u, vehicle),
        op.longitudinal_state,
        op.longitudinal_control,
        dt=dt,
    )
    lat_lin = linearize(
        lambda x, u: lateral_state_derivative(x, u, vehicle),
        op.lateral_state,
        op.lateral_control,
        dt=dt,
    )
    return {"operating_point": op, "longitudinal": lon_lin, "lateral": lat_lin}
