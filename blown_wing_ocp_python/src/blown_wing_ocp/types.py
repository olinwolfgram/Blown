from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PropulsionData:
    n_props: int
    diameter_m: float
    centers_y_m: np.ndarray
    axial_x_m: float
    thrust_axis_body: np.ndarray
    rpm_grid: np.ndarray
    thrust_n_grid: np.ndarray
    cp_static_grid: np.ndarray


@dataclass(frozen=True)
class VehicleParameters:
    name: str
    mass_kg: float
    gravity_mps2: float
    rho_kgpm3: float
    span_m: float
    chord_m: float
    area_m2: float
    aspect_ratio: float
    oswald_e: float
    wing_incidence_rad: float
    wing_washout_rad: float
    cg_x_m: float
    wing_ac_x_m: float
    tail_arm_m: float
    htail_area_m2: float
    htail_incidence_rad: float
    ixx_kgm2: float
    iyy_kgm2: float
    izz_kgm2: float
    ixz_kgm2: float
    elevator_limit_rad: float
    aileron_limit_rad: float
    rudder_limit_rad: float
    flap_limit_rad: float
    propulsion: PropulsionData
    aero: dict
    blown: dict
    references: dict


@dataclass(frozen=True)
class TrimResult:
    airspeed_mps: float
    alpha_rad: float
    theta_rad: float
    rpm: float
    elevator_rad: float
    state: np.ndarray
    control: np.ndarray
    residual: np.ndarray
    success: bool
    message: str


@dataclass(frozen=True)
class OCPResult:
    t_s: np.ndarray
    x_hist: np.ndarray
    u_hist: np.ndarray
    objective: float
    success: bool
    message: str
    solver_output: dict
