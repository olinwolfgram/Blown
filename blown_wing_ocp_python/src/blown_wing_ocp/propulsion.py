from __future__ import annotations

import numpy as np

from .types import VehicleParameters


def thrust_per_prop_n(rpm: float, vehicle: VehicleParameters) -> float:
    grid_rpm = vehicle.propulsion.rpm_grid
    grid_thrust = vehicle.propulsion.thrust_n_grid
    rpm_clamped = float(np.clip(rpm, grid_rpm.min(), grid_rpm.max()))
    return float(np.interp(rpm_clamped, grid_rpm, grid_thrust))


def total_thrust_n(rpm: float, vehicle: VehicleParameters) -> float:
    return vehicle.propulsion.n_props * thrust_per_prop_n(rpm, vehicle)


def total_disk_area_m2(vehicle: VehicleParameters) -> float:
    radius = 0.5 * vehicle.propulsion.diameter_m
    return vehicle.propulsion.n_props * np.pi * radius**2
