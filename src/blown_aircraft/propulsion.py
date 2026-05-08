from __future__ import annotations

import numpy as np

from .geometry import Vehicle


def thrust_per_prop_n(rpm: float, vehicle: Vehicle) -> float:
    rpm_grid = np.asarray(vehicle.propulsion["rpm_grid"], dtype=float)
    thrust_grid = np.asarray(vehicle.propulsion["thrust_n_grid"], dtype=float)
    rpm_clamped = float(np.clip(rpm, rpm_grid.min(), rpm_grid.max()))
    return float(np.interp(rpm_clamped, rpm_grid, thrust_grid))


def prop_positions_body(vehicle: Vehicle) -> np.ndarray:
    centers_y = np.asarray(vehicle.propulsion["centers_y_m"], dtype=float)
    x = float(vehicle.propulsion["axial_x_m"]) - float(vehicle.geometry["cg_x_m"])
    z = float(vehicle.propulsion.get("vertical_z_m", 0.0))
    return np.column_stack(
        [
            np.full_like(centers_y, x, dtype=float),
            centers_y,
            np.full_like(centers_y, z, dtype=float),
        ]
    )


def propulsion_forces_and_moments(control: np.ndarray, vehicle: Vehicle) -> tuple[np.ndarray, np.ndarray, dict]:
    """Compute total propulsion force and moment in body axes.

    Control ordering:
    [rpm_1, ..., rpm_10, delta_e, delta_a, delta_r, delta_f]

    Propeller thrust acts along body +x. Differential thrust creates yaw
    moment through the spanwise moment arms.
    """

    n_props = int(vehicle.propulsion["n_props"])
    rpm_vec = np.asarray(control[:n_props], dtype=float)
    positions = prop_positions_body(vehicle)

    total_force = np.zeros(3, dtype=float)
    total_moment = np.zeros(3, dtype=float)
    per_prop_thrusts = []
    for idx in range(n_props):
        thrust = thrust_per_prop_n(float(rpm_vec[idx]), vehicle)
        force = np.array([thrust, 0.0, 0.0], dtype=float)
        moment = np.cross(positions[idx], force)
        total_force += force
        total_moment += moment
        per_prop_thrusts.append(thrust)

    diagnostics = {
        "per_prop_thrust_n": np.asarray(per_prop_thrusts, dtype=float),
        "rpm_vector": rpm_vec.copy(),
        "left_thrust_n": float(np.sum(per_prop_thrusts[: n_props // 2])),
        "right_thrust_n": float(np.sum(per_prop_thrusts[n_props // 2 :])),
    }
    return total_force, total_moment, diagnostics


def collective_rpm_to_full_control(
    rpm_collective: float,
    surface_controls: np.ndarray,
    vehicle: Vehicle,
) -> np.ndarray:
    n_props = int(vehicle.propulsion["n_props"])
    return np.concatenate([np.full(n_props, rpm_collective, dtype=float), np.asarray(surface_controls, dtype=float)])


def split_rpm_to_full_control(
    rpm_left: float,
    rpm_right: float,
    surface_controls: np.ndarray,
    vehicle: Vehicle,
) -> np.ndarray:
    n_props = int(vehicle.propulsion["n_props"])
    half = n_props // 2
    rpm_vec = np.concatenate(
        [np.full(half, rpm_left, dtype=float), np.full(n_props - half, rpm_right, dtype=float)]
    )
    return np.concatenate([rpm_vec, np.asarray(surface_controls, dtype=float)])
