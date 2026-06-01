from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from blown_aircraft.geometry import load_vehicle
from blown_aircraft.linearize import linearize
from blown_aircraft.lqr import design_lqr
from blown_aircraft.operating_point import build_symmetric_cruise_operating_point
from blown_aircraft.rigid_body_ac import total_forces_and_moments


def longitudinal_state_derivative_10motor(
    x_lon: np.ndarray,
    u_lon: np.ndarray,
    vehicle,
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


def main() -> None:
    vehicle = load_vehicle()
    op = build_symmetric_cruise_operating_point(
        vehicle,
        speed_mps=10.0,
        flight_path_angle_rad=0.0,
        flap_rad=0.0,
    )
    x_trim = op.longitudinal_state.copy()
    n_props = int(vehicle.propulsion["n_props"])
    rpm_trim = np.asarray(op.full_control[:n_props], dtype=float)
    flap_trim = float(op.longitudinal_control[2])
    elevator_trim = float(op.longitudinal_control[1])

    u_trim = np.concatenate([rpm_trim, np.array([elevator_trim], dtype=float)])

    lon_lin = linearize(
        lambda x, u: longitudinal_state_derivative_10motor(x, u, vehicle, flap_trim_rad=flap_trim),
        x_trim,
        u_trim,
        dt=0.05,
    )

    state_idx = (2, 3, 4, 5)
    input_idx = tuple(range(n_props + 1))

    q_mat = np.diag([4.0, 8.0, 30.0, 12.0])
    r_diag = np.concatenate([np.full(n_props, 1.0e-7, dtype=float), np.array([2.0], dtype=float)])
    r_mat = np.diag(r_diag)

    lqr_ct = design_lqr(
        lon_lin["A"],
        lon_lin["B"],
        q_mat,
        r_mat,
        state_indices=state_idx,
        input_indices=input_idx,
        discrete_time=False,
    )
    lqr_dt = design_lqr(
        lon_lin["Ad"],
        lon_lin["Bd"],
        q_mat,
        r_mat,
        state_indices=state_idx,
        input_indices=input_idx,
        discrete_time=True,
    )

    print("Cruise longitudinal operating point (10-motor input demo)")
    print(f"  u                 : {x_trim[2]:.6f} m/s")
    print(f"  w                 : {x_trim[3]:.6f} m/s")
    print(f"  theta             : {np.rad2deg(x_trim[4]):.6f} deg")
    print(f"  q                 : {np.rad2deg(x_trim[5]):.6f} deg/s")
    print(f"  elevator trim     : {np.rad2deg(elevator_trim):.6f} deg")
    print(f"  flap trim         : {np.rad2deg(flap_trim):.6f} deg")
    print(f"  per-motor trim RPM: {rpm_trim}")
    print(f"  trim residual     : {op.longitudinal_trim_residual}")

    print("\nSelected longitudinal subsystem for 10-motor LQR")
    print("  states: [u, w, theta, q]")
    print("  inputs: [rpm_1, ..., rpm_10, elevator]")
    print("\nMotor-input interpretation")
    print("  The trim point remains symmetric, but the controller may distribute")
    print("  propulsion corrections across all ten motors instead of a single")
    print("  collective-RPM channel.")

    print("\nContinuous-time 10-motor longitudinal LQR gain K")
    print(lqr_ct.k_gain)
    print("Continuous-time closed-loop eigenvalues")
    print(lqr_ct.eigenvalues)

    print("\nDiscrete-time 10-motor longitudinal LQR gain K")
    print(lqr_dt.k_gain)
    print("Discrete-time closed-loop eigenvalues")
    print(lqr_dt.eigenvalues)


if __name__ == "__main__":
    main()
