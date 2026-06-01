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


def lateral_state_derivative_10motor(
    x_lat: np.ndarray,
    u_lat: np.ndarray,
    vehicle,
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


def main() -> None:
    vehicle = load_vehicle()
    op = build_symmetric_cruise_operating_point(
        vehicle,
        speed_mps=10.0,
        flight_path_angle_rad=0.0,
        flap_rad=0.0,
    )
    x_trim = op.lateral_state.copy()
    n_props = int(vehicle.propulsion["n_props"])
    rpm_trim = np.asarray(op.full_control[:n_props], dtype=float)
    elevator_trim = float(op.longitudinal_control[1])
    flap_trim = float(op.longitudinal_control[2])
    w_trim = float(op.longitudinal_state[3])
    theta_trim = float(op.longitudinal_state[4])

    u_trim = np.concatenate([rpm_trim, np.array([0.0, 0.0], dtype=float)])

    lat_lin = linearize(
        lambda x, u: lateral_state_derivative_10motor(
            x,
            u,
            vehicle,
            w_trim_mps=w_trim,
            theta_trim_rad=theta_trim,
            elevator_trim_rad=elevator_trim,
            flap_trim_rad=flap_trim,
        ),
        x_trim,
        u_trim,
        dt=0.05,
    )

    state_idx = (2, 3, 4, 5, 6, 7)
    input_idx = tuple(range(n_props + 2))

    q_mat = np.diag([8.0, 6.0, 25.0, 8.0, 30.0, 18.0])
    r_diag = np.concatenate([np.full(n_props, 1.0e-4, dtype=float), np.array([2.0, 2.0], dtype=float)])
    r_mat = np.diag(r_diag)

    lqr = design_lqr(
        lat_lin["A"],
        lat_lin["B"],
        q_mat,
        r_mat,
        state_indices=state_idx,
        input_indices=input_idx,
        discrete_time=False,
    )

    print("Cruise lateral operating point (10-motor input demo)")
    print(f"  u trim              : {x_trim[2]:.6f} m/s")
    print(f"  v trim              : {x_trim[3]:.6f} m/s")
    print(f"  phi trim            : {np.rad2deg(x_trim[4]):.6f} deg")
    print(f"  psi trim            : {np.rad2deg(x_trim[5]):.6f} deg")
    print(f"  p trim              : {np.rad2deg(x_trim[6]):.6f} deg/s")
    print(f"  r trim              : {np.rad2deg(x_trim[7]):.6f} deg/s")
    print(f"  fixed elevator trim : {np.rad2deg(elevator_trim):.6f} deg")
    print(f"  fixed flap trim     : {np.rad2deg(flap_trim):.6f} deg")
    print(f"  per-motor trim RPM  : {rpm_trim}")
    print(f"  lateral trim residual: {op.lateral_trim_residual}")

    print("\nSelected lateral subsystem for 10-motor LQR")
    print("  states: [u, v, phi, psi, p, r]")
    print("  inputs: [rpm_1, ..., rpm_10, aileron, rudder]")
    print("\nMotor-input interpretation")
    print("  Unlike the differential-thrust regulator, this demo exposes the")
    print("  physical RPM of each motor directly in the LQR input vector.")
    print("  Symmetric forward-flight thrust is preserved by the trim RPM baseline,")
    print("  while the controller computes per-motor perturbations about that point.")

    print("\nFull reduced lateral A")
    print(lat_lin["A"])
    print()
    print("10-motor lateral B")
    print(lat_lin["B"])
    print()
    print("Continuous-time 10-motor lateral LQR gain K")
    print(lqr.k_gain)
    print("Continuous-time closed-loop eigenvalues")
    print(lqr.eigenvalues)


if __name__ == "__main__":
    main()
