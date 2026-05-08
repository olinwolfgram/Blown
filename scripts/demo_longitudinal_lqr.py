from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from blown_aircraft.geometry import load_vehicle
from blown_aircraft.lqr import design_lqr
from blown_aircraft.operating_point import linearize_about_cruise


def main() -> None:
    vehicle = load_vehicle()
    result = linearize_about_cruise(vehicle, speed_mps=10.0, flight_path_angle_rad=0.0, flap_rad=0.0, dt=0.05)
    op = result["operating_point"]
    lon = result["longitudinal"]

    # Longitudinal reduced state ordering:
    # [x, h, u, w, theta, q]
    #
    # For the first LQR pass, regulate the dynamic states [u, w, theta, q]
    # using [collective RPM, elevator]. Flap is treated as a configuration
    # parameter rather than an active fast control in this inner-loop design.
    state_idx = (2, 3, 4, 5)
    input_idx = (0, 1)

    q_mat = np.diag([4.0, 8.0, 30.0, 12.0])
    r_mat = np.diag([1e-6, 2.0])

    lqr_ct = design_lqr(
        lon["A"],
        lon["B"],
        q_mat,
        r_mat,
        state_indices=state_idx,
        input_indices=input_idx,
        discrete_time=False,
    )
    lqr_dt = design_lqr(
        lon["Ad"],
        lon["Bd"],
        q_mat,
        r_mat,
        state_indices=state_idx,
        input_indices=input_idx,
        discrete_time=True,
    )

    print("Cruise longitudinal operating point")
    print(f"  u            : {op.longitudinal_state[2]:.6f} m/s")
    print(f"  w            : {op.longitudinal_state[3]:.6f} m/s")
    print(f"  theta        : {np.rad2deg(op.longitudinal_state[4]):.6f} deg")
    print(f"  q            : {np.rad2deg(op.longitudinal_state[5]):.6f} deg/s")
    print(f"  collective rpm: {op.longitudinal_control[0]:.6f}")
    print(f"  elevator     : {np.rad2deg(op.longitudinal_control[1]):.6f} deg")
    print(f"  flap         : {np.rad2deg(op.longitudinal_control[2]):.6f} deg")
    print(f"  trim residual: {op.longitudinal_trim_residual}")

    print("\nSelected longitudinal subsystem for LQR")
    print("  states: [u, w, theta, q]")
    print("  inputs: [collective RPM, elevator]")

    print("\nContinuous-time LQR gain K")
    print(lqr_ct.k_gain)
    print("Continuous-time closed-loop eigenvalues")
    print(lqr_ct.eigenvalues)

    print("\nDiscrete-time LQR gain K")
    print(lqr_dt.k_gain)
    print("Discrete-time closed-loop eigenvalues")
    print(lqr_dt.eigenvalues)


if __name__ == "__main__":
    main()
