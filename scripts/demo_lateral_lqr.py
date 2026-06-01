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
    result = linearize_about_cruise(
        vehicle,
        speed_mps=10.0,
        flight_path_angle_rad=0.0,
        flap_rad=0.0,
        dt=0.05,
    )
    op = result["operating_point"]
    lat = result["lateral"]
    a_lat = lat["A"]
    b_lat = lat["B"]

    # Build a three-input lateral control channel:
    # [delta_rpm_diff, aileron, rudder]
    # where rpm_left = rpm_trim - delta_rpm_diff and
    # rpm_right = rpm_trim + delta_rpm_diff.
    b_diff = (b_lat[:, [1]] - b_lat[:, [0]]).copy()
    b_ctrl = np.hstack([b_diff, b_lat[:, [2]], b_lat[:, [3]]])

    # Use the dynamic planar states and exclude x/y position.
    # State ordering in the planar lateral model:
    # [x, y, u, v, phi, psi, p, r]
    state_idx = (2, 3, 4, 5, 6, 7)  # [u, v, phi, psi, p, r]
    q_mat = np.diag([8.0, 6.0, 25.0, 8.0, 30.0, 18.0])
    r_mat = np.diag([1.0e-3, 2.0, 2.0])

    lqr = design_lqr(
        a_lat,
        b_ctrl,
        q_mat,
        r_mat,
        state_indices=state_idx,
        input_indices=(0, 1, 2),
        discrete_time=False,
    )

    print("Cruise lateral operating point")
    print(f"  rpm left trim         : {op.lateral_control[0]:.6f}")
    print(f"  rpm right trim        : {op.lateral_control[1]:.6f}")
    print(f"  aileron trim          : {np.rad2deg(op.lateral_control[2]):.6f} deg")
    print(f"  rudder trim           : {np.rad2deg(op.lateral_control[3]):.6f} deg")
    print(f"  lateral trim residual : {op.lateral_trim_residual}")
    print()

    print("Lateral model note")
    print("  The controller now uses the lateral actuator set")
    print("  [delta_rpm_diff, aileron, rudder].")
    print()

    print("Selected lateral subsystem for LQR")
    print("  states: [u, v, phi, psi, p, r]")
    print("  inputs: [delta_rpm_diff, aileron, rudder]")
    print()

    print("Full reduced lateral A")
    print(a_lat)
    print()
    print("Effective lateral-control B matrix")
    print(b_ctrl)
    print()
    print("Continuous-time lateral LQR gain K")
    print(lqr.k_gain)
    print("Continuous-time closed-loop eigenvalues")
    print(lqr.eigenvalues)


if __name__ == "__main__":
    main()
