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

    # Build a differential-thrust input channel:
    # rpm_left = rpm_trim - delta_rpm_diff
    # rpm_right = rpm_trim + delta_rpm_diff
    # so the effective input column is B_right - B_left.
    b_diff = (b_lat[:, [1]] - b_lat[:, [0]]).copy()

    # Use the dynamic lateral-directional states and exclude y-position.
    # State ordering in the reduced lateral model:
    # [y, v, phi, psi, p, r]
    state_idx = (1, 2, 3, 4, 5)  # [v, phi, psi, p, r]
    q_mat = np.diag([6.0, 25.0, 8.0, 30.0, 18.0])
    r_mat = np.array([[1.0e-3]])

    lqr = design_lqr(
        a_lat,
        b_diff,
        q_mat,
        r_mat,
        state_indices=state_idx,
        input_indices=(0,),
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
    print("  The current controller is intentionally built around differential thrust.")
    print("  Aileron and rudder derivatives are present in the aerodynamic model now,")
    print("  but this first-pass LQR uses the differential-thrust channel only.")
    print()

    print("Selected lateral subsystem for LQR")
    print("  states: [v, phi, psi, p, r]")
    print("  input : [delta_rpm_diff]")
    print()

    print("Full reduced lateral A")
    print(a_lat)
    print()
    print("Effective differential-thrust B column")
    print(b_diff)
    print()
    print("Continuous-time lateral LQR gain K")
    print(lqr.k_gain)
    print("Continuous-time closed-loop eigenvalues")
    print(lqr.eigenvalues)


if __name__ == "__main__":
    main()
