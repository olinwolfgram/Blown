from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from blown_aircraft.geometry import load_vehicle
from blown_aircraft.operating_point import linearize_about_cruise


def main() -> None:
    vehicle = load_vehicle()
    result = linearize_about_cruise(vehicle, speed_mps=10.0, flight_path_angle_rad=0.0, flap_rad=0.0, dt=0.05)
    op = result["operating_point"]
    lon = result["longitudinal"]
    lat = result["lateral"]

    print("Cruise operating point")
    print(f"  speed      : {op.speed_mps:.3f} m/s")
    print(f"  alpha      : {np.rad2deg(np.arctan2(op.longitudinal_state[3], op.longitudinal_state[2])):.3f} deg")
    print(f"  theta      : {np.rad2deg(op.longitudinal_state[4]):.3f} deg")
    print(f"  rpm        : {op.longitudinal_control[0]:.3f}")
    print(f"  elevator   : {np.rad2deg(op.longitudinal_control[1]):.3f} deg")
    print(f"  flap       : {np.rad2deg(op.longitudinal_control[2]):.3f} deg")
    print(f"  lon xdot   : {op.longitudinal_state_derivative}")
    print(f"  lat xdot   : {op.lateral_state_derivative}")
    print(f"  lon trim residual [u_dot, w_dot, q_dot] : {op.longitudinal_trim_residual}")
    print(f"  lat trim residual [v_dot, p_dot, r_dot] : {op.lateral_trim_residual}")

    print("\nLongitudinal A")
    print(lon["A"])
    print("\nLongitudinal B")
    print(lon["B"])
    print("\nLateral A")
    print(lat["A"])
    print("\nLateral B")
    print(lat["B"])


if __name__ == "__main__":
    main()
