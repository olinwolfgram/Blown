from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from blown_aircraft.aerodynamics import aerodynamic_forces_and_moments
from blown_aircraft.geometry import load_vehicle
from blown_aircraft.operating_point import build_symmetric_cruise_operating_point
from blown_aircraft.rigid_body_ac import total_forces_and_moments


def main() -> None:
    vehicle = load_vehicle()
    op = build_symmetric_cruise_operating_point(
        vehicle,
        speed_mps=10.0,
        flight_path_angle_rad=0.0,
        flap_rad=0.0,
    )

    state = op.full_state.copy()
    control_trim = op.full_control.copy()
    n_props = int(vehicle.propulsion["n_props"])

    trim_de_rad = float(control_trim[n_props])
    trim_de_deg = np.rad2deg(trim_de_rad)

    print("Cruise elevator sweep about trimmed operating point")
    print(f"  speed                 : {op.speed_mps:.4f} m/s")
    print(f"  trim elevator         : {trim_de_deg:.4f} deg")
    print(f"  trim rpm              : {op.longitudinal_control[0]:.4f}")
    print()
    print("Pitch-driving parameters")
    print(f"  wing cm0              : {vehicle.aero['cm0']:.6f}")
    print(f"  wing cm_alpha_per_rad : {vehicle.aero['cm_alpha_per_rad']:.6f}")
    print(f"  wing cm_q             : {vehicle.aero['cm_q']:.6f}")
    print(f"  blown pitch_gain      : {vehicle.blown['pitch_gain']:.6f}")
    print(f"  blown flap_pitch_gain : {vehicle.blown['flap_pitch_gain']:.6f}")
    print(f"  downwash_gradient     : {vehicle.aero['downwash_gradient']:.6f}")
    print(f"  tail_cl_de_per_rad    : {vehicle.aero['tail_cl_de_per_rad']:.6f}")
    print(f"  htail incidence       : {vehicle.geometry['htail_incidence_deg']:.6f} deg")
    print()

    print(
        f"{'de (deg)':>10} {'My total':>12} {'My wing':>12} {'My tail':>12} "
        f"{'wing sec':>12} {'tail CL':>12}"
    )
    print("-" * 74)

    for delta_deg in np.linspace(-8.0, 8.0, 17):
        control = control_trim.copy()
        control[n_props] = trim_de_rad + np.deg2rad(delta_deg)

        _, m_total, _ = total_forces_and_moments(state, control, vehicle)
        _, _, aero_diag = aerodynamic_forces_and_moments(state, control, vehicle)

        my_total = float(m_total[1])
        my_wing = float(aero_diag["wing"]["moment_body_nm"][1])
        my_tail = float(aero_diag["tail"]["moment_body_nm"][1])
        my_wing_section = float(aero_diag["wing"]["section_pitch_moment_nm"])
        tail_cl = float(aero_diag["tail"]["cl_tail"])

        print(
            f"{trim_de_deg + delta_deg:10.4f} {my_total:12.6f} {my_wing:12.6f} "
            f"{my_tail:12.6f} {my_wing_section:12.6f} {tail_cl:12.6f}"
        )


if __name__ == "__main__":
    main()
