from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from blown_aircraft.aerodynamics import aerodynamic_forces_and_moments, airdata_from_body_velocity
from blown_aircraft.geometry import load_vehicle
from blown_aircraft.operating_point import build_symmetric_cruise_operating_point
from blown_aircraft.propulsion import propulsion_forces_and_moments
from blown_aircraft.rigid_body_ac import gravity_force_body, total_forces_and_moments


def main() -> None:
    vehicle = load_vehicle()
    op = build_symmetric_cruise_operating_point(vehicle, speed_mps=10.0, flight_path_angle_rad=0.0, flap_rad=0.0)

    state = op.full_state
    control = op.full_control

    faero, maero, aero_diag = aerodynamic_forces_and_moments(state, control, vehicle)
    fprop, mprop, prop_diag = propulsion_forces_and_moments(control, vehicle)
    fgrav = gravity_force_body(float(state[6]), float(state[7]), vehicle)
    ftotal, mtotal, _ = total_forces_and_moments(state, control, vehicle)

    air = airdata_from_body_velocity(float(state[3]), float(state[4]), float(state[5]))
    alpha_body_deg = np.rad2deg(air["alpha"])
    wing_inc_deg = float(vehicle.geometry["wing_incidence_deg"])
    wing_alpha_eff_deg = alpha_body_deg + wing_inc_deg
    theta_deg = np.rad2deg(float(state[7]))
    elevator_deg = np.rad2deg(float(op.longitudinal_control[1]))
    flap_deg = np.rad2deg(float(op.longitudinal_control[2]))

    print("Cruise trim diagnostic")
    print(f"  speed                 : {op.speed_mps:.4f} m/s")
    print(f"  body alpha            : {alpha_body_deg:.4f} deg")
    print(f"  wing incidence        : {wing_inc_deg:.4f} deg")
    print(f"  effective wing alpha  : {wing_alpha_eff_deg:.4f} deg")
    print(f"  theta                 : {theta_deg:.4f} deg")
    print(f"  collective rpm        : {op.longitudinal_control[0]:.4f}")
    print(f"  elevator              : {elevator_deg:.4f} deg")
    print(f"  flap                  : {flap_deg:.4f} deg")
    print()
    print("Reduced longitudinal trim residual [u_dot, w_dot, q_dot]")
    print(f"  {op.longitudinal_trim_residual}")
    print()
    print("Force breakdown in body axes [X, Y, Z] N")
    print(f"  aerodynamic           : {faero}")
    print(f"  propulsion            : {fprop}")
    print(f"  gravity               : {fgrav}")
    print(f"  total                 : {ftotal}")
    print()
    print("Moment breakdown in body axes [L, M, N] N-m")
    print(f"  aerodynamic           : {maero}")
    print(f"  propulsion            : {mprop}")
    print(f"  total                 : {mtotal}")
    print()
    print("Aerodynamic surface totals")
    print(f"  wing force            : {np.array(aero_diag['wing']['force_body_n'])}")
    print(f"  wing moment           : {np.array(aero_diag['wing']['moment_body_nm'])}")
    print(f"  wing arm pitch M      : {aero_diag['wing']['arm_pitch_moment_nm']:.6f} N-m")
    print(f"  wing section pitch M  : {aero_diag['wing']['section_pitch_moment_nm']:.6f} N-m")
    wing_pitch = aero_diag["wing"]["section_pitch_components_nm"]
    print(f"    cm0 contribution    : {wing_pitch['cm0']:.6f} N-m")
    print(f"    cm_alpha contribution: {wing_pitch['cm_alpha']:.6f} N-m")
    print(f"    cm_q contribution   : {wing_pitch['cm_q']:.6f} N-m")
    print(f"    flap contribution   : {wing_pitch['flap']:.6f} N-m")
    print(f"    blown contribution  : {wing_pitch['blown']:.6f} N-m")
    print(f"  tail force            : {np.array(aero_diag['tail']['force_body_n'])}")
    print(f"  tail moment           : {np.array(aero_diag['tail']['moment_body_nm'])}")
    print(f"  wing blown pitch M    : {aero_diag['wing']['blown_pitch_moment_nm']:.6f} N-m")
    print(f"  wing flap pitch M     : {aero_diag['wing']['flap_pitch_moment_nm']:.6f} N-m")
    print()
    print("Wing drag decomposition")
    wing_drag = aero_diag["wing"]["drag_components_n"]
    print(f"  parasite/profile drag : {wing_drag['parasite_profile']:.6f} N")
    print(f"  induced drag          : {wing_drag['induced']:.6f} N")
    print(f"  blown drag increment  : {wing_drag['blown']:.6f} N")
    print(f"  flap drag increment   : {wing_drag['flap']:.6f} N")
    print(f"  total wing drag       : {wing_drag['total']:.6f} N")
    print(f"  tail drag             : {aero_diag['tail']['drag_tail_n']:.6f} N")
    print()
    print("Aerodynamic submodel diagnostics")
    print(f"  qbar                  : {aero_diag['wing']['qbar_pa']:.6f} Pa")
    print(f"  wing strips           : {len(aero_diag['wing']['strips'])}")
    print(f"  tail alpha            : {np.rad2deg(aero_diag['tail']['alpha_tail_rad']):.4f} deg")
    print(f"  tail cl               : {aero_diag['tail']['cl_tail']:.6f}")
    print()
    strips = aero_diag["wing"]["strips"]
    wake_ratios = np.array([s["wake_ratio"] for s in strips], dtype=float)
    cls = np.array([s["cl"] for s in strips], dtype=float)
    flap_flags = np.array([float(s["flap"]) for s in strips], dtype=float)
    print("Wing-strip summary")
    print(f"  wake ratio min/max    : {wake_ratios.min():.6f} / {wake_ratios.max():.6f}")
    print(f"  strip cl min/max      : {cls.min():.6f} / {cls.max():.6f}")
    print(f"  flap-active strips    : {int(flap_flags.sum())} of {len(flap_flags)}")
    print()
    print("Wing family totals")
    for family_name, family_diag in aero_diag["wing"]["families"].items():
        print(
            f"  {family_name:16s} area={family_diag['area_m2']:.4f} m^2, "
            f"n={family_diag['n_strips']:2d}, q=[{family_diag['q_local_min_pa']:.2f}, {family_diag['q_local_max_pa']:.2f}] Pa, "
            f"CL_mean={family_diag['cl_mean']:.4f}, CD_mean={family_diag['cd_mean']:.4f}, CM_mean={family_diag['cm_mean']:.4f}"
        )
    print()
    print("Propulsion summary")
    print(f"  left thrust           : {prop_diag['left_thrust_n']:.6f} N")
    print(f"  right thrust          : {prop_diag['right_thrust_n']:.6f} N")
    print(f"  total thrust          : {np.sum(prop_diag['per_prop_thrust_n']):.6f} N")


if __name__ == "__main__":
    main()
