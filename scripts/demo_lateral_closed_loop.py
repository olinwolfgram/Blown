from __future__ import annotations

import sys
from pathlib import Path

import argparse
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from blown_aircraft.geometry import load_vehicle
from blown_aircraft.lateral import lateral_state_derivative
from blown_aircraft.lqr import design_lqr
from blown_aircraft.operating_point import linearize_about_cruise
from blown_aircraft.plotting import (
    plot_lateral_closed_loop_response,
    save_figure,
)


def rk4_step(x: np.ndarray, u: np.ndarray, dt: float, vehicle, *, w_trim_mps: float, theta_trim_rad: float) -> np.ndarray:
    f = lambda xk: lateral_state_derivative(xk, u, vehicle, w_trim_mps=w_trim_mps, theta_trim_rad=theta_trim_rad)
    k1 = f(x)
    k2 = f(x + 0.5 * dt * k1)
    k3 = f(x + 0.5 * dt * k2)
    k4 = f(x + dt * k3)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Nonlinear lateral closed-loop demo about the cruise trim.")
    parser.add_argument("--t-final", type=float, default=40.0, help="Simulation duration in seconds.")
    parser.add_argument("--dt", type=float, default=0.02, help="Simulation step in seconds.")
    parser.add_argument("--show", action="store_true", help="Display figures interactively.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    vehicle = load_vehicle()
    result = linearize_about_cruise(vehicle, speed_mps=10.0, flight_path_angle_rad=0.0, flap_rad=0.0, dt=0.05)
    op = result["operating_point"]
    a_lat = result["lateral"]["A"]
    b_lat = result["lateral"]["B"]
    b_diff = (b_lat[:, [1]] - b_lat[:, [0]]).copy()
    b_ctrl = np.hstack([b_diff, b_lat[:, [2]], b_lat[:, [3]]])

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

    t_final = float(args.t_final)
    dt = float(args.dt)
    t = np.arange(0.0, t_final + 0.5 * dt, dt)

    x_trim = op.lateral_state.copy()
    u_trim = op.lateral_control.copy()
    w_trim_mps = float(op.longitudinal_state[3])
    theta_trim_rad = float(op.longitudinal_state[4])
    rpm_trim = float(u_trim[0])
    rpm_grid = np.asarray(vehicle.propulsion["rpm_grid"], dtype=float)
    rpm_min = float(rpm_grid.min())
    rpm_max = float(rpm_grid.max())
    lim = vehicle.control_limits_rad

    x0 = x_trim.copy()
    x0[2] += 0.25  # delta u [m/s]
    x0[3] += 0.40  # delta v [m/s]
    x0[4] += np.deg2rad(5.0)  # delta phi [rad]
    x0[5] += np.deg2rad(8.0)  # delta psi [rad]
    x0[6] += np.deg2rad(4.0)  # delta p [rad/s]
    x0[7] += np.deg2rad(2.0)  # delta r [rad/s]

    x_hist = np.zeros((len(t), 8), dtype=float)
    u_hist = np.zeros((len(t), 4), dtype=float)
    u_cmd_hist = np.zeros((len(t), 3), dtype=float)
    x_hist[0] = x0

    for k in range(len(t) - 1):
        xk = x_hist[k]
        dx_sub = xk[list(state_idx)] - x_trim[list(state_idx)]
        delta_u = np.asarray(-lqr.k_gain @ dx_sub, dtype=float).reshape(-1)
        delta_rpm_diff = float(delta_u[0])
        delta_da = float(delta_u[1])
        delta_dr = float(delta_u[2])

        rpm_left = float(np.clip(rpm_trim - delta_rpm_diff, rpm_min, rpm_max))
        rpm_right = float(np.clip(rpm_trim + delta_rpm_diff, rpm_min, rpm_max))
        da_cmd = float(np.clip(u_trim[2] + delta_da, -lim["aileron"], lim["aileron"]))
        dr_cmd = float(np.clip(u_trim[3] + delta_dr, -lim["rudder"], lim["rudder"]))
        uk = np.array([rpm_left, rpm_right, da_cmd, dr_cmd], dtype=float)

        u_hist[k] = uk
        u_cmd_hist[k] = np.array([delta_rpm_diff, da_cmd - u_trim[2], dr_cmd - u_trim[3]], dtype=float)
        x_hist[k + 1] = rk4_step(xk, uk, dt, vehicle, w_trim_mps=w_trim_mps, theta_trim_rad=theta_trim_rad)

    dx_sub = x_hist[-1, list(state_idx)] - x_trim[list(state_idx)]
    delta_u = np.asarray(-lqr.k_gain @ dx_sub, dtype=float).reshape(-1)
    delta_rpm_diff = float(delta_u[0])
    delta_da = float(delta_u[1])
    delta_dr = float(delta_u[2])
    da_cmd = float(np.clip(u_trim[2] + delta_da, -lim["aileron"], lim["aileron"]))
    dr_cmd = float(np.clip(u_trim[3] + delta_dr, -lim["rudder"], lim["rudder"]))
    u_cmd_hist[-1] = np.array([delta_rpm_diff, da_cmd - u_trim[2], dr_cmd - u_trim[3]], dtype=float)
    u_hist[-1] = np.array(
        [
            float(np.clip(rpm_trim - delta_rpm_diff, rpm_min, rpm_max)),
            float(np.clip(rpm_trim + delta_rpm_diff, rpm_min, rpm_max)),
            da_cmd,
            dr_cmd,
        ],
        dtype=float,
    )

    x_ref = np.tile(x_trim, (len(t), 1))
    x_dev = x_hist - x_ref
    print("Closed-loop lateral nonlinear demo")
    print(f"  simulation time       : {t_final:.2f} s")
    print(f"  dt                    : {dt:.3f} s")
    print(f"  initial delta u       : {x0[2] - x_trim[2]:.4f} m/s")
    print(f"  initial delta v       : {x0[3] - x_trim[3]:.4f} m/s")
    print(f"  initial delta phi     : {np.rad2deg(x0[4] - x_trim[4]):.4f} deg")
    print(f"  initial delta psi     : {np.rad2deg(x0[5] - x_trim[5]):.4f} deg")
    print(f"  initial delta p       : {np.rad2deg(x0[6] - x_trim[6]):.4f} deg/s")
    print(f"  initial delta r       : {np.rad2deg(x0[7] - x_trim[7]):.4f} deg/s")
    print()
    print("Final state deviation from lateral trim")
    print(f"  delta x               : {x_dev[-1, 0]:.6f} m")
    print(f"  delta y               : {x_dev[-1, 1]:.6f} m")
    print(f"  delta u               : {x_dev[-1, 2]:.6f} m/s")
    print(f"  delta v               : {x_dev[-1, 3]:.6f} m/s")
    print(f"  delta phi             : {np.rad2deg(x_dev[-1, 4]):.6f} deg")
    print(f"  delta psi             : {np.rad2deg(x_dev[-1, 5]):.6f} deg")
    print(f"  delta p               : {np.rad2deg(x_dev[-1, 6]):.6f} deg/s")
    print(f"  delta r               : {np.rad2deg(x_dev[-1, 7]):.6f} deg/s")
    print()
    print(f"  max |delta rpm diff|  : {np.max(np.abs(u_cmd_hist[:, 0])):.3f}")
    print(f"  max |delta aileron|   : {np.max(np.abs(np.rad2deg(u_cmd_hist[:, 1]))):.3f} deg")
    print(f"  max |delta rudder|    : {np.max(np.abs(np.rad2deg(u_cmd_hist[:, 2]))):.3f} deg")

    u_ref_plot = np.array([0.0, 0.0, 0.0], dtype=float)
    fig, _ = plot_lateral_closed_loop_response(t, x_dev, x_ref, u_cmd_hist, u_ref_plot)

    output_dir = REPO_ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_figure(fig, output_dir / "lateral_closed_loop_states.png")

    if args.show and plt.get_backend().lower() != "agg":
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
