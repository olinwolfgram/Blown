from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from blown_aircraft.finite_horizon import solve_finite_horizon_lqr
from blown_aircraft.flight_history import (
    lake_lagunita_reference,
    local_offsets_to_geodetic,
    timestamped_history_path,
    write_flight_history_csv,
)
from blown_aircraft.geometry import load_vehicle
from blown_aircraft.lateral import lateral_state_derivative
from blown_aircraft.lqr import design_lqr
from blown_aircraft.operating_point import build_symmetric_cruise_operating_point, linearize_about_cruise


def rk4_step(x: np.ndarray, u: np.ndarray, dt: float, vehicle, *, w_trim_mps: float, theta_trim_rad: float) -> np.ndarray:
    f = lambda xk: lateral_state_derivative(xk, u, vehicle, w_trim_mps=w_trim_mps, theta_trim_rad=theta_trim_rad)
    k1 = f(x)
    k2 = f(x + 0.5 * dt * k1)
    k3 = f(x + 0.5 * dt * k2)
    k4 = f(x + dt * k3)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export lateral closed-loop flight history for Cesium playback.")
    parser.add_argument("--controller", choices=("lqr", "finite"), default="lqr")
    parser.add_argument("--lat", type=float, default=lake_lagunita_reference()["lat_deg"])
    parser.add_argument("--lon", type=float, default=lake_lagunita_reference()["lon_deg"])
    parser.add_argument("--alt", type=float, default=lake_lagunita_reference()["alt_m"])
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--t-final", type=float, default=40.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    vehicle = load_vehicle()
    op = build_symmetric_cruise_operating_point(vehicle, speed_mps=10.0, flight_path_angle_rad=0.0, flap_rad=0.0)
    lat_lin = linearize_about_cruise(vehicle, speed_mps=10.0, flight_path_angle_rad=0.0, flap_rad=0.0, dt=args.dt)["lateral"]
    lon_trim = op.longitudinal_state.copy()
    lon_ctrl = op.longitudinal_control.copy()

    a_lat = lat_lin["A"]
    b_lat = lat_lin["B"]
    b_diff = (b_lat[:, [1]] - b_lat[:, [0]]).copy()
    ad_lat = lat_lin["Ad"]
    bd_diff = (lat_lin["Bd"][:, [1]] - lat_lin["Bd"][:, [0]]).copy()

    state_idx = (2, 3, 4, 5, 6, 7)
    q_mat = np.diag([8.0, 6.0, 25.0, 8.0, 30.0, 18.0])
    r_mat = np.array([[1.0e-3]])

    t = np.arange(0.0, args.t_final + 0.5 * args.dt, args.dt)
    horizon_steps = len(t) - 1

    if args.controller == "lqr":
        controller = design_lqr(
            a_lat,
            b_diff,
            q_mat,
            r_mat,
            state_indices=state_idx,
            input_indices=(0,),
            discrete_time=False,
        )
        finite_horizon = None
    else:
        finite_horizon = solve_finite_horizon_lqr(
            ad_lat,
            bd_diff,
            q_mat,
            r_mat,
            25.0 * q_mat,
            horizon_steps,
            state_indices=state_idx,
            input_indices=(0,),
        )
        controller = None

    x_trim = op.lateral_state.copy()
    u_trim = op.lateral_control.copy()
    w_trim_mps = float(op.longitudinal_state[3])
    theta_trim_rad = float(op.longitudinal_state[4])
    rpm_trim = float(u_trim[0])
    rpm_grid = np.asarray(vehicle.propulsion["rpm_grid"], dtype=float)
    rpm_min = float(rpm_grid.min())
    rpm_max = float(rpm_grid.max())

    x0 = x_trim.copy()
    x0[2] += 0.25
    x0[3] += 0.40
    x0[4] += np.deg2rad(5.0)
    x0[5] += np.deg2rad(8.0)
    x0[6] += np.deg2rad(4.0)
    x0[7] += np.deg2rad(2.0)

    x_hist = np.zeros((len(t), 8), dtype=float)
    u_hist = np.zeros((len(t), 4), dtype=float)
    u_diff_hist = np.zeros(len(t), dtype=float)
    x_hist[0] = x0

    for k in range(len(t) - 1):
        xk = x_hist[k]
        dx_sub = xk[list(state_idx)] - x_trim[list(state_idx)]
        if args.controller == "lqr":
            delta_rpm_diff = float((-controller.k_gain @ dx_sub).item())
        else:
            delta_rpm_diff = float((-finite_horizon.k_seq[k] @ dx_sub).item())

        rpm_left = float(np.clip(rpm_trim - delta_rpm_diff, rpm_min, rpm_max))
        rpm_right = float(np.clip(rpm_trim + delta_rpm_diff, rpm_min, rpm_max))
        uk = np.array([rpm_left, rpm_right, 0.0, 0.0], dtype=float)

        u_hist[k] = uk
        u_diff_hist[k] = delta_rpm_diff
        x_hist[k + 1] = rk4_step(xk, uk, args.dt, vehicle, w_trim_mps=w_trim_mps, theta_trim_rad=theta_trim_rad)

    u_hist[-1] = u_hist[-2]
    u_diff_hist[-1] = u_diff_hist[-2]
    x_ref = np.tile(x_trim, (len(t), 1))
    x_dev = x_hist - x_ref

    east_m = x_hist[:, 0]
    north_m = x_hist[:, 1]
    up_m = np.full(len(t), lon_trim[1], dtype=float)
    lat_deg, lon_deg, alt_m = local_offsets_to_geodetic(east_m, north_m, up_m, args.lat, args.lon, args.alt)

    phi_deg = np.rad2deg(x_hist[:, 4])
    psi_deg = np.rad2deg(x_hist[:, 5])
    heading_deg = 90.0 + psi_deg
    pitch_deg = np.full(len(t), np.rad2deg(lon_trim[4]), dtype=float)

    fieldnames = [
        "time_s",
        "lat_deg",
        "lon_deg",
        "alt_m",
        "east_m",
        "north_m",
        "up_m",
        "x_m",
        "y_m",
        "h_m",
        "u_mps",
        "v_mps",
        "w_mps",
        "roll_deg",
        "pitch_deg",
        "yaw_deg",
        "heading_deg",
        "p_deg_s",
        "q_deg_s",
        "r_deg_s",
        "collective_rpm",
        "rpm_left",
        "rpm_right",
        "elevator_deg",
        "aileron_deg",
        "rudder_deg",
        "flap_deg",
        "controller",
        "delta_y_m",
        "delta_v_mps",
        "delta_phi_deg",
        "delta_psi_deg",
        "delta_p_deg_s",
        "delta_r_deg_s",
        "delta_rpm_diff",
    ]

    rows = []
    for k in range(len(t)):
        collective = 0.5 * (u_hist[k, 0] + u_hist[k, 1])
        rows.append(
            {
                "time_s": f"{t[k]:.6f}",
                "lat_deg": f"{lat_deg[k]:.9f}",
                "lon_deg": f"{lon_deg[k]:.9f}",
                "alt_m": f"{alt_m[k]:.6f}",
                "east_m": f"{east_m[k]:.6f}",
                "north_m": f"{north_m[k]:.6f}",
                "up_m": f"{up_m[k]:.6f}",
                "x_m": f"{east_m[k]:.6f}",
                "y_m": f"{x_hist[k, 1]:.6f}",
                "h_m": f"{lon_trim[1]:.6f}",
                "u_mps": f"{x_hist[k, 2]:.6f}",
                "v_mps": f"{x_hist[k, 3]:.6f}",
                "w_mps": f"{lon_trim[3]:.6f}",
                "roll_deg": f"{phi_deg[k]:.6f}",
                "pitch_deg": f"{pitch_deg[k]:.6f}",
                "yaw_deg": f"{psi_deg[k]:.6f}",
                "heading_deg": f"{heading_deg[k]:.6f}",
                "p_deg_s": f"{np.rad2deg(x_hist[k, 6]):.6f}",
                "q_deg_s": "0.000000",
                "r_deg_s": f"{np.rad2deg(x_hist[k, 7]):.6f}",
                "collective_rpm": f"{collective:.6f}",
                "rpm_left": f"{u_hist[k, 0]:.6f}",
                "rpm_right": f"{u_hist[k, 1]:.6f}",
                "elevator_deg": f"{np.rad2deg(lon_ctrl[1]):.6f}",
                "aileron_deg": "0.000000",
                "rudder_deg": "0.000000",
                "flap_deg": f"{np.rad2deg(lon_ctrl[2]):.6f}",
                "controller": args.controller,
                "delta_y_m": f"{x_dev[k, 1]:.6f}",
                "delta_v_mps": f"{x_dev[k, 3]:.6f}",
                "delta_phi_deg": f"{np.rad2deg(x_dev[k, 4]):.6f}",
                "delta_psi_deg": f"{np.rad2deg(x_dev[k, 5]):.6f}",
                "delta_p_deg_s": f"{np.rad2deg(x_dev[k, 6]):.6f}",
                "delta_r_deg_s": f"{np.rad2deg(x_dev[k, 7]):.6f}",
                "delta_rpm_diff": f"{u_diff_hist[k]:.6f}",
            }
        )

    out_path = timestamped_history_path(REPO_ROOT / "outputs" / "flight_history", f"lateral_{args.controller}_control_flight")
    write_flight_history_csv(out_path, rows, fieldnames)
    print(f"Saved lateral flight history to {out_path}")


if __name__ == "__main__":
    main()
