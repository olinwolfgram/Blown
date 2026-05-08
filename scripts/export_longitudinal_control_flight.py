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
from blown_aircraft.longitudinal import longitudinal_state_derivative
from blown_aircraft.lqr import design_lqr
from blown_aircraft.operating_point import build_symmetric_cruise_operating_point, linearize_about_cruise


def rk4_step(x: np.ndarray, u: np.ndarray, dt: float, vehicle) -> np.ndarray:
    f = lambda xk: longitudinal_state_derivative(xk, u, vehicle)
    k1 = f(x)
    k2 = f(x + 0.5 * dt * k1)
    k3 = f(x + 0.5 * dt * k2)
    k4 = f(x + dt * k3)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def reference_trajectory(op, t: np.ndarray) -> np.ndarray:
    x_ref = np.zeros((len(t), 6), dtype=float)
    x0 = op.longitudinal_state
    dx0 = op.longitudinal_state_derivative
    for k, tk in enumerate(t):
        x_ref[k, 0] = x0[0] + dx0[0] * tk
        x_ref[k, 1] = x0[1] + dx0[1] * tk
        x_ref[k, 2:] = x0[2:]
    return x_ref


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export longitudinal closed-loop flight history for Cesium playback.")
    parser.add_argument("--controller", choices=("lqr", "finite"), default="lqr")
    parser.add_argument("--lat", type=float, default=lake_lagunita_reference()["lat_deg"])
    parser.add_argument("--lon", type=float, default=lake_lagunita_reference()["lon_deg"])
    parser.add_argument("--alt", type=float, default=lake_lagunita_reference()["alt_m"])
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--t-final", type=float, default=12.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    vehicle = load_vehicle()
    op = build_symmetric_cruise_operating_point(vehicle, speed_mps=10.0, flight_path_angle_rad=0.0, flap_rad=0.0)
    lin = linearize_about_cruise(vehicle, speed_mps=10.0, flight_path_angle_rad=0.0, flap_rad=0.0, dt=args.dt)["longitudinal"]

    state_idx = (2, 3, 4, 5)
    input_idx = (0, 1)
    q_mat = np.diag([4.0, 8.0, 30.0, 12.0])
    r_mat = np.diag([1.0e-6, 2.0])

    t = np.arange(0.0, args.t_final + 0.5 * args.dt, args.dt)
    horizon_steps = len(t) - 1

    if args.controller == "lqr":
        controller = design_lqr(
            lin["A"],
            lin["B"],
            q_mat,
            r_mat,
            state_indices=state_idx,
            input_indices=input_idx,
            discrete_time=False,
        )
        finite_horizon = None
    else:
        finite_horizon = solve_finite_horizon_lqr(
            lin["Ad"],
            lin["Bd"],
            q_mat,
            r_mat,
            25.0 * q_mat,
            horizon_steps,
            state_indices=state_idx,
            input_indices=input_idx,
        )
        controller = None

    x_trim = op.longitudinal_state.copy()
    u_trim = op.longitudinal_control.copy()
    x0 = x_trim.copy()
    x0[2] += -0.75
    x0[3] += 0.20
    x0[4] += np.deg2rad(4.0)
    x0[5] += np.deg2rad(1.5)

    rpm_grid = np.asarray(vehicle.propulsion["rpm_grid"], dtype=float)
    rpm_min = float(rpm_grid.min())
    rpm_max = float(rpm_grid.max())
    lim = vehicle.control_limits_rad

    x_hist = np.zeros((len(t), 6), dtype=float)
    u_hist = np.zeros((len(t), 3), dtype=float)
    x_hist[0] = x0

    for k in range(len(t) - 1):
        xk = x_hist[k]
        dx_sub = xk[list(state_idx)] - x_trim[list(state_idx)]
        if args.controller == "lqr":
            du = -controller.k_gain @ dx_sub
        else:
            du = -finite_horizon.k_seq[k] @ dx_sub

        uk = u_trim.copy()
        uk[0] = float(np.clip(u_trim[0] + du[0], rpm_min, rpm_max))
        uk[1] = float(np.clip(u_trim[1] + du[1], -lim["elevator"], lim["elevator"]))
        uk[2] = float(np.clip(u_trim[2], 0.0, lim["flap"]))
        u_hist[k] = uk
        x_hist[k + 1] = rk4_step(xk, uk, args.dt, vehicle)

    u_hist[-1] = u_hist[-2]
    x_ref = reference_trajectory(op, t)
    x_dev = x_hist - x_ref

    east_m = x_hist[:, 0]
    north_m = np.zeros(len(t), dtype=float)
    up_m = x_hist[:, 1]
    lat_deg, lon_deg, alt_m = local_offsets_to_geodetic(east_m, north_m, up_m, args.lat, args.lon, args.alt)

    theta_deg = np.rad2deg(x_hist[:, 4])
    heading_deg = np.full(len(t), 90.0, dtype=float)

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
        "delta_x_m",
        "delta_h_m",
        "delta_u_mps",
        "delta_w_mps",
        "delta_theta_deg",
        "delta_q_deg_s",
    ]

    rows = []
    for k in range(len(t)):
        rows.append(
            {
                "time_s": f"{t[k]:.6f}",
                "lat_deg": f"{lat_deg[k]:.9f}",
                "lon_deg": f"{lon_deg[k]:.9f}",
                "alt_m": f"{alt_m[k]:.6f}",
                "east_m": f"{east_m[k]:.6f}",
                "north_m": f"{north_m[k]:.6f}",
                "up_m": f"{up_m[k]:.6f}",
                "x_m": f"{x_hist[k, 0]:.6f}",
                "y_m": "0.000000",
                "h_m": f"{x_hist[k, 1]:.6f}",
                "u_mps": f"{x_hist[k, 2]:.6f}",
                "v_mps": "0.000000",
                "w_mps": f"{x_hist[k, 3]:.6f}",
                "roll_deg": "0.000000",
                "pitch_deg": f"{theta_deg[k]:.6f}",
                "yaw_deg": "0.000000",
                "heading_deg": f"{heading_deg[k]:.6f}",
                "p_deg_s": "0.000000",
                "q_deg_s": f"{np.rad2deg(x_hist[k, 5]):.6f}",
                "r_deg_s": "0.000000",
                "collective_rpm": f"{u_hist[k, 0]:.6f}",
                "rpm_left": f"{u_hist[k, 0]:.6f}",
                "rpm_right": f"{u_hist[k, 0]:.6f}",
                "elevator_deg": f"{np.rad2deg(u_hist[k, 1]):.6f}",
                "aileron_deg": "0.000000",
                "rudder_deg": "0.000000",
                "flap_deg": f"{np.rad2deg(u_hist[k, 2]):.6f}",
                "controller": args.controller,
                "delta_x_m": f"{x_dev[k, 0]:.6f}",
                "delta_h_m": f"{x_dev[k, 1]:.6f}",
                "delta_u_mps": f"{x_dev[k, 2]:.6f}",
                "delta_w_mps": f"{x_dev[k, 3]:.6f}",
                "delta_theta_deg": f"{np.rad2deg(x_dev[k, 4]):.6f}",
                "delta_q_deg_s": f"{np.rad2deg(x_dev[k, 5]):.6f}",
            }
        )

    out_path = timestamped_history_path(REPO_ROOT / "outputs" / "flight_history", f"longitudinal_{args.controller}_control_flight")
    write_flight_history_csv(out_path, rows, fieldnames)
    print(f"Saved longitudinal flight history to {out_path}")


if __name__ == "__main__":
    main()
