from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from blown_aircraft.flight_history import (
    lake_lagunita_reference,
    local_offsets_to_geodetic,
    timestamped_history_path,
    write_flight_history_csv,
)
from blown_aircraft.geometry import load_vehicle
from blown_aircraft.linearize import linearize
from blown_aircraft.lqr import design_lqr
from blown_aircraft.operating_point import build_symmetric_cruise_operating_point
from blown_aircraft.rigid_body_ac import total_forces_and_moments
from blown_aircraft.plotting import save_figure


def longitudinal_state_derivative_10motor(
    x_lon: np.ndarray,
    u_lon: np.ndarray,
    vehicle,
    *,
    flap_trim_rad: float,
) -> np.ndarray:
    x_fwd, h, u, w, theta, q = np.asarray(x_lon, dtype=float)
    u_lon = np.asarray(u_lon, dtype=float)
    n_props = int(vehicle.propulsion["n_props"])
    rpm_vec = u_lon[:n_props]
    delta_e = float(u_lon[n_props])

    full_state = np.array([x_fwd, 0.0, -h, u, 0.0, w, 0.0, theta, 0.0, 0.0, q, 0.0], dtype=float)
    full_control = np.concatenate(
        [rpm_vec, np.array([delta_e, 0.0, 0.0, flap_trim_rad], dtype=float)],
        dtype=float,
    )

    force, moment, _ = total_forces_and_moments(full_state, full_control, vehicle)
    u_dot = force[0] / vehicle.mass_kg - q * w
    w_dot = force[2] / vehicle.mass_kg + q * u
    q_dot = moment[1] / vehicle.inertia[1, 1]
    theta_dot = q
    x_dot = u * np.cos(theta) + w * np.sin(theta)
    h_dot = u * np.sin(theta) - w * np.cos(theta)
    return np.array([x_dot, h_dot, u_dot, w_dot, theta_dot, q_dot], dtype=float)


def rk4_step(x: np.ndarray, u: np.ndarray, dt: float, vehicle, *, flap_trim_rad: float) -> np.ndarray:
    f = lambda xk: longitudinal_state_derivative_10motor(xk, u, vehicle, flap_trim_rad=flap_trim_rad)
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


def plot_state_response(t: np.ndarray, x_dev: np.ndarray, x_ref: np.ndarray) -> plt.Figure:
    labels = [
        r"$\Delta x$ (m)",
        r"$\Delta h$ (m)",
        r"$\Delta u$ (m/s)",
        r"$\Delta w$ (m/s)",
        r"$\Delta \theta$ (deg)",
        r"$\Delta q$ (deg/s)",
    ]
    x_plot = x_dev.copy()
    x_plot[:, 4:] = np.rad2deg(x_plot[:, 4:])

    fig, axes = plt.subplots(3, 2, figsize=(11, 9), dpi=120, constrained_layout=True)
    axes = axes.reshape(3, 2)
    for idx, ax in enumerate(axes.flat):
        ax.plot(t, x_plot[:, idx], linewidth=2.0, color="tab:blue")
        ax.plot(t, np.zeros_like(t), "--", linewidth=1.5, color="tab:gray")
        ax.set_ylabel(labels[idx])
        ax.set_xlabel("Time (s)")
        ax.grid(True, alpha=0.3)

    trim_text = (
        f"Trim: u={x_ref[0,2]:.2f} m/s, w={x_ref[0,3]:.2f} m/s, "
        f"theta={np.rad2deg(x_ref[0,4]):.2f} deg, q={np.rad2deg(x_ref[0,5]):.2f} deg/s"
    )
    fig.suptitle("Closed-Loop Longitudinal Response About Cruise Trim (10-Motor LQR)", fontsize=16)
    fig.text(0.5, 0.01, trim_text, ha="center", va="bottom", fontsize=9)
    return fig


def plot_control_response(t: np.ndarray, rpm_hist: np.ndarray, elevator_hist: np.ndarray, rpm_trim: np.ndarray, elevator_trim: float) -> plt.Figure:
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), dpi=120, constrained_layout=True)

    ax = axes[0]
    motor_lines = []
    for idx in range(rpm_hist.shape[1]):
        line = ax.plot(t, rpm_hist[:, idx], linewidth=1.2, alpha=0.9, label=f"M{idx + 1}")[0]
        motor_lines.append(line)
    trim_line = ax.plot(t, np.full_like(t, rpm_trim[0]), "--", linewidth=1.6, color="k", label="Trim RPM")[0]
    ax.set_ylabel("Motor RPM")
    ax.set_xlabel("Time (s)")
    ax.set_title("Per-Motor RPM Commands")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(t, np.rad2deg(elevator_hist), linewidth=2.0, color="tab:orange", label="Command")
    ax.plot(t, np.full_like(t, np.rad2deg(elevator_trim)), "--", linewidth=1.5, color="tab:gray", label="Trim")
    ax.set_ylabel("Elevator (deg)")
    ax.set_xlabel("Time (s)")
    ax.set_title("Elevator Command")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    axes[0].legend(
        motor_lines + [trim_line],
        [f"M{k + 1}" for k in range(rpm_hist.shape[1])] + ["Trim RPM"],
        loc="upper center",
        ncol=6,
        bbox_to_anchor=(0.5, -0.28),
        frameon=False,
    )
    fig.suptitle("Longitudinal 10-Motor Control Histories", fontsize=16)
    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Nonlinear longitudinal closed-loop demo with 10 individual motor inputs.")
    parser.add_argument("--t-final", type=float, default=12.0, help="Simulation duration in seconds.")
    parser.add_argument("--dt", type=float, default=0.02, help="Simulation step in seconds.")
    parser.add_argument("--lat", type=float, default=lake_lagunita_reference()["lat_deg"])
    parser.add_argument("--lon", type=float, default=lake_lagunita_reference()["lon_deg"])
    parser.add_argument("--alt", type=float, default=lake_lagunita_reference()["alt_m"])
    parser.add_argument("--show", action="store_true", help="Display figures interactively.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    vehicle = load_vehicle()
    op = build_symmetric_cruise_operating_point(vehicle, speed_mps=10.0, flight_path_angle_rad=0.0, flap_rad=0.0)
    n_props = int(vehicle.propulsion["n_props"])
    rpm_trim = np.asarray(op.full_control[:n_props], dtype=float)
    flap_trim = float(op.longitudinal_control[2])
    elevator_trim = float(op.longitudinal_control[1])

    x_trim = op.longitudinal_state.copy()
    u_trim = np.concatenate([rpm_trim, np.array([elevator_trim], dtype=float)])

    lin = linearize(
        lambda x, u: longitudinal_state_derivative_10motor(x, u, vehicle, flap_trim_rad=flap_trim),
        x_trim,
        u_trim,
        dt=0.05,
    )

    q_mat = np.diag([4.0, 8.0, 30.0, 12.0])
    r_diag = np.concatenate([np.full(n_props, 1.0e-7, dtype=float), np.array([2.0], dtype=float)])
    r_mat = np.diag(r_diag)
    state_idx = (2, 3, 4, 5)
    input_idx = tuple(range(n_props + 1))
    lqr = design_lqr(
        lin["A"],
        lin["B"],
        q_mat,
        r_mat,
        state_indices=state_idx,
        input_indices=input_idx,
        discrete_time=False,
    )

    t_final = float(args.t_final)
    dt = float(args.dt)
    t = np.arange(0.0, t_final + 0.5 * dt, dt)

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
    rpm_hist = np.zeros((len(t), n_props), dtype=float)
    elevator_hist = np.zeros(len(t), dtype=float)
    x_hist[0] = x0

    for k in range(len(t) - 1):
        xk = x_hist[k]
        dx_sub = xk[list(state_idx)] - x_trim[list(state_idx)]
        du = np.asarray(-lqr.k_gain @ dx_sub, dtype=float).reshape(-1)
        rpm_cmd = np.clip(rpm_trim + du[:n_props], rpm_min, rpm_max)
        elevator_cmd = float(np.clip(elevator_trim + du[n_props], -lim["elevator"], lim["elevator"]))
        uk = np.concatenate([rpm_cmd, np.array([elevator_cmd], dtype=float)])

        rpm_hist[k] = rpm_cmd
        elevator_hist[k] = elevator_cmd
        x_hist[k + 1] = rk4_step(xk, uk, dt, vehicle, flap_trim_rad=flap_trim)

    dx_sub = x_hist[-1, list(state_idx)] - x_trim[list(state_idx)]
    du = np.asarray(-lqr.k_gain @ dx_sub, dtype=float).reshape(-1)
    rpm_hist[-1] = np.clip(rpm_trim + du[:n_props], rpm_min, rpm_max)
    elevator_hist[-1] = float(np.clip(elevator_trim + du[n_props], -lim["elevator"], lim["elevator"]))

    x_ref = reference_trajectory(op, t)
    x_dev = x_hist - x_ref

    print("Closed-loop longitudinal nonlinear demo (10-motor input)")
    print(f"  simulation time       : {t_final:.2f} s")
    print(f"  dt                    : {dt:.3f} s")
    print(f"  final delta u         : {x_dev[-1, 2]:.6f} m/s")
    print(f"  final delta w         : {x_dev[-1, 3]:.6f} m/s")
    print(f"  final delta theta     : {np.rad2deg(x_dev[-1, 4]):.6f} deg")
    print(f"  final delta q         : {np.rad2deg(x_dev[-1, 5]):.6f} deg/s")
    print(f"  max motor RPM         : {np.max(rpm_hist):.3f}")
    print(f"  min motor RPM         : {np.min(rpm_hist):.3f}")
    print(f"  max elevator          : {np.rad2deg(np.max(elevator_hist)):.3f} deg")
    print(f"  min elevator          : {np.rad2deg(np.min(elevator_hist)):.3f} deg")

    fig_states = plot_state_response(t, x_dev, x_ref)
    fig_controls = plot_control_response(t, rpm_hist, elevator_hist, rpm_trim, elevator_trim)

    output_dir = REPO_ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_figure(fig_states, output_dir / "longitudinal_closed_loop_10motor_states.png")
    save_figure(fig_controls, output_dir / "longitudinal_closed_loop_10motor_controls.png")

    east_m = x_hist[:, 0]
    north_m = np.zeros(len(t), dtype=float)
    up_m = x_hist[:, 1]
    lat_deg, lon_deg, alt_m = local_offsets_to_geodetic(east_m, north_m, up_m, args.lat, args.lon, args.alt)
    theta_deg = np.rad2deg(x_hist[:, 4])
    q_deg_s = np.rad2deg(x_hist[:, 5])
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
        *[f"rpm_{i + 1}" for i in range(n_props)],
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
    half = n_props // 2
    rows = []
    for k in range(len(t)):
        row = {
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
            "q_deg_s": f"{q_deg_s[k]:.6f}",
            "r_deg_s": "0.000000",
            "collective_rpm": f"{np.mean(rpm_hist[k]):.6f}",
            "rpm_left": f"{np.mean(rpm_hist[k, :half]):.6f}",
            "rpm_right": f"{np.mean(rpm_hist[k, half:]):.6f}",
            "elevator_deg": f"{np.rad2deg(elevator_hist[k]):.6f}",
            "aileron_deg": "0.000000",
            "rudder_deg": "0.000000",
            "flap_deg": f"{np.rad2deg(flap_trim):.6f}",
            "controller": "lqr_10motor",
            "delta_x_m": f"{x_dev[k, 0]:.6f}",
            "delta_h_m": f"{x_dev[k, 1]:.6f}",
            "delta_u_mps": f"{x_dev[k, 2]:.6f}",
            "delta_w_mps": f"{x_dev[k, 3]:.6f}",
            "delta_theta_deg": f"{np.rad2deg(x_dev[k, 4]):.6f}",
            "delta_q_deg_s": f"{np.rad2deg(x_dev[k, 5]):.6f}",
        }
        for i in range(n_props):
            row[f"rpm_{i + 1}"] = f"{rpm_hist[k, i]:.6f}"
        rows.append(row)

    out_path = timestamped_history_path(output_dir / "flight_history", "longitudinal_lqr_10motor_flight")
    write_flight_history_csv(out_path, rows, fieldnames)
    print(f"  flight log            : {out_path}")

    if args.show and plt.get_backend().lower() != "agg":
        plt.show()
    else:
        plt.close(fig_states)
        plt.close(fig_controls)


if __name__ == "__main__":
    main()
