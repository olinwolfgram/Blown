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
from blown_aircraft.plotting import save_figure
from blown_aircraft.rigid_body_ac import total_forces_and_moments


def lateral_state_derivative_10motor(
    x_lat: np.ndarray,
    u_lat: np.ndarray,
    vehicle,
    *,
    w_trim_mps: float,
    theta_trim_rad: float,
    elevator_trim_rad: float,
    flap_trim_rad: float,
) -> np.ndarray:
    x_pos, y_pos, u, v, phi, psi, p, r = np.asarray(x_lat, dtype=float)
    u_lat = np.asarray(u_lat, dtype=float)
    n_props = int(vehicle.propulsion["n_props"])
    rpm_vec = u_lat[:n_props]
    delta_a = float(u_lat[n_props])
    delta_r = float(u_lat[n_props + 1])

    full_state = np.array([x_pos, y_pos, 0.0, u, v, w_trim_mps, phi, theta_trim_rad, psi, p, 0.0, r], dtype=float)
    full_control = np.concatenate(
        [rpm_vec, np.array([elevator_trim_rad, delta_a, delta_r, flap_trim_rad], dtype=float)],
        dtype=float,
    )

    force, moment, _ = total_forces_and_moments(full_state, full_control, vehicle)
    x_dot = u * np.cos(psi) - v * np.sin(psi)
    y_dot = u * np.sin(psi) + v * np.cos(psi)
    u_dot = force[0] / vehicle.mass_kg + r * v
    v_dot = force[1] / vehicle.mass_kg - r * u
    phi_dot = p
    psi_dot = r
    p_dot = moment[0] / vehicle.inertia[0, 0]
    r_dot = moment[2] / vehicle.inertia[2, 2]
    return np.array([x_dot, y_dot, u_dot, v_dot, phi_dot, psi_dot, p_dot, r_dot], dtype=float)


def rk4_step(
    x: np.ndarray,
    u: np.ndarray,
    dt: float,
    vehicle,
    *,
    w_trim_mps: float,
    theta_trim_rad: float,
    elevator_trim_rad: float,
    flap_trim_rad: float,
) -> np.ndarray:
    f = lambda xk: lateral_state_derivative_10motor(
        xk,
        u,
        vehicle,
        w_trim_mps=w_trim_mps,
        theta_trim_rad=theta_trim_rad,
        elevator_trim_rad=elevator_trim_rad,
        flap_trim_rad=flap_trim_rad,
    )
    k1 = f(x)
    k2 = f(x + 0.5 * dt * k1)
    k3 = f(x + 0.5 * dt * k2)
    k4 = f(x + dt * k3)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def plot_state_response(t: np.ndarray, x_dev: np.ndarray, x_ref: np.ndarray) -> plt.Figure:
    labels = [
        r"$\Delta x$ (m)",
        r"$\Delta y$ (m)",
        r"$\Delta u$ (m/s)",
        r"$\Delta v$ (m/s)",
        r"$\Delta \phi$ (deg)",
        r"$\Delta \psi$ (deg)",
        r"$\Delta p$ (deg/s)",
        r"$\Delta r$ (deg/s)",
    ]
    x_plot = x_dev.copy()
    x_plot[:, 4:] = np.rad2deg(x_plot[:, 4:])

    fig, axes = plt.subplots(4, 2, figsize=(11, 11), dpi=120, constrained_layout=True)
    axes = axes.reshape(4, 2)
    for idx, ax in enumerate(axes.flat):
        ax.plot(t, x_plot[:, idx], linewidth=2.0, color="tab:purple")
        ax.plot(t, np.zeros_like(t), "--", linewidth=1.5, color="tab:gray")
        ax.set_ylabel(labels[idx])
        ax.set_xlabel("Time (s)")
        ax.grid(True, alpha=0.3)

    trim_text = (
        f"Trim: u={x_ref[0,2]:.2f} m/s, v={x_ref[0,3]:.2f} m/s, "
        f"phi={np.rad2deg(x_ref[0,4]):.2f} deg, psi={np.rad2deg(x_ref[0,5]):.2f} deg"
    )
    fig.suptitle("Closed-Loop Lateral Response About Cruise Trim (10-Motor LQR)", fontsize=16)
    fig.text(0.5, 0.01, trim_text, ha="center", va="bottom", fontsize=9)
    return fig


def plot_control_response(
    t: np.ndarray,
    rpm_hist: np.ndarray,
    aileron_hist: np.ndarray,
    rudder_hist: np.ndarray,
    rpm_trim: np.ndarray,
    aileron_trim: float,
    rudder_trim: float,
) -> plt.Figure:
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), dpi=120, constrained_layout=True)

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
    ax.plot(t, np.rad2deg(aileron_hist), linewidth=2.0, color="tab:orange", label="Command")
    ax.plot(t, np.full_like(t, np.rad2deg(aileron_trim)), "--", linewidth=1.5, color="tab:gray", label="Trim")
    ax.set_ylabel("Aileron (deg)")
    ax.set_xlabel("Time (s)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    ax = axes[2]
    ax.plot(t, np.rad2deg(rudder_hist), linewidth=2.0, color="tab:green", label="Command")
    ax.plot(t, np.full_like(t, np.rad2deg(rudder_trim)), "--", linewidth=1.5, color="tab:gray", label="Trim")
    ax.set_ylabel("Rudder (deg)")
    ax.set_xlabel("Time (s)")
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
    fig.suptitle("Lateral 10-Motor Control Histories", fontsize=16)
    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Nonlinear lateral closed-loop demo with 10 individual motor inputs.")
    parser.add_argument("--t-final", type=float, default=10.0, help="Simulation duration in seconds.")
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
    elevator_trim = float(op.longitudinal_control[1])
    flap_trim = float(op.longitudinal_control[2])
    w_trim = float(op.longitudinal_state[3])
    theta_trim = float(op.longitudinal_state[4])

    x_trim = op.lateral_state.copy()
    u_trim = np.concatenate([rpm_trim, np.array([0.0, 0.0], dtype=float)])

    lin = linearize(
        lambda x, u: lateral_state_derivative_10motor(
            x,
            u,
            vehicle,
            w_trim_mps=w_trim,
            theta_trim_rad=theta_trim,
            elevator_trim_rad=elevator_trim,
            flap_trim_rad=flap_trim,
        ),
        x_trim,
        u_trim,
        dt=0.05,
    )

    q_mat = np.diag([8.0, 6.0, 25.0, 8.0, 30.0, 18.0])
    r_diag = np.concatenate([np.full(n_props, 1.0e-4, dtype=float), np.array([2.0, 2.0], dtype=float)])
    r_mat = np.diag(r_diag)
    state_idx = (2, 3, 4, 5, 6, 7)
    input_idx = tuple(range(n_props + 2))
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
    x0[2] += 0.25
    x0[3] += 0.40
    x0[4] += np.deg2rad(5.0)
    x0[5] += np.deg2rad(8.0)
    x0[6] += np.deg2rad(4.0)
    x0[7] += np.deg2rad(2.0)

    rpm_grid = np.asarray(vehicle.propulsion["rpm_grid"], dtype=float)
    rpm_min = float(rpm_grid.min())
    rpm_max = float(rpm_grid.max())
    lim = vehicle.control_limits_rad

    x_hist = np.zeros((len(t), 8), dtype=float)
    rpm_hist = np.zeros((len(t), n_props), dtype=float)
    aileron_hist = np.zeros(len(t), dtype=float)
    rudder_hist = np.zeros(len(t), dtype=float)
    x_hist[0] = x0

    for k in range(len(t) - 1):
        xk = x_hist[k]
        dx_sub = xk[list(state_idx)] - x_trim[list(state_idx)]
        du = np.asarray(-lqr.k_gain @ dx_sub, dtype=float).reshape(-1)
        rpm_cmd = np.clip(rpm_trim + du[:n_props], rpm_min, rpm_max)
        aileron_cmd = float(np.clip(du[n_props], -lim["aileron"], lim["aileron"]))
        rudder_cmd = float(np.clip(du[n_props + 1], -lim["rudder"], lim["rudder"]))
        uk = np.concatenate([rpm_cmd, np.array([aileron_cmd, rudder_cmd], dtype=float)])

        rpm_hist[k] = rpm_cmd
        aileron_hist[k] = aileron_cmd
        rudder_hist[k] = rudder_cmd
        x_hist[k + 1] = rk4_step(
            xk,
            uk,
            dt,
            vehicle,
            w_trim_mps=w_trim,
            theta_trim_rad=theta_trim,
            elevator_trim_rad=elevator_trim,
            flap_trim_rad=flap_trim,
        )

    dx_sub = x_hist[-1, list(state_idx)] - x_trim[list(state_idx)]
    du = np.asarray(-lqr.k_gain @ dx_sub, dtype=float).reshape(-1)
    rpm_hist[-1] = np.clip(rpm_trim + du[:n_props], rpm_min, rpm_max)
    aileron_hist[-1] = float(np.clip(du[n_props], -lim["aileron"], lim["aileron"]))
    rudder_hist[-1] = float(np.clip(du[n_props + 1], -lim["rudder"], lim["rudder"]))

    x_ref = np.tile(x_trim, (len(t), 1))
    x_dev = x_hist - x_ref

    print("Closed-loop lateral nonlinear demo (10-motor input)")
    print(f"  simulation time       : {t_final:.2f} s")
    print(f"  dt                    : {dt:.3f} s")
    print(f"  final delta u         : {x_dev[-1, 2]:.6f} m/s")
    print(f"  final delta v         : {x_dev[-1, 3]:.6f} m/s")
    print(f"  final delta phi       : {np.rad2deg(x_dev[-1, 4]):.6f} deg")
    print(f"  final delta psi       : {np.rad2deg(x_dev[-1, 5]):.6f} deg")
    print(f"  final delta p         : {np.rad2deg(x_dev[-1, 6]):.6f} deg/s")
    print(f"  final delta r         : {np.rad2deg(x_dev[-1, 7]):.6f} deg/s")
    print(f"  max motor RPM         : {np.max(rpm_hist):.3f}")
    print(f"  min motor RPM         : {np.min(rpm_hist):.3f}")
    print(f"  max |aileron|         : {np.rad2deg(np.max(np.abs(aileron_hist))):.3f} deg")
    print(f"  max |rudder|          : {np.rad2deg(np.max(np.abs(rudder_hist))):.3f} deg")

    fig_states = plot_state_response(t, x_dev, x_ref)
    fig_controls = plot_control_response(t, rpm_hist, aileron_hist, rudder_hist, rpm_trim, 0.0, 0.0)

    output_dir = REPO_ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_figure(fig_states, output_dir / "lateral_closed_loop_10motor_states.png")
    save_figure(fig_controls, output_dir / "lateral_closed_loop_10motor_controls.png")

    east_m = x_hist[:, 0]
    north_m = x_hist[:, 1]
    up_m = np.full(len(t), op.longitudinal_state[1], dtype=float)
    lat_deg, lon_deg, alt_m = local_offsets_to_geodetic(east_m, north_m, up_m, args.lat, args.lon, args.alt)
    phi_deg = np.rad2deg(x_hist[:, 4])
    psi_deg = np.rad2deg(x_hist[:, 5])
    heading_deg = 90.0 + psi_deg
    pitch_deg = np.full(len(t), np.rad2deg(theta_trim), dtype=float)
    p_deg_s = np.rad2deg(x_hist[:, 6])
    r_deg_s = np.rad2deg(x_hist[:, 7])

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
        "delta_y_m",
        "delta_v_mps",
        "delta_phi_deg",
        "delta_psi_deg",
        "delta_p_deg_s",
        "delta_r_deg_s",
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
            "y_m": f"{x_hist[k, 1]:.6f}",
            "h_m": f"{op.longitudinal_state[1]:.6f}",
            "u_mps": f"{x_hist[k, 2]:.6f}",
            "v_mps": f"{x_hist[k, 3]:.6f}",
            "w_mps": f"{w_trim:.6f}",
            "roll_deg": f"{phi_deg[k]:.6f}",
            "pitch_deg": f"{pitch_deg[k]:.6f}",
            "yaw_deg": f"{psi_deg[k]:.6f}",
            "heading_deg": f"{heading_deg[k]:.6f}",
            "p_deg_s": f"{p_deg_s[k]:.6f}",
            "q_deg_s": "0.000000",
            "r_deg_s": f"{r_deg_s[k]:.6f}",
            "collective_rpm": f"{np.mean(rpm_hist[k]):.6f}",
            "rpm_left": f"{np.mean(rpm_hist[k, :half]):.6f}",
            "rpm_right": f"{np.mean(rpm_hist[k, half:]):.6f}",
            "elevator_deg": f"{np.rad2deg(elevator_trim):.6f}",
            "aileron_deg": f"{np.rad2deg(aileron_hist[k]):.6f}",
            "rudder_deg": f"{np.rad2deg(rudder_hist[k]):.6f}",
            "flap_deg": f"{np.rad2deg(flap_trim):.6f}",
            "controller": "lqr_10motor",
            "delta_y_m": f"{x_dev[k, 1]:.6f}",
            "delta_v_mps": f"{x_dev[k, 3]:.6f}",
            "delta_phi_deg": f"{np.rad2deg(x_dev[k, 4]):.6f}",
            "delta_psi_deg": f"{np.rad2deg(x_dev[k, 5]):.6f}",
            "delta_p_deg_s": f"{np.rad2deg(x_dev[k, 6]):.6f}",
            "delta_r_deg_s": f"{np.rad2deg(x_dev[k, 7]):.6f}",
        }
        for i in range(n_props):
            row[f"rpm_{i + 1}"] = f"{rpm_hist[k, i]:.6f}"
        rows.append(row)

    out_path = timestamped_history_path(output_dir / "flight_history", "lateral_lqr_10motor_flight")
    write_flight_history_csv(out_path, rows, fieldnames)
    print(f"  flight log            : {out_path}")

    if args.show and plt.get_backend().lower() != "agg":
        plt.show()
    else:
        plt.close(fig_states)
        plt.close(fig_controls)


if __name__ == "__main__":
    main()
