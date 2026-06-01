from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from blown_aircraft.geometry import load_vehicle
from blown_aircraft.jax_longitudinal import build_jax_longitudinal_dynamics
from blown_aircraft.longitudinal import longitudinal_state_derivative
from blown_aircraft.operating_point import build_symmetric_cruise_operating_point
from blown_aircraft.plotting import save_figure
from blown_aircraft.scp import solve_scp
from blown_aircraft.tvlqr import design_tvlqr, rollout_tvlqr


def parse_waypoint_string(spec: str, x_start: float, h_start: float, x_final_default: float) -> list[tuple[float, float]]:
    text = spec.strip()
    if not text:
        return [(x_final_default, h_start)]

    waypoints: list[tuple[float, float]] = []
    for chunk in text.split(";"):
        fields = chunk.strip().split(":")
        if len(fields) != 2:
            raise ValueError(
                "Longitudinal waypoints must use 'x:h' pairs separated by ';', "
                f"got {chunk!r}"
            )
        x_val = x_start + float(fields[0])
        h_val = h_start + float(fields[1])
        waypoints.append((x_val, h_val))
    return waypoints


def waypoint_schedule(
    t: np.ndarray,
    x0: np.ndarray,
    speed_nominal: float,
    waypoints: list[tuple[float, float]],
    *,
    q_running_diag: np.ndarray,
    q_waypoint_diag: np.ndarray,
    q_terminal_diag: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    nx = x0.size
    horizon_steps = len(t) - 1
    x_ref = np.tile(x0, (len(t), 1))
    q_seq = np.zeros((horizon_steps, nx, nx), dtype=float)
    qf_mat = np.diag(q_terminal_diag)

    if np.any(q_running_diag > 0.0):
        q_seq[:] = np.diag(q_running_diag)

    path_points = [(float(x0[0]), float(x0[1]))]
    for x_wp, h_wp in waypoints:
        path_points.append((x_wp, h_wp))
    path_points_arr = np.asarray(path_points, dtype=float)
    path_times = [float(t[0])]

    for i, (x_wp, h_wp) in enumerate(waypoints):
        dt_wp = max((x_wp - float(x0[0])) / max(speed_nominal, 1e-3), 0.0)
        idx = int(np.clip(np.round(dt_wp / max(float(t[1] - t[0]), 1e-6)), 1, len(t) - 1))
        x_ref[idx, 0] = x_wp
        x_ref[idx, 1] = h_wp
        path_times.append(float(np.clip(dt_wp, t[0], t[-1])))
        if i == len(waypoints) - 1:
            x_ref[-1, 0] = x_wp
            x_ref[-1, 1] = h_wp
        else:
            q_seq[min(idx, horizon_steps - 1)] += np.diag(q_waypoint_diag)

    mission_path = np.zeros((len(t), 2), dtype=float)
    mission_path[:, 0] = np.interp(t, np.asarray(path_times, dtype=float), path_points_arr[:, 0])
    mission_path[:, 1] = np.interp(t, np.asarray(path_times, dtype=float), path_points_arr[:, 1])
    return x_ref, q_seq, qf_mat, mission_path


def build_control_reference(u_trim: np.ndarray, horizon_steps: int) -> np.ndarray:
    return np.repeat(u_trim[None, :], horizon_steps, axis=0)


def rk4_step(x: np.ndarray, u: np.ndarray, dt: float, vehicle) -> np.ndarray:
    f = lambda xk, uk: longitudinal_state_derivative(xk, uk, vehicle)
    k1 = f(x, u)
    k2 = f(x + 0.5 * dt * k1, u)
    k3 = f(x + 0.5 * dt * k2, u)
    k4 = f(x + dt * k3, u)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def plot_planner_tracker_results(
    t: np.ndarray,
    x_nom: np.ndarray,
    u_nom: np.ndarray,
    x_track: np.ndarray,
    u_track: np.ndarray,
    mission_path: np.ndarray,
    waypoints: list[tuple[float, float]],
) -> tuple[plt.Figure, plt.Figure]:
    fig_states, axes = plt.subplots(3, 3, figsize=(14, 10), dpi=120, constrained_layout=True)
    axes = axes.reshape(3, 3)
    labels = ["x (m)", "h (m)", "u (m/s)", "w (m/s)", r"$\theta$ (deg)", r"$q$ (deg/s)"]
    x_nom_plot = x_nom.copy()
    x_track_plot = x_track.copy()
    x_nom_plot[:, 4:] = np.rad2deg(x_nom_plot[:, 4:])
    x_track_plot[:, 4:] = np.rad2deg(x_track_plot[:, 4:])

    for idx, label in enumerate(labels):
        ax = axes.flat[idx]
        ax.plot(t, x_nom_plot[:, idx], "--", linewidth=1.8, color="tab:orange", label="SCP nominal")
        ax.plot(t, x_track_plot[:, idx], linewidth=2.0, color="tab:blue", label="TVLQR tracked")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
        if idx >= 3:
            ax.set_xlabel("Time (s)")

    u_nom_plot = u_nom.copy()
    u_track_plot = u_track.copy()
    u_nom_plot[:, 1:] = np.rad2deg(u_nom_plot[:, 1:])
    u_track_plot[:, 1:] = np.rad2deg(u_track_plot[:, 1:])
    control_labels = ["Collective RPM", "Elevator (deg)", "Flap (deg)"]
    for j, label in enumerate(control_labels):
        ax = axes.flat[6 + j]
        ax.plot(t, u_nom_plot[:, j], "--", linewidth=1.8, color="tab:orange", label="SCP nominal")
        ax.plot(t, u_track_plot[:, j], linewidth=2.0, color="tab:blue", label="TVLQR tracked")
        ax.set_ylabel(label)
        ax.set_xlabel("Time (s)")
        ax.grid(True, alpha=0.3)

    axes[0, 0].legend(loc="best")
    axes[2, 0].legend(loc="best")
    fig_states.suptitle("Longitudinal SCP Nominal Trajectory and TVLQR Tracking", fontsize=16)

    fig_path, axes_path = plt.subplots(2, 2, figsize=(12, 8), dpi=120, constrained_layout=True)
    ax = axes_path[0, 0]
    ax.plot(mission_path[:, 0], mission_path[:, 1], "--", linewidth=1.8, color="tab:orange", label="Mission path")
    ax.plot(x_nom[:, 0], x_nom[:, 1], linewidth=2.0, color="tab:green", label="SCP nominal")
    ax.plot(x_track[:, 0], x_track[:, 1], linewidth=2.2, color="tab:blue", label="TVLQR tracked")
    wp_arr = np.asarray(waypoints, dtype=float)
    ax.scatter(wp_arr[:, 0], wp_arr[:, 1], color="tab:red", s=40, label="Waypoints")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("h (m)")
    ax.set_title("Waypoint Mission in x-h Plane")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    axes_path[0, 1].plot(t, x_nom[:, 0], "--", linewidth=1.8, color="tab:green", label="Nominal x")
    axes_path[0, 1].plot(t, x_track[:, 0], linewidth=2.0, color="tab:blue", label="Tracked x")
    axes_path[0, 1].plot(t, mission_path[:, 0], ":", linewidth=1.6, color="tab:orange", label="Mission x")
    axes_path[0, 1].set_xlabel("Time (s)")
    axes_path[0, 1].set_ylabel("x (m)")
    axes_path[0, 1].set_title("Longitudinal Position")
    axes_path[0, 1].grid(True, alpha=0.3)
    axes_path[0, 1].legend(loc="best")

    axes_path[1, 0].plot(t, x_nom[:, 1], "--", linewidth=1.8, color="tab:green", label="Nominal h")
    axes_path[1, 0].plot(t, x_track[:, 1], linewidth=2.0, color="tab:blue", label="Tracked h")
    axes_path[1, 0].plot(t, mission_path[:, 1], ":", linewidth=1.6, color="tab:orange", label="Mission h")
    axes_path[1, 0].set_xlabel("Time (s)")
    axes_path[1, 0].set_ylabel("h (m)")
    axes_path[1, 0].set_title("Altitude")
    axes_path[1, 0].grid(True, alpha=0.3)
    axes_path[1, 0].legend(loc="best")

    path_err = np.linalg.norm(x_track[:, :2] - x_nom[:, :2], axis=1)
    axes_path[1, 1].plot(t, path_err, linewidth=2.0, color="tab:purple")
    axes_path[1, 1].set_xlabel("Time (s)")
    axes_path[1, 1].set_ylabel("Tracking error to nominal (m)")
    axes_path[1, 1].set_title("TVLQR Tracking Error")
    axes_path[1, 1].grid(True, alpha=0.3)

    fig_path.suptitle("Longitudinal Planner-Tracker Geometry", fontsize=16)
    return fig_states, fig_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Longitudinal SCP planner with TVLQR trajectory tracking.")
    parser.add_argument("--show", action="store_true", help="Display matplotlib windows at the end of the run.")
    parser.add_argument("--dt", type=float, default=0.05, help="Simulation step for the reduced dynamics.")
    parser.add_argument("--t-final", type=float, default=8.0, help="Optimization horizon in seconds.")
    parser.add_argument("--max-iter", type=int, default=12, help="Maximum SCP iterations.")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-iteration SCP progress output.")
    parser.add_argument(
        "--waypoints",
        default="18:-0.75;35:-0.75;58:0.0",
        help="Relative longitudinal waypoints as 'x:h;x:h;...'. Example: '20:-1.0;45:0.0'.",
    )
    parser.add_argument("--solver", choices=("CLARABEL", "OSQP", "SCS"), default="CLARABEL")
    parser.add_argument("--derivatives", choices=("jax", "finite-diff"), default="jax")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    vehicle = load_vehicle()
    dt = float(args.dt)
    requested_t_final = float(args.t_final)

    op = build_symmetric_cruise_operating_point(vehicle, speed_mps=10.0, flight_path_angle_rad=0.0, flap_rad=0.0)
    lon_trim_resid = np.asarray(op.longitudinal_trim_residual, dtype=float)
    x0_plan = op.longitudinal_state.copy()
    u_trim_full = op.longitudinal_control.copy()
    u_trim = u_trim_full[:2].copy()
    if args.derivatives == "jax":
        dynamics, dynamics_jacobian = build_jax_longitudinal_dynamics(vehicle, dt)
    else:
        def dynamics(xk: np.ndarray, uk_reduced: np.ndarray) -> np.ndarray:
            uk_full = u_trim_full.copy()
            uk_full[:2] = uk_reduced
            return rk4_step(xk, uk_full, dt, vehicle)

        dynamics_jacobian = None

    speed_nominal = max(float(op.longitudinal_state_derivative[0]), 1e-3)
    x_final_default = float(x0_plan[0] + speed_nominal * requested_t_final)
    mission_waypoints = parse_waypoint_string(args.waypoints, float(x0_plan[0]), float(x0_plan[1]), x_final_default)
    farthest_x = max(float(x_wp) for x_wp, _ in mission_waypoints)
    mission_distance = max(0.0, farthest_x - float(x0_plan[0]))
    recommended_t_final = max(requested_t_final, 1.15 * mission_distance / speed_nominal)
    t_final = float(recommended_t_final)
    t = np.arange(0.0, t_final + 0.5 * dt, dt)
    horizon_steps = len(t) - 1

    rpm_grid = np.asarray(vehicle.propulsion["rpm_grid"], dtype=float)
    lim = vehicle.control_limits_rad
    u_lower = np.array([float(rpm_grid.min()), -lim["elevator"]], dtype=float)
    u_upper = np.array([float(rpm_grid.max()), lim["elevator"]], dtype=float)
    x_bound = 1.0e6
    x_lower = np.array([-x_bound, -120.0, 4.0, -5.0, np.deg2rad(-20.0), np.deg2rad(-45.0)], dtype=float)
    x_upper = np.array([x_bound, 120.0, 20.0, 5.0, np.deg2rad(20.0), np.deg2rad(45.0)], dtype=float)

    x_ref, q_plan, qf_plan, mission_path = waypoint_schedule(
        t,
        x0_plan,
        speed_nominal,
        mission_waypoints,
        q_running_diag=np.array([0.0, 0.0, 0.8, 2.0, 1.5, 2.0], dtype=float),
        q_waypoint_diag=np.array([200.0, 350.0, 0.0, 0.0, 0.0, 0.0], dtype=float),
        q_terminal_diag=np.array([350.0, 550.0, 20.0, 30.0, 25.0, 30.0], dtype=float),
    )
    u_ref = build_control_reference(u_trim, horizon_steps)
    u_init = build_control_reference(u_trim, horizon_steps)
    r_plan = np.repeat(np.diag([5.0e-7, 0.10])[None, :, :], horizon_steps, axis=0)
    rd_plan = np.diag([1.0e-6, 0.35])
    rho_x = np.array([1.5, 1.2, 0.75, 0.6, np.deg2rad(5.0), np.deg2rad(8.0)], dtype=float)
    rho_u = np.array([150.0, np.deg2rad(2.0)], dtype=float)

    print("Longitudinal SCP planner + TVLQR tracker")
    print("  planner decision vars  : x_k = [x, h, u, w, theta, q], u_k = [collective RPM, elevator]")
    print("  tracker architecture   : TVLQR about the SCP nominal trajectory")
    print(f"  actuator bounds        : RPM in [{u_lower[0]:.1f}, {u_upper[0]:.1f}], elevator in [{np.rad2deg(u_lower[1]):.1f}, {np.rad2deg(u_upper[1]):.1f}] deg")
    print(f"  trust region           : rho_x={rho_x}, rho_u={rho_u}")
    print(f"  trim residual          : [u_dot={lon_trim_resid[0]:+.3e}, w_dot={lon_trim_resid[1]:+.3e}, q_dot={lon_trim_resid[2]:+.3e}]")
    if t_final > requested_t_final + 1.0e-9:
        print(f"  horizon adjustment     : requested {requested_t_final:.3f} s, auto-extended to {t_final:.3f} s for waypoint reachability")
    print(f"  derivatives            : {args.derivatives}")
    if dt > 0.1:
        print("  warning                : dt > 0.1 s can destabilize the initial longitudinal trim rollout.")

    plan_result = solve_scp(
        dynamics,
        x0_plan,
        u_init,
        x_ref,
        u_ref,
        q_plan,
        r_plan,
        qf_plan,
        rd_mat=rd_plan,
        u_lower=u_lower,
        u_upper=u_upper,
        x_lower=x_lower,
        x_upper=x_upper,
        trust_region_state=rho_x,
        trust_region_input=rho_u,
        max_iter=int(args.max_iter),
        tol=1.0e-3,
        solver=str(args.solver),
        verbose=not args.quiet,
        dynamics_jacobian=dynamics_jacobian,
    )

    x_nom = plan_result.x_seq
    u_nom_red = plan_result.u_seq
    u_nom = np.zeros((len(t), 3), dtype=float)
    u_nom[:-1, :2] = u_nom_red
    u_nom[:-1, 2] = u_trim_full[2]
    u_nom[-1] = u_nom[-2]

    q_track = np.diag([30.0, 45.0, 8.0, 18.0, 14.0, 16.0])
    r_track = np.diag([1.0e-6, 0.25])
    qf_track = 8.0 * q_track
    tvlqr_plan = design_tvlqr(
        dynamics,
        x_nom,
        u_nom_red,
        q_track,
        r_track,
        qf_track,
        dynamics_jacobian=dynamics_jacobian,
    )

    x0_track = x_nom[0].copy()
    x0_track[2] += -0.25
    x0_track[3] += 0.08
    x0_track[4] += np.deg2rad(1.5)
    x0_track[5] += np.deg2rad(0.8)
    x_track, u_track_red = rollout_tvlqr(
        dynamics,
        x0_track,
        tvlqr_plan,
        u_lower=u_lower,
        u_upper=u_upper,
    )

    u_track = np.zeros((len(t), 3), dtype=float)
    u_track[:-1, :2] = u_track_red
    u_track[:-1, 2] = u_trim_full[2]
    u_track[-1] = u_track[-2]

    terminal_wp = np.asarray(mission_waypoints[-1], dtype=float)
    nominal_terminal_err = x_nom[-1, :2] - terminal_wp
    tracked_terminal_err = x_track[-1, :2] - terminal_wp
    tracking_err = x_track - x_nom

    print()
    print("SCP planner result")
    print(f"  converged              : {plan_result.converged}")
    print(f"  termination reason     : {plan_result.termination_reason}")
    print(f"  iterations             : {plan_result.iterations}")
    print(f"  initial cost           : {plan_result.cost_history[0]:.6f}")
    print(f"  final cost             : {plan_result.cost_history[-1]:.6f}")
    print(f"  nominal terminal err x : {nominal_terminal_err[0]:.6f} m")
    print(f"  nominal terminal err h : {nominal_terminal_err[1]:.6f} m")
    print()
    print("TVLQR tracker result")
    print(f"  tracked terminal err x : {tracked_terminal_err[0]:.6f} m")
    print(f"  tracked terminal err h : {tracked_terminal_err[1]:.6f} m")
    print(f"  rms tracking err x     : {np.sqrt(np.mean(tracking_err[:, 0] ** 2)):.6f} m")
    print(f"  rms tracking err h     : {np.sqrt(np.mean(tracking_err[:, 1] ** 2)):.6f} m")
    print(f"  rms tracking err theta : {np.rad2deg(np.sqrt(np.mean(tracking_err[:, 4] ** 2))):.6f} deg")
    print(f"  max collective RPM     : {np.max(u_track[:, 0]):.3f}")
    print(f"  min collective RPM     : {np.min(u_track[:, 0]):.3f}")
    print(f"  max elevator           : {np.rad2deg(np.max(u_track[:, 1])):.3f} deg")
    print(f"  min elevator           : {np.rad2deg(np.min(u_track[:, 1])):.3f} deg")

    fig_states, fig_path = plot_planner_tracker_results(t, x_nom, u_nom, x_track, u_track, mission_path, mission_waypoints)
    output_dir = REPO_ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_figure(fig_states, output_dir / "longitudinal_scp_tvlqr_states.png")
    save_figure(fig_path, output_dir / "longitudinal_scp_tvlqr_path.png")

    if args.show and plt.get_backend().lower() != "agg":
        plt.show()
    else:
        plt.close(fig_states)
        plt.close(fig_path)


if __name__ == "__main__":
    main()
