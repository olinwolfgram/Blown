from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from blown_aircraft.geometry import load_vehicle
from blown_aircraft.jax_lateral import build_jax_lateral_dynamics
from blown_aircraft.lateral import lateral_state_derivative
from blown_aircraft.operating_point import build_symmetric_cruise_operating_point
from blown_aircraft.plotting import save_figure
from blown_aircraft.scp import solve_scp
from blown_aircraft.tvlqr import design_tvlqr, rollout_tvlqr


def rk4_step(x: np.ndarray, u: np.ndarray, dt: float, vehicle, *, w_trim_mps: float, theta_trim_rad: float) -> np.ndarray:
    f = lambda xk: lateral_state_derivative(xk, u, vehicle, w_trim_mps=w_trim_mps, theta_trim_rad=theta_trim_rad)
    k1 = f(x)
    k2 = f(x + 0.5 * dt * k1)
    k3 = f(x + 0.5 * dt * k2)
    k4 = f(x + dt * k3)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def finite_difference_jacobians(
    dynamics,
    x: np.ndarray,
    u: np.ndarray,
    eps_x: float = 1e-5,
    eps_u: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    u = np.asarray(u, dtype=float)
    nx = x.size
    nu = u.size
    a_mat = np.zeros((nx, nx), dtype=float)
    b_mat = np.zeros((nx, nu), dtype=float)

    for i in range(nx):
        dx = np.zeros(nx, dtype=float)
        dx[i] = eps_x
        a_mat[:, i] = (dynamics(x + dx, u) - dynamics(x - dx, u)) / (2.0 * eps_x)

    for j in range(nu):
        du = np.zeros(nu, dtype=float)
        du[j] = eps_u
        b_mat[:, j] = (dynamics(x, u + du) - dynamics(x, u - du)) / (2.0 * eps_u)

    return a_mat, b_mat


def parse_waypoint_string(spec: str, x_start: float, y_start: float, x_final_default: float) -> list[tuple[float, float]]:
    text = spec.strip()
    if not text:
        return [(x_final_default, y_start)]

    waypoints: list[tuple[float, float]] = []
    for chunk in text.split(";"):
        fields = chunk.strip().split(":")
        if len(fields) != 2:
            raise ValueError(
                "Lateral waypoints must use 'x:y' pairs separated by ';', "
                f"got {chunk!r}"
            )
        x_val = x_start + float(fields[0])
        y_val = y_start + float(fields[1])
        waypoints.append((x_val, y_val))
    return waypoints


def build_planar_lateral_dynamics(vehicle, dt: float, *, w_trim_mps: float, theta_trim_rad: float, derivatives: str = "jax"):
    base_dynamics, base_jacobian = build_jax_lateral_dynamics(
        vehicle,
        dt,
        w_trim_mps=w_trim_mps,
        theta_trim_rad=theta_trim_rad,
    )

    if derivatives == "finite-diff":
        def dynamics(x_lat: np.ndarray, u_lat: np.ndarray) -> np.ndarray:
            return rk4_step(x_lat, u_lat, dt, vehicle, w_trim_mps=w_trim_mps, theta_trim_rad=theta_trim_rad)

        def jacobian(x_lat: np.ndarray, u_lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            return finite_difference_jacobians(dynamics, x_lat, u_lat)

        return dynamics, jacobian

    return base_dynamics, base_jacobian


def build_waypoint_objective(
    t: np.ndarray,
    x0_aug: np.ndarray,
    waypoints: list[tuple[float, float]],
    *,
    q_running_diag: np.ndarray,
    q_waypoint_diag: np.ndarray,
    q_terminal_diag: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    horizon_steps = len(t) - 1
    nx = x0_aug.size
    q_seq = np.zeros((horizon_steps, nx, nx), dtype=float)
    if np.any(q_running_diag > 0.0):
        q_seq[:] = np.diag(q_running_diag)

    start_xy = np.array([x0_aug[0], x0_aug[1]], dtype=float)
    path_points = np.vstack([start_xy, np.asarray(waypoints, dtype=float)])
    if len(path_points) < 2:
        raise ValueError("At least one waypoint is required for the lateral mission.")

    x_ref = np.tile(x0_aug, (len(t), 1))
    mission_path = np.zeros((len(t), 2), dtype=float)
    mission_path[:, 0] = np.interp(t, np.linspace(t[0], t[-1], len(path_points)), path_points[:, 0])
    mission_path[:, 1] = np.interp(t, np.linspace(t[0], t[-1], len(path_points)), path_points[:, 1])

    x_start = float(path_points[0, 0])
    x_goal = float(path_points[-1, 0])
    x_span = max(x_goal - x_start, 1.0e-6)
    waypoint_indices: list[int] = []
    for i, (x_wp, y_wp) in enumerate(waypoints):
        progress = np.clip((float(x_wp) - x_start) / x_span, 0.0, 1.0)
        idx = int(np.clip(np.round(progress * horizon_steps), 1, len(t) - 1))
        waypoint_indices.append(idx)
        x_ref[idx, 0] = float(x_wp)
        x_ref[idx, 1] = float(y_wp)

        if i == 0:
            dx_path = float(x_wp - x_start)
            dy_path = float(y_wp - path_points[0, 1])
        else:
            dx_path = float(x_wp - waypoints[i - 1][0])
            dy_path = float(y_wp - waypoints[i - 1][1])
        x_ref[idx, 5] = np.arctan2(dy_path, max(dx_path, 1.0e-6))
        x_ref[idx, 7] = 0.0
        q_seq[min(idx, horizon_steps - 1)] += np.diag(q_waypoint_diag)

    qf_mat = np.diag(q_terminal_diag)
    x_ref[-1, 0] = path_points[-1, 0]
    x_ref[-1, 1] = path_points[-1, 1]
    if len(waypoints) >= 2:
        dx_terminal = float(waypoints[-1][0] - waypoints[-2][0])
        dy_terminal = float(waypoints[-1][1] - waypoints[-2][1])
    else:
        dx_terminal = float(waypoints[-1][0] - x_start)
        dy_terminal = float(waypoints[-1][1] - path_points[0, 1])
    x_ref[-1, 5] = np.arctan2(dy_terminal, max(dx_terminal, 1.0e-6))
    x_ref[-1, 7] = 0.0
    return x_ref, q_seq, qf_mat, mission_path


def plot_planner_tracker_results(
    t: np.ndarray,
    x_nom: np.ndarray,
    u_nom: np.ndarray,
    x_track: np.ndarray,
    u_track: np.ndarray,
    mission_path: np.ndarray,
    waypoints: list[tuple[float, float]],
) -> tuple[plt.Figure, plt.Figure]:
    fig_states, axes = plt.subplots(4, 3, figsize=(14, 12), dpi=120, constrained_layout=True)
    axes = axes.reshape(4, 3)
    labels = ["x (m)", "y (m)", "u (m/s)", "v (m/s)", r"$\phi$ (deg)", r"$\psi$ (deg)", r"$p$ (deg/s)", r"$r$ (deg/s)"]
    x_nom_plot = x_nom.copy()
    x_track_plot = x_track.copy()
    x_nom_plot[:, 4:] = np.rad2deg(x_nom_plot[:, 4:])
    x_track_plot[:, 4:] = np.rad2deg(x_track_plot[:, 4:])

    for idx, label in enumerate(labels):
        ax = axes.flat[idx]
        ax.plot(t, x_nom_plot[:, idx], "--", linewidth=1.8, color="tab:orange", label="SCP nominal")
        ax.plot(t, x_track_plot[:, idx], linewidth=2.0, color="tab:purple", label="TVLQR tracked")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
        if idx >= 5:
            ax.set_xlabel("Time (s)")

    u_nom_plot = u_nom.copy()
    u_track_plot = u_track.copy()
    u_nom_plot[:, 2:] = np.rad2deg(u_nom_plot[:, 2:])
    u_track_plot[:, 2:] = np.rad2deg(u_track_plot[:, 2:])
    control_labels = ["Left RPM", "Right RPM", "Aileron (deg)", "Rudder (deg)"]
    for j, label in enumerate(control_labels):
        ax = axes.flat[7 + j]
        ax.plot(t, u_nom_plot[:, j], "--", linewidth=1.8, color="tab:orange", label="SCP nominal")
        ax.plot(t, u_track_plot[:, j], linewidth=2.0, color="tab:blue", label="TVLQR tracked")
        ax.set_ylabel(label)
        ax.set_xlabel("Time (s)")
        ax.grid(True, alpha=0.3)

    axes[0, 0].legend(loc="best")
    axes[3, 1].legend(loc="best")
    fig_states.suptitle("Lateral SCP Nominal Trajectory and TVLQR Tracking", fontsize=16)

    fig_path, axes_path = plt.subplots(2, 2, figsize=(12, 8), dpi=120, constrained_layout=True)
    ax = axes_path[0, 0]
    ax.plot(mission_path[:, 0], mission_path[:, 1], "--", linewidth=1.8, color="tab:orange", label="Mission path")
    ax.plot(x_nom[:, 0], x_nom[:, 1], linewidth=2.0, color="tab:green", label="SCP nominal")
    ax.plot(x_track[:, 0], x_track[:, 1], linewidth=2.2, color="tab:purple", label="TVLQR tracked")
    wp_arr = np.asarray(waypoints, dtype=float)
    ax.scatter(wp_arr[:, 0], wp_arr[:, 1], color="tab:red", s=40, label="Waypoints")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Planar Waypoint Mission")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    axes_path[0, 1].plot(t, x_nom[:, 0], "--", linewidth=1.8, color="tab:green", label="Nominal x")
    axes_path[0, 1].plot(t, x_track[:, 0], linewidth=2.0, color="tab:purple", label="Tracked x")
    axes_path[0, 1].plot(t, mission_path[:, 0], ":", linewidth=1.6, color="tab:orange", label="Mission x")
    axes_path[0, 1].set_xlabel("Time (s)")
    axes_path[0, 1].set_ylabel("x (m)")
    axes_path[0, 1].set_title("Longitudinal Position")
    axes_path[0, 1].grid(True, alpha=0.3)
    axes_path[0, 1].legend(loc="best")

    axes_path[1, 0].plot(t, x_nom[:, 1], "--", linewidth=1.8, color="tab:green", label="Nominal y")
    axes_path[1, 0].plot(t, x_track[:, 1], linewidth=2.0, color="tab:purple", label="Tracked y")
    axes_path[1, 0].plot(t, mission_path[:, 1], ":", linewidth=1.6, color="tab:orange", label="Mission y")
    axes_path[1, 0].set_xlabel("Time (s)")
    axes_path[1, 0].set_ylabel("y (m)")
    axes_path[1, 0].set_title("Lateral Position")
    axes_path[1, 0].grid(True, alpha=0.3)
    axes_path[1, 0].legend(loc="best")

    path_err = np.linalg.norm(x_track[:, :2] - x_nom[:, :2], axis=1)
    axes_path[1, 1].plot(t, path_err, linewidth=2.0, color="tab:purple")
    axes_path[1, 1].set_xlabel("Time (s)")
    axes_path[1, 1].set_ylabel("Tracking error to nominal (m)")
    axes_path[1, 1].set_title("TVLQR Tracking Error")
    axes_path[1, 1].grid(True, alpha=0.3)

    fig_path.suptitle("Lateral Planner-Tracker Geometry", fontsize=16)
    return fig_states, fig_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lateral SCP planner with TVLQR trajectory tracking.")
    parser.add_argument("--show", action="store_true", help="Display matplotlib windows at the end of the run.")
    parser.add_argument("--dt", type=float, default=0.05, help="Simulation step for the reduced dynamics.")
    parser.add_argument("--t-final", type=float, default=8.0, help="Optimization horizon in seconds.")
    parser.add_argument("--max-iter", type=int, default=10, help="Maximum SCP iterations.")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-iteration SCP progress output.")
    parser.add_argument(
        "--waypoints",
        default="20:0.8;40:1.5;60:2.0",
        help="Relative planar waypoints as 'x:y;x:y;...'. Example: '20:1.5;45:0.0'.",
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
    lat_trim_resid = np.asarray(op.lateral_trim_residual, dtype=float)
    x0_plan = op.lateral_state.copy()
    u_trim = op.lateral_control.copy()
    w_trim_mps = float(op.longitudinal_state[3])
    theta_trim_rad = float(op.longitudinal_state[4])
    dynamics, dynamics_jacobian = build_planar_lateral_dynamics(
        vehicle,
        dt,
        w_trim_mps=w_trim_mps,
        theta_trim_rad=theta_trim_rad,
        derivatives=str(args.derivatives),
    )

    u_trim_mps = float(x0_plan[2])
    x_final_default = float(x0_plan[0] + u_trim_mps * requested_t_final)
    mission_waypoints = parse_waypoint_string(args.waypoints, float(x0_plan[0]), float(x0_plan[1]), x_final_default)
    farthest_x = max(float(x_wp) for x_wp, _ in mission_waypoints)
    mission_distance = max(0.0, farthest_x - float(x0_plan[0]))
    recommended_t_final = max(requested_t_final, 1.15 * mission_distance / max(u_trim_mps, 1e-3))
    t_final = float(recommended_t_final)
    t = np.arange(0.0, t_final + 0.5 * dt, dt)
    horizon_steps = len(t) - 1

    rpm_grid = np.asarray(vehicle.propulsion["rpm_grid"], dtype=float)
    lim = vehicle.control_limits_rad
    u_lower = np.array([float(rpm_grid.min()), float(rpm_grid.min()), -lim["aileron"], -lim["rudder"]], dtype=float)
    u_upper = np.array([float(rpm_grid.max()), float(rpm_grid.max()), lim["aileron"], lim["rudder"]], dtype=float)
    x_lower = np.array([-20.0, -100.0, 4.0, -5.0, np.deg2rad(-20.0), np.deg2rad(-45.0), np.deg2rad(-35.0), np.deg2rad(-35.0)], dtype=float)
    x_upper = np.array([120.0, 100.0, 20.0, 5.0, np.deg2rad(20.0), np.deg2rad(45.0), np.deg2rad(35.0), np.deg2rad(35.0)], dtype=float)

    x_ref, q_plan, qf_plan, mission_path = build_waypoint_objective(
        t,
        x0_plan,
        mission_waypoints,
        q_running_diag=np.array([0.0, 0.0, 1.5, 2.0, 4.0, 3.0, 4.0, 4.0], dtype=float),
        q_waypoint_diag=np.array([700.0, 900.0, 80.0, 0.0, 0.0, 40.0, 0.0, 0.0], dtype=float),
        q_terminal_diag=np.array([1200.0, 1400.0, 120.0, 6.0, 14.0, 12.0, 12.0, 12.0], dtype=float),
    )
    u_ref = np.repeat(u_trim[None, :], horizon_steps, axis=0)
    u_init = np.repeat(u_trim[None, :], horizon_steps, axis=0)
    r_plan = np.repeat(np.diag([5.0e-7, 5.0e-7, 0.20, 0.20])[None, :, :], horizon_steps, axis=0)
    rd_plan = np.diag([1.0e-6, 1.0e-6, 0.25, 0.25])
    rho_x = np.array([0.9, 0.7, 0.5, 0.35, np.deg2rad(3.0), np.deg2rad(4.0), np.deg2rad(4.0), np.deg2rad(4.0)], dtype=float)
    rho_u = np.array([120.0, 120.0, np.deg2rad(1.5), np.deg2rad(1.5)], dtype=float)

    print("Lateral SCP planner + TVLQR tracker")
    print("  planner decision vars  : x_k = [x, y, u, v, phi, psi, p, r], u_k = [rpm_left, rpm_right, aileron, rudder]")
    print("  tracker architecture   : TVLQR about the SCP nominal trajectory")
    print("  planner objective      : sparse waypoint/terminal penalties with running dynamic-state regularization")
    print(f"  planar model           : dynamic x-y kinematics with stateful forward speed, trim u0={u_trim_mps:.3f} m/s")
    print(f"  actuator bounds        : RPM in [{u_lower[0]:.1f}, {u_upper[0]:.1f}], aileron in [{np.rad2deg(u_lower[2]):.1f}, {np.rad2deg(u_upper[2]):.1f}] deg, rudder in [{np.rad2deg(u_lower[3]):.1f}, {np.rad2deg(u_upper[3]):.1f}] deg")
    print(f"  trust region           : rho_x={rho_x}, rho_u={rho_u}")
    print(f"  trim residual          : [u_dot={lat_trim_resid[0]:+.3e}, v_dot={lat_trim_resid[1]:+.3e}, psi_dot={lat_trim_resid[2]:+.3e}, r_dot={lat_trim_resid[3]:+.3e}]")
    if t_final > requested_t_final + 1.0e-9:
        print(f"  horizon adjustment     : requested {requested_t_final:.3f} s, auto-extended to {t_final:.3f} s for waypoint reachability")
    print(f"  derivatives            : {args.derivatives}")

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
    u_nom = np.zeros((len(t), 4), dtype=float)
    u_nom[:-1] = plan_result.u_seq
    u_nom[-1] = u_nom[-2]

    q_track = np.diag([40.0, 50.0, 20.0, 4.0, 10.0, 14.0, 10.0, 10.0])
    r_track = np.diag([1.0e-6, 1.0e-6, 0.15, 0.15])
    qf_track = 8.0 * q_track
    tvlqr_plan = design_tvlqr(
        dynamics,
        x_nom,
        plan_result.u_seq,
        q_track,
        r_track,
        qf_track,
        dynamics_jacobian=dynamics_jacobian,
    )

    x0_track = x_nom[0].copy()
    x0_track[1] += 0.15
    x0_track[2] += 0.15
    x0_track[3] += 0.05
    x0_track[4] += np.deg2rad(1.0)
    x0_track[5] += np.deg2rad(1.0)
    x0_track[6] += np.deg2rad(0.8)
    x0_track[7] += np.deg2rad(0.8)
    x_track, u_track_red = rollout_tvlqr(
        dynamics,
        x0_track,
        tvlqr_plan,
        u_lower=u_lower,
        u_upper=u_upper,
    )
    u_track = np.zeros((len(t), 4), dtype=float)
    u_track[:-1] = u_track_red
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
    print(f"  nominal terminal err y : {nominal_terminal_err[1]:.6f} m")
    print()
    print("TVLQR tracker result")
    print(f"  tracked terminal err x : {tracked_terminal_err[0]:.6f} m")
    print(f"  tracked terminal err y : {tracked_terminal_err[1]:.6f} m")
    print(f"  rms tracking err x     : {np.sqrt(np.mean(tracking_err[:, 0] ** 2)):.6f} m")
    print(f"  rms tracking err y     : {np.sqrt(np.mean(tracking_err[:, 1] ** 2)):.6f} m")
    print(f"  rms tracking err psi   : {np.rad2deg(np.sqrt(np.mean(tracking_err[:, 5] ** 2))):.6f} deg")
    print(f"  max |aileron|          : {np.rad2deg(np.max(np.abs(u_track[:, 2]))):.3f} deg")
    print(f"  max |rudder|           : {np.rad2deg(np.max(np.abs(u_track[:, 3]))):.3f} deg")

    fig_states, fig_path = plot_planner_tracker_results(t, x_nom, u_nom, x_track, u_track, mission_path, mission_waypoints)
    output_dir = REPO_ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_figure(fig_states, output_dir / "lateral_scp_tvlqr_states.png")
    save_figure(fig_path, output_dir / "lateral_scp_tvlqr_path.png")

    if args.show and plt.get_backend().lower() != "agg":
        plt.show()
    else:
        plt.close(fig_states)
        plt.close(fig_path)


if __name__ == "__main__":
    main()
