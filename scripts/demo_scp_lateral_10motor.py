from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from blown_aircraft.geometry import load_vehicle
from blown_aircraft.operating_point import build_symmetric_cruise_operating_point
from blown_aircraft.plotting import save_figure
from blown_aircraft.reduced_10motor import lateral_state_derivative_10motor
from blown_aircraft.scp import solve_scp


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


def parse_terminal(spec: str) -> tuple[float, float]:
    result = {"x": 20.0, "y": 5.0}
    if spec.strip():
        for chunk in spec.split(","):
            key, value = chunk.split(":", 1)
            result[key.strip().lower()] = float(value.strip())
    return float(result["x"]), float(result["y"])


def plot_state_response(t: np.ndarray, x_hist: np.ndarray, x_target: np.ndarray) -> plt.Figure:
    labels = ["x (m)", "y (m)", "u (m/s)", "v (m/s)", r"$\phi$ (deg)", r"$\psi$ (deg)", r"$p$ (deg/s)", r"$r$ (deg/s)"]
    x_plot = x_hist.copy()
    x_plot[:, 4:] = np.rad2deg(x_plot[:, 4:])
    target_plot = x_target.copy()
    target_plot[4:] = np.rad2deg(target_plot[4:])

    fig, axes = plt.subplots(4, 2, figsize=(11, 11), dpi=120, constrained_layout=True)
    for idx, ax in enumerate(np.asarray(axes).reshape(-1)):
        ax.plot(t, x_plot[:, idx], linewidth=2.0, color="tab:purple", label="SCP nominal")
        ax.axhline(target_plot[idx], linestyle=":", linewidth=1.5, color="tab:green", label="Terminal target" if idx == 0 else None)
        ax.set_ylabel(labels[idx])
        ax.set_xlabel("Time (s)")
        ax.grid(True, alpha=0.3)
    axes.flat[0].legend(loc="best")
    fig.suptitle("Lateral 10-Motor SCP State History", fontsize=16)
    return fig


def plot_control_response(
    t: np.ndarray,
    rpm_hist: np.ndarray,
    aileron_hist: np.ndarray,
    rudder_hist: np.ndarray,
    rpm_trim: np.ndarray,
) -> plt.Figure:
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), dpi=120, constrained_layout=True)

    motor_lines = []
    ax = axes[0]
    for idx in range(rpm_hist.shape[1]):
        line = ax.plot(t[:-1], rpm_hist[:, idx], linewidth=1.2, alpha=0.9, label=f"M{idx + 1}")[0]
        motor_lines.append(line)
    trim_line = ax.plot(t[:-1], np.full_like(t[:-1], rpm_trim[0]), "--", linewidth=1.6, color="k", label="Trim RPM")[0]
    ax.set_ylabel("Motor RPM")
    ax.set_xlabel("Time (s)")
    ax.set_title("Per-Motor RPM Commands")
    ax.grid(True, alpha=0.3)
    ax.legend(
        motor_lines + [trim_line],
        [f"M{k + 1}" for k in range(rpm_hist.shape[1])] + ["Trim RPM"],
        loc="upper center",
        ncol=6,
        bbox_to_anchor=(0.5, -0.28),
        frameon=False,
    )

    ax = axes[1]
    ax.plot(t[:-1], np.rad2deg(aileron_hist), linewidth=2.0, color="tab:orange", label="Command")
    ax.plot(t[:-1], np.zeros_like(t[:-1]), "--", linewidth=1.5, color="tab:gray", label="Trim")
    ax.set_ylabel("Aileron (deg)")
    ax.set_xlabel("Time (s)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    ax = axes[2]
    ax.plot(t[:-1], np.rad2deg(rudder_hist), linewidth=2.0, color="tab:green", label="Command")
    ax.plot(t[:-1], np.zeros_like(t[:-1]), "--", linewidth=1.5, color="tab:gray", label="Trim")
    ax.set_ylabel("Rudder (deg)")
    ax.set_xlabel("Time (s)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    fig.suptitle("Lateral 10-Motor SCP Control History", fontsize=16)
    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reduced lateral 10-motor SCP point-to-point demo.")
    parser.add_argument("--terminal", default="x:20,y:5", help="Relative terminal target, e.g. 'x:20,y:5'.")
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--t-final", type=float, default=None)
    parser.add_argument("--max-iter", type=int, default=5)
    parser.add_argument("--solver", choices=("CLARABEL", "OSQP", "SCS"), default="CLARABEL")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--show", action="store_true")
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

    x0 = op.lateral_state.copy()
    u_trim = np.concatenate([rpm_trim, np.array([0.0, 0.0], dtype=float)])

    dx_target, dy_target = parse_terminal(args.terminal)
    x_target = x0.copy()
    x_target[0] += dx_target
    x_target[1] += dy_target

    speed_nominal = max(float(x0[2]), 1e-3)
    lateral_ratio = abs(dy_target) / max(abs(dx_target), 1.0)
    tf_guess = max(1.5, 1.15 * np.hypot(dx_target, dy_target) / speed_nominal * (1.0 + min(1.0, 0.8 * lateral_ratio)))
    t_final = float(args.t_final) if args.t_final is not None else tf_guess
    horizon_steps = max(10, int(np.ceil(t_final / float(args.dt))))
    dt = t_final / horizon_steps
    t = np.linspace(0.0, t_final, horizon_steps + 1)

    dynamics = lambda x, u: rk4_step(
        x,
        u,
        dt,
        vehicle,
        w_trim_mps=w_trim,
        theta_trim_rad=theta_trim,
        elevator_trim_rad=elevator_trim,
        flap_trim_rad=flap_trim,
    )
    u_init = np.repeat(u_trim[None, :], horizon_steps, axis=0)
    u_ref = u_init.copy()

    q_running = np.diag([0.0, 0.0, 1.5, 2.0, 4.0, 3.0, 4.0, 4.0])
    q_plan = np.repeat(q_running[None, :, :], horizon_steps, axis=0)
    r_diag = np.concatenate([np.full(n_props, 1.0e-4, dtype=float), np.array([2.0, 2.0], dtype=float)])
    r_plan = np.repeat(np.diag(r_diag)[None, :, :], horizon_steps, axis=0)
    rd_diag = np.concatenate([np.full(n_props, 2.5e-4, dtype=float), np.array([0.25, 0.25], dtype=float)])
    qf_diag = np.array([1200.0, 1400.0, 120.0, 6.0, 14.0, 12.0, 12.0, 12.0], dtype=float)

    rpm_grid = np.asarray(vehicle.propulsion["rpm_grid"], dtype=float)
    lim = vehicle.control_limits_rad
    u_lower = np.concatenate([np.full(n_props, float(rpm_grid.min())), np.array([-lim["aileron"], -lim["rudder"]])])
    u_upper = np.concatenate([np.full(n_props, float(rpm_grid.max())), np.array([lim["aileron"], lim["rudder"]])])
    x_lower = np.array([-120.0, -100.0, 4.0, -5.0, np.deg2rad(-20.0), np.deg2rad(-45.0), np.deg2rad(-35.0), np.deg2rad(-35.0)], dtype=float)
    x_upper = np.array([120.0, 100.0, 20.0, 5.0, np.deg2rad(20.0), np.deg2rad(45.0), np.deg2rad(35.0), np.deg2rad(35.0)], dtype=float)
    rho_x = np.array([0.9, 0.7, 0.5, 0.35, np.deg2rad(3.0), np.deg2rad(4.0), np.deg2rad(4.0), np.deg2rad(4.0)], dtype=float)
    rho_u = np.concatenate([np.full(n_props, 120.0, dtype=float), np.array([np.deg2rad(1.5), np.deg2rad(1.5)], dtype=float)])

    print("Lateral 10-motor SCP point-to-point demo")
    print("  state                 : [x, y, u, v, phi, psi, p, r]")
    print("  control               : [rpm_1, ..., rpm_10, aileron, rudder]")
    print(f"  terminal target       : x={x_target[0]:.3f} m, y={x_target[1]:.3f} m")
    print(f"  horizon               : {t_final:.3f} s over {horizon_steps} steps")

    result = solve_scp(
        dynamics,
        x0,
        u_init,
        x_ref=None,
        u_ref=u_ref,
        q_mat=q_plan,
        r_mat=r_plan,
        qf_mat=np.diag(qf_diag),
        x_terminal_ref=x_target,
        terminal_state_indices=(0, 1),
        terminal_state_tolerance=np.array([0.25, 0.25], dtype=float),
        rd_mat=np.diag(rd_diag),
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
    )

    x_nom = result.x_seq
    u_nom = result.u_seq

    print()
    print("SCP result")
    print(f"  converged             : {result.converged}")
    print(f"  termination reason    : {result.termination_reason}")
    print(f"  iterations            : {result.iterations}")
    print(f"  initial cost          : {result.cost_history[0]:.6f}")
    print(f"  final cost            : {result.cost_history[-1]:.6f}")
    print(f"  terminal state        : x={x_nom[-1,0]:.6f} m, y={x_nom[-1,1]:.6f} m")
    print(f"  terminal error        : dx={x_nom[-1,0]-x_target[0]:.6f} m, dy={x_nom[-1,1]-x_target[1]:.6f} m")

    fig_states = plot_state_response(t, x_nom, x_target)
    fig_controls = plot_control_response(t, u_nom[:, :n_props], u_nom[:, n_props], u_nom[:, n_props + 1], rpm_trim)

    output_dir = REPO_ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_figure(fig_states, output_dir / "lateral_scp_10motor_states.png")
    save_figure(fig_controls, output_dir / "lateral_scp_10motor_controls.png")

    if args.show and plt.get_backend().lower() != "agg":
        plt.show()
    else:
        plt.close(fig_states)
        plt.close(fig_controls)


if __name__ == "__main__":
    main()
