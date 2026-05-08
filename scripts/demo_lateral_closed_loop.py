from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from blown_aircraft.geometry import load_vehicle
from blown_aircraft.lateral import lateral_state_derivative
from blown_aircraft.lqr import design_lqr
from blown_aircraft.operating_point import linearize_about_cruise
from blown_aircraft.plotting import (
    animate_lateral_aircraft,
    animate_lateral_rear_view,
    plot_lateral_closed_loop_response,
    save_figure,
)


def rk4_step(x: np.ndarray, u: np.ndarray, dt: float, vehicle) -> np.ndarray:
    f = lambda xk: lateral_state_derivative(xk, u, vehicle)
    k1 = f(x)
    k2 = f(x + 0.5 * dt * k1)
    k3 = f(x + 0.5 * dt * k2)
    k4 = f(x + dt * k3)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def main() -> None:
    vehicle = load_vehicle()
    result = linearize_about_cruise(vehicle, speed_mps=10.0, flight_path_angle_rad=0.0, flap_rad=0.0, dt=0.05)
    op = result["operating_point"]
    a_lat = result["lateral"]["A"]
    b_lat = result["lateral"]["B"]
    b_diff = (b_lat[:, [1]] - b_lat[:, [0]]).copy()

    state_idx = (1, 2, 3, 4, 5)  # [v, phi, psi, p, r]
    q_mat = np.diag([6.0, 25.0, 8.0, 30.0, 18.0])
    r_mat = np.array([[1.0e-3]])
    lqr = design_lqr(
        a_lat,
        b_diff,
        q_mat,
        r_mat,
        state_indices=state_idx,
        input_indices=(0,),
        discrete_time=False,
    )

    t_final = 40.0
    dt = 0.02
    t = np.arange(0.0, t_final + 0.5 * dt, dt)

    x_trim = op.lateral_state.copy()
    u_trim = op.lateral_control.copy()
    rpm_trim = float(u_trim[0])
    rpm_grid = np.asarray(vehicle.propulsion["rpm_grid"], dtype=float)
    rpm_min = float(rpm_grid.min())
    rpm_max = float(rpm_grid.max())

    x0 = x_trim.copy()
    x0[1] += 0.40  # delta v [m/s]
    x0[2] += np.deg2rad(5.0)  # delta phi [rad]
    x0[3] += np.deg2rad(8.0)  # delta psi [rad]
    x0[4] += np.deg2rad(4.0)  # delta p [rad/s]
    x0[5] += np.deg2rad(2.0)  # delta r [rad/s]

    x_hist = np.zeros((len(t), 6), dtype=float)
    u_hist = np.zeros((len(t), 4), dtype=float)
    u_diff_hist = np.zeros(len(t), dtype=float)
    x_hist[0] = x0

    for k in range(len(t) - 1):
        xk = x_hist[k]
        dx_sub = xk[list(state_idx)] - x_trim[list(state_idx)]
        delta_rpm_diff = float((-lqr.k_gain @ dx_sub).item())

        rpm_left = float(np.clip(rpm_trim - delta_rpm_diff, rpm_min, rpm_max))
        rpm_right = float(np.clip(rpm_trim + delta_rpm_diff, rpm_min, rpm_max))
        uk = np.array([rpm_left, rpm_right, 0.0, 0.0], dtype=float)

        u_hist[k] = uk
        u_diff_hist[k] = delta_rpm_diff
        x_hist[k + 1] = rk4_step(xk, uk, dt, vehicle)

    dx_sub = x_hist[-1, list(state_idx)] - x_trim[list(state_idx)]
    u_diff_hist[-1] = float((-lqr.k_gain @ dx_sub).item())
    u_hist[-1] = np.array(
        [
            float(np.clip(rpm_trim - u_diff_hist[-1], rpm_min, rpm_max)),
            float(np.clip(rpm_trim + u_diff_hist[-1], rpm_min, rpm_max)),
            0.0,
            0.0,
        ],
        dtype=float,
    )

    x_ref = np.tile(x_trim, (len(t), 1))
    x_dev = x_hist - x_ref
    x_forward = op.speed_mps * t

    print("Closed-loop lateral nonlinear demo")
    print(f"  simulation time       : {t_final:.2f} s")
    print(f"  dt                    : {dt:.3f} s")
    print(f"  initial delta v       : {x0[1] - x_trim[1]:.4f} m/s")
    print(f"  initial delta phi     : {np.rad2deg(x0[2] - x_trim[2]):.4f} deg")
    print(f"  initial delta psi     : {np.rad2deg(x0[3] - x_trim[3]):.4f} deg")
    print(f"  initial delta p       : {np.rad2deg(x0[4] - x_trim[4]):.4f} deg/s")
    print(f"  initial delta r       : {np.rad2deg(x0[5] - x_trim[5]):.4f} deg/s")
    print()
    print("Final state deviation from lateral trim")
    print(f"  delta y               : {x_dev[-1, 0]:.6f} m")
    print(f"  delta v               : {x_dev[-1, 1]:.6f} m/s")
    print(f"  delta phi             : {np.rad2deg(x_dev[-1, 2]):.6f} deg")
    print(f"  delta psi             : {np.rad2deg(x_dev[-1, 3]):.6f} deg")
    print(f"  delta p               : {np.rad2deg(x_dev[-1, 4]):.6f} deg/s")
    print(f"  delta r               : {np.rad2deg(x_dev[-1, 5]):.6f} deg/s")
    print()
    print(f"  max |delta rpm diff|  : {np.max(np.abs(u_diff_hist)):.3f}")

    fig, _ = plot_lateral_closed_loop_response(t, x_dev, x_ref, u_diff_hist)
    fig_anim, ani = animate_lateral_aircraft(t, x_forward, x_hist[:, 0], x_hist[:, 3], x_hist[:, 2], vehicle)
    fig_anim_zoom, ani_zoom = animate_lateral_aircraft(
        t,
        x_forward,
        x_hist[:, 0],
        x_hist[:, 3],
        x_hist[:, 2],
        vehicle,
        follow_vehicle=True,
        window_width=8.0,
        window_height=6.0,
        title="Lateral-Plane Animation (Zoomed)",
    )
    fig_rear, ani_rear = animate_lateral_rear_view(
        t,
        x_hist[:, 2],
        x_hist[:, 3],
        vehicle,
        title="Lateral Rear View Animation",
    )

    output_dir = REPO_ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_figure(fig, output_dir / "lateral_closed_loop_states.png")
    save_figure(fig_anim, output_dir / "lateral_closed_loop_animation_frame.png")
    save_figure(fig_anim_zoom, output_dir / "lateral_closed_loop_zoomed_animation_frame.png")
    save_figure(fig_rear, output_dir / "lateral_closed_loop_rear_view_frame.png")
    try:
        ani.save(output_dir / "lateral_closed_loop.gif", writer="pillow", fps=max(1, int(round(1.0 / dt))))
        print(f"Saved animation to {output_dir / 'lateral_closed_loop.gif'}")
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"Could not save GIF animation with pillow: {exc}")
    try:
        ani_zoom.save(output_dir / "lateral_closed_loop_zoomed.gif", writer="pillow", fps=max(1, int(round(1.0 / dt))))
        print(f"Saved zoomed animation to {output_dir / 'lateral_closed_loop_zoomed.gif'}")
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"Could not save zoomed GIF animation with pillow: {exc}")
    try:
        ani_rear.save(output_dir / "lateral_closed_loop_rear_view.gif", writer="pillow", fps=max(1, int(round(1.0 / dt))))
        print(f"Saved rear-view animation to {output_dir / 'lateral_closed_loop_rear_view.gif'}")
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"Could not save rear-view GIF animation with pillow: {exc}")

    if plt.get_backend().lower() != "agg":
        plt.show()
    else:
        plt.close(fig)
        plt.close(fig_anim)
        plt.close(fig_anim_zoom)
        plt.close(fig_rear)


if __name__ == "__main__":
    main()
