from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from blown_aircraft.finite_horizon import solve_finite_horizon_lqr
from blown_aircraft.geometry import load_vehicle
from blown_aircraft.longitudinal import longitudinal_state_derivative
from blown_aircraft.operating_point import build_symmetric_cruise_operating_point, linearize_about_cruise
from blown_aircraft.plotting import (
    animate_longitudinal_aircraft,
    plot_longitudinal_closed_loop_response,
    save_figure,
)


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
    parser = argparse.ArgumentParser(description="Finite-horizon longitudinal LQR demo about cruise trim.")
    parser.add_argument("--save-gifs", action="store_true", help="Save GIF animations in addition to PNGs.")
    parser.add_argument("--show", action="store_true", help="Display matplotlib windows at the end of the run.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    vehicle = load_vehicle()
    dt = 0.05
    t_final = 12.0
    t = np.arange(0.0, t_final + 0.5 * dt, dt)
    horizon_steps = len(t) - 1

    op = build_symmetric_cruise_operating_point(vehicle, speed_mps=10.0, flight_path_angle_rad=0.0, flap_rad=0.0)
    lin = linearize_about_cruise(vehicle, speed_mps=10.0, flight_path_angle_rad=0.0, flap_rad=0.0, dt=dt)["longitudinal"]

    state_idx = (2, 3, 4, 5)  # [u, w, theta, q]
    input_idx = (0, 1)  # [collective RPM, elevator]
    q_mat = np.diag([4.0, 8.0, 30.0, 12.0])
    r_mat = np.diag([1.0e-6, 2.0])
    qf_mat = 25.0 * q_mat

    fh = solve_finite_horizon_lqr(
        lin["Ad"],
        lin["Bd"],
        q_mat,
        r_mat,
        qf_mat,
        horizon_steps,
        state_indices=state_idx,
        input_indices=input_idx,
    )

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

    for k in range(horizon_steps):
        xk = x_hist[k]
        dx_sub = xk[list(state_idx)] - x_trim[list(state_idx)]
        du = -fh.k_seq[k] @ dx_sub

        uk = u_trim.copy()
        uk[0] = float(np.clip(u_trim[0] + du[0], rpm_min, rpm_max))
        uk[1] = float(np.clip(u_trim[1] + du[1], -lim["elevator"], lim["elevator"]))
        uk[2] = float(np.clip(u_trim[2], 0.0, lim["flap"]))

        u_hist[k] = uk
        x_hist[k + 1] = rk4_step(xk, uk, dt, vehicle)

    u_hist[-1] = u_hist[-2]
    x_ref = reference_trajectory(op, t)
    x_dev = x_hist - x_ref

    print("Finite-horizon longitudinal LQR demo")
    print(f"  horizon steps         : {horizon_steps}")
    print(f"  dt                    : {dt:.3f} s")
    print(f"  terminal weight scale : 25.0")
    print(f"  initial delta u       : {x0[2] - x_trim[2]:.4f} m/s")
    print(f"  initial delta w       : {x0[3] - x_trim[3]:.4f} m/s")
    print(f"  initial delta theta   : {np.rad2deg(x0[4] - x_trim[4]):.4f} deg")
    print(f"  initial delta q       : {np.rad2deg(x0[5] - x_trim[5]):.4f} deg/s")
    print()
    print("First and last feedback gains")
    print("K_0 =")
    print(fh.k_seq[0])
    print("K_{N-1} =")
    print(fh.k_seq[-1])
    print()
    print("Final state deviation from trimmed cruise trajectory")
    print(f"  delta x               : {x_dev[-1, 0]:.6f} m")
    print(f"  delta h               : {x_dev[-1, 1]:.6f} m")
    print(f"  delta u               : {x_dev[-1, 2]:.6f} m/s")
    print(f"  delta w               : {x_dev[-1, 3]:.6f} m/s")
    print(f"  delta theta           : {np.rad2deg(x_dev[-1, 4]):.6f} deg")
    print(f"  delta q               : {np.rad2deg(x_dev[-1, 5]):.6f} deg/s")
    print()
    print(f"  max collective RPM    : {np.max(u_hist[:, 0]):.3f}")
    print(f"  min collective RPM    : {np.min(u_hist[:, 0]):.3f}")
    print(f"  max elevator          : {np.rad2deg(np.max(u_hist[:, 1])):.3f} deg")
    print(f"  min elevator          : {np.rad2deg(np.min(u_hist[:, 1])):.3f} deg")

    fig_states, _ = plot_longitudinal_closed_loop_response(t, x_dev, x_ref, x_hist, u_hist, u_trim)
    fig_anim, ani = animate_longitudinal_aircraft(t, x_hist, vehicle)
    fig_anim_zoom, ani_zoom = animate_longitudinal_aircraft(
        t,
        x_hist,
        vehicle,
        follow_vehicle=True,
        window_width=3.0,
        window_height=2.0,
        title="Finite-Horizon Longitudinal Animation (Zoomed)",
    )

    output_dir = REPO_ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_figure(fig_states, output_dir / "longitudinal_finite_horizon_states.png")
    save_figure(fig_anim, output_dir / "longitudinal_finite_horizon_animation_frame.png")
    save_figure(fig_anim_zoom, output_dir / "longitudinal_finite_horizon_zoomed_animation_frame.png")

    if args.save_gifs:
        try:
            ani.save(output_dir / "longitudinal_finite_horizon.gif", writer="pillow", fps=max(1, int(round(1.0 / dt))))
            print(f"Saved animation to {output_dir / 'longitudinal_finite_horizon.gif'}")
        except Exception as exc:  # pragma: no cover
            print(f"Could not save GIF animation with pillow: {exc}")
        try:
            ani_zoom.save(
                output_dir / "longitudinal_finite_horizon_zoomed.gif",
                writer="pillow",
                fps=max(1, int(round(1.0 / dt))),
            )
            print(f"Saved zoomed animation to {output_dir / 'longitudinal_finite_horizon_zoomed.gif'}")
        except Exception as exc:  # pragma: no cover
            print(f"Could not save zoomed GIF animation with pillow: {exc}")

    if args.show and plt.get_backend().lower() != "agg":
        plt.show()
    else:
        plt.close(fig_states)
        plt.close(fig_anim)
        plt.close(fig_anim_zoom)


if __name__ == "__main__":
    main()
