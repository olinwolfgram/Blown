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
from blown_aircraft.lateral import lateral_state_derivative
from blown_aircraft.operating_point import linearize_about_cruise
from blown_aircraft.plotting import animate_lateral_aircraft, plot_lateral_closed_loop_response, save_figure


def rk4_step(x: np.ndarray, u: np.ndarray, dt: float, vehicle) -> np.ndarray:
    f = lambda xk: lateral_state_derivative(xk, u, vehicle)
    k1 = f(x)
    k2 = f(x + 0.5 * dt * k1)
    k3 = f(x + 0.5 * dt * k2)
    k4 = f(x + dt * k3)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Finite-horizon lateral LQR demo about cruise trim.")
    parser.add_argument("--save-gifs", action="store_true", help="Save GIF animations in addition to PNGs.")
    parser.add_argument("--show", action="store_true", help="Display matplotlib windows at the end of the run.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    vehicle = load_vehicle()
    dt = 0.05
    t_final = 20.0
    t = np.arange(0.0, t_final + 0.5 * dt, dt)
    horizon_steps = len(t) - 1

    result = linearize_about_cruise(vehicle, speed_mps=10.0, flight_path_angle_rad=0.0, flap_rad=0.0, dt=dt)
    op = result["operating_point"]
    lat = result["lateral"]
    a_lat = lat["Ad"]
    b_lat = lat["Bd"]
    b_diff = (b_lat[:, [1]] - b_lat[:, [0]]).copy()

    state_idx = (1, 2, 3, 4, 5)  # [v, phi, psi, p, r]
    q_mat = np.diag([6.0, 25.0, 8.0, 30.0, 18.0])
    r_mat = np.array([[1.0e-3]])
    qf_mat = 25.0 * q_mat

    fh = solve_finite_horizon_lqr(
        a_lat,
        b_diff,
        q_mat,
        r_mat,
        qf_mat,
        horizon_steps,
        state_indices=state_idx,
        input_indices=(0,),
    )

    x_trim = op.lateral_state.copy()
    u_trim = op.lateral_control.copy()
    rpm_trim = float(u_trim[0])
    rpm_grid = np.asarray(vehicle.propulsion["rpm_grid"], dtype=float)
    rpm_min = float(rpm_grid.min())
    rpm_max = float(rpm_grid.max())

    x0 = x_trim.copy()
    x0[1] += 0.40
    x0[2] += np.deg2rad(5.0)
    x0[3] += np.deg2rad(8.0)
    x0[4] += np.deg2rad(4.0)
    x0[5] += np.deg2rad(2.0)

    x_hist = np.zeros((len(t), 6), dtype=float)
    u_hist = np.zeros((len(t), 4), dtype=float)
    u_diff_hist = np.zeros(len(t), dtype=float)
    x_hist[0] = x0

    for k in range(horizon_steps):
        xk = x_hist[k]
        dx_sub = xk[list(state_idx)] - x_trim[list(state_idx)]
        delta_rpm_diff = float((-fh.k_seq[k] @ dx_sub).item())

        rpm_left = float(np.clip(rpm_trim - delta_rpm_diff, rpm_min, rpm_max))
        rpm_right = float(np.clip(rpm_trim + delta_rpm_diff, rpm_min, rpm_max))
        uk = np.array([rpm_left, rpm_right, 0.0, 0.0], dtype=float)

        u_hist[k] = uk
        u_diff_hist[k] = delta_rpm_diff
        x_hist[k + 1] = rk4_step(xk, uk, dt, vehicle)

    u_hist[-1] = u_hist[-2]
    u_diff_hist[-1] = u_diff_hist[-2]
    x_ref = np.tile(x_trim, (len(t), 1))
    x_dev = x_hist - x_ref
    x_forward = op.speed_mps * t

    print("Finite-horizon lateral LQR demo")
    print(f"  horizon steps         : {horizon_steps}")
    print(f"  dt                    : {dt:.3f} s")
    print(f"  terminal weight scale : 25.0")
    print(f"  initial delta v       : {x0[1] - x_trim[1]:.4f} m/s")
    print(f"  initial delta phi     : {np.rad2deg(x0[2] - x_trim[2]):.4f} deg")
    print(f"  initial delta psi     : {np.rad2deg(x0[3] - x_trim[3]):.4f} deg")
    print(f"  initial delta p       : {np.rad2deg(x0[4] - x_trim[4]):.4f} deg/s")
    print(f"  initial delta r       : {np.rad2deg(x0[5] - x_trim[5]):.4f} deg/s")
    print()
    print("First and last feedback gains")
    print("K_0 =")
    print(fh.k_seq[0])
    print("K_{N-1} =")
    print(fh.k_seq[-1])
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
        title="Finite-Horizon Lateral Animation (Zoomed)",
    )

    output_dir = REPO_ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_figure(fig, output_dir / "lateral_finite_horizon_states.png")
    save_figure(fig_anim, output_dir / "lateral_finite_horizon_animation_frame.png")
    save_figure(fig_anim_zoom, output_dir / "lateral_finite_horizon_zoomed_animation_frame.png")

    if args.save_gifs:
        try:
            ani.save(output_dir / "lateral_finite_horizon.gif", writer="pillow", fps=max(1, int(round(1.0 / dt))))
            print(f"Saved animation to {output_dir / 'lateral_finite_horizon.gif'}")
        except Exception as exc:  # pragma: no cover
            print(f"Could not save GIF animation with pillow: {exc}")
        try:
            ani_zoom.save(
                output_dir / "lateral_finite_horizon_zoomed.gif",
                writer="pillow",
                fps=max(1, int(round(1.0 / dt))),
            )
            print(f"Saved zoomed animation to {output_dir / 'lateral_finite_horizon_zoomed.gif'}")
        except Exception as exc:  # pragma: no cover
            print(f"Could not save zoomed GIF animation with pillow: {exc}")

    if args.show and plt.get_backend().lower() != "agg":
        plt.show()
    else:
        plt.close(fig)
        plt.close(fig_anim)
        plt.close(fig_anim_zoom)


if __name__ == "__main__":
    main()
