from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .types import OCPResult, TrimResult


def save_ocp_plots(result: OCPResult, trim: TrimResult, out_dir: str | Path) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    t = result.t_s
    x = result.x_hist
    u = result.u_hist

    fig1, ax = plt.subplots(3, 2, figsize=(11, 8), constrained_layout=True)
    labels = ["x (m)", "h (m)", "u (m/s)", "w (m/s)", "theta (deg)", "q (deg/s)"]
    series = [
        x[0, :],
        x[1, :],
        x[2, :],
        x[3, :],
        np.rad2deg(x[4, :]),
        np.rad2deg(x[5, :]),
    ]
    for axis, label, values in zip(ax.flat, labels, series):
        axis.plot(t, values, linewidth=2)
        axis.set_ylabel(label)
        axis.grid(True, alpha=0.3)
    for axis in ax[-1, :]:
        axis.set_xlabel("time (s)")
    path1 = out_dir / "ocp_states.png"
    fig1.savefig(path1, dpi=160)
    plt.close(fig1)
    paths.append(path1)

    fig2, ax2 = plt.subplots(2, 1, figsize=(10, 6), constrained_layout=True)
    ax2[0].plot(t[:-1], u[0, :], linewidth=2)
    ax2[0].axhline(trim.rpm, linestyle="--", alpha=0.5)
    ax2[0].set_ylabel("RPM")
    ax2[0].grid(True, alpha=0.3)
    ax2[1].plot(t[:-1], np.rad2deg(u[1, :]), linewidth=2)
    ax2[1].axhline(np.rad2deg(trim.elevator_rad), linestyle="--", alpha=0.5)
    ax2[1].set_ylabel("elevator (deg)")
    ax2[1].set_xlabel("time (s)")
    ax2[1].grid(True, alpha=0.3)
    path2 = out_dir / "ocp_controls.png"
    fig2.savefig(path2, dpi=160)
    plt.close(fig2)
    paths.append(path2)

    fig3, ax3 = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax3.plot(x[0, :], x[1, :], linewidth=2)
    ax3.set_xlabel("forward distance x (m)")
    ax3.set_ylabel("altitude h (m)")
    ax3.grid(True, alpha=0.3)
    path3 = out_dir / "ocp_trajectory.png"
    fig3.savefig(path3, dpi=160)
    plt.close(fig3)
    paths.append(path3)

    return paths
