from __future__ import annotations

from pathlib import Path

import matplotlib.animation as animation
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms
import matplotlib.pyplot as plt
import numpy as np

from .geometry import Vehicle


def _rot2(theta: float) -> np.ndarray:
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=float)


def _transform_points(points: np.ndarray, origin: np.ndarray, angle: float) -> np.ndarray:
    return points @ _rot2(angle).T + origin


def _airframe_layout(vehicle: Vehicle) -> dict[str, np.ndarray | float]:
    geom = vehicle.geometry
    prop = vehicle.propulsion

    chord = float(geom["chord_m"])
    span = float(geom["span_m"])
    semispan = 0.5 * span
    wing_x = float(geom["wing_ac_x_m"]) - float(geom["cg_x_m"])
    tail_x = wing_x - float(geom["tail_arm_m"])
    nose_x = wing_x + 1.35 * chord
    tail_end_x = tail_x - 0.28 * chord
    tail_span = min(0.62 * span, 0.95)
    flap_half = float(geom["flap_span_fraction"]) * semispan
    leading_x = wing_x + 0.25 * chord
    trailing_x = leading_x - chord

    return {
        "chord": chord,
        "span": span,
        "semispan": semispan,
        "wing_x": wing_x,
        "tail_x": tail_x,
        "nose_x": nose_x,
        "tail_end_x": tail_end_x,
        "tail_span": tail_span,
        "flap_half": flap_half,
        "leading_x": leading_x,
        "trailing_x": trailing_x,
        "prop_x": -(float(prop["axial_x_m"]) - float(geom["cg_x_m"])),
        "prop_y": np.asarray(prop["centers_y_m"], dtype=float),
        "prop_d": float(prop["diameter_m"]),
    }


def plot_longitudinal_closed_loop_response(
    t: np.ndarray,
    x_dev: np.ndarray,
    x_ref: np.ndarray,
    x_abs: np.ndarray,
    u_hist: np.ndarray,
    u_ref: np.ndarray,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot longitudinal state and control histories against trim references."""

    state_labels = [
        r"$\Delta x$ (m)",
        r"$\Delta h$ (m)",
        r"$\Delta u$ (m/s)",
        r"$\Delta w$ (m/s)",
        r"$\Delta \theta$ (deg)",
        r"$\Delta q$ (deg/s)",
    ]
    x_plot = x_dev.copy()
    x_plot[:, 4] = np.rad2deg(x_plot[:, 4])
    x_plot[:, 5] = np.rad2deg(x_plot[:, 5])

    control_labels = [
        "Collective RPM",
        r"Elevator (deg)",
        r"Flap (deg)",
    ]
    u_plot = u_hist.copy()
    u_plot[:, 1:] = np.rad2deg(u_plot[:, 1:])
    u_ref_plot = u_ref.copy()
    u_ref_plot[1:] = np.rad2deg(u_ref_plot[1:])

    fig, axes = plt.subplots(3, 3, figsize=(14, 10), dpi=120, constrained_layout=True)
    axes = axes.reshape(3, 3)

    for idx in range(6):
        ax = axes.flat[idx]
        ax.plot(t, x_plot[:, idx], linewidth=2, color="tab:blue", label="Closed-loop")
        ax.plot(t, np.zeros_like(t), "--", linewidth=1.5, color="tab:gray", label="Trim reference")
        ax.set_ylabel(state_labels[idx])
        ax.grid(True, alpha=0.3)
        if idx >= 3:
            ax.set_xlabel("Time (s)")

    for j in range(3):
        ax = axes.flat[6 + j]
        ax.plot(t, u_plot[:, j], linewidth=2, color="tab:orange", label="Command")
        ax.plot(t, np.full_like(t, u_ref_plot[j]), "--", linewidth=1.5, color="tab:gray", label="Trim reference")
        ax.set_ylabel(control_labels[j])
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Time (s)")

    axes[0, 0].legend(loc="best")
    axes[2, 0].legend(loc="best")

    trim_text = (
        "Absolute trim state: "
        f"x={x_ref[0,0]:.2f} m, h={x_ref[0,1]:.2f} m, u={x_ref[0,2]:.2f} m/s, "
        f"w={x_ref[0,3]:.2f} m/s, theta={np.rad2deg(x_ref[0,4]):.2f} deg, "
        f"q={np.rad2deg(x_ref[0,5]):.2f} deg/s"
    )
    fig.suptitle("Closed-Loop Longitudinal Response About Cruise Trim", fontsize=16)
    fig.text(0.5, 0.01, trim_text, ha="center", va="bottom", fontsize=9)
    return fig, axes


def plot_lateral_closed_loop_response(
    t: np.ndarray,
    x_dev: np.ndarray,
    x_ref: np.ndarray,
    u_diff_hist: np.ndarray,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot lateral closed-loop state histories against trim references."""

    state_labels = [
        r"$\Delta y$ (m)",
        r"$\Delta v$ (m/s)",
        r"$\Delta \phi$ (deg)",
        r"$\Delta \psi$ (deg)",
        r"$\Delta p$ (deg/s)",
        r"$\Delta r$ (deg/s)",
    ]

    x_plot = x_dev.copy()
    x_plot[:, 2:] = np.rad2deg(x_plot[:, 2:])

    fig, axes = plt.subplots(3, 3, figsize=(14, 10), dpi=120, constrained_layout=True)
    axes = axes.reshape(3, 3)

    for idx in range(6):
        ax = axes.flat[idx]
        ax.plot(t, x_plot[:, idx], linewidth=2, color="tab:purple", label="Closed-loop")
        ax.plot(t, np.zeros_like(t), "--", linewidth=1.5, color="tab:gray", label="Trim reference")
        ax.set_ylabel(state_labels[idx])
        ax.grid(True, alpha=0.3)
        if idx >= 3:
            ax.set_xlabel("Time (s)")

    ax_diff = axes.flat[6]
    ax_diff.plot(t, u_diff_hist, linewidth=2, color="tab:orange", label=r"$\Delta$RPM diff")
    ax_diff.plot(t, np.zeros_like(t), "--", linewidth=1.5, color="tab:gray", label="Trim reference")
    ax_diff.set_ylabel(r"$\Delta$RPM diff")
    ax_diff.set_xlabel("Time (s)")
    ax_diff.grid(True, alpha=0.3)

    axes.flat[7].axis("off")
    axes.flat[8].axis("off")
    axes[0, 0].legend(loc="best")
    ax_diff.legend(loc="best")

    trim_text = (
        "Absolute trim state: "
        f"y={x_ref[0,0]:.2f} m, v={x_ref[0,1]:.2f} m/s, phi={np.rad2deg(x_ref[0,2]):.2f} deg, "
        f"psi={np.rad2deg(x_ref[0,3]):.2f} deg, p={np.rad2deg(x_ref[0,4]):.2f} deg/s, "
        f"r={np.rad2deg(x_ref[0,5]):.2f} deg/s"
    )
    fig.suptitle("Closed-Loop Lateral Response About Cruise Trim", fontsize=16)
    fig.text(0.5, 0.01, trim_text, ha="center", va="bottom", fontsize=9)
    return fig, axes


def animate_longitudinal_aircraft(
    t: np.ndarray,
    x_abs: np.ndarray,
    vehicle: Vehicle,
    *,
    follow_vehicle: bool = False,
    window_width: float = 3.0,
    window_height: float = 2.0,
    title: str = "Longitudinal Plane Animation",
) -> tuple[plt.Figure, animation.FuncAnimation]:
    """Animate the aircraft motion in the longitudinal plane."""

    x = x_abs[:, 0]
    h = x_abs[:, 1]
    theta = x_abs[:, 4]

    layout = _airframe_layout(vehicle)
    chord = float(layout["chord"])
    wing_inc = np.deg2rad(float(vehicle.geometry["wing_incidence_deg"]))
    tail_inc = np.deg2rad(float(vehicle.geometry["htail_incidence_deg"]))
    tail_x = float(layout["tail_x"])
    nose_x = float(layout["nose_x"])
    tail_end_x = float(layout["tail_end_x"])
    leading_x = float(layout["leading_x"])
    trailing_x = float(layout["trailing_x"])

    fig, ax = plt.subplots(figsize=(12, 5), dpi=120, constrained_layout=True)
    x_pad = max(5.0, 0.1 * max(np.ptp(x), 1.0))
    h_pad = max(2.0, 0.2 * max(np.ptp(h), 1.0))
    ax.set_xlim(np.min(x) - x_pad, np.max(x) + x_pad)
    ax.set_ylim(np.min(h) - h_pad, np.max(h) + h_pad)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("h (m)")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.set_aspect("equal", adjustable="box")

    trace = ax.plot([], [], "--", color="tab:orange", linewidth=1.8, label="CG trace")[0]
    fuselage_patch = mpatches.Polygon(np.zeros((7, 2)), closed=True, facecolor="k", alpha=0.95, edgecolor="none")
    fuselage_boom = ax.plot([], [], "-", color="k", linewidth=3, label="Fuselage")[0]
    cg_marker = ax.plot([], [], "o", color="tab:red", markersize=6, label="CG")[0]
    cg_label = ax.text(0.0, 0.0, "CG", color="tab:red", fontsize=9, weight="bold")
    wing_patch = mpatches.Polygon(np.zeros((4, 2)), closed=True, facecolor="tab:blue", alpha=0.9, edgecolor="none")
    tail_patch = mpatches.Polygon(np.zeros((4, 2)), closed=True, facecolor="tab:green", alpha=0.7, edgecolor="none")
    flap_patch = mpatches.Polygon(np.zeros((4, 2)), closed=True, facecolor="tab:orange", alpha=0.9, edgecolor="none")
    aileron_patch = mpatches.Polygon(np.zeros((4, 2)), closed=True, facecolor="tab:purple", alpha=0.9, edgecolor="none")
    for patch in (wing_patch, tail_patch, flap_patch, aileron_patch, fuselage_patch):
        ax.add_patch(patch)
    timestamp = ax.text(0.02, 0.93, "", transform=ax.transAxes)
    attitude = ax.text(0.02, 0.87, "", transform=ax.transAxes)
    ax.legend(loc="upper right")

    wing_poly = np.array(
        [
            [leading_x, -0.06],
            [trailing_x, -0.06],
            [trailing_x, 0.06],
            [leading_x, 0.06],
        ],
        dtype=float,
    )
    tail_poly = np.array(
        [
            [tail_x + 0.16 * chord, -0.035],
            [tail_x - 0.20 * chord, -0.035],
            [tail_x - 0.20 * chord, 0.035],
            [tail_x + 0.16 * chord, 0.035],
        ],
        dtype=float,
    )
    flap_poly = np.array(
        [
            [trailing_x + 0.34 * chord, -0.045],
            [trailing_x, -0.045],
            [trailing_x, 0.005],
            [trailing_x + 0.34 * chord, 0.005],
        ],
        dtype=float,
    )
    aileron_poly = np.array(
        [
            [trailing_x + 0.30 * chord, 0.015],
            [trailing_x, 0.015],
            [trailing_x, 0.055],
            [trailing_x + 0.30 * chord, 0.055],
        ],
        dtype=float,
    )
    fuselage_poly = np.array(
        [
            [nose_x, 0.0],
            [leading_x + 0.06 * chord, -0.025],
            [trailing_x + 0.08 * chord, -0.025],
            [trailing_x - 0.35 * chord, -0.010],
            [trailing_x - 0.35 * chord, 0.010],
            [trailing_x + 0.08 * chord, 0.025],
            [leading_x + 0.06 * chord, 0.025],
        ],
        dtype=float,
    )
    fuselage_boom_seg = np.array([[tail_end_x, 0.0], [trailing_x - 0.35 * chord, 0.0]], dtype=float)

    def animate_frame(k: int):
        xk = x[k]
        hk = h[k]
        thetak = theta[k]
        origin = np.array([xk, hk], dtype=float)

        trace.set_data(x[: k + 1], h[: k + 1])
        cg_marker.set_data([xk], [hk])
        cg_label.set_position((xk + 0.05, hk + 0.06))

        fuselage_fill = _transform_points(fuselage_poly, origin, thetak)
        fus_boom = _transform_points(fuselage_boom_seg, origin, thetak)
        wing_fill = _transform_points(wing_poly, origin, thetak + wing_inc)
        tail_fill = _transform_points(tail_poly, origin, thetak + tail_inc)
        flap_fill = _transform_points(flap_poly, origin, thetak + wing_inc)
        aileron_fill = _transform_points(aileron_poly, origin, thetak + wing_inc)

        fuselage_patch.set_xy(fuselage_fill)
        fuselage_boom.set_data(fus_boom[:, 0], fus_boom[:, 1])
        wing_patch.set_xy(wing_fill)
        tail_patch.set_xy(tail_fill)
        flap_patch.set_xy(flap_fill)
        aileron_patch.set_xy(aileron_fill)

        if follow_vehicle:
            ax.set_xlim(xk - 0.5 * window_width, xk + 0.5 * window_width)
            ax.set_ylim(hk - 0.5 * window_height, hk + 0.5 * window_height)

        timestamp.set_text(f"t = {t[k]:.2f} s")
        attitude.set_text(f"theta = {np.rad2deg(thetak):.2f} deg")
        return (
            trace,
            fuselage_patch,
            fuselage_boom,
            cg_marker,
            cg_label,
            wing_patch,
            tail_patch,
            flap_patch,
            aileron_patch,
            timestamp,
            attitude,
        )

    dt = float(t[1] - t[0]) if len(t) > 1 else 0.05
    ani = animation.FuncAnimation(fig, animate_frame, frames=len(t), interval=dt * 1000.0, blit=True)
    return fig, ani


def animate_lateral_aircraft(
    t: np.ndarray,
    x_forward: np.ndarray,
    y_lateral: np.ndarray,
    psi: np.ndarray,
    phi: np.ndarray,
    vehicle: Vehicle,
    *,
    follow_vehicle: bool = False,
    window_width: float = 6.0,
    window_height: float = 4.0,
    title: str = "Lateral-Plane Animation",
) -> tuple[plt.Figure, animation.FuncAnimation]:
    """Animate aircraft motion in the horizontal plane."""

    layout = _airframe_layout(vehicle)
    chord = float(layout["chord"])
    tail_x = float(layout["tail_x"])
    nose_x = float(layout["nose_x"])
    tail_end_x = float(layout["tail_end_x"])
    semispan = float(layout["semispan"])
    tail_span = float(layout["tail_span"])
    flap_half = float(layout["flap_half"])
    leading_x = float(layout["leading_x"])
    trailing_x = float(layout["trailing_x"])
    prop_x = float(layout["prop_x"])
    prop_y = np.asarray(layout["prop_y"], dtype=float)
    prop_d = float(layout["prop_d"])

    fig, ax = plt.subplots(figsize=(12, 6), dpi=120, constrained_layout=True)
    x_pad = max(5.0, 0.08 * max(np.ptp(x_forward), 1.0))
    y_pad = max(3.0, 0.25 * max(np.ptp(y_lateral), 1.0))
    ax.set_xlim(np.min(x_forward) - x_pad, np.max(x_forward) + x_pad)
    ax.set_ylim(np.min(y_lateral) - y_pad, np.max(y_lateral) + y_pad)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.set_aspect("equal", adjustable="box")

    trace = ax.plot([], [], "--", color="tab:orange", linewidth=1.8, label="CG trace")[0]
    fuselage_patch = mpatches.Polygon(np.zeros((7, 2)), closed=True, facecolor="k", alpha=0.95, edgecolor="none")
    fuselage_boom = ax.plot([], [], "-", color="k", linewidth=3, label="Fuselage")[0]
    cg_marker = ax.plot([], [], "o", color="tab:red", markersize=6, label="CG")[0]
    cg_label = ax.text(0.0, 0.0, "CG", color="tab:red", fontsize=9, weight="bold")
    wing_patch = mpatches.Rectangle((trailing_x, -semispan), leading_x - trailing_x, 2.0 * semispan, facecolor="tab:blue", alpha=0.9, edgecolor="none")
    tail_patch = mpatches.Rectangle((tail_x - 0.16 * chord, -0.5 * tail_span), 0.18 * chord, tail_span, facecolor="tab:green", alpha=0.7, edgecolor="none")
    flap_left = mpatches.Rectangle((trailing_x, -flap_half), 0.34 * chord, flap_half, facecolor="tab:orange", alpha=0.9, edgecolor="none")
    flap_right = mpatches.Rectangle((trailing_x, 0.0), 0.34 * chord, flap_half, facecolor="tab:orange", alpha=0.9, edgecolor="none")
    aileron_span = max(semispan - flap_half, 0.0)
    aileron_left = mpatches.Rectangle((trailing_x, -semispan), 0.30 * chord, aileron_span, facecolor="tab:purple", alpha=0.9, edgecolor="none")
    aileron_right = mpatches.Rectangle((trailing_x, flap_half), 0.30 * chord, aileron_span, facecolor="tab:purple", alpha=0.9, edgecolor="none")
    for patch in (wing_patch, tail_patch, flap_left, flap_right, aileron_left, aileron_right, fuselage_patch):
        ax.add_patch(patch)
    prop_artists = [
        ax.plot([], [], "-", color="tab:gray", linewidth=3.0, alpha=0.85, label="Props" if i == 0 else None)[0]
        for i in range(len(prop_y))
    ]
    timestamp = ax.text(0.02, 0.93, "", transform=ax.transAxes)
    bank_text = ax.text(0.02, 0.87, "", transform=ax.transAxes)
    ax.legend(loc="upper right")

    fuselage_poly = np.array(
        [
            [nose_x, 0.0],
            [leading_x + 0.06 * chord, -0.055],
            [trailing_x + 0.08 * chord, -0.055],
            [trailing_x - 0.35 * chord, -0.020],
            [trailing_x - 0.35 * chord, 0.020],
            [trailing_x + 0.08 * chord, 0.055],
            [leading_x + 0.06 * chord, 0.055],
        ],
        dtype=float,
    )
    fuselage_boom_seg = np.array([[tail_end_x, 0.0], [trailing_x - 0.35 * chord, 0.0]], dtype=float)
    prop_segments = []
    for yi in prop_y:
        prop_segments.append(np.array([[prop_x, yi - 0.5 * prop_d], [prop_x, yi + 0.5 * prop_d]], dtype=float))

    def animate_frame(k: int):
        xk = x_forward[k]
        yk = y_lateral[k]
        psik = psi[k]
        phik = phi[k]
        origin = np.array([xk, yk], dtype=float)

        trace.set_data(x_forward[: k + 1], y_lateral[: k + 1])
        cg_marker.set_data([xk], [yk])
        cg_label.set_position((xk + 0.08, yk + 0.08))

        fuselage_fill = _transform_points(fuselage_poly, origin, psik)
        fus_boom = _transform_points(fuselage_boom_seg, origin, psik)
        fuselage_patch.set_xy(fuselage_fill)
        fuselage_boom.set_data(fus_boom[:, 0], fus_boom[:, 1])

        base_transform = mtransforms.Affine2D().rotate(psik).translate(origin[0], origin[1]) + ax.transData
        for patch in (wing_patch, tail_patch, flap_left, flap_right, aileron_left, aileron_right):
            patch.set_transform(base_transform)

        for artist, seg in zip(prop_artists, prop_segments):
            seg_world = _transform_points(seg, origin, psik)
            artist.set_data(seg_world[:, 0], seg_world[:, 1])

        if follow_vehicle:
            ax.set_xlim(xk - 0.5 * window_width, xk + 0.5 * window_width)
            ax.set_ylim(yk - 0.5 * window_height, yk + 0.5 * window_height)

        timestamp.set_text(f"t = {t[k]:.2f} s")
        bank_text.set_text(f"phi = {np.rad2deg(phik):.2f} deg")
        return (
            trace,
            fuselage_patch,
            fuselage_boom,
            cg_marker,
            cg_label,
            wing_patch,
            tail_patch,
            flap_left,
            flap_right,
            aileron_left,
            aileron_right,
            *prop_artists,
            timestamp,
            bank_text,
        )

    dt = float(t[1] - t[0]) if len(t) > 1 else 0.05
    ani = animation.FuncAnimation(fig, animate_frame, frames=len(t), interval=dt * 1000.0, blit=True)
    return fig, ani


def animate_lateral_rear_view(
    t: np.ndarray,
    phi: np.ndarray,
    psi: np.ndarray,
    vehicle: Vehicle,
    *,
    title: str = "Lateral Rear View Animation",
) -> tuple[plt.Figure, animation.FuncAnimation]:
    """Animate a rear-view y-z projection for lateral motion intuition.

    This view is normal to the body x-axis and is meant to make roll visually
    obvious. Yaw does not naturally appear strongly in a pure rear view, so it
    is represented by a lateral nose offset and explicit text annotation.
    """

    layout = _airframe_layout(vehicle)
    semispan = float(layout["semispan"])
    tail_span = float(layout["tail_span"])
    chord = float(layout["chord"])

    fuselage_half_height = 0.16 * chord
    fuselage_half_width = 0.12 * chord
    wing_thickness = 0.035
    tail_thickness = 0.024
    yaw_scale = 0.045 * float(layout["span"])

    fig, ax = plt.subplots(figsize=(6.5, 6.5), dpi=120, constrained_layout=True)
    y_lim = 1.25 * semispan
    z_lim = max(0.45, 1.1 * semispan)
    ax.set_xlim(-y_lim, y_lim)
    ax.set_ylim(-z_lim, z_lim)
    ax.set_xlabel("y (m)")
    ax.set_ylabel("z (m)")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.set_aspect("equal", adjustable="box")

    wing_patch = mpatches.Polygon(np.zeros((4, 2)), closed=True, facecolor="tab:blue", alpha=0.9, edgecolor="none")
    tail_patch = mpatches.Polygon(np.zeros((4, 2)), closed=True, facecolor="tab:green", alpha=0.7, edgecolor="none")
    fuselage_patch = mpatches.Polygon(np.zeros((6, 2)), closed=True, facecolor="k", alpha=0.95, edgecolor="none")
    cg_marker = ax.plot([], [], "o", color="tab:red", markersize=6, label="CG")[0]
    nose_marker = ax.plot([], [], "o", color="tab:gray", markersize=5, label="Nose projection")[0]
    trace = ax.plot([], [], "--", color="tab:orange", linewidth=1.6, label="Nose yaw trace")[0]
    timestamp = ax.text(0.03, 0.94, "", transform=ax.transAxes)
    angle_text = ax.text(0.03, 0.87, "", transform=ax.transAxes)
    for patch in (wing_patch, tail_patch, fuselage_patch):
        ax.add_patch(patch)
    ax.legend(loc="upper right")

    wing_base = np.array(
        [
            [-semispan, -wing_thickness],
            [semispan, -wing_thickness],
            [semispan, wing_thickness],
            [-semispan, wing_thickness],
        ],
        dtype=float,
    )
    tail_half = 0.5 * tail_span
    tail_base = np.array(
        [
            [-tail_half, -tail_thickness],
            [tail_half, -tail_thickness],
            [tail_half, tail_thickness],
            [-tail_half, tail_thickness],
        ],
        dtype=float,
    )
    fuselage_base = np.array(
        [
            [-fuselage_half_width, -fuselage_half_height],
            [0.0, -1.15 * fuselage_half_height],
            [fuselage_half_width, -fuselage_half_height],
            [fuselage_half_width, fuselage_half_height],
            [0.0, 1.15 * fuselage_half_height],
            [-fuselage_half_width, fuselage_half_height],
        ],
        dtype=float,
    )

    yaw_trace_y = np.zeros_like(t)
    yaw_trace_z = np.zeros_like(t)

    def animate_frame(k: int):
        phik = phi[k]
        psik = psi[k]

        rot = _rot2(phik)
        wing_patch.set_xy(wing_base @ rot.T)
        tail_patch.set_xy(0.55 * (tail_base @ rot.T))

        nose_offset = yaw_scale * np.sin(psik)
        fuselage_shape = fuselage_base.copy()
        fuselage_shape[1, 0] += nose_offset
        fuselage_shape[2, 0] += 0.55 * nose_offset
        fuselage_shape[3, 0] += 0.55 * nose_offset
        fuselage_shape[4, 0] += nose_offset
        fuselage_patch.set_xy(fuselage_shape)

        cg_marker.set_data([0.0], [0.0])
        nose_marker.set_data([nose_offset], [0.0])
        yaw_trace_y[k] = nose_offset
        trace.set_data(yaw_trace_y[: k + 1], yaw_trace_z[: k + 1])

        timestamp.set_text(f"t = {t[k]:.2f} s")
        angle_text.set_text(f"phi = {np.rad2deg(phik):.2f} deg\npsi = {np.rad2deg(psik):.2f} deg")

        return (
            wing_patch,
            tail_patch,
            fuselage_patch,
            cg_marker,
            nose_marker,
            trace,
            timestamp,
            angle_text,
        )

    dt = float(t[1] - t[0]) if len(t) > 1 else 0.05
    ani = animation.FuncAnimation(fig, animate_frame, frames=len(t), interval=dt * 1000.0, blit=True)
    return fig, ani


def save_figure(fig: plt.Figure, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
