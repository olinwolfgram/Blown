from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from blown_aircraft.geometry import Vehicle, load_vehicle


OUTPUT_DIR = REPO_ROOT / "outputs"


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")


def rot2(theta: float) -> np.ndarray:
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=float)


def transform_points(points: np.ndarray, origin: np.ndarray, angle: float) -> np.ndarray:
    return points @ rot2(angle).T + origin


def local_airframe_layout(vehicle: Vehicle) -> dict[str, np.ndarray | float]:
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

    flap_frac = float(geom["flap_span_fraction"])
    flap_half = flap_frac * semispan
    ail_half = semispan - flap_half
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
        "ail_half": ail_half,
        "leading_x": leading_x,
        "trailing_x": trailing_x,
        "prop_x": -(float(prop["axial_x_m"]) - float(geom["cg_x_m"])),
        "prop_y": np.asarray(prop["centers_y_m"], dtype=float),
        "prop_d": float(prop["diameter_m"]),
    }


def make_preview_trajectories() -> dict[str, np.ndarray]:
    t = np.linspace(0.0, 4.0, 81)
    x_long = 1.7 * t
    h_long = 0.12 * np.sin(1.4 * t)
    theta_long = np.deg2rad(4.0) * np.sin(1.1 * t) * np.exp(-0.12 * t)

    x_lat = 2.2 * t
    y_lat = 0.85 * np.sin(1.1 * t) * np.exp(-0.08 * t)
    psi_lat = np.deg2rad(10.0) * np.sin(0.95 * t) * np.exp(-0.1 * t)
    phi_lat = np.deg2rad(8.0) * np.sin(1.2 * t + 0.4) * np.exp(-0.1 * t)

    return {
        "t": t,
        "x_long": x_long,
        "h_long": h_long,
        "theta_long": theta_long,
        "x_lat": x_lat,
        "y_lat": y_lat,
        "psi_lat": psi_lat,
        "phi_lat": phi_lat,
    }


def animate_longitudinal_geometry_preview(
    t: np.ndarray,
    x: np.ndarray,
    h: np.ndarray,
    theta: np.ndarray,
    vehicle: Vehicle,
    *,
    follow_vehicle: bool = False,
    window_width: float = 3.6,
    window_height: float = 2.4,
    title: str = "Geometry Preview: Longitudinal",
) -> tuple[plt.Figure, animation.FuncAnimation]:
    layout = local_airframe_layout(vehicle)
    chord = float(layout["chord"])
    wing_x = float(layout["wing_x"])
    tail_x = float(layout["tail_x"])
    nose_x = float(layout["nose_x"])
    tail_end_x = float(layout["tail_end_x"])
    tail_span = float(layout["tail_span"])
    leading_x = float(layout["leading_x"])
    trailing_x = float(layout["trailing_x"])

    wing_inc = np.deg2rad(float(vehicle.geometry["wing_incidence_deg"]))
    tail_inc = np.deg2rad(float(vehicle.geometry["htail_incidence_deg"]))

    fig, ax = plt.subplots(figsize=(12, 5.5), dpi=120, constrained_layout=True)
    x_pad = max(1.0, 0.15 * max(np.ptp(x), 1.0))
    h_pad = 0.8
    ax.set_xlim(np.min(x) - x_pad, np.max(x) + x_pad)
    ax.set_ylim(np.min(h) - h_pad, np.max(h) + h_pad)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("h (m)")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.set_aspect("equal", adjustable="box")

    trace = ax.plot([], [], "--", color="tab:orange", linewidth=1.6, label="CG trace")[0]
    fuselage_patch = mpatches.Polygon(np.zeros((6, 2)), closed=True, facecolor="k", alpha=0.95, edgecolor="none")
    fuselage_boom = ax.plot([], [], "-", color="k", linewidth=3.0, label="Fuselage")[0]
    cg = ax.plot([], [], "o", color="tab:red", markersize=6, label="CG")[0]
    cg_label = ax.text(0.0, 0.0, "CG", color="tab:red", fontsize=9, weight="bold")
    wing_patch = mpatches.Polygon(np.zeros((4, 2)), closed=True, facecolor="tab:blue", alpha=0.9, edgecolor="none")
    tail_patch = mpatches.Polygon(np.zeros((4, 2)), closed=True, facecolor="tab:green", alpha=0.7, edgecolor="none")
    flap_patch = mpatches.Polygon(np.zeros((4, 2)), closed=True, facecolor="tab:orange", alpha=0.9, edgecolor="none")
    aileron_patch = mpatches.Polygon(np.zeros((4, 2)), closed=True, facecolor="tab:purple", alpha=0.9, edgecolor="none")
    ax.add_patch(wing_patch)
    ax.add_patch(tail_patch)
    ax.add_patch(flap_patch)
    ax.add_patch(aileron_patch)
    ax.add_patch(fuselage_patch)
    timestamp = ax.text(0.02, 0.93, "", transform=ax.transAxes)
    attitude = ax.text(0.02, 0.87, "", transform=ax.transAxes)
    ax.legend(loc="upper right", ncol=2)

    wing_seg = np.array(
        [
            [wing_x - 0.48 * chord, 0.0],
            [wing_x + 0.32 * chord, 0.0],
        ],
        dtype=float,
    )
    tail_seg = np.array(
        [
            [tail_x - 0.24 * chord, 0.0],
            [tail_x + 0.20 * chord, 0.0],
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

    def animate_frame(k: int):
        origin = np.array([x[k], h[k]], dtype=float)
        th = theta[k]

        trace.set_data(x[: k + 1], h[: k + 1])
        cg.set_data([origin[0]], [origin[1]])
        cg_label.set_position((origin[0] + 0.05, origin[1] + 0.06))

        fuselage_fill = transform_points(fuselage_poly, origin, th)
        fus_boom = transform_points(fuselage_boom_seg, origin, th)
        wing_fill = transform_points(wing_poly, origin, th + wing_inc)
        tail_fill = transform_points(tail_poly, origin, th + tail_inc)
        flap_fill = transform_points(flap_poly, origin, th + wing_inc)
        aileron_fill = transform_points(aileron_poly, origin, th + wing_inc)

        fuselage_patch.set_xy(fuselage_fill)
        fuselage_boom.set_data(fus_boom[:, 0], fus_boom[:, 1])
        wing_patch.set_xy(wing_fill)
        tail_patch.set_xy(tail_fill)
        flap_patch.set_xy(flap_fill)
        aileron_patch.set_xy(aileron_fill)

        if follow_vehicle:
            ax.set_xlim(origin[0] - 0.5 * window_width, origin[0] + 0.5 * window_width)
            ax.set_ylim(origin[1] - 0.5 * window_height, origin[1] + 0.5 * window_height)

        timestamp.set_text(f"t = {t[k]:.2f} s")
        attitude.set_text(f"theta = {np.rad2deg(th):.2f} deg")
        return (
            trace,
            fuselage_patch,
            fuselage_boom,
            cg,
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


def animate_lateral_geometry_preview(
    t: np.ndarray,
    x_forward: np.ndarray,
    y_lateral: np.ndarray,
    psi: np.ndarray,
    phi: np.ndarray,
    vehicle: Vehicle,
    *,
    follow_vehicle: bool = False,
    window_width: float = 4.8,
    window_height: float = 3.6,
    title: str = "Geometry Preview: Lateral / Top View",
) -> tuple[plt.Figure, animation.FuncAnimation]:
    layout = local_airframe_layout(vehicle)
    wing_x = float(layout["wing_x"])
    tail_x = float(layout["tail_x"])
    nose_x = float(layout["nose_x"])
    tail_end_x = float(layout["tail_end_x"])
    semispan = float(layout["semispan"])
    tail_span = float(layout["tail_span"])
    flap_half = float(layout["flap_half"])
    prop_x = float(layout["prop_x"])
    prop_y = np.asarray(layout["prop_y"], dtype=float)
    prop_d = float(layout["prop_d"])
    chord = float(layout["chord"])
    leading_x = float(layout["leading_x"])
    trailing_x = float(layout["trailing_x"])

    fig, ax = plt.subplots(figsize=(12, 6), dpi=120, constrained_layout=True)
    x_pad = max(1.0, 0.15 * max(np.ptp(x_forward), 1.0))
    y_pad = 1.6
    ax.set_xlim(np.min(x_forward) - x_pad, np.max(x_forward) + x_pad)
    ax.set_ylim(np.min(y_lateral) - y_pad, np.max(y_lateral) + y_pad)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.set_aspect("equal", adjustable="box")

    trace = ax.plot([], [], "--", color="tab:orange", linewidth=1.6, label="CG trace")[0]
    fuselage_patch = mpatches.Polygon(np.zeros((7, 2)), closed=True, facecolor="k", alpha=0.95, edgecolor="none")
    fuselage_boom = ax.plot([], [], "-", color="k", linewidth=3.0, label="Fuselage")[0]
    cg = ax.plot([], [], "o", color="tab:red", markersize=6, label="CG")[0]
    cg_label = ax.text(0.0, 0.0, "CG", color="tab:red", fontsize=9, weight="bold")
    wing_patch = mpatches.Rectangle((trailing_x, -semispan), leading_x - trailing_x, 2.0 * semispan, facecolor="tab:blue", alpha=0.9, edgecolor="none")
    tail_chord = 0.18 * chord
    tail_patch = mpatches.Rectangle((tail_x - 0.16 * chord, -0.5 * tail_span), tail_chord, tail_span, facecolor="tab:green", alpha=0.7, edgecolor="none")
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
    ax.legend(loc="upper right", ncol=2)

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
    wing_seg = np.array([[wing_x, -semispan], [wing_x, semispan]], dtype=float)
    tail_seg = np.array([[tail_x, -0.5 * tail_span], [tail_x, 0.5 * tail_span]], dtype=float)

    prop_segments = []
    for yi in prop_y:
        prop_segments.append(
            np.array(
                [
                    [prop_x, yi - 0.5 * prop_d],
                    [prop_x, yi + 0.5 * prop_d],
                ],
                dtype=float,
            )
        )

    def animate_frame(k: int):
        origin = np.array([x_forward[k], y_lateral[k]], dtype=float)
        hdg = psi[k]
        bank = phi[k]

        trace.set_data(x_forward[: k + 1], y_lateral[: k + 1])
        cg.set_data([origin[0]], [origin[1]])
        cg_label.set_position((origin[0] + 0.08, origin[1] + 0.08))

        fuselage_fill = transform_points(fuselage_poly, origin, hdg)
        fus_boom = transform_points(fuselage_boom_seg, origin, hdg)

        fuselage_patch.set_xy(fuselage_fill)
        fuselage_boom.set_data(fus_boom[:, 0], fus_boom[:, 1])

        base_transform = mtransforms.Affine2D().rotate(hdg).translate(origin[0], origin[1]) + ax.transData
        for patch in (wing_patch, tail_patch, flap_left, flap_right, aileron_left, aileron_right):
            patch.set_transform(base_transform)

        for artist, seg in zip(prop_artists, prop_segments):
            seg_world = transform_points(seg, origin, hdg)
            artist.set_data(seg_world[:, 0], seg_world[:, 1])

        if follow_vehicle:
            ax.set_xlim(origin[0] - 0.5 * window_width, origin[0] + 0.5 * window_width)
            ax.set_ylim(origin[1] - 0.5 * window_height, origin[1] + 0.5 * window_height)

        timestamp.set_text(f"t = {t[k]:.2f} s")
        bank_text.set_text(f"phi = {np.rad2deg(bank):.2f} deg")
        return (
            trace,
            fuselage_patch,
            fuselage_boom,
            cg,
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


def write_preview_outputs(*, stills_only: bool = False, view: str = "all") -> None:
    vehicle = load_vehicle()
    traj = make_preview_trajectories()

    if view in ("all", "longitudinal"):
        fig_lon, ani_lon = animate_longitudinal_geometry_preview(
            traj["t"],
            traj["x_long"],
            traj["h_long"],
            traj["theta_long"],
            vehicle,
            follow_vehicle=False,
            title="Geometry Preview: Longitudinal (Wide View)",
        )
        save_figure(fig_lon, OUTPUT_DIR / "geometry_preview_longitudinal_frame.png")
        if not stills_only:
            ani_lon.save(OUTPUT_DIR / "geometry_preview_longitudinal.gif", writer="pillow", fps=20)
        plt.close(fig_lon)

        fig_lon_zoom, ani_lon_zoom = animate_longitudinal_geometry_preview(
            traj["t"],
            traj["x_long"],
            traj["h_long"],
            traj["theta_long"],
            vehicle,
            follow_vehicle=True,
            window_width=2.6,
            window_height=1.8,
            title="Geometry Preview: Longitudinal (Zoomed)",
        )
        save_figure(fig_lon_zoom, OUTPUT_DIR / "geometry_preview_longitudinal_zoomed_frame.png")
        if not stills_only:
            ani_lon_zoom.save(OUTPUT_DIR / "geometry_preview_longitudinal_zoomed.gif", writer="pillow", fps=20)
        plt.close(fig_lon_zoom)

    if view in ("all", "lateral"):
        fig_lat, ani_lat = animate_lateral_geometry_preview(
            traj["t"],
            traj["x_lat"],
            traj["y_lat"],
            traj["psi_lat"],
            traj["phi_lat"],
            vehicle,
            follow_vehicle=False,
            title="Geometry Preview: Lateral / Top View (Wide View)",
        )
        save_figure(fig_lat, OUTPUT_DIR / "geometry_preview_lateral_frame.png")
        if not stills_only:
            ani_lat.save(OUTPUT_DIR / "geometry_preview_lateral.gif", writer="pillow", fps=20)
        plt.close(fig_lat)

        fig_lat_zoom, ani_lat_zoom = animate_lateral_geometry_preview(
            traj["t"],
            traj["x_lat"],
            traj["y_lat"],
            traj["psi_lat"],
            traj["phi_lat"],
            vehicle,
            follow_vehicle=True,
            window_width=3.2,
            window_height=2.8,
            title="Geometry Preview: Lateral / Top View (Zoomed)",
        )
        save_figure(fig_lat_zoom, OUTPUT_DIR / "geometry_preview_lateral_zoomed_frame.png")
        if not stills_only:
            ani_lat_zoom.save(OUTPUT_DIR / "geometry_preview_lateral_zoomed.gif", writer="pillow", fps=20)
        plt.close(fig_lat_zoom)

    print("Saved geometry preview artifacts:")
    if view in ("all", "longitudinal"):
        print(f"  {OUTPUT_DIR / 'geometry_preview_longitudinal_frame.png'}")
        if not stills_only:
            print(f"  {OUTPUT_DIR / 'geometry_preview_longitudinal.gif'}")
        print(f"  {OUTPUT_DIR / 'geometry_preview_longitudinal_zoomed_frame.png'}")
        if not stills_only:
            print(f"  {OUTPUT_DIR / 'geometry_preview_longitudinal_zoomed.gif'}")
    if view in ("all", "lateral"):
        print(f"  {OUTPUT_DIR / 'geometry_preview_lateral_frame.png'}")
        if not stills_only:
            print(f"  {OUTPUT_DIR / 'geometry_preview_lateral.gif'}")
        print(f"  {OUTPUT_DIR / 'geometry_preview_lateral_zoomed_frame.png'}")
        if not stills_only:
            print(f"  {OUTPUT_DIR / 'geometry_preview_lateral_zoomed.gif'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast vehicle-geometry preview for animation styling.")
    parser.add_argument(
        "--stills-only",
        action="store_true",
        help="Only save PNG frames, skip GIF generation for faster iteration.",
    )
    parser.add_argument(
        "--view",
        choices=("all", "longitudinal", "lateral"),
        default="all",
        help="Limit preview generation to one view family.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    write_preview_outputs(stills_only=args.stills_only, view=args.view)
