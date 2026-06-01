from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from blown_aircraft.geometry import Vehicle, load_vehicle
from blown_aircraft.trim import solve_longitudinal_trim


def set_nested(mapping: dict, path: tuple[str, ...], value: float) -> None:
    target = mapping
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value


def clone_with_override(vehicle: Vehicle, path: tuple[str, ...], value: float) -> Vehicle:
    raw = copy.deepcopy(vehicle.raw)
    set_nested(raw, path, value)
    return Vehicle(raw=raw)


def main() -> None:
    base_vehicle = load_vehicle()
    base_trim = solve_longitudinal_trim(base_vehicle, speed_mps=10.0, flight_path_angle_rad=0.0)
    base_elev_deg = np.rad2deg(base_trim["control"][1])

    print("Cruise trim parameter sensitivity")
    print(f"  baseline elevator trim : {base_elev_deg:.4f} deg")
    print(f"  baseline alpha         : {np.rad2deg(base_trim['alpha_rad']):.4f} deg")
    print(f"  baseline rpm           : {base_trim['control'][0]:.4f}")
    print()

    studies = [
        {
            "label": "Tail incidence (deg)",
            "path": ("geometry", "htail_incidence_deg"),
            "values": [
                float(base_vehicle.geometry["htail_incidence_deg"]) - 2.0,
                float(base_vehicle.geometry["htail_incidence_deg"]) - 1.0,
                float(base_vehicle.geometry["htail_incidence_deg"]),
                float(base_vehicle.geometry["htail_incidence_deg"]) + 1.0,
                float(base_vehicle.geometry["htail_incidence_deg"]) + 2.0,
            ],
        },
        {
            "label": "Wing cm0",
            "path": ("aero", "cm0"),
            "values": [0.5, 0.75, 1.0, 1.25, 1.5],
            "mode": "scale",
        },
        {
            "label": "Wing cm_alpha_per_rad",
            "path": ("aero", "cm_alpha_per_rad"),
            "values": [0.5, 0.75, 1.0, 1.25, 1.5],
            "mode": "scale",
        },
        {
            "label": "Blown pitch_gain",
            "path": ("blown_wing_model", "pitch_gain"),
            "values": [0.5, 0.75, 1.0, 1.25, 1.5],
            "mode": "scale",
        },
    ]

    for study in studies:
        label = study["label"]
        path = study["path"]
        mode = study.get("mode", "absolute")
        base_value = float(base_vehicle.raw[path[0]][path[1]])

        print(label)
        print(
            f"{'setting':>14} {'elev (deg)':>12} {'delta elev':>12} "
            f"{'alpha (deg)':>12} {'rpm':>10} {'|resid|':>10}"
        )
        print("-" * 76)

        for v in study["values"]:
            applied = base_value * v if mode == "scale" else float(v)
            vehicle = clone_with_override(base_vehicle, path, applied)
            trim = solve_longitudinal_trim(vehicle, speed_mps=10.0, flight_path_angle_rad=0.0)

            elev_deg = np.rad2deg(trim["control"][1])
            alpha_deg = np.rad2deg(trim["alpha_rad"])
            rpm = float(trim["control"][0])
            resid = float(np.linalg.norm(trim["residual"]))

            setting_label = f"{v:.2f}x" if mode == "scale" else f"{applied:.4f}"
            print(
                f"{setting_label:>14} {elev_deg:12.4f} {elev_deg - base_elev_deg:12.4f} "
                f"{alpha_deg:12.4f} {rpm:10.2f} {resid:10.5f}"
            )
        print()


if __name__ == "__main__":
    main()
