from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .types import PropulsionData, VehicleParameters


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VEHICLE_PATH = PACKAGE_ROOT / "data" / "frozen_vehicle" / "aa146_rank1_vehicle.json"


def _deg2rad(value: float) -> float:
    return float(np.deg2rad(value))


def load_vehicle(path: str | Path) -> VehicleParameters:
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))

    propulsion = PropulsionData(
        n_props=int(raw["propulsion"]["n_props"]),
        diameter_m=float(raw["propulsion"]["diameter_m"]),
        centers_y_m=np.asarray(raw["propulsion"]["centers_y_m"], dtype=float),
        axial_x_m=float(raw["propulsion"]["axial_x_m"]),
        thrust_axis_body=np.asarray(raw["propulsion"]["thrust_axis_body"], dtype=float),
        rpm_grid=np.asarray(raw["propulsion"]["rpm_grid"], dtype=float),
        thrust_n_grid=np.asarray(raw["propulsion"]["thrust_n_grid"], dtype=float),
        cp_static_grid=np.asarray(raw["propulsion"]["cp_static_grid"], dtype=float),
    )

    return VehicleParameters(
        name=raw["name"],
        mass_kg=float(raw["mass_kg"]),
        gravity_mps2=float(raw["gravity_mps2"]),
        rho_kgpm3=float(raw["rho_kgpm3"]),
        span_m=float(raw["geometry"]["span_m"]),
        chord_m=float(raw["geometry"]["chord_m"]),
        area_m2=float(raw["geometry"]["area_m2"]),
        aspect_ratio=float(raw["geometry"]["aspect_ratio"]),
        oswald_e=float(raw["geometry"]["oswald_e"]),
        wing_incidence_rad=_deg2rad(raw["geometry"]["wing_incidence_deg"]),
        wing_washout_rad=_deg2rad(raw["geometry"]["wing_washout_deg"]),
        cg_x_m=float(raw["geometry"]["cg_x_m"]),
        wing_ac_x_m=float(raw["geometry"]["wing_ac_x_m"]),
        tail_arm_m=float(raw["geometry"]["tail_arm_m"]),
        htail_area_m2=float(raw["geometry"]["htail_area_m2"]),
        htail_incidence_rad=_deg2rad(raw["geometry"]["htail_incidence_deg"]),
        ixx_kgm2=float(raw["inertia_kgm2"]["ixx"]),
        iyy_kgm2=float(raw["inertia_kgm2"]["iyy"]),
        izz_kgm2=float(raw["inertia_kgm2"]["izz"]),
        ixz_kgm2=float(raw["inertia_kgm2"]["ixz"]),
        elevator_limit_rad=_deg2rad(raw["controls"]["elevator_limit_deg"]),
        aileron_limit_rad=_deg2rad(raw["controls"]["aileron_limit_deg"]),
        rudder_limit_rad=_deg2rad(raw["controls"]["rudder_limit_deg"]),
        flap_limit_rad=_deg2rad(raw["controls"]["flap_limit_deg"]),
        propulsion=propulsion,
        aero=raw["aero"],
        blown=raw["blown_wing_model"],
        references=raw["references"],
    )


def load_default_vehicle() -> VehicleParameters:
    return load_vehicle(DEFAULT_VEHICLE_PATH)


def freeze_vehicle_from_capstone(
    stage3_csv: str | Path,
    stage2_csv: str | Path,
    ecalc_csv: str | Path,
    output_json: str | Path,
) -> Path:
    """Create a compact frozen vehicle JSON from capstone CSV artifacts.

    This exporter intentionally keeps only the data needed by the simplified
    Python OCP project. Aerodynamic derivatives remain explicit assumptions in
    the frozen JSON so they can be reviewed and tuned directly.
    """

    stage3_csv = Path(stage3_csv)
    stage2_csv = Path(stage2_csv)
    ecalc_csv = Path(ecalc_csv)
    output_json = Path(output_json)

    with stage3_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    design = next(row for row in rows if row.get("status") == "SUCCESS")

    with stage2_csv.open(newline="", encoding="utf-8") as f:
        stage2_rows = list(csv.DictReader(f))
    prop_row = next(
        row
        for row in stage2_rows
        if int(float(row["n_props"])) == int(float(design["n_props"]))
        and abs(float(row["prop_diameter_in"]) - float(design["prop_diameter_in"])) < 1e-9
        and abs(float(row["prop_pitch_ratio"]) - float(design["prop_pitch_ratio"])) < 1e-9
        and row["prop_family"] == design["prop_family"]
    )

    with ecalc_csv.open(newline="", encoding="utf-8") as f:
        prop_rows = list(csv.DictReader(f))

    payload = {
        "name": "AA146 Rank-1 DAE51 blown-wing vehicle",
        "mass_kg": float(design["gross_flight_mass_kg"]),
        "gravity_mps2": 9.80665,
        "rho_kgpm3": 1.225,
        "geometry": {
            "span_m": float(design["wing_span_m"]),
            "chord_m": float(design["wing_chord_m"]),
            "area_m2": float(design["wing_area_m2"]),
            "aspect_ratio": float(design["wing_aspect_ratio"]),
            "oswald_e": 0.66,
            "wing_incidence_deg": float(design["main_wing_incidence_deg"]),
            "wing_washout_deg": float(design["main_wing_washout_deg"]),
            "cg_x_m": float(design["cg_x_m"]),
            "wing_ac_x_m": 0.25 * float(design["wing_chord_m"]),
            "tail_arm_m": float(design["tail_arm_m"]),
            "htail_area_m2": float(design["htail_area_m2"]),
            "htail_incidence_deg": float(design["htail_incidence_deg"]),
        },
        "inertia_kgm2": {
            "ixx": 0.50,
            "iyy": 0.80,
            "izz": 1.20,
            "ixz": 0.05,
        },
        "controls": {
            "elevator_limit_deg": float(design["elevator_max_deflection_deg"]),
            "aileron_limit_deg": 20.0,
            "rudder_limit_deg": float(design["rudder_max_deflection_deg"]),
            "flap_limit_deg": float(design["flap_deflection_slow_deg"]),
        },
        "propulsion": {
            "n_props": int(float(design["n_props"])),
            "diameter_m": float(design["prop_diameter_in"]) * 0.0254,
            "centers_y_m": [float(v) for v in prop_row["prop_centers_m"].split(";")],
            "axial_x_m": float(design["prop_axial_x_m"]),
            "thrust_axis_body": [1.0, 0.0, 0.0],
            "rpm_grid": [float(row["rpm"]) for row in prop_rows],
            "thrust_n_grid": [float(row["thrust_n"]) for row in prop_rows],
            "cp_static_grid": [float(row["cp_static"]) for row in prop_rows],
        },
        "aero": {
            "cl0": 0.18,
            "cl_alpha_per_rad": 5.10,
            "cl_q": 3.50,
            "cl_de_per_rad": 0.35,
            "cd0": 0.065,
            "cm0": 0.03,
            "cm_alpha_per_rad": -0.85,
            "cm_q": -14.0,
            "cm_de_per_rad": -1.10,
            "cy_beta_per_rad": -0.45,
            "cl_beta_per_rad": -0.08,
            "cn_beta_per_rad": 0.07,
            "cl_p": -0.45,
            "cn_r": -0.20,
        },
        "blown_wing_model": {
            "blown_span_fraction": float(prop_row["blown_span_fraction"]),
            "prop_to_wing_leading_edge_m": max(-float(design["prop_axial_x_m"]), 0.0),
            "wake_decay_length_m": 0.75 * float(design["wing_chord_m"]),
            "lift_gain": 0.22,
            "drag_gain": 0.03,
            "pitch_gain": -0.045,
            "flap_lift_gain": 1.20,
            "flap_drag_gain": 0.40,
            "flap_pitch_gain": -0.60,
        },
        "references": {
            "aerodynamics": "Anderson, Fundamentals of Aerodynamics, 5th ed.",
            "flight_dynamics": "Nelson, Flight Stability and Automatic Control",
            "airfoil_sections": "Abbott and von Doenhoff, Theory of Wing Sections",
            "blown_wing": "Agrawal et al. (2019), Wind Tunnel Testing of a Blown Flap Wing",
            "upstream_geometry": "AA146 capstone Stage 2 and Stage 3 output CSV artifacts",
        },
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_json


def freeze_vehicle_cli() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    capstone_root = repo_root / "AA146-Capstone"
    stage3_csv = capstone_root / "outputs" / "stage3_aerosandbox_top_designs.csv"
    stage2_csv = capstone_root / "outputs" / "stage2_prop_span_report.csv"
    ecalc_csv = (
        capstone_root
        / "outputs"
        / "ecalc_prop_analysis"
        / "x2302_1500kv_3s_5p5x3p5_3b"
        / "ecalc_static_partial_load.csv"
    )
    out = freeze_vehicle_from_capstone(stage3_csv, stage2_csv, ecalc_csv, DEFAULT_VEHICLE_PATH)
    print(f"Wrote frozen vehicle JSON to {out}")
