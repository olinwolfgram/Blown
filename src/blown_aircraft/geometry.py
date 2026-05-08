from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Vehicle:
    raw: dict

    @property
    def mass_kg(self) -> float:
        return float(self.raw["mass_kg"])

    @property
    def gravity_mps2(self) -> float:
        return float(self.raw["gravity_mps2"])

    @property
    def rho_kgpm3(self) -> float:
        return float(self.raw["rho_kgpm3"])

    @property
    def geometry(self) -> dict:
        return self.raw["geometry"]

    @property
    def aero(self) -> dict:
        return self.raw["aero"]

    @property
    def propulsion(self) -> dict:
        return self.raw["propulsion"]

    @property
    def blown(self) -> dict:
        return self.raw["blown_wing_model"]

    @property
    def controls(self) -> dict:
        return self.raw["controls"]

    @property
    def inertia(self) -> np.ndarray:
        i = self.raw["inertia_kgm2"]
        return np.array(
            [
                [float(i["ixx"]), 0.0, -float(i["ixz"])],
                [0.0, float(i["iyy"]), 0.0],
                [-float(i["ixz"]), 0.0, float(i["izz"])],
            ],
            dtype=float,
        )

    @property
    def control_limits_rad(self) -> dict:
        c = self.controls
        return {
            "elevator": np.deg2rad(float(c["elevator_limit_deg"])),
            "aileron": np.deg2rad(float(c["aileron_limit_deg"])),
            "rudder": np.deg2rad(float(c["rudder_limit_deg"])),
            "flap": np.deg2rad(float(c["flap_limit_deg"])),
        }


def default_vehicle_path() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "vehicle" / "aa146_rank1_vehicle.json"


def load_vehicle(path: str | Path | None = None) -> Vehicle:
    if path is None:
        path = default_vehicle_path()
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return Vehicle(raw=raw)
