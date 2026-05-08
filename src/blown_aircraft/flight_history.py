from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np


EARTH_RADIUS_M = 6378137.0


def lake_lagunita_reference() -> dict[str, float]:
    return {
        "name": "Lake Lagunita, Stanford",
        "lat_deg": 37.423273,
        "lon_deg": -122.176076,
        "alt_m": 40.0,
    }


def local_offsets_to_geodetic(
    east_m: np.ndarray,
    north_m: np.ndarray,
    up_m: np.ndarray,
    lat0_deg: float,
    lon0_deg: float,
    alt0_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    east = np.asarray(east_m, dtype=float)
    north = np.asarray(north_m, dtype=float)
    up = np.asarray(up_m, dtype=float)

    lat0_rad = np.deg2rad(lat0_deg)
    lat_deg = lat0_deg + np.rad2deg(north / EARTH_RADIUS_M)
    lon_deg = lon0_deg + np.rad2deg(east / (EARTH_RADIUS_M * np.cos(lat0_rad)))
    alt_m = alt0_m + up
    return lat_deg, lon_deg, alt_m


def timestamped_history_path(output_dir: Path, stem: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return output_dir / f"{stamp}_{stem}.csv"


def write_flight_history_csv(path: str | Path, rows: Iterable[dict[str, object]], fieldnames: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
