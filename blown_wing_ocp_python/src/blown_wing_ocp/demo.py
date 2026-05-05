from __future__ import annotations

from pathlib import Path

import numpy as np

from .ocp import LongitudinalOCPConfig, solve_longitudinal_ocp
from .plotting import save_ocp_plots
from .trim import solve_longitudinal_trim
from .vehicle import load_default_vehicle


def main() -> None:
    vehicle = load_default_vehicle()
    print(f"Loaded vehicle: {vehicle.name}")
    print(f"Mass = {vehicle.mass_kg:.3f} kg | span = {vehicle.span_m:.3f} m | props = {vehicle.propulsion.n_props}")

    trim = solve_longitudinal_trim(vehicle, speed_mps=10.0, flight_path_angle_rad=0.0, rpm_guess=9900.0)
    print("\nTrim result")
    print(f"  success   : {trim.success}")
    print(f"  message   : {trim.message}")
    print(f"  alpha (deg): {np.rad2deg(trim.alpha_rad):.3f}")
    print(f"  theta (deg): {np.rad2deg(trim.theta_rad):.3f}")
    print(f"  rpm        : {trim.rpm:.1f}")
    print(f"  elevator   : {np.rad2deg(trim.elevator_rad):.3f} deg")
    print(f"  residual   : {trim.residual}")

    cfg = LongitudinalOCPConfig()
    ocp = solve_longitudinal_ocp(vehicle, trim, cfg)
    print("\nOCP result")
    print(f"  success   : {ocp.success}")
    print(f"  message   : {ocp.message}")
    print(f"  objective : {ocp.objective:.6g}")
    print(f"  solver    : {ocp.solver_output}")

    out_dir = Path.cwd() / "outputs"
    paths = save_ocp_plots(ocp, trim, out_dir)
    print("\nSaved figures:")
    for path in paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
