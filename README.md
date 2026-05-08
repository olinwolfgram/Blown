# Blown-Wing Aircraft Dynamics and Cesium Playback

This repository contains the rebuilt Python project for the blown-wing aircraft
controls work. The current focus is:

1. a frozen vehicle geometry / parameter JSON,
2. a rigid-body aircraft model,
3. separate longitudinal and lateral reduced dynamics,
4. trim and linearization tooling,
5. infinite-horizon and finite-horizon LQR demos,
6. CSV export for replay,
7. a browser-based Cesium playback app.

The old work has been archived under `OLD/` and `ghost/`.

## Repository layout

```text
data/
  vehicle/
src/
  blown_aircraft/
scripts/
blown_sim/
```

## Core modules

- `src/blown_aircraft/geometry.py`
- `src/blown_aircraft/propulsion.py`
- `src/blown_aircraft/aerodynamics.py`
- `src/blown_aircraft/rigid_body_ac.py`
- `src/blown_aircraft/longitudinal.py`
- `src/blown_aircraft/lateral.py`
- `src/blown_aircraft/trim.py`
- `src/blown_aircraft/linearize.py`
- `src/blown_aircraft/lqr.py`
- `src/blown_aircraft/finite_horizon.py`
- `src/blown_aircraft/flight_history.py`

## Python setup

From the repo root:

```powershell
python -m pip install -e .
```

If you prefer a plain dependency install:

```powershell
python -m pip install numpy scipy
```

## Vehicle data

The current frozen aircraft configuration lives in:

- `data/vehicle/aa146_rank1_vehicle.json`

That JSON contains the geometry, propulsion data, and aerodynamic parameters
used by the Python model.

## Recommended workflow

The normal control-design flow is:

1. inspect trim,
2. linearize about cruise,
3. compute infinite-horizon LQR,
4. run nonlinear closed-loop demos,
5. compute finite-horizon LQR,
6. export a flight-history CSV,
7. replay it in Cesium.

## 1. Cruise trim diagnostics

Inspect the current symmetric cruise trim and force / moment breakdown:

```powershell
python scripts/inspect_cruise_trim.py
```

This prints:

- cruise speed,
- trim angle of attack,
- trim pitch angle,
- collective RPM,
- elevator trim,
- force balance,
- moment balance,
- wing-family strip totals,
- drag decomposition.

## 2. Cruise linearization

Compute the cruise operating point and local linearization:

```powershell
python scripts/demo_cruise_linearization.py
```

This prints:

- longitudinal trim residual,
- lateral trim residual,
- longitudinal `A`, `B`,
- lateral `A`, `B`.

## 3. Infinite-horizon LQR

### Longitudinal LQR

```powershell
python scripts/demo_longitudinal_lqr.py
```

This computes the longitudinal inner-loop LQR about cruise trim and prints:

- trim operating point,
- chosen state / input subset,
- continuous-time gain `K`,
- continuous closed-loop eigenvalues,
- discrete-time gain `K`,
- discrete closed-loop eigenvalues.

### Lateral LQR

```powershell
python scripts/demo_lateral_lqr.py
```

This computes the lateral differential-thrust LQR and prints:

- lateral trim point,
- effective differential-thrust input model,
- continuous-time gain `K`,
- continuous closed-loop eigenvalues.

## 4. Nonlinear closed-loop demos

### Longitudinal nonlinear closed-loop demo

```powershell
python scripts/demo_longitudinal_closed_loop.py
```

This runs the nonlinear longitudinal model with the LQR controller and saves:

- state-response plots,
- animation frames,
- GIF playback.

### Lateral nonlinear closed-loop demo

```powershell
python scripts/demo_lateral_closed_loop.py
```

This runs the nonlinear lateral model with the differential-thrust LQR and saves:

- state-response plots,
- top-view animation,
- zoomed animation,
- rear-view animation.

## 5. Finite-horizon LQR

### Longitudinal finite-horizon LQR

```powershell
python scripts/demo_finite_horizon_longitudinal.py
```

Optional GIF export:

```powershell
python scripts/demo_finite_horizon_longitudinal.py --save-gifs
```

### Lateral finite-horizon LQR

```powershell
python scripts/demo_finite_horizon_lateral.py
```

Optional GIF export:

```powershell
python scripts/demo_finite_horizon_lateral.py --save-gifs
```

These scripts compute the backward Riccati recursion, apply the resulting
time-varying gains to the nonlinear reduced models, and save plots / animation
frames.

## 6. Parameter sweeps and diagnostics

### Elevator pitch-moment sweep

```powershell
python scripts/sweep_elevator_pitch_moment.py
```

### Trim sensitivity sweep

```powershell
python scripts/sweep_trim_parameter_sensitivity.py
```

### Vehicle geometry preview

```powershell
python scripts/preview_vehicle_geometry.py --stills-only
```

Full preview GIFs:

```powershell
python scripts/preview_vehicle_geometry.py
```

## 7. Export flight history CSVs for Cesium

The exporters create timestamped CSVs in:

- `outputs/flight_history/`

### Longitudinal LQR flight history

```powershell
python scripts/export_longitudinal_control_flight.py
```

### Longitudinal finite-horizon flight history

```powershell
python scripts/export_longitudinal_control_flight.py --controller finite
```

### Lateral LQR flight history

```powershell
python scripts/export_lateral_control_flight.py
```

### Lateral finite-horizon flight history

```powershell
python scripts/export_lateral_control_flight.py --controller finite
```

Optional origin override for either exporter:

```powershell
python scripts/export_longitudinal_control_flight.py --lat 37.423273 --lon -122.176076 --alt 40.0
```

The CSV includes:

- time,
- latitude / longitude / altitude,
- local east / north / up offsets,
- position and attitude,
- body velocities,
- body rates,
- control-surface commands,
- left / right / collective RPM values,
- deviation states used for control interpretation.

## 8. Cesium playback

The browser playback app lives in:

- `blown_sim/`

### Start the local server

From the repo root:

```powershell
python blown_sim/serve.py
```

If port `8008` is busy, the server automatically moves to the next open port.

### Open the Cesium app

Open the URL printed by the server, for example:

- `http://localhost:8008/index.html`

### In the browser

1. paste your Cesium Ion token,
2. optionally change the world origin,
3. browse to a CSV from `outputs/flight_history/`,
4. click `Visualize Flight`.

The default origin is:

- Lake Lagunita, Stanford
- latitude `37.423273`
- longitude `-122.176076`
- altitude `40.0 m`

### Cesium tuning knobs

The model-viewer adjustments live near the top of:

- `blown_sim/app.js`

Useful constants:

- `MODEL_HEADING_CORRECTION_DEG`
- `MODEL_PITCH_CORRECTION_DEG`
- `MODEL_ROLL_CORRECTION_DEG`
- `PROP_VISUAL_RPM_SCALE`
- `PROP_ROTATION_AXIS`

These let you tune:

- model orientation,
- prop rotation speed,
- prop rotation axis.

## Quick command summary

### Trim / linearization

```powershell
python scripts/inspect_cruise_trim.py
python scripts/demo_cruise_linearization.py
```

### Infinite-horizon LQR

```powershell
python scripts/demo_longitudinal_lqr.py
python scripts/demo_lateral_lqr.py
python scripts/demo_longitudinal_closed_loop.py
python scripts/demo_lateral_closed_loop.py
```

### Finite-horizon LQR

```powershell
python scripts/demo_finite_horizon_longitudinal.py
python scripts/demo_finite_horizon_lateral.py
```

### Cesium export + playback

```powershell
python scripts/export_longitudinal_control_flight.py
python scripts/export_lateral_control_flight.py
python blown_sim/serve.py
```

## Notes

- The Python model is the source of truth for all dynamics and control logic.
- Cesium is only used for playback / visualization.
- `outputs/` is intentionally ignored by Git.
- `OLD/`, `ghost/`, `reference_cesium/`, `AA146-Capstone/`, and
  `simulink-agentic-toolkit/` are not part of the minimal runnable Python/Cesium
  project snapshot.
