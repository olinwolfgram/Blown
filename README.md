# Blown-Wing Aircraft Controls

This repository contains a blown-wing aircraft model, trim solver, split longitudinal and lateral reduced-order dynamics, LQR regulators, reduced-order SCP planner-tracker demos, actuator-rich 10-motor variants, and a Cesium playback viewer.

The frozen aircraft geometry, propulsion layout, and aerodynamic coefficients come from the upstream AA capstone vehicle definition. The control work here is built on top of that fixed vehicle model.

## Repository layout

```text
data/
  vehicle/
src/
  blown_aircraft/
scripts/
blown_sim/
examples/
  plots/
```

## Python setup

From the repository root:

```powershell
python -m pip install -e .
```

If a plain install is easier:

```powershell
python -m pip install numpy scipy matplotlib cvxpy clarabel jax jaxlib
```

## Core modeling modules

- `src/blown_aircraft/geometry.py`
- `src/blown_aircraft/aerodynamics.py`
- `src/blown_aircraft/propulsion.py`
- `src/blown_aircraft/rigid_body_ac.py`
- `src/blown_aircraft/longitudinal.py`
- `src/blown_aircraft/lateral.py`
- `src/blown_aircraft/operating_point.py`
- `src/blown_aircraft/linearize.py`
- `src/blown_aircraft/lqr.py`
- `src/blown_aircraft/scp.py`
- `src/blown_aircraft/tvlqr.py`
- `src/blown_aircraft/reduced_10motor.py`
- `src/blown_aircraft/jax_longitudinal.py`
- `src/blown_aircraft/jax_lateral.py`
- `src/blown_aircraft/flight_history.py`

## Baseline workflow

The split-dynamics workflow is:

1. compute a symmetric cruise trim,
2. linearize about that operating point,
3. design longitudinal and lateral LQR regulators,
4. run nonlinear closed-loop simulations,
5. generate reduced-order SCP nominals,
6. track those nominals with TVLQR,
7. export selected runs to Cesium-compatible flight logs.

## Trim and linearization

Inspect cruise trim:

```powershell
python scripts/inspect_cruise_trim.py
```

Inspect cruise linearizations:

```powershell
python scripts/demo_cruise_linearization.py
```

## Six main split-dynamics demos

These are the six cases that cover longitudinal and lateral LQR, reduced-order SCP + TVLQR, and reduced-order 10-motor SCP.

### 1. Longitudinal LQR

```powershell
python scripts/demo_longitudinal_lqr.py
python scripts/demo_longitudinal_closed_loop.py
```

Main figure:

- `examples/plots/longitudinal_closed_loop_states.png`

### 2. Lateral LQR

```powershell
python scripts/demo_lateral_lqr.py
python scripts/demo_lateral_closed_loop.py
```

Main figure:

- `examples/plots/lateral_closed_loop_states.png`

### 3. Longitudinal SCP + TVLQR

```powershell
python scripts/demo_scp_longitudinal.py --max-iter 6 --t-final 8 --dt 0.05 --solver CLARABEL --derivatives jax
```

Main figures:

- `examples/plots/longitudinal_scp_tvlqr_states.png`
- `examples/plots/longitudinal_scp_tvlqr_path.png`

### 4. Lateral SCP + TVLQR

```powershell
python scripts/demo_scp_lateral.py --max-iter 6 --t-final 8 --dt 0.05 --solver CLARABEL --derivatives jax
```

Main figures:

- `examples/plots/lateral_scp_tvlqr_states.png`
- `examples/plots/lateral_scp_tvlqr_path.png`

### 5. Longitudinal 10-motor SCP

```powershell
python scripts/demo_scp_longitudinal_10motor.py --max-iter 5 --dt 0.1 --solver CLARABEL --terminal "x:20,h:0"
```

Main figures:

- `examples/plots/longitudinal_scp_10motor_states.png`
- `examples/plots/longitudinal_scp_10motor_controls.png`

### 6. Lateral 10-motor SCP

```powershell
python scripts/demo_scp_lateral_10motor.py --max-iter 5 --dt 0.1 --solver CLARABEL --terminal "x:20,y:5"
```

Main figures:

- `examples/plots/lateral_scp_10motor_states.png`
- `examples/plots/lateral_scp_10motor_controls.png`

The 10-motor SCP demos are exploratory. They expose individual motor commands directly, but they are more sensitive to horizon length, terminal targets, and trust-region tuning than the compact-input reduced-order cases.

## 10-motor LQR extensions

Two additional actuator-rich LQR studies expose all ten motor RPMs directly while keeping the same reduced state definitions.

### Longitudinal 10-motor LQR

```powershell
python scripts/demo_longitudinal_lqr_10motor.py
python scripts/demo_longitudinal_closed_loop_10motor.py
```

### Lateral 10-motor LQR

```powershell
python scripts/demo_lateral_lqr_10motor.py
python scripts/demo_lateral_closed_loop_10motor.py
```

These runs also create flight logs that can be loaded directly in the Cesium viewer.

## Cesium playback

The Cesium viewer lives in `blown_sim/`.

Start the local server:

```powershell
python blown_sim/serve.py
```

Then open the printed URL, paste a Cesium Ion token, and load a flight-history CSV from `outputs/flight_history/`.

Useful flight-log generators for Cesium:

```powershell
python scripts/demo_longitudinal_closed_loop_10motor.py
python scripts/demo_lateral_closed_loop_10motor.py
```

The viewer now includes a sidebar control display that shows:

- live per-motor RPM deviations relative to trim,
- live elevator / aileron / rudder / flap deviations,
- trim/reference labels at the bar centers.

Additional Cesium notes are in:

- `blown_sim/README.md`

## Plot archive included in the repo

Tracked example figures for the main demos are stored in:

- `examples/plots/`

Those figures are copied from the generated `outputs/` directory so the repository shows representative results even though `outputs/` itself remains git-ignored.

## Notes

- The Python model is the source of truth for all dynamics and control logic.
- Cesium is only used for playback and visualization.
- `outputs/` remains ignored because it contains generated files, logs, and large artifacts.
- The split longitudinal and lateral models are the main control-design path in this repository.
