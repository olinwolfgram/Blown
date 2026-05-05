# Blown-Wing OCP (Python)

This is a fresh, simplified Python-first restart of the blown-wing controls project.

The goal is to keep the project academically clean and easy to share:

1. import a frozen vehicle definition from the AA146 capstone outputs,
2. define the aircraft equations of motion from forces and moments,
3. build a reduced nonlinear optimal control problem from those dynamics,
4. solve and visualize a sample trajectory.

The old MATLAB / Simulink work remains untouched in the parent project. This folder is the new clean start.

## Project Scope

This restart intentionally avoids carrying over the full strip-theory and Simulink plumbing. Instead it uses:

- a frozen vehicle parameter file,
- a compact coefficient-based aerodynamic model,
- a momentum-theory-inspired blown-wing lift increment,
- standard rigid-body 6DOF equations of motion,
- a reduced longitudinal nonlinear program for the first OCP studies.

That keeps the project small enough for Colab and class sharing while preserving the main optimization story.

## Folder Layout

```text
blown_wing_ocp_python/
  data/frozen_vehicle/
  docs/
  src/blown_wing_ocp/
```

## Quick Start: Local

From this folder:

```bash
pip install -e .
python -m blown_wing_ocp.demo
```

## Quick Start: Google Colab

Upload this folder to Drive or GitHub, then in Colab:

```python
%cd /content
!git clone <your-repo-url>
%cd /content/<repo>/blown_wing_ocp_python
!pip install -e .
!python -m blown_wing_ocp.demo
```

## What the Demo Does

The demo script:

1. loads the frozen AA146-derived vehicle,
2. solves a steady longitudinal trim point near cruise,
3. formulates a direct-transcription nonlinear optimal control problem,
4. solves a climb-like trajectory,
5. saves plots to `outputs/`.

## Files You’ll Touch First

- [src/blown_wing_ocp/vehicle.py](C:/Users/olinw/Documents/BlownWing/blown_wing_ocp_python/src/blown_wing_ocp/vehicle.py)
- [src/blown_wing_ocp/aero.py](C:/Users/olinw/Documents/BlownWing/blown_wing_ocp_python/src/blown_wing_ocp/aero.py)
- [src/blown_wing_ocp/dynamics.py](C:/Users/olinw/Documents/BlownWing/blown_wing_ocp_python/src/blown_wing_ocp/dynamics.py)
- [src/blown_wing_ocp/trim.py](C:/Users/olinw/Documents/BlownWing/blown_wing_ocp_python/src/blown_wing_ocp/trim.py)
- [src/blown_wing_ocp/ocp.py](C:/Users/olinw/Documents/BlownWing/blown_wing_ocp_python/src/blown_wing_ocp/ocp.py)
- [docs/modeling_notes.md](C:/Users/olinw/Documents/BlownWing/blown_wing_ocp_python/docs/modeling_notes.md)

## Design Philosophy

This project is meant to stay in an AA222 frame:

- transparent nonlinear model,
- explicit assumptions,
- tunable surrogate parameters,
- finite-horizon optimal control formulation,
- clear separation between model choice and optimization results.

## Notes

- The aerodynamic coefficients in the frozen vehicle file are intentionally low-order and documented.
- The blown-wing increment is a surrogate, not CFD.
- The full 6DOF model exists for clarity and future extensions, but the first optimization problem is longitudinal.
