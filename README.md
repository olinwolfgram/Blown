# Blown-Wing Aircraft Dynamics Reset

This repository has been reset around a new aircraft-modeling-first structure.

The project now starts from:

1. a frozen JSON geometry and vehicle description derived from the AA146 capstone,
2. a full rigid-body aircraft model,
3. explicit propulsion and aerodynamic submodels,
4. separate longitudinal and lateral reduced models,
5. operating-point linearization utilities for later control design.

The old work has been archived under [OLD/](C:/Users/olinw/Documents/BlownWing/OLD) and [ghost/](C:/Users/olinw/Documents/BlownWing/ghost).

## New Structure

```text
data/
  vehicle/
docs/
src/
  blown_aircraft/
```

## Core Modules

- [rigid_body_ac.py](C:/Users/olinw/Documents/BlownWing/src/blown_aircraft/rigid_body_ac.py)
- [aerodynamics.py](C:/Users/olinw/Documents/BlownWing/src/blown_aircraft/aerodynamics.py)
- [propulsion.py](C:/Users/olinw/Documents/BlownWing/src/blown_aircraft/propulsion.py)
- [longitudinal.py](C:/Users/olinw/Documents/BlownWing/src/blown_aircraft/longitudinal.py)
- [lateral.py](C:/Users/olinw/Documents/BlownWing/src/blown_aircraft/lateral.py)
- [linearize.py](C:/Users/olinw/Documents/BlownWing/src/blown_aircraft/linearize.py)

## Key Idea

We are treating the aircraft as a rigid body with forces and moments

\[
m\dot{\mathbf{V}}_b = \mathbf{F}_b - \boldsymbol{\omega}_b \times m \mathbf{V}_b
\]

\[
\mathbf{I}\dot{\boldsymbol{\omega}}_b =
\mathbf{M}_b - \boldsymbol{\omega}_b \times (\mathbf{I}\boldsymbol{\omega}_b)
\]

where the total force and moment are the sum of:

- aerodynamic wing-strip contributions,
- tail forces and moments,
- propulsion forces and moments,
- gravity.

## What Comes Next

The next control-design steps are:

1. solve trim conditions,
2. linearize about trim,
3. build LQR about those trim conditions,
4. use a nonlinear trajectory optimizer if we want large transitions between operating points.

For static trim, a small nonlinear root solve or constrained NLP is usually enough. We do **not** need SCP just to find trim. SCP or direct transcription becomes useful for trajectory problems, transition maneuvers, and large-envelope planning.
