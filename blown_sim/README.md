# Blown Wing Cesium Playback

This folder contains a browser-based Cesium playback viewer for the aircraft
state histories exported from the control demos.

## What it expects

The viewer is built around the CSV files produced by:

- `python scripts/export_longitudinal_control_flight.py`
- `python scripts/export_lateral_control_flight.py`

Those exporters write timestamped files into:

- `outputs/flight_history/`

Examples:

- `*_longitudinal_lqr_control_flight.csv`
- `*_lateral_lqr_control_flight.csv`

The CSV includes:

- time
- geodetic position
- local East/North/Up offsets
- body attitude
- body rates
- translational velocities
- control-surface commands
- motor RPM values

If `east_m`, `north_m`, and `up_m` are present, the Cesium app can relocate the
same flight history to any starting latitude/longitude/altitude entered in the
GUI.

## Default start point

The viewer initializes the origin fields to:

- Lake Lagunita, Stanford
- latitude `37.423273`
- longitude `-122.176076`
- altitude `40.0 m`

## How to run

1. Export one of the control histories:

   ```powershell
   python scripts/export_longitudinal_control_flight.py
   ```

   or

   ```powershell
   python scripts/export_lateral_control_flight.py
   ```

2. Start the local static server:

   ```powershell
   python blown_sim/serve.py
   ```

3. In the browser:
   - paste your Cesium Ion token
   - optionally change the origin latitude / longitude / altitude
   - browse to one of the CSV files in `outputs/flight_history/`
   - click `Visualize Flight`

## Notes

- The viewer uses a simple glTF aircraft model at:
  - `blown_sim/static/models/egg_dropper.gltf`
- The Python dynamics remain the source of truth. Cesium is only used for
  playback and visualization.
