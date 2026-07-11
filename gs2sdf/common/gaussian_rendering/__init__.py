"""Vendored subset of gs_sensor_core (sibling repo) needed to load and
render a trained 2DGS ply for depth-only fusion (gs2mesh/render_validate_fuse.py).

Copied rather than imported as a library dependency, so gs2sdf doesn't
require gs_sensors to be installed/importable. Only the modules actually
used by this project's rendering path are included — not the full
gs_sensor_core package (ROS integration, LiDAR rendering, culling LOD,
etc. are all out of scope here).
"""
