# gs2sdf

Converts a trained 2DGS model into physics-ready collision geometry for a Gazebo world. See CLAUDE.md, PHASE1.md, PHASE2.md.

## Pipeline stages

### gs2mesh (Phase 1)

- `load_ply_extract_centers.py` — load a trained 2DGS `.ply`, extract Gaussian centers, optional opacity floater filter.
- `ransac_ground_plane.py` — fit the floor plane via up-axis-aware RANSAC (peels off non-horizontal planes like walls until one matches the declared `--up-axis`).
- `prune_floor_gaussians.py` — remove the floor and everything below it from the original ply (half-space cut on the fitted plane), writes a pruned trained-model copy.
- `fit_plane_primitive.py` — fit position/orientation/bounded extent of the floor as an oriented rectangle (PCA-aligned to the footprint), for a later Gazebo `<plane>` primitive.
- `load_camera_poses.py` — parse training camera poses/intrinsics from `cameras.json` (camera-to-world convention), ordered by capture id.
- `build_extended_pose_set.py` — insert interpolated poses (SLERP rotation, linear translation) between adjacent real poses, strictly bounded by the real trajectory.
- `render_validate_fuse.py` — stream-render depth per pose (`diff-surfel-rasterization`, vendored as a submodule under `third_party/`), validate interpolated poses by reprojecting flanking real depth maps, fuse into an Open3D `VoxelBlockGrid` (sparse/hashed TSDF, save/load-able — the legacy `ScalableTSDFVolume` can't cross a process boundary).

Depends on `gs_sensor_core` (sibling repo, already pip-installed editable) for ply loading and rendering.
