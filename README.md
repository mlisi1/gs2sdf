# gs2sdf

Converts a trained 2DGS model into physics-ready collision geometry for a Gazebo world. See CLAUDE.md, PHASE1.md, PHASE2.md.

## Pipeline stages

### gs2mesh (Phase 1)

- `load_ply_extract_centers.py` — load a trained 2DGS `.ply`, extract Gaussian centers, optional opacity floater filter.
- `ransac_ground_plane.py` — fit the floor plane via up-axis-aware RANSAC (peels off non-horizontal planes like walls until one matches the declared `--up-axis`).
- `prune_floor_gaussians.py` — remove the floor and everything below it from the original ply (half-space cut on the fitted plane), writes a pruned trained-model copy.
