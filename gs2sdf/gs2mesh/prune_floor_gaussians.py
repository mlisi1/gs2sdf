"""Prune floor Gaussians from a trained 2DGS ply.

Input: original trained 2DGS .ply, fitted plane (plane.json from
       ransac_ground_plane.py)
Output:
  <output_dir>/pruned.ply         — full-property trained-model ply with
                                    floor Gaussians removed (original ply
                                    untouched, still needed by gs_sensors
                                    for appearance rendering)
  <output_dir>/pruned_preview.ply — colored point-cloud preview of the
                                    survivors (same DC-color technique as
                                    load_ply_extract_centers.py) for
                                    visual comparison against centers.ply
                                    — a stand-in for the doc's "render
                                    original vs. pruned" inspection, since
                                    no renderer exists in the pipeline yet
                                    (that's Step 8)

A 2D Gaussian is a flat disk, not a point: pruning by center position
alone leaves splats whose center clears the cutoff but whose disk still
reaches down to (or past) the actual floor. So a Gaussian is first
pruned if the *closest point on its disk* to the plane — not its center
— is at or below `distance_threshold` (disk radius = `extent_sigma`
standard deviations of scale_0/scale_1).

That alone wasn't enough: rendering pruned.ply still showed most of the
floor. Checking the surviving near-floor splats directly showed they're
mostly small, near-vertical disks (median 72 degrees from horizontal),
not large near-horizontal ones — i.e. not a disk-extent problem at all.
The remaining floor appearance is disconnected debris: small clusters of
splats near the floor with no connection to any larger structure (wall
bases, furniture, real clutter would all be part of bigger connected
groups).

A second pass drops near-floor survivors (within `--near-floor-band` of
the plane) that have fewer than `--cluster-min-points` neighbors within
`--cluster-eps`. Critically, that neighbor count is searched against the
*whole* kept point cloud, not just other near-floor points: clustering
within the band alone (an earlier, wrong version of this) would judge a
chair leg's bottom segment purely by its neighbors inside the band,
which can look like a small orphaned cluster even though the leg
plainly connects to a well-populated chair just above the cutoff —
chopping the base off of every object touching the floor, not just
genuinely isolated debris.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from gs2sdf.common.ply_io import load_xyz_opacity_color


def disk_min_signed_dist(data, normal: np.ndarray, extent_sigma: float) -> np.ndarray:
    xyz = np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float64)
    center_signed_dist = xyz @ normal

    # Raw ply quaternion order is (w, x, y, z) and unnormalized (see
    # notes/ply_format.md); scipy wants (x, y, z, w).
    raw_quat = np.stack([data["rot_0"], data["rot_1"], data["rot_2"], data["rot_3"]], axis=1)
    quat_xyzw = np.column_stack([raw_quat[:, 1:4], raw_quat[:, 0]])
    quat_xyzw /= np.linalg.norm(quat_xyzw, axis=1, keepdims=True)
    rotmats = Rotation.from_quat(quat_xyzw).as_matrix()
    u = rotmats[:, :, 0]
    v = rotmats[:, :, 1]

    scale = extent_sigma * np.exp(np.stack([data["scale_0"], data["scale_1"]], axis=1).astype(np.float64))
    proj_u = scale[:, 0] * (u @ normal)
    proj_v = scale[:, 1] * (v @ normal)
    projected_extent = np.sqrt(proj_u**2 + proj_v**2)

    return center_signed_dist - projected_extent


def remove_isolated_near_floor_clusters(
    xyz: np.ndarray,
    normal: np.ndarray,
    d: float,
    keep_mask: np.ndarray,
    near_floor_band: float,
    eps: float,
    min_points: int,
) -> np.ndarray:
    signed_dist = xyz @ normal + d
    candidate = keep_mask & (signed_dist > 0) & (signed_dist < near_floor_band)
    candidate_idx = np.nonzero(candidate)[0]
    if len(candidate_idx) == 0:
        return keep_mask

    # Neighbor search against every kept point, not just other near-floor
    # candidates, so a candidate connected to structure just above the
    # band (e.g. the rest of a chair leg) is correctly found as connected.
    kept_idx = np.nonzero(keep_mask)[0]
    tree = cKDTree(xyz[kept_idx])
    neighbor_counts = tree.query_ball_point(xyz[candidate_idx], r=eps, return_length=True, workers=-1)

    isolated_idx = candidate_idx[neighbor_counts < min_points]
    new_keep_mask = keep_mask.copy()
    new_keep_mask[isolated_idx] = False
    print(f"Near-floor isolation filter: {len(candidate_idx)} candidates, {len(isolated_idx)} isolated (dropped)")
    return new_keep_mask


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Path to the original trained 2DGS .ply")
    parser.add_argument("plane_json", type=Path, help="Path to plane.json from ransac_ground_plane.py")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--extent-sigma",
        type=float,
        default=3.0,
        help="Standard deviations of scale_0/scale_1 counted as a splat's practical disk radius",
    )
    parser.add_argument(
        "--near-floor-band",
        type=float,
        required=True,
        help="Height above the plane, among first-pass survivors, to check for isolated debris clusters",
    )
    parser.add_argument(
        "--cluster-eps",
        type=float,
        required=True,
        help="DBSCAN neighborhood radius for the near-floor isolated-debris filter (scene-dependent)",
    )
    parser.add_argument(
        "--cluster-min-points",
        type=int,
        required=True,
        help="Minimum connected splats to NOT be considered isolated debris near the floor",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    plane = json.loads(args.plane_json.read_text())
    normal = np.array([plane["a"], plane["b"], plane["c"]])
    d = plane["d"]
    distance_threshold = plane["distance_threshold"]

    ply = PlyData.read(str(args.input))
    data = ply["vertex"].data

    min_signed_dist = disk_min_signed_dist(data, normal, args.extent_sigma) + d
    keep_mask = min_signed_dist > distance_threshold

    xyz = np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float64)
    keep_mask = remove_isolated_near_floor_clusters(
        xyz, normal, d, keep_mask, args.near_floor_band, args.cluster_eps, args.cluster_min_points
    )

    pruned_data = data[keep_mask]
    pruned_element = PlyElement.describe(pruned_data, "vertex")
    pruned_path = args.output_dir / "pruned.ply"
    PlyData([pruned_element], text=ply.text, comments=ply.comments).write(str(pruned_path))

    xyz_kept, _, color_kept = load_xyz_opacity_color(pruned_data)
    preview = o3d.geometry.PointCloud()
    preview.points = o3d.utility.Vector3dVector(xyz_kept)
    preview.colors = o3d.utility.Vector3dVector(color_kept)
    preview_path = args.output_dir / "pruned_preview.ply"
    o3d.io.write_point_cloud(str(preview_path), preview)

    print(f"Pruned {int((~keep_mask).sum())} / {len(keep_mask)} Gaussians (floor disk extent + below-plane)")
    print(f"Kept {int(keep_mask.sum())}")
    print(f"Wrote {pruned_path}")
    print(f"Wrote {preview_path}")


if __name__ == "__main__":
    main()
