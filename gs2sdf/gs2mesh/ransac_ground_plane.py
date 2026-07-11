"""RANSAC ground plane fitting.

Input: Gaussian center point cloud (output of load_ply_extract_centers.py)
Output:
  <output_dir>/plane.json         — fitted plane equation ax+by+cz+d=0
  <output_dir>/inlier_mask.npy    — boolean mask into the input point cloud
  <output_dir>/inliers_highlighted.ply — input cloud with RANSAC inliers
                                         colored red, non-inlier points
                                         below the plane colored orange,
                                         everything else gray

Plain single-shot RANSAC will happily fit the largest coplanar surface in
the scene regardless of orientation — in practice this is often a wall,
not the floor. GS training frames aren't guaranteed to be gravity-aligned
per capture, so the caller must state the scene's up axis explicitly;
candidate planes are then accepted only if their normal is close to it,
peeling off and retrying on non-matching planes (e.g. walls) otherwise.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d

INLIER_COLOR = (1.0, 0.0, 0.0)
BELOW_PLANE_COLOR = (1.0, 0.5, 0.0)
OTHER_COLOR = (0.6, 0.6, 0.6)

UP_AXES = {
    "x": np.array([1.0, 0.0, 0.0]),
    "-x": np.array([-1.0, 0.0, 0.0]),
    "y": np.array([0.0, 1.0, 0.0]),
    "-y": np.array([0.0, -1.0, 0.0]),
    "z": np.array([0.0, 0.0, 1.0]),
    "-z": np.array([0.0, 0.0, -1.0]),
}


def fit_ground_plane(
    pcd: o3d.geometry.PointCloud,
    distance_threshold: float,
    ransac_n: int,
    num_iterations: int,
    up_axis: np.ndarray,
    up_axis_tolerance_deg: float,
    max_attempts: int,
):
    cos_tolerance = np.cos(np.deg2rad(up_axis_tolerance_deg))
    remaining_indices = np.arange(len(pcd.points))
    working_pcd = pcd

    for attempt in range(1, max_attempts + 1):
        plane_model, local_inliers = working_pcd.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations,
        )
        normal = np.array(plane_model[:3])
        normal /= np.linalg.norm(normal)
        alignment = np.dot(normal, up_axis)

        if abs(alignment) >= cos_tolerance:
            if alignment < 0:
                plane_model = tuple(-c for c in plane_model)
            global_inliers = remaining_indices[local_inliers]
            mask = np.zeros(len(pcd.points), dtype=bool)
            mask[global_inliers] = True
            return plane_model, mask, attempt

        keep = np.ones(len(remaining_indices), dtype=bool)
        keep[local_inliers] = False
        remaining_indices = remaining_indices[keep]
        working_pcd = working_pcd.select_by_index(local_inliers, invert=True)

    raise RuntimeError(
        f"No plane aligned with up axis {up_axis.tolist()} within "
        f"{up_axis_tolerance_deg} deg found in {max_attempts} attempts"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Path to centers.ply")
    parser.add_argument("output_dir", type=Path, help="Directory to write outputs into")
    parser.add_argument(
        "--distance-threshold",
        type=float,
        required=True,
        help="RANSAC inlier distance for ground plane fitting (scene-dependent)",
    )
    parser.add_argument(
        "--up-axis",
        choices=sorted(UP_AXES),
        required=True,
        help="Scene's up axis, e.g. 'z' or '-z' (not assumed gravity-aligned per capture)",
    )
    parser.add_argument(
        "--up-axis-tolerance-deg",
        type=float,
        default=15.0,
        help="Max angle between a candidate plane's normal and the up axis to accept it",
    )
    parser.add_argument(
        "--max-plane-attempts",
        type=int,
        default=10,
        help="Max non-matching planes (e.g. walls) to peel off before giving up",
    )
    parser.add_argument("--ransac-n", type=int, default=3, help="Points per RANSAC sample")
    parser.add_argument("--num-iterations", type=int, default=1000)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    pcd = o3d.io.read_point_cloud(str(args.input))
    plane_model, mask, attempts = fit_ground_plane(
        pcd,
        args.distance_threshold,
        args.ransac_n,
        args.num_iterations,
        UP_AXES[args.up_axis],
        args.up_axis_tolerance_deg,
        args.max_plane_attempts,
    )
    a, b, c, d = plane_model

    plane_path = args.output_dir / "plane.json"
    plane_path.write_text(
        json.dumps(
            {
                "a": a,
                "b": b,
                "c": c,
                "d": d,
                "distance_threshold": args.distance_threshold,
                "up_axis": args.up_axis,
                "up_axis_tolerance_deg": args.up_axis_tolerance_deg,
                "attempts": attempts,
                "num_inliers": int(mask.sum()),
                "num_points": len(mask),
            },
            indent=2,
        )
    )

    mask_path = args.output_dir / "inlier_mask.npy"
    np.save(mask_path, mask)

    # Signed distance along the (up-facing) plane normal: negative means a
    # point sits below the floor, on the opposite side from "up".
    points = np.asarray(pcd.points)
    signed_dist = points @ np.array([a, b, c]) + d
    below_mask = (signed_dist < 0) & ~mask

    colors = np.full((len(points), 3), OTHER_COLOR)
    colors[below_mask] = BELOW_PLANE_COLOR
    colors[mask] = INLIER_COLOR
    highlighted = o3d.geometry.PointCloud()
    highlighted.points = pcd.points
    highlighted.colors = o3d.utility.Vector3dVector(colors)
    highlighted_path = args.output_dir / "inliers_highlighted.ply"
    o3d.io.write_point_cloud(str(highlighted_path), highlighted)

    print(f"Below-plane (non-inlier): {int(below_mask.sum())} / {len(mask)}")

    print(f"Plane: {a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f} = 0 (found on attempt {attempts})")
    print(f"Inliers: {int(mask.sum())} / {len(mask)}")
    print(f"Wrote {plane_path}")
    print(f"Wrote {mask_path}")
    print(f"Wrote {highlighted_path}")


if __name__ == "__main__":
    main()
