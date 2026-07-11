"""Fit final plane primitive parameters (position, orientation, bounded extent).

Input: RANSAC floor inlier points (centers.ply + inlier_mask.npy from
       ransac_ground_plane.py), fitted plane (plane.json)
Output:
  <output_dir>/plane_primitive.json — position (3D point at the
                                       footprint center), orientation
                                       (quaternion xyzw; local Z = plane
                                       normal, local X/Y = PCA-aligned
                                       in-plane axes), extent (size along
                                       local X/Y) — for a later Gazebo
                                       <plane> primitive (Phase 2, not
                                       done here)

Orientation and on-plane position come from the RANSAC inliers alone,
but extent comes from the whole scene's footprint (outlier-cleaned)
projected onto those same axes: the floor's own point sampling doesn't
reliably reach every corner of a room (occlusion under furniture, sparse
coverage near walls), but the room's other geometry — walls, furniture —
shares the same horizontal footprint the floor should span.
  <output_dir>/plane_extent_overlay.ply — the pruned preview point cloud
                                          (Step 4 output) with the fitted
                                          rectangle's perimeter overlaid
                                          in a bright color, for visual
                                          inspection

No clustering step has run yet (single-floor scenes only for now — see
PHASE1.md Step 3), so this treats the whole RANSAC inlier set as one
cluster/one plane primitive.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

OVERLAY_COLOR = (0.0, 1.0, 1.0)


def in_plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reference = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(reference, normal)) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])
    u = reference - np.dot(reference, normal) * normal
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    return u, v


def fit_plane_primitive(inlier_points: np.ndarray, scene_points: np.ndarray, plane: dict) -> dict:
    normal = np.array([plane["a"], plane["b"], plane["c"]])
    normal /= np.linalg.norm(normal)
    d = plane["d"]

    # Project inliers exactly onto the plane, removing residual RANSAC
    # noise perpendicular to it before computing orientation/position.
    signed_dist = inlier_points @ normal + d
    on_plane = inlier_points - signed_dist[:, None] * normal

    u0, v0 = in_plane_basis(normal)
    centroid = on_plane.mean(axis=0)
    coords2d = np.column_stack([(on_plane - centroid) @ u0, (on_plane - centroid) @ v0])

    # Align local axes to the floor's own principal direction (e.g. a
    # rotated room's walls) instead of the arbitrary reference basis.
    eigvals, eigvecs = np.linalg.eigh(np.cov(coords2d.T))
    principal = eigvecs[:, np.argmax(eigvals)]
    u = principal[0] * u0 + principal[1] * v0
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)

    # Extent comes from the whole scene's footprint along the same axes,
    # not just the floor inliers (see module docstring).
    scene2d = np.column_stack([(scene_points - centroid) @ u, (scene_points - centroid) @ v])
    lo, hi = scene2d.min(axis=0), scene2d.max(axis=0)
    extent = hi - lo
    footprint_center = centroid + ((lo[0] + hi[0]) / 2) * u + ((lo[1] + hi[1]) / 2) * v

    rotation_matrix = np.column_stack([u, v, normal])
    quat_xyzw = Rotation.from_matrix(rotation_matrix).as_quat()

    return {
        "position": footprint_center.tolist(),
        "orientation_xyzw": quat_xyzw.tolist(),
        "normal": normal.tolist(),
        "extent": extent.tolist(),
        "u_axis": u.tolist(),
        "v_axis": v.tolist(),
    }


def rectangle_perimeter_points(primitive: dict, samples_per_edge: int = 100) -> np.ndarray:
    position = np.array(primitive["position"])
    u = np.array(primitive["u_axis"])
    v = np.array(primitive["v_axis"])
    half_u, half_v = np.array(primitive["extent"]) / 2

    corners = [
        position - half_u * u - half_v * v,
        position + half_u * u - half_v * v,
        position + half_u * u + half_v * v,
        position - half_u * u + half_v * v,
    ]
    edges = []
    for i in range(4):
        start, end = corners[i], corners[(i + 1) % 4]
        t = np.linspace(0.0, 1.0, samples_per_edge)[:, None]
        edges.append(start[None, :] * (1 - t) + end[None, :] * t)
    return np.concatenate(edges, axis=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("centers_ply", type=Path, help="Path to centers.ply")
    parser.add_argument("inlier_mask", type=Path, help="Path to inlier_mask.npy")
    parser.add_argument("plane_json", type=Path, help="Path to plane.json")
    parser.add_argument("pruned_preview_ply", type=Path, help="Path to pruned_preview.ply (Step 4 output)")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--outlier-nb-neighbors", type=int, default=20,
        help="Neighbors sampled per point for statistical outlier removal on the scene footprint",
    )
    parser.add_argument(
        "--outlier-std-ratio", type=float, default=2.0,
        help="Std-dev threshold for statistical outlier removal on the scene footprint",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    plane = json.loads(args.plane_json.read_text())
    pcd = o3d.io.read_point_cloud(str(args.centers_ply))
    points = np.asarray(pcd.points)
    inlier_mask = np.load(args.inlier_mask)
    inlier_points = points[inlier_mask]

    # Drop flying-blob outliers before they inflate the footprint bbox.
    cleaned_pcd, _ = pcd.remove_statistical_outlier(
        nb_neighbors=args.outlier_nb_neighbors, std_ratio=args.outlier_std_ratio
    )
    scene_points = np.asarray(cleaned_pcd.points)

    primitive = fit_plane_primitive(inlier_points, scene_points, plane)

    primitive_path = args.output_dir / "plane_primitive.json"
    primitive_path.write_text(json.dumps(primitive, indent=2))

    perimeter = rectangle_perimeter_points(primitive)
    pruned_preview = o3d.io.read_point_cloud(str(args.pruned_preview_ply))
    base_points = np.asarray(pruned_preview.points)
    base_colors = np.asarray(pruned_preview.colors)

    overlay_points = np.concatenate([base_points, perimeter], axis=0)
    overlay_colors = np.concatenate(
        [base_colors, np.tile(OVERLAY_COLOR, (len(perimeter), 1))], axis=0
    )
    overlay = o3d.geometry.PointCloud()
    overlay.points = o3d.utility.Vector3dVector(overlay_points)
    overlay.colors = o3d.utility.Vector3dVector(overlay_colors)
    overlay_path = args.output_dir / "plane_extent_overlay.ply"
    o3d.io.write_point_cloud(str(overlay_path), overlay)

    print(f"Position: {primitive['position']}")
    print(f"Extent (u x v): {primitive['extent']}")
    print(f"Wrote {primitive_path}")
    print(f"Wrote {overlay_path}")


if __name__ == "__main__":
    main()
