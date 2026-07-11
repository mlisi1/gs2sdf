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

A Gaussian is pruned if its center's signed distance along the plane's
up-facing normal is <= distance_threshold — this single condition covers
both the RANSAC inlier band (the floor itself) and everything below it,
per PHASE1.md's indoor-only planar-floor assumption.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
from plyfile import PlyData, PlyElement

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from gs2sdf.common.ply_io import load_xyz_opacity_color


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Path to the original trained 2DGS .ply")
    parser.add_argument("plane_json", type=Path, help="Path to plane.json from ransac_ground_plane.py")
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    plane = json.loads(args.plane_json.read_text())
    normal_and_offset = np.array([plane["a"], plane["b"], plane["c"], plane["d"]])
    distance_threshold = plane["distance_threshold"]

    ply = PlyData.read(str(args.input))
    data = ply["vertex"].data

    xyz = np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float64)
    signed_dist = np.column_stack([xyz, np.ones(len(xyz))]) @ normal_and_offset
    keep_mask = signed_dist > distance_threshold

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

    print(f"Pruned {int((~keep_mask).sum())} / {len(keep_mask)} Gaussians (floor + below-plane)")
    print(f"Kept {int(keep_mask.sum())}")
    print(f"Wrote {pruned_path}")
    print(f"Wrote {preview_path}")


if __name__ == "__main__":
    main()
