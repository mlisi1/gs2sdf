"""Load 2DGS ply, extract Gaussian centers.

Input: trained 2DGS .ply
Output: <output_dir>/centers.ply — a point cloud of Gaussian centers,
colored by each Gaussian's DC spherical-harmonic term for visual sanity
checking, optionally opacity-filtered to drop obvious floaters.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
from plyfile import PlyData

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from gs2sdf.common.ply_io import load_xyz_opacity_color


def load_centers(ply_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vertex = PlyData.read(str(ply_path))["vertex"]
    return load_xyz_opacity_color(vertex)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Path to trained 2DGS .ply")
    parser.add_argument("output_dir", type=Path, help="Directory to write centers.ply into")
    parser.add_argument(
        "--min-opacity",
        type=float,
        default=None,
        help="Drop Gaussians with activated opacity below this value (optional floater filter)",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    xyz, opacity, color = load_centers(args.input)
    kept = np.ones(len(xyz), dtype=bool)
    if args.min_opacity is not None:
        kept &= opacity >= args.min_opacity

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz[kept])
    pcd.colors = o3d.utility.Vector3dVector(color[kept])

    out_path = args.output_dir / "centers.ply"
    o3d.io.write_point_cloud(str(out_path), pcd)

    print(f"Loaded {len(xyz)} Gaussians, kept {int(kept.sum())} after filtering")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
