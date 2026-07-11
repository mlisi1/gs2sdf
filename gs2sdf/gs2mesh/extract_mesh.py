"""Extract a raw triangle mesh from the fused TSDF volume.

Input: tsdf_volume.npz (render_validate_fuse.py)
Output: <output_dir>/raw_mesh.ply — unprocessed mesh reconstruction.
        Non-manifold geometry and tiny disconnected islands are Step 10's
        job (mesh cleanup), not this one — per PHASE1.md, inspecting this
        raw mesh is optional; it's fine to go straight to Step 10 if it
        already looks reasonable.

Uses point cloud extraction + Poisson surface reconstruction, not
VoxelBlockGrid's own extract_triangle_mesh (marching cubes):
extract_triangle_mesh reproducibly crashes with a CUDA "illegal memory
access" — on both GPU and CPU, regardless of weight_threshold,
estimated_vertex_number, or hashmap capacity — on a fully-integrated,
room-scale volume in this Open3D build (0.19.0). extract_point_cloud
works reliably on the exact same data and already includes surface
normals from the TSDF gradient, which Poisson reconstruction needs.

Caveat: unlike marching cubes, Poisson fits a single global smooth
implicit function, so it tends to close up real holes (areas genuinely
never observed, e.g. under furniture, outside camera coverage) with
extrapolated surface rather than leaving them open. Trimming the
lowest-density vertices (Open3D's own documented pattern — density
roughly tracks how far a region is from real supporting points) removes
the most poorly-supported extrapolated regions, but this doesn't fully
recover marching cubes' behavior of just leaving unobserved space empty
— expect somewhat rounder/smoother results, particularly at real
coverage gaps.
"""

import argparse
from pathlib import Path

import matplotlib.cm
import numpy as np
import open3d as o3d


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tsdf_volume", type=Path, help="Path to tsdf_volume.npz")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--weight-threshold",
        type=float,
        default=3.0,
        help="Minimum accumulated integration weight for a voxel to be surfaced (Open3D's own default)",
    )
    parser.add_argument(
        "--poisson-depth",
        type=int,
        default=9,
        help="Poisson reconstruction octree depth — higher is finer detail at more compute/memory cost",
    )
    parser.add_argument(
        "--density-quantile-threshold",
        type=float,
        default=0.01,
        help="Fraction of lowest-density (most extrapolated/unsupported) vertices to trim after Poisson reconstruction",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    vbg = o3d.t.geometry.VoxelBlockGrid.load(str(args.tsdf_volume))
    pcd_t = vbg.extract_point_cloud(weight_threshold=args.weight_threshold)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_t.point.positions.cpu().numpy())
    pcd.normals = o3d.utility.Vector3dVector(pcd_t.point.normals.cpu().numpy())

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=args.poisson_depth
    )

    # Color by Poisson density before trimming, so low-confidence
    # (extrapolated, far-from-data) regions are visible for inspection —
    # normalized min-max per mesh, viridis: dark/purple = low density
    # (least supported), yellow = high density (well supported).
    densities = np.asarray(densities)
    density_norm = (densities - densities.min()) / (densities.max() - densities.min())
    mesh.vertex_colors = o3d.utility.Vector3dVector(matplotlib.cm.viridis(density_norm)[:, :3])

    cutoff = np.quantile(densities, args.density_quantile_threshold)
    low_density_vertices = densities < cutoff
    mesh.remove_vertices_by_mask(low_density_vertices)

    mesh_path = args.output_dir / "raw_mesh.ply"
    o3d.io.write_triangle_mesh(str(mesh_path), mesh)

    print(f"Point cloud: {len(pcd.points)} points")
    print(f"Trimmed {int(low_density_vertices.sum())} low-density vertices")
    print(f"Vertices: {len(mesh.vertices)}")
    print(f"Triangles: {len(mesh.triangles)}")
    print(f"Wrote {mesh_path} (vertex-colored by Poisson density: dark=low-confidence/extrapolated, yellow=well-supported)")


if __name__ == "__main__":
    main()
