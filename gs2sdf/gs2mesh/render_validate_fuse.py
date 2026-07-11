"""Streaming render, validate, and fuse.

Input: pruned 2DGS ply (prune_floor_gaussians.py), extended pose set
       (build_extended_pose_set.py)
Output:
  <output_dir>/tsdf_volume.npz — populated sparse/hashed TSDF volume
                                 (Open3D VoxelBlockGrid — see note below)
  <output_dir>/rejection_overlays/*.png — per interpolated pose, a
                                 downsampled depth thumbnail with pixels
                                 rejected by the consistency check tinted
                                 red

For each pose: render depth only (no RGB — color isn't needed anywhere
downstream for collision geometry) from the pruned ply. Interpolated
poses are additionally checked against their two flanking real poses'
depth maps, reprojected into the interpolated view; pixels whose direct
depth disagrees with either available reprojection by more than
`--consistency-tolerance` are zeroed before integration. Real poses are
trusted as training-supported ground truth and integrated directly.

The loop processes one real-to-real "gap" at a time: render real pose A,
render real pose B, render+validate+integrate each interpolated pose
between them (using A's and B's depth), then integrate A and discard its
depth, keeping only B (renamed A) for the next gap — at most two real
depth maps plus one transient interpolated depth map in memory at once,
bounding peak memory regardless of trajectory length.

Uses Open3D's tensor-based VoxelBlockGrid rather than the legacy
ScalableTSDFVolume named in PHASE1.md: same sparse/hashed architecture,
but the legacy class has no save/load, and this step's output has to
survive a process boundary for Step 9 (marching cubes) to load
independently, per this project's one-step-one-script convention.

Uses octree-based frustum culling (gs2sdf.common.gaussian_rendering,
vendored from the sibling gs_sensors repo's gs_sensor_core — copied in
directly rather than depended on as an installed library) rather than
rendering the full model every frame: without it, every render() call
activates (sigmoid/exp/normalize) and rasterizes all ~7M splats
regardless of camera frustum, which is the actual per-frame memory/
compute driver on the hot path, not the octree's own (negligible)
memory footprint. The octree is built/cached once against the pruned
ply and the model is permuted into leaf-contiguous order once at
startup, not per frame.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
from tqdm import tqdm
import torch
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gs2sdf.common.gaussian_rendering.culling import load_or_build_octree
from gs2sdf.common.gaussian_rendering.frames import Pose
from gs2sdf.common.gaussian_rendering.models.ply_loader import load_gaussian_model
from gs2sdf.common.gaussian_rendering.render.pipeline import CameraRasterizer
from gs2sdf.common.gaussian_rendering.camera_profiles.schema import CameraProfile


def w2c_matrix(position: list, rotation_c2w: list) -> np.ndarray:
    r_c2w = np.array(rotation_c2w)
    r_w2c = r_c2w.T
    t_w2c = -r_w2c @ np.array(position)
    m = np.eye(4)
    m[:3, :3] = r_w2c
    m[:3, 3] = t_w2c
    return m


def render_depth(rasterizer: CameraRasterizer, pose: dict) -> np.ndarray:
    quat_xyzw = Rotation.from_matrix(np.array(pose["rotation_c2w"])).as_quat()
    gs_pose = Pose(position=np.array(pose["position"]), orientation=quat_xyzw)
    return rasterizer.render(gs_pose).depth


def backproject(depth: np.ndarray, fx: float, fy: float, cx: float, cy: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vs, us = np.nonzero(depth > 0)
    d = depth[vs, us]
    x = (us - cx) / fx * d
    y = (vs - cy) / fy * d
    return np.stack([x, y, d], axis=1), vs, us


def reproject_depth(
    depth_src: np.ndarray,
    intr: tuple,
    rotation_c2w_src: list,
    position_src: list,
    rotation_c2w_dst: list,
    position_dst: list,
    dst_h: int,
    dst_w: int,
) -> np.ndarray:
    """Unproject depth_src into world space and re-render it (z-buffered)
    from the destination pose's camera, producing what depth_src implies
    the destination view should see."""
    fx, fy, cx, cy = intr
    pts_cam_src, _, _ = backproject(depth_src, fx, fy, cx, cy)
    reproj = np.zeros((dst_h, dst_w), dtype=np.float32)
    if len(pts_cam_src) == 0:
        return reproj

    r_src = np.array(rotation_c2w_src)
    r_dst = np.array(rotation_c2w_dst)
    pts_world = pts_cam_src @ r_src.T + np.array(position_src)
    pts_cam_dst = (pts_world - np.array(position_dst)) @ r_dst

    z = pts_cam_dst[:, 2]
    in_front = z > 1e-6
    safe_z = np.where(in_front, z, 1.0)
    u = np.round(fx * pts_cam_dst[:, 0] / safe_z + cx).astype(np.int64)
    v = np.round(fy * pts_cam_dst[:, 1] / safe_z + cy).astype(np.int64)
    in_bounds = in_front & (u >= 0) & (u < dst_w) & (v >= 0) & (v < dst_h)

    flat = np.full(dst_h * dst_w, np.inf, dtype=np.float32)
    idx = v[in_bounds] * dst_w + u[in_bounds]
    np.minimum.at(flat, idx, z[in_bounds].astype(np.float32))
    flat[np.isinf(flat)] = 0.0
    return flat.reshape(dst_h, dst_w)


def consistency_mask(depth_direct: np.ndarray, reproj_a: np.ndarray, reproj_b: np.ndarray, tolerance: float) -> np.ndarray:
    """Keep a pixel unless it disagrees with an available flanking
    reprojection by more than tolerance. A pixel with no reprojection
    coverage from either flank can't be checked, so it's kept (benefit
    of the doubt) rather than invented a rejection for."""
    has_a, has_b = reproj_a > 0, reproj_b > 0
    bad_a = has_a & (np.abs(depth_direct - reproj_a) > tolerance)
    bad_b = has_b & (np.abs(depth_direct - reproj_b) > tolerance)
    return (depth_direct > 0) & ~(bad_a | bad_b)


def save_rejection_overlay(path: Path, depth: np.ndarray, valid_mask: np.ndarray, max_dim: int) -> None:
    h, w = depth.shape
    stride = max(1, max(h, w) // max_dim)
    small_depth = depth[::stride, ::stride]
    small_mask = valid_mask[::stride, ::stride]
    dmax = small_depth.max() if small_depth.max() > 0 else 1.0
    gray = np.clip(small_depth / dmax * 255, 0, 255).astype(np.uint8)
    rgb = np.stack([gray, gray, gray], axis=-1)
    rejected = (~small_mask) & (small_depth > 0)
    rgb[rejected] = [255, 0, 0]
    o3d.io.write_image(str(path), o3d.geometry.Image(np.ascontiguousarray(rgb)))


def integrate_pose(vbg, intrinsic_t, device, depth: np.ndarray, pose: dict, depth_max: float) -> None:
    depth_img = o3d.t.geometry.Image(depth.astype(np.float32)).to(device)
    extrinsic_t = o3d.core.Tensor(w2c_matrix(pose["position"], pose["rotation_c2w"]), dtype=o3d.core.Dtype.Float64)
    block_coords = vbg.compute_unique_block_coordinates(
        depth_img, intrinsic_t, extrinsic_t, depth_scale=1.0, depth_max=depth_max
    )
    vbg.integrate(block_coords, depth_img, intrinsic_t, extrinsic_t, depth_scale=1.0, depth_max=depth_max)


def truncate_to_max_real(poses: list[dict], max_real_poses: int) -> list[dict]:
    real_seen = 0
    for i, pose in enumerate(poses):
        if not pose["is_interpolated"]:
            real_seen += 1
            if real_seen == max_real_poses:
                return poses[: i + 1]
    return poses


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pruned_ply", type=Path)
    parser.add_argument("extended_poses_json", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--voxel-size", type=float, required=True)
    parser.add_argument("--truncation-distance", type=float, required=True)
    parser.add_argument("--consistency-tolerance", type=float, required=True)
    parser.add_argument("--opacity-threshold", type=float, default=0.0)
    parser.add_argument("--max-depth", type=float, default=50.0)
    parser.add_argument("--downsample-render", action="store_true")
    parser.add_argument("--downsample-factor", type=int, default=4)
    parser.add_argument("--chunk-volume", action="store_true")
    parser.add_argument("--block-count", type=int, default=100_000)
    parser.add_argument("--leaf-max", type=int, default=5000, help="Octree leaf size for frustum culling (gs_sensor_core default; finer isn't necessarily more accurate, see notes/)")
    parser.add_argument("--max-real-poses", type=int, default=None, help="Debug: process only the first N real poses (and interpolated poses between them)")
    parser.add_argument("--overlay-max-dim", type=int, default=480)
    parser.add_argument("--log-every-interpolated", type=int, default=1, help="Log a rejection overlay for every Nth interpolated pose")
    args = parser.parse_args()

    if args.chunk_volume:
        raise NotImplementedError(
            "--chunk-volume is not implemented yet (opt-in only, for scenes too large "
            "for a single TSDF volume — not needed for this project's test scenes so far)"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir = args.output_dir / "rejection_overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    poses = json.loads(args.extended_poses_json.read_text())["poses"]
    if args.max_real_poses is not None:
        poses = truncate_to_max_real(poses, args.max_real_poses)

    sample = next(p for p in poses if not p["is_interpolated"])
    width, height = sample["width"], sample["height"]
    fx, fy, cx, cy = sample["fx"], sample["fy"], sample["cx"], sample["cy"]
    if args.downsample_render:
        f = args.downsample_factor
        width, height = max(1, width // f), max(1, height // f)
        fx, fy, cx, cy = fx / f, fy / f, cx / f, cy / f
    intr = (fx, fy, cx, cy)

    model = load_gaussian_model(str(args.pruned_ply), device="cuda", opacity_threshold=args.opacity_threshold)

    octree = load_or_build_octree(
        str(args.pruned_ply),
        model.get_xyz.detach().cpu().numpy(),
        leaf_max=args.leaf_max,
        build_index=True,
        opacity_threshold=args.opacity_threshold,
    )
    perm = torch.from_numpy(octree.flat_indices).long().to(model.xyz.device)
    model.reorder_(perm)

    profile = CameraProfile(width=width, height=height, fx=fx, fy=fy, cx=cx, cy=cy, frame_id="render", update_rate=1.0)
    rasterizer = CameraRasterizer(model, profile, device="cuda", octree=octree, culling_enabled=True)

    device = o3d.core.Device("CUDA:0")
    vbg = o3d.t.geometry.VoxelBlockGrid(
        attr_names=["tsdf", "weight"],
        attr_dtypes=[o3d.core.Dtype.Float32, o3d.core.Dtype.Float32],
        attr_channels=[(1,), (1,)],
        voxel_size=args.voxel_size,
        block_resolution=16,
        block_count=args.block_count,
        device=device,
    )
    intrinsic_t = o3d.core.Tensor(
        [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=o3d.core.Dtype.Float64
    )

    num_real = sum(1 for p in poses if not p["is_interpolated"])
    num_interp = len(poses) - num_real

    prev_real, prev_depth = None, None
    pending_interp: list[dict] = []
    interp_count = 0
    real_count = 0

    progress = tqdm(total=num_real, desc="Fusing", unit="real pose")
    for pose in poses:
        if not pose["is_interpolated"]:
            depth = render_depth(rasterizer, pose)

            if prev_real is not None:
                for k, ip in enumerate(pending_interp):
                    depth_ip = render_depth(rasterizer, ip)
                    reproj_a = reproject_depth(prev_depth, intr, prev_real["rotation_c2w"], prev_real["position"], ip["rotation_c2w"], ip["position"], height, width)
                    reproj_b = reproject_depth(depth, intr, pose["rotation_c2w"], pose["position"], ip["rotation_c2w"], ip["position"], height, width)
                    valid = consistency_mask(depth_ip, reproj_a, reproj_b, args.consistency_tolerance)
                    depth_ip_masked = np.where(valid, depth_ip, 0.0).astype(np.float32)
                    integrate_pose(vbg, intrinsic_t, device, depth_ip_masked, ip, args.max_depth)

                    interp_count += 1
                    if interp_count % args.log_every_interpolated == 0:
                        save_rejection_overlay(overlay_dir / f"{ip['img_name']}.png", depth_ip, valid, args.overlay_max_dim)

                integrate_pose(vbg, intrinsic_t, device, prev_depth, prev_real, args.max_depth)

            prev_real, prev_depth = pose, depth
            pending_interp = []
            real_count += 1
            progress.set_postfix(interpolated=f"{interp_count}/{num_interp}")
            progress.update(1)
            if real_count % 20 == 0:
                # Frustum culling makes num_rendered vary per pose, so
                # PyTorch's caching allocator grows its reserved pool to
                # whatever the single busiest frame so far needed, and
                # never shrinks it back on its own — release what's
                # actually idle periodically so it doesn't sit stacked
                # on top of the VoxelBlockGrid's own (real, legitimate)
                # growth for the rest of the run.
                torch.cuda.empty_cache()
        else:
            pending_interp.append(pose)
    progress.close()

    if prev_real is not None:
        integrate_pose(vbg, intrinsic_t, device, prev_depth, prev_real, args.max_depth)

    # The VoxelBlockGrid is largest right here, at the very end of the
    # run — free PyTorch's idle reserved memory before save() needs
    # headroom to serialize the whole hash-map structure.
    torch.cuda.empty_cache()
    volume_path = args.output_dir / "tsdf_volume.npz"
    vbg.save(str(volume_path))

    print(f"Integrated {real_count} real + {interp_count} interpolated poses")
    print(f"Wrote {volume_path}")
    print(f"Wrote rejection overlays to {overlay_dir}")


if __name__ == "__main__":
    main()
