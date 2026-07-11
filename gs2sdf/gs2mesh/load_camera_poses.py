"""Load training camera poses & intrinsics.

Input: cameras.json — the 3DGS-convention camera export used by this
       project's training pipeline (a list of per-camera entries with
       id, img_name, width, height, fx, fy, position, rotation).
       `position` is the camera center in world coordinates; `rotation`
       is the 3x3 camera-to-world rotation matrix, in the OpenCV/COLMAP
       optical convention (camera-local x-right, y-down, z-forward), so
       a point in camera space maps to world via
       X_world = rotation @ X_cam + position. No cx/cy is present in
       this export, so the principal point is assumed to be the image
       center.
Output:
  <output_dir>/poses.json — parsed camera trajectory, ordered by id
                             (assumed to be capture order)
  <output_dir>/trajectory_overlay.ply — camera positions (green) and
                             short forward-direction stubs (magenta)
                             overlaid on the pruned scene preview, for
                             visually confirming the trajectory aligns
                             with and looks into the scene
"""

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d

POSITION_COLOR = (1.0, 0.0, 0.0)
FORWARD_COLOR = (0.0, 0.3, 1.0)
STUB_LENGTH = 0.5
STUB_SAMPLES = 10


def load_poses(cameras_json_path: Path) -> list[dict]:
    raw_cameras = json.loads(cameras_json_path.read_text())
    cameras = []
    for cam in sorted(raw_cameras, key=lambda c: c["id"]):
        cameras.append(
            {
                "id": cam["id"],
                "img_name": cam["img_name"],
                "width": cam["width"],
                "height": cam["height"],
                "fx": cam["fx"],
                "fy": cam["fy"],
                "cx": cam.get("cx", cam["width"] / 2),
                "cy": cam.get("cy", cam["height"] / 2),
                "position": cam["position"],
                "rotation_c2w": cam["rotation"],
            }
        )
    return cameras


def trajectory_points(cameras: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    positions = np.array([cam["position"] for cam in cameras])

    stubs = []
    for cam in cameras:
        position = np.array(cam["position"])
        forward = np.array(cam["rotation_c2w"]) @ np.array([0.0, 0.0, 1.0])
        t = np.linspace(0.0, 1.0, STUB_SAMPLES)[:, None]
        stubs.append(position[None, :] * (1 - t) + (position + STUB_LENGTH * forward)[None, :] * t)
    forward_stubs = np.concatenate(stubs, axis=0)

    return positions, forward_stubs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cameras_json", type=Path, help="Path to cameras.json")
    parser.add_argument("scene_preview_ply", type=Path, help="Scene point cloud to overlay the trajectory on")
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    cameras = load_poses(args.cameras_json)

    poses_path = args.output_dir / "poses.json"
    poses_path.write_text(json.dumps({"num_cameras": len(cameras), "cameras": cameras}, indent=2))

    positions, forward_stubs = trajectory_points(cameras)

    scene = o3d.io.read_point_cloud(str(args.scene_preview_ply))
    scene_points = np.asarray(scene.points)
    scene_colors = np.asarray(scene.colors)

    overlay_points = np.concatenate([scene_points, positions, forward_stubs], axis=0)
    overlay_colors = np.concatenate(
        [
            scene_colors,
            np.tile(POSITION_COLOR, (len(positions), 1)),
            np.tile(FORWARD_COLOR, (len(forward_stubs), 1)),
        ],
        axis=0,
    )
    overlay = o3d.geometry.PointCloud()
    overlay.points = o3d.utility.Vector3dVector(overlay_points)
    overlay.colors = o3d.utility.Vector3dVector(overlay_colors)
    overlay_path = args.output_dir / "trajectory_overlay.ply"
    o3d.io.write_point_cloud(str(overlay_path), overlay)

    print(f"Loaded {len(cameras)} camera poses")
    print(f"Wrote {poses_path}")
    print(f"Wrote {overlay_path}")


if __name__ == "__main__":
    main()
