"""Build extended pose set: interpolate poses between adjacent real ones.

Input: poses.json (output of load_camera_poses.py)
Output:
  <output_dir>/extended_poses.json — original poses plus interpolated
                                      poses inserted between each
                                      adjacent pair (SLERP rotation,
                                      linear position interpolation),
                                      each tagged is_interpolated; for
                                      interpolated poses, flank_ids
                                      records the two real poses they're
                                      between (Step 8 needs these for
                                      reprojection consistency checks)
  <output_dir>/extended_trajectory_overlay.ply — real poses (red) vs.
                                      interpolated poses (yellow)
                                      overlaid on the scene, to confirm
                                      interpolated ones hug the real
                                      trajectory rather than cutting
                                      corners

Interpolated poses stay strictly between adjacent real poses — never
extrapolating beyond the real trajectory.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation, Slerp

REAL_COLOR = (1.0, 0.0, 0.0)
INTERPOLATED_COLOR = (1.0, 1.0, 0.0)


def interpolate_pose(pose_a: dict, pose_b: dict, t: float) -> dict:
    pos_a, pos_b = np.array(pose_a["position"]), np.array(pose_b["position"])
    position = (1 - t) * pos_a + t * pos_b

    rotations = Rotation.from_matrix([pose_a["rotation_c2w"], pose_b["rotation_c2w"]])
    slerp = Slerp([0.0, 1.0], rotations)
    rotation_c2w = slerp([t]).as_matrix()[0]

    return {
        "img_name": f"interp_{pose_a['id']}_{pose_b['id']}_t{t:.3f}",
        "width": pose_a["width"],
        "height": pose_a["height"],
        "fx": (1 - t) * pose_a["fx"] + t * pose_b["fx"],
        "fy": (1 - t) * pose_a["fy"] + t * pose_b["fy"],
        "cx": (1 - t) * pose_a["cx"] + t * pose_b["cx"],
        "cy": (1 - t) * pose_a["cy"] + t * pose_b["cy"],
        "position": position.tolist(),
        "rotation_c2w": rotation_c2w.tolist(),
        "is_interpolated": True,
        "interp_t": t,
        "flank_ids": [pose_a["id"], pose_b["id"]],
    }


def build_extended_poses(cameras: list[dict], num_interpolated: int) -> list[dict]:
    extended = []
    for i, cam in enumerate(cameras):
        real = dict(cam)
        real["is_interpolated"] = False
        extended.append(real)
        if i + 1 < len(cameras):
            for k in range(1, num_interpolated + 1):
                t = k / (num_interpolated + 1)
                extended.append(interpolate_pose(cam, cameras[i + 1], t))
    return extended


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("poses_json", type=Path, help="Path to poses.json")
    parser.add_argument("scene_preview_ply", type=Path, help="Scene point cloud to overlay the trajectory on")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--num-interpolated",
        type=int,
        required=True,
        help="Interpolated poses to insert between each adjacent pair of real poses (scene-dependent)",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    cameras = json.loads(args.poses_json.read_text())["cameras"]
    extended = build_extended_poses(cameras, args.num_interpolated)

    extended_path = args.output_dir / "extended_poses.json"
    extended_path.write_text(
        json.dumps(
            {
                "num_real": len(cameras),
                "num_interpolated": len(extended) - len(cameras),
                "poses": extended,
            },
            indent=2,
        )
    )

    real_positions = np.array([p["position"] for p in extended if not p["is_interpolated"]])
    interp_positions = np.array([p["position"] for p in extended if p["is_interpolated"]])

    scene = o3d.io.read_point_cloud(str(args.scene_preview_ply))
    scene_points = np.asarray(scene.points)
    scene_colors = np.asarray(scene.colors)

    overlay_points = np.concatenate([scene_points, real_positions, interp_positions], axis=0)
    overlay_colors = np.concatenate(
        [
            scene_colors,
            np.tile(REAL_COLOR, (len(real_positions), 1)),
            np.tile(INTERPOLATED_COLOR, (len(interp_positions), 1)),
        ],
        axis=0,
    )
    overlay = o3d.geometry.PointCloud()
    overlay.points = o3d.utility.Vector3dVector(overlay_points)
    overlay.colors = o3d.utility.Vector3dVector(overlay_colors)
    overlay_path = args.output_dir / "extended_trajectory_overlay.ply"
    o3d.io.write_point_cloud(str(overlay_path), overlay)

    print(f"Real poses: {len(real_positions)}, interpolated: {len(interp_positions)}")
    print(f"Wrote {extended_path}")
    print(f"Wrote {overlay_path}")


if __name__ == "__main__":
    main()
