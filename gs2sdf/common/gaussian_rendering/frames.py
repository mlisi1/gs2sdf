"""GS-training-frame <-> world transform (load + invert).

A Sim(3) transform (rotation + uniform scale + translation) that maps a
world-frame pose into the frame the Gaussian-Splat model was trained in.
This project's own pipeline operates entirely within the GS-training
frame throughout (see notes/), so GSFrameTransform.identity() /
gs_scale=1.0 defaults apply — this class is vendored for completeness
and to keep Pose's shape identical to gs_sensor_core's, not because
gs2sdf currently performs a non-identity world<->GS transform anywhere.

Vendored from gs_sensors/gs_sensor_core/frames.py.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from gs2sdf.common.gaussian_rendering.rotations import quat_mul, quat_normalize, quat_to_rotmat, rotmat_to_quat


@dataclass(frozen=True)
class Pose:
    """Quaternion order (x, y, z, w), matching geometry_msgs/Quaternion."""
    position: np.ndarray     # (3,)
    orientation: np.ndarray  # (4,)


@dataclass(frozen=True)
class GSFrameTransform:
    """gs_pose = scale * (rotation @ world_pose + translation); orientation is
    rotated by `rotation` only (never scaled)."""
    rotation: np.ndarray     # (3, 3), orthonormal
    translation: np.ndarray  # (3,)
    scale: float

    @staticmethod
    def identity() -> "GSFrameTransform":
        return GSFrameTransform(np.eye(3), np.zeros(3), 1.0)

    def apply(self, pose: Pose) -> Pose:
        position = self.scale * (self.rotation @ pose.position + self.translation)
        rot_quat = rotmat_to_quat(self.rotation)
        orientation = quat_normalize(quat_mul(rot_quat, pose.orientation))
        return Pose(position=position, orientation=orientation)

    def inverse(self) -> "GSFrameTransform":
        r_inv = self.rotation.T
        s_inv = 1.0 / self.scale
        t_inv = -self.scale * (r_inv @ self.translation)
        return GSFrameTransform(r_inv, t_inv, s_inv)

    def to_dict(self) -> dict:
        return {
            "rotation": self.rotation.tolist(),
            "translation": self.translation.tolist(),
            "scale": self.scale,
        }

    @staticmethod
    def from_dict(data: dict) -> "GSFrameTransform":
        return GSFrameTransform(
            rotation=np.asarray(data["rotation"], dtype=np.float64),
            translation=np.asarray(data["translation"], dtype=np.float64),
            scale=float(data["scale"]),
        )


def save_gs_frame_transform(path: str | Path, transform: GSFrameTransform) -> None:
    with open(path, "w") as f:
        json.dump({"gs_T_world": transform.to_dict()}, f, indent=2)


def load_gs_frame_transform(path: str | Path) -> GSFrameTransform:
    with open(path) as f:
        data = json.load(f)
    return GSFrameTransform.from_dict(data["gs_T_world"])
