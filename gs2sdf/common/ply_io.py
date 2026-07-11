"""Shared helpers for reading trained 2DGS ply files.

opacity is stored pre-sigmoid (raw logit); DC-only color is an
approximation (ignores higher SH bands) used purely for visual sanity
checks, not radiance-accurate reconstruction. See notes/ply_format.md.
"""

import numpy as np

SH_C0 = 0.28209479177387814


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def dc_to_rgb(f_dc: np.ndarray) -> np.ndarray:
    return np.clip(0.5 + SH_C0 * f_dc, 0.0, 1.0)


def load_xyz_opacity_color(vertex) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """vertex: a plyfile PlyElement or a structured numpy array with the
    same named-field access (x,y,z,opacity,f_dc_0..2)."""
    xyz = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float64)
    opacity = sigmoid(np.asarray(vertex["opacity"], dtype=np.float64))
    color = dc_to_rgb(
        np.stack([vertex["f_dc_0"], vertex["f_dc_1"], vertex["f_dc_2"]], axis=1)
    )
    return xyz, opacity, color
