"""Narrow public interface: pose in, RGB (+ depth) out.

Vendored from gs_sensors/gs_sensor_core/render/pipeline.py.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from gs2sdf.common.gaussian_rendering.camera_profiles.schema import CameraProfile
from gs2sdf.common.gaussian_rendering.culling import Octree
from gs2sdf.common.gaussian_rendering.frames import Pose
from gs2sdf.common.gaussian_rendering.models.gaussian_model import GaussianModel
from gs2sdf.common.gaussian_rendering.render.camera import build_camera
from gs2sdf.common.gaussian_rendering.render.rasterizer import GaussianRasterizerWrapper


@dataclass
class RenderResult:
    rgb: np.ndarray             # (H, W, 3) uint8
    depth: np.ndarray | None    # (H, W) float32, metric meters
    num_rendered: int           # splats actually rendered this frame (post-culling)
    timings: dict[str, float] | None = None  # stage -> ms, only populated when profile=True


class CameraRasterizer:
    """One instance per simulated camera. Holds the loaded model + this
    camera's intrinsics profile; `render()` is the per-frame entry point."""

    def __init__(self, model: GaussianModel, profile: CameraProfile,
                 gs_scale: float = 1.0, publish_depth: bool = True, device: str = "cuda",
                 octree: Octree | None = None, culling_enabled: bool = True,
                 culling_narrow_phase: bool = False, culling_margin: float = 0.0,
                 screen_size_culling: bool = False, screen_size_min_pixels: float = 1.0,
                 octree_lod: bool = False, lod_leaf_pixel_threshold: float = 16.0):
        self.profile = profile
        self.gs_scale = gs_scale
        self.publish_depth = publish_depth
        self.device = device
        self._rasterizer = GaussianRasterizerWrapper(
            model, device=device, octree=octree, culling_enabled=culling_enabled,
            culling_narrow_phase=culling_narrow_phase,
            culling_margin=culling_margin, screen_size_culling=screen_size_culling,
            screen_size_min_pixels=screen_size_min_pixels, octree_lod=octree_lod,
            lod_leaf_pixel_threshold=lod_leaf_pixel_threshold)

    def render(self, pose_gs: Pose, profile: bool = False) -> RenderResult:
        """`pose_gs` must already be in GS-training space (see frames.py) and
        in the optical-frame axis convention."""
        import torch

        camera = build_camera(pose_gs, self.profile, device=self.device)
        with torch.no_grad():
            output = self._rasterizer.render(camera, profile=profile)
            t0 = time.perf_counter()
            rgb = (output.rgb.clamp(0., 1.)
                   .permute(1, 2, 0).mul(255).byte().cpu().numpy())
            depth = None
            if self.publish_depth:
                depth = (output.depth / self.gs_scale).cpu().numpy().astype(np.float32)
            if output.timings is not None:
                output.timings["copy_to_cpu"] = (time.perf_counter() - t0) * 1000.0
        return RenderResult(rgb=rgb, depth=depth, num_rendered=output.num_rendered, timings=output.timings)
