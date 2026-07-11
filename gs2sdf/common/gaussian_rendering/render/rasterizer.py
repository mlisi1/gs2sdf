"""Wraps diff-surfel-rasterization directly: SH evaluation + the CUDA
tile-sort/alpha-composite kernel + extraction of depth from its extra-
channels output buffer.

The channel layout of the rasterizer's second output tensor (7 channels) is
fixed by the CUDA kernel itself (cuda_rasterizer/auxiliary.h): DEPTH_OFFSET=0
(accumulated depth*weight), ALPHA_OFFSET=1, NORMAL_OFFSET=2..4,
MIDDEPTH_OFFSET=5, DISTORTION_OFFSET=6. Only depth (accumulated depth /
alpha, i.e. the expected depth) is extracted here.

Vendored from gs_sensors/gs_sensor_core/render/rasterizer.py. LOD proxy
support (octree_lod) is kept (harmless if unused) since it's cheap and
already wired through the constructor, but gs2sdf never enables it.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import torch

from gs2sdf.common.gaussian_rendering.culling import (
    Octree,
    visible_leaf_mask_torch,
    visible_point_mask_exact_torch,
    visible_point_mask_screen_size_torch,
)
from gs2sdf.common.gaussian_rendering.models.gaussian_model import GaussianModel
from gs2sdf.common.gaussian_rendering.render.camera import RenderCamera
from gs2sdf.common.gaussian_rendering.sh_utils import C0, eval_sh

_DEPTH_OFFSET = 0
_ALPHA_OFFSET = 1
_XYZ, _OPACITY, _SCALING, _ROTATION, _FEATURES_DC, _FEATURES_REST = range(6)


@dataclass
class RenderOutput:
    rgb: torch.Tensor       # [3, H, W], float, ~[0, 1]
    depth: torch.Tensor     # [H, W], float, GS-training-space units
    num_rendered: int       # splats actually passed to the rasterizer this frame
    timings: dict[str, float] | None = None  # stage -> ms, only populated when profile=True


class GaussianRasterizerWrapper:
    """One instance per loaded model -- the model is loaded once, this is
    reused every frame with a new camera. `octree`/`culling_enabled` are
    optional: with no octree, every splat is rendered every frame.

    PRECONDITION when an octree is supplied: `model` must already be
    permuted into that octree's leaf-contiguous order via
    `model.reorder_(octree.flat_indices)` -- the caller's job."""

    def __init__(self, model: GaussianModel, device: str = "cuda",
                 octree: Octree | None = None, culling_enabled: bool = True,
                 culling_narrow_phase: bool = False, culling_margin: float = 0.0,
                 screen_size_culling: bool = False, screen_size_min_pixels: float = 1.0,
                 octree_lod: bool = False, lod_leaf_pixel_threshold: float = 16.0):
        from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer
        self._settings_cls = GaussianRasterizationSettings
        self._rasterizer_cls = GaussianRasterizer

        self.model = model
        self.device = device
        self.octree = octree
        self.culling_enabled = culling_enabled
        self.culling_narrow_phase = culling_narrow_phase
        self.culling_margin = culling_margin
        self.screen_size_culling = screen_size_culling
        self.screen_size_min_pixels = screen_size_min_pixels
        self.octree_lod = octree_lod
        self.lod_leaf_pixel_threshold = lod_leaf_pixel_threshold
        self.background = torch.zeros(3, dtype=torch.float32, device=device)
        self.last_visible_count = model.num_points

        self._has_octree = culling_enabled and octree is not None
        self._node_aabbs_gpu = None
        self._node_offsets_gpu = None
        if self._has_octree:
            self._node_aabbs_gpu = torch.from_numpy(octree.node_aabbs).to(device)
            self._node_offsets_gpu = torch.from_numpy(octree.node_offsets).to(device)

        self._proxy_xyz_gpu = None
        self._proxy_scale_gpu = None
        self._proxy_rotation_gpu = None
        self._proxy_opacity_gpu = None
        self._proxy_features_dc_gpu = None
        self._leaf_center_gpu = None
        self._leaf_radius_gpu = None
        if self._has_octree and octree_lod and octree.has_lod:
            self._proxy_xyz_gpu = torch.from_numpy(octree.proxy_xyz).to(device)
            self._proxy_scale_gpu = torch.from_numpy(octree.proxy_scale).to(device)
            self._proxy_rotation_gpu = torch.from_numpy(octree.proxy_rotation).to(device)
            self._proxy_opacity_gpu = torch.from_numpy(octree.proxy_opacity).to(device)
            self._proxy_features_dc_gpu = torch.from_numpy(octree.proxy_features_dc).to(device)
            self._leaf_center_gpu = (self._node_aabbs_gpu[:, :3] + self._node_aabbs_gpu[:, 3:]) * 0.5
            self._leaf_radius_gpu = (
                (self._node_aabbs_gpu[:, 3:] - self._node_aabbs_gpu[:, :3]) * 0.5
            ).amax(dim=-1, keepdim=True)

    def _gather_leaf_slices(self, leaf_mask: torch.Tensor):
        """Builds one index tensor covering every True leaf's contiguous
        point range in the model's own (reorder_-permuted) order, and does
        a single `tensor[index]` gather per raw field -- touches only the
        visible K points, never the full N."""
        visible = torch.nonzero(leaf_mask, as_tuple=True)[0]
        if visible.numel() == 0:
            z = lambda t: t.new_zeros((0,) + t.shape[1:])
            return (z(self.model.xyz), z(self.model.raw_opacity), z(self.model.raw_scaling),
                    z(self.model.raw_rotation), z(self.model.features_dc), z(self.model.features_rest))
        starts = self._node_offsets_gpu[visible]
        ends = self._node_offsets_gpu[visible + 1]
        lengths = ends - starts
        total = int(lengths.sum().item())
        idx = torch.repeat_interleave(starts, lengths) + (
            torch.arange(total, device=starts.device)
            - torch.repeat_interleave(torch.cumsum(lengths, 0) - lengths, lengths)
        )
        return (self.model.xyz[idx], self.model.raw_opacity[idx], self.model.raw_scaling[idx],
                self.model.raw_rotation[idx], self.model.features_dc[idx], self.model.features_rest[idx])

    def _lod_split(self, leaf_vis: torch.Tensor | None, camera: RenderCamera):
        """Splits frustum-visible leaves into (leaf_fine, leaf_coarse)
        based on each leaf's own projected screen size. Returns (leaf_vis,
        None) unchanged if LOD isn't active this frame -- always the case
        for gs2sdf, which never enables octree_lod."""
        if leaf_vis is None or not (self.octree_lod and self._proxy_xyz_gpu is not None):
            return leaf_vis, None
        focal_x = camera.width / (2.0 * math.tan(camera.fov_x * 0.5))
        focal_y = camera.height / (2.0 * math.tan(camera.fov_y * 0.5))
        leaf_full_detail = visible_point_mask_screen_size_torch(
            self._leaf_center_gpu, self._leaf_radius_gpu, camera.world_view_transform,
            focal_x, focal_y, cutoff=1.0, min_pixel_radius=self.lod_leaf_pixel_threshold,
        )
        leaf_coarse = leaf_vis & ~leaf_full_detail
        leaf_fine = leaf_vis & leaf_full_detail
        return leaf_fine, leaf_coarse

    def _append_proxies(self, leaf_coarse: torch.Tensor | None,
                         means3D: torch.Tensor, opacity: torch.Tensor,
                         scales: torch.Tensor, rotations: torch.Tensor):
        if leaf_coarse is None:
            return means3D, opacity, scales, rotations, None
        proxy_idx = torch.nonzero(leaf_coarse, as_tuple=True)[0]
        if proxy_idx.numel() == 0:
            return means3D, opacity, scales, rotations, proxy_idx
        means3D = torch.cat([means3D, self._proxy_xyz_gpu[proxy_idx]], dim=0)
        opacity = torch.cat([opacity, self._proxy_opacity_gpu[proxy_idx]], dim=0)
        scales = torch.cat([scales, self._proxy_scale_gpu[proxy_idx]], dim=0)
        rotations = torch.cat([rotations, self._proxy_rotation_gpu[proxy_idx]], dim=0)
        return means3D, opacity, scales, rotations, proxy_idx

    def _compute_colors(self, means3D: torch.Tensor, shs: torch.Tensor, camera: RenderCamera) -> torch.Tensor:
        degree = self.model.active_sh_degree
        if degree > 0:
            dirs = means3D - camera.camera_center
            dirs = dirs / (dirs.norm(dim=1, keepdim=True) + 1e-8)
            sh_dim = (degree + 1) ** 2
            colors = eval_sh(degree, shs.transpose(1, 2)[:, :, :sh_dim], dirs)
            return torch.clamp_min(colors + 0.5, 0.0)
        return torch.clamp_min(C0 * shs[:, 0, :] + 0.5, 0.0)

    def _colors_with_proxies(self, means3D_full: torch.Tensor, shs: torch.Tensor,
                              camera: RenderCamera, proxy_idx: torch.Tensor | None) -> torch.Tensor:
        colors = self._compute_colors(means3D_full, shs, camera)
        if proxy_idx is not None and proxy_idx.numel() > 0:
            proxy_colors = torch.clamp_min(C0 * self._proxy_features_dc_gpu[proxy_idx, 0, :] + 0.5, 0.0)
            colors = torch.cat([colors, proxy_colors], dim=0)
        return colors

    def _sync(self) -> None:
        if self.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()

    def render(self, camera: RenderCamera, profile: bool = False) -> RenderOutput:
        timings: dict[str, float] | None = {} if profile else None
        t = time.perf_counter()

        def lap(name: str) -> None:
            nonlocal t
            if timings is None:
                return
            self._sync()
            now = time.perf_counter()
            timings[name] = (now - t) * 1000.0
            t = now

        model = self.model

        leaf_vis = None
        if self._has_octree:
            leaf_vis = visible_leaf_mask_torch(self._node_aabbs_gpu, camera.full_proj_transform)
        lap("cull")

        leaf_fine, leaf_coarse = self._lod_split(leaf_vis, camera)
        lap("lod_select")

        raw = (
            self._gather_leaf_slices(leaf_fine) if leaf_fine is not None else
            (model.xyz, model.raw_opacity, model.raw_scaling,
             model.raw_rotation, model.features_dc, model.features_rest)
        )

        def filter_raw(keep: torch.Tensor) -> None:
            nonlocal raw
            raw = tuple(field[keep] for field in raw)

        if self.culling_narrow_phase and leaf_fine is not None:
            keep = visible_point_mask_exact_torch(
                raw[_XYZ].float(), camera.full_proj_transform, margin=self.culling_margin)
            filter_raw(keep)
        lap("narrow_cull")

        if self.screen_size_culling and leaf_fine is not None:
            focal_x = camera.width / (2.0 * math.tan(camera.fov_x * 0.5))
            focal_y = camera.height / (2.0 * math.tan(camera.fov_y * 0.5))
            keep = visible_point_mask_screen_size_torch(
                raw[_XYZ].float(), torch.exp(raw[_SCALING].float()), camera.world_view_transform,
                focal_x, focal_y, min_pixel_radius=self.screen_size_min_pixels,
            )
            filter_raw(keep)
        lap("screen_size_cull")

        means3D, opacity, scales, rotations, shs = model._activate(*raw)
        n_full = means3D.shape[0]
        means3D, opacity, scales, rotations, proxy_idx = self._append_proxies(
            leaf_coarse, means3D, opacity, scales, rotations)
        lap("gather")

        self.last_visible_count = int(means3D.shape[0])
        means2D = torch.zeros_like(means3D)
        colors = self._colors_with_proxies(means3D[:n_full], shs, camera, proxy_idx)
        lap("sh_eval")

        raster_settings = self._settings_cls(
            image_height=int(camera.height),
            image_width=int(camera.width),
            tanfovx=math.tan(camera.fov_x * 0.5),
            tanfovy=math.tan(camera.fov_y * 0.5),
            bg=self.background,
            scale_modifier=1.0,
            viewmatrix=camera.world_view_transform,
            projmatrix=camera.full_proj_transform,
            sh_degree=model.active_sh_degree,
            campos=camera.camera_center,
            prefiltered=False,
            debug=False,
        )
        rasterizer = self._rasterizer_cls(raster_settings=raster_settings)

        rendered_image, _radii, allmap = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=None,
            colors_precomp=colors,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=None,
        )
        lap("rasterize")

        alpha = allmap[_ALPHA_OFFSET:_ALPHA_OFFSET + 1]
        depth = torch.nan_to_num(
            allmap[_DEPTH_OFFSET:_DEPTH_OFFSET + 1] / alpha, nan=0.0, posinf=0.0, neginf=0.0
        )
        lap("depth_extract")
        return RenderOutput(
            rgb=rendered_image, depth=depth.squeeze(0),
            num_rendered=self.last_visible_count, timings=timings,
        )
