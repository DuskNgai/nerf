from typing import Any

from omegaconf import DictConfig
import torch
import torch.nn as nn
import torch.nn.functional as F

from coach_pl.configuration import configurable
from nerf.model.scene.build import NERF_SCENE_REGISTRY
from nerf.model.scene.scene import Scene
from nerf.model.utils.ray_box import ray_box_intersection

__all__ = ["TriplaneScene"]


@NERF_SCENE_REGISTRY.register()
class TriplaneScene(Scene):
    """
    Using 3 planes to represent the scene.
    """

    @configurable
    def __init__(self,
        aabb_scale: float,
        resolution: int,
        feature_dim: int,
        num_samples_per_ray: int,
    ) -> None:
        super().__init__()

        self.aabb_scale = aabb_scale
        self.feature_dim = feature_dim
        self.num_samples_per_ray = num_samples_per_ray

        self.triplane = nn.Parameter(torch.zeros(3, feature_dim, resolution, resolution))
        self.register_buffer("aabb", torch.Tensor([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]) * self.aabb_scale)

    @classmethod
    def from_config(cls, cfg: DictConfig, **kwargs) -> dict[str, Any]:
        return {
            "aabb_scale": cfg.MODEL.SCENE.AABB_SCALE,
            "resolution": cfg.MODEL.SCENE.RESOLUTION,
            "feature_dim": kwargs["feature_dim"],
            "num_samples_per_ray": cfg.MODEL.SCENE.NUM_SAMPLES_PER_RAY,
        }

    def forward(self, batch: dict[str, Any]) -> None:
        origin = batch["origin"]       # [B, 3]
        direction = batch["direction"] # [B, 3]

        # Ray-box intersection
        aabb_min, aabb_max = self.aabb[0], self.aabb[1] # [3]
        near, far, invalid = ray_box_intersection(origin, direction, aabb_min, aabb_max)

        # Ray marching
        ramp = torch.linspace(0.0, 1.0, self.num_samples_per_ray + 1, device=origin.device).unsqueeze(0) # [1, N + 1]
        t = ((far.unsqueeze(-1) - near.unsqueeze(-1)) * ramp + near.unsqueeze(-1)).unsqueeze(-1) # [B, N + 1, 1]
        xyz = (origin.unsqueeze(-2) + direction.unsqueeze(-2) * t)[:, :-1] # [B, N, 3]

        xyz_normalized = xyz / self.aabb_scale
        xyz_normalized = xyz_normalized.reshape(1, -1, self.num_samples_per_ray, 3) # [1, B, N, 3]

        xy_feature = F.grid_sample(self.triplane[0:1], xyz_normalized[..., :2], align_corners=True, mode="bilinear", padding_mode="zeros")
        yz_feature = F.grid_sample(self.triplane[1:2], xyz_normalized[..., 1:], align_corners=True, mode="bilinear", padding_mode="zeros")
        zx_feature = F.grid_sample(self.triplane[2:3], xyz_normalized[..., [2, 0]], align_corners=True, mode="bilinear", padding_mode="zeros")

        feature = xy_feature + yz_feature + zx_feature
        feature = feature.reshape(self.feature_dim, -1, self.num_samples_per_ray).permute(1, 2, 0) # [B, N, C]

        return {
            "t": t,
            "invalid": invalid,
            "xyz": xyz,
            "direction": direction,
            "feature": feature,
        }
