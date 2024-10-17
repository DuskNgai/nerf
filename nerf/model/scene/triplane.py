from typing import Any

from omegaconf import DictConfig
import torch
import torch.nn as nn

from coach_pl.configuration import configurable
from model.scene.build import NERF_SCENE_REGISTRY
from model.scene.scene import Scene

__all__ = ["TriplaneScene"]


@NERF_SCENE_REGISTRY.register()
class TriplaneScene(Scene):
    """
    Using 3 planes to represent the scene.
    """

    @configurable
    def __init__(self,
        aabb: torch.Tensor,
        resolution: int,
        feature_dim: int
    ) -> None:
        super().__init__()

        self.triplane = nn.Parameter(torch.zeros(3, feature_dim, resolution, resolution))
        self.register_buffer("aabb", aabb)

    @classmethod
    def from_config(cls, cfg: DictConfig, **kwargs) -> dict[str, Any]:
        return {
            "aabb": torch.Tensor(cfg.MODEL.SCENE.AABB),
            "resolution": cfg.MODEL.SCENE.RESOLUTION,
            "feature_dim": kwargs["feature_dim"],
        }

    def forward(self, batch: dict[str, Any]) -> None:
        ...
