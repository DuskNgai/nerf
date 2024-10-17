from typing import Any

from omegaconf import DictConfig
import torch
import torch.nn as nn

from coach_pl.configuration import configurable
from coach_pl.model import MODEL_REGISTRY
from nerf.model.renderer import build_renderer, Renderer
from nerf.model.scene import build_scene, Scene
from nerf.model.utils import SphericalHarmonics

__all__ = ["NeuralRadianceField"]


@MODEL_REGISTRY.register()
class NeuralRadianceField(nn.Module):
    """
    The Neural Radiance Field. A model that contains the following components:

        1. Scene: Computes the attributes of the scene.
        2. Renderer: Renders the attributes into the output images.
    """
    @configurable
    def __init__(self,
        scene: Scene,
        renderer: Renderer,
        spherical_harmonics: SphericalHarmonics
    ) -> None:
        super().__init__()

        self.scene = scene
        self.renderer = renderer
        self.spherical_harmonics = spherical_harmonics

    @classmethod
    def from_config(cls, cfg: DictConfig) -> dict[str, Any]:
        sh_degrees = cfg.MODEL.SH_DEGREES
        spherical_harmonics = SphericalHarmonics(sh_degrees)
        feature_dim = 1 + 3 * len(spherical_harmonics)

        return {
            "scene": build_scene(cfg, feature_dim=feature_dim),
            "renderer": build_renderer(cfg),
            "spherical_harmonics": spherical_harmonics,
        }

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        scene_inputs = self.prepare_scene_inputs(batch)
        scene_outputs = self.scene(scene_inputs)

        renderer_inputs = self.prepare_renderer_inputs(scene_outputs)
        renderer_outputs = self.renderer(renderer_inputs)
        return renderer_outputs

    def prepare_scene_inputs(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return batch

    def prepare_renderer_inputs(self, scene_outputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        direction = scene_outputs["direction"] # [B, 3]
        invalid = scene_outputs["invalid"] # [B]
        feature = scene_outputs["feature"] # [B, N, C]

        (
            raw_density,
            sh_coeff,
            other
        ) = feature.split([1, 3 * len(self.spherical_harmonics), feature.shape[-1] - 3 * len(self.spherical_harmonics) - 1], dim=-1)

        density = torch.exp(raw_density[..., 0:1])
        sh_coeff = sh_coeff.reshape(*feature.shape[:-1], 3, len(self.spherical_harmonics)) # [B, N, 3, sh ** 2]
        color = self.spherical_harmonics(sh_coeff, direction.unsqueeze(1))

        return {
            "t": scene_outputs["t"],
            "invalid": invalid,
            "xyz": scene_outputs["xyz"],
            "density": density,
            "color": color,
            "feature": other,
        }
