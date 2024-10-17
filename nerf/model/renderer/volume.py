from typing import Any

from omegaconf import DictConfig
import torch

from coach_pl.configuration import configurable
from nerf.model.renderer.build import NERF_RENDERER_REGISTRY
from nerf.model.renderer.renderer import Renderer

__all__ = ["VolumeRenderer"]


@NERF_RENDERER_REGISTRY.register()
class VolumeRenderer(Renderer):
    @configurable
    def __init__(self,
        bg_color: torch.Tensor,
        render_depth: bool,
        render_normal: bool,
        render_feature: bool,
    ) -> None:
        super().__init__(bg_color)

        self.render_depth = render_depth
        self.render_normal = render_normal
        self.render_feature = render_feature

    @classmethod
    def from_config(cls, cfg: DictConfig, **kwargs) -> dict[str, Any]:
        return {
            "bg_color": torch.Tensor(cfg.MODEL.BG_COLOR),
            "render_depth": cfg.MODEL.RENDERER.RENDER_DEPTH,
            "render_normal": cfg.MODEL.RENDERER.RENDER_NORMAL,
            "render_feature": cfg.MODEL.RENDERER.RENDER_FEATURE
        }

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        t = batch["t"]             # [B, N + 1, 1]
        invalid = batch["invalid"] # [B]
        density = batch["density"] # [B, N, 1]
        color = batch["color"]     # [B, N, 3]
        feature = batch["feature"] # [B, N, D]

        t[invalid] = 0.0
        delta = torch.diff(t, dim=-2) # [B, N, 1]
        transmittance = torch.exp(torch.cumsum(-density * delta, dim=-2)) # [B, N, 1]
        weight = torch.diff(transmittance, dim=-2, prepend=torch.ones_like(delta[:, :1])) # [B, N, 1]
        final_transmittance = transmittance[:, -1] # [B, 1]

        # Color rendering
        final_color = (weight * color).sum(dim=-2) + final_transmittance * self.bg_color # [B, 3]

        # Prepare the output
        output = {
            "transmittance": final_transmittance,
            "color": final_color
        }

        if self.render_depth:
            depth = (weight * t).sum(dim=-2)
            output["depth"] = depth

        if self.render_normal:
            normal = (weight * normal).sum(dim=-2) # [B, 3]
            output["normal"] = normal

        if self.render_feature:
            feature = (weight * feature).sum(dim=-2) # [B, 3]
            output["feature"] = feature

        return output
