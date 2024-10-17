from omegaconf import DictConfig
from pytorch_lightning.trainer.states import RunningStage
import torch

from coach_pl.configuration import configurable
from coach_pl.dataset.transform import TRANSFORM_REGISTRY
from nerf.dataset.transform.base import BaseRayGenerator

__all__ = ["OrthographicRayGenerator"]


@TRANSFORM_REGISTRY.register()
class OrthographicRayGenerator(BaseRayGenerator):
    @configurable
    def __init__(self) -> None:
        super().__init__()


    @classmethod
    def from_config(cls, cfg: DictConfig, stage: RunningStage) -> dict:
        return {}

    def __call__(self,
        view_port_x: float,
        view_port_y: float,
        H: int,
        W: int,
        poses: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = self.generate_image_coordinates(H, W)
        x, y = self.image_coordinates_to_screen_coordinates(x, y, H, W)
        xy_hom = self.screen_coordinates_to_view_coordinates(x, y, view_port_x, view_port_y, H, W)

        origin = xy_hom
        direction = torch.nn.functional.normalize(poses[:3, 2]).reshape(1, 1, 3).expand(H, W, -1) # [H, W, 3]

        return origin, direction

    @classmethod
    def screen_coordinates_to_view_coordinates(cls, x: torch.Tensor, y: torch.Tensor, view_port_x: float, view_port_y: float, H: int, W: int) -> torch.Tensor:
        half_H, half_W = H / 2, W / 2
        x = (x - half_W) / half_W * view_port_x
        y = (y - half_H) / half_H * view_port_y
        return torch.stack([x, y, torch.zeros_like(x)], dim=-1)
