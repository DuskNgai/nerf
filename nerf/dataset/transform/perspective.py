import math

from omegaconf import DictConfig
from pytorch_lightning.trainer.states import RunningStage
import torch

from coach_pl.configuration import configurable
from coach_pl.dataset.transform import TRANSFORM_REGISTRY
from nerf.dataset.transform.base import BaseRayGenerator

__all__ = ["PerspectiveRayGenerator"]


@TRANSFORM_REGISTRY.register()
class PerspectiveRayGenerator(BaseRayGenerator):
    @configurable
    def __init__(self) -> None:
        super().__init__()

    @classmethod
    def from_config(cls, cfg: DictConfig, stage: RunningStage) -> dict:
        return {}

    def __call__(self,
        camera_angle_x: float,
        camera_angle_y: float,
        H: int,
        W: int,
        poses: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = self.generate_image_coordinates(H, W)
        x, y = self.image_coordinates_to_screen_coordinates(x, y, H, W)
        xy_hom = self.screen_coordinates_to_view_coordinates(x, y, camera_angle_x, camera_angle_y, H, W)

        origin = poses[:, :3, 3].reshape(-1, 1, 1, 3).expand(-1, H, W, -1)
        direction = torch.nn.functional.normalize(
            torch.einsum("bij, hwi -> bhwj", poses[:, :3, :3].transpose(-2, -1), xy_hom)
        , p=2, dim=-1)

        return origin, direction

    @classmethod
    def screen_coordinates_to_view_coordinates(cls, x: torch.Tensor, y: torch.Tensor, camera_angle_x: float, camera_angle_y: float, H: int, W: int) -> torch.Tensor:
        half_H, half_W = H / 2, W / 2
        x = (x - half_W) / (half_W / math.tan(camera_angle_x / 2))
        y = (y - half_H) / (half_H / math.tan(camera_angle_y / 2))
        # OpenGL looks down the negative z-axis, so we negate z.
        return torch.stack([x, y, -torch.ones_like(x)], dim=-1)
