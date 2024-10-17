from abc import ABCMeta, abstractmethod

import torch

__all__ = ["BaseRayGenerator"]


class BaseRayGenerator(metaclass=ABCMeta):
    """
    Generate rays within OpenGL coordinate system.
    """
    def __init__(self) -> None:
        pass

    @abstractmethod
    def __call__(self) -> None:
        """
        Compute the origin and direction of the rays.
        """
        raise NotImplementedError

    @classmethod
    def generate_image_coordinates(cls, H: int, W: int) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = torch.meshgrid(
            torch.arange(W, dtype=torch.float32),
            torch.arange(H, dtype=torch.float32),
            indexing="xy"
        )
        return x, y

    @classmethod
    def image_coordinates_to_screen_coordinates(cls, x: torch.Tensor, y: torch.Tensor, H: int, W: int) -> tuple[torch.Tensor, torch.Tensor]:
        return x, (H - 1) - y

    @classmethod
    @abstractmethod
    def screen_coordinates_to_view_coordinates(cls) -> None:
        raise NotImplementedError
