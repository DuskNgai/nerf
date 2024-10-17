from abc import ABCMeta, abstractmethod

import torch
import torch.nn as nn

__all__ = ["Renderer"]


class Renderer(nn.Module, metaclass=ABCMeta):
    def __init__(self, bg_color: torch.Tensor) -> None:
        super().__init__()

        self.register_buffer("bg_color", bg_color)

    @abstractmethod
    def forward(self) -> None:
        """
        Compute the output image from the scene attributes.
        """
        raise NotImplementedError
