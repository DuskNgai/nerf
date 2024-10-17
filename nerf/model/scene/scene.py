from abc import ABCMeta, abstractmethod

import torch.nn as nn

__all__ = ["Scene"]


class Scene(nn.Module, metaclass=ABCMeta):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self) -> None:
        """
        Sample points from the scene.
        """
        raise NotImplementedError
