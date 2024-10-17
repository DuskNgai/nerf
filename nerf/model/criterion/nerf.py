from typing import Any
from omegaconf import DictConfig
import torch
from torch import nn

from coach_pl.configuration import configurable
from coach_pl.model.criterion import CRITERION_REGISTRY

__all__ = ["NeuralRadianceFieldCriterion"]


@CRITERION_REGISTRY.register()
class NeuralRadianceFieldCriterion(nn.Module):
    @configurable
    def __init__(self) -> None:
        super().__init__()
        self.mse_loss = nn.MSELoss()

    @classmethod
    def from_config(cls, cfg: DictConfig) -> dict[str, Any]:
        return {}

    def forward(self, predicted: dict[str, torch.Tensor], target: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        loss_dict = {}
        loss_dict["color_loss"] = self.mse_loss(predicted["color"], target["color"])
        loss_dict["loss"] = sum(loss_dict.values())
        return loss_dict
