from omegaconf import DictConfig
import torch
from torch import nn
from torchmetrics.image import (
    PeakSignalNoiseRatio,
    LearnedPerceptualImagePatchSimilarity,
    StructuralSimilarityIndexMeasure
)

from coach_pl.configuration import configurable
from coach_pl.model.metric import METRIC_REGISTRY

__all__ = ["NeuralRadianceFieldMetric"]


@METRIC_REGISTRY.register()
class NeuralRadianceFieldMetric(nn.Module):
    """
    Compute the metrics for Neural Radiance Field (NeRF) model, which includes:

        1. Peak Signal-to-Noise Ratio (PSNR)
        2. Learned Perceptual Image Patch Similarity (LPIPS)
        3. Structural Similarity Index Measure (SSIM)

    It is only used for evaluating a whole image.
    """

    @configurable
    def __init__(self) -> None:
        super().__init__()

        self.psnr = PeakSignalNoiseRatio()
        self.lpips = LearnedPerceptualImagePatchSimilarity()
        self.ssim = StructuralSimilarityIndexMeasure()

    @classmethod
    def from_config(cls, cfg: DictConfig) -> dict:
        return {}

    def forward(self, predicted: dict[str, torch.Tensor], target: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        predicted_color, target_color = predicted["color"].permute(0, 3, 1, 2), target["color"].permute(0, 3, 1, 2)
        return {
            "psnr": self.psnr(predicted_color, target_color),
            "lpips": self.lpips(predicted_color, target_color),
            "ssim": self.ssim(predicted_color, target_color)
        }
