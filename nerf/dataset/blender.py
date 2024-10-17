import json
from pathlib import Path
from typing import Any

from omegaconf import DictConfig
from pytorch_lightning.trainer.states import RunningStage
import torch
from torch.utils.data import Dataset
import tqdm

from coach_pl.configuration import configurable
from coach_pl.dataset import DATASET_REGISTRY
from coach_pl.dataset.transform import build_transform
from nerf.dataset.transform import BaseRayGenerator
from nerf.dataset.utils.image_io import load_image

__all__ = ["BlenderDataset"]


@DATASET_REGISTRY.register()
class BlenderDataset(Dataset):
    """
    A dataset for Blender generated images and poses, which contains:

        1. transforms_{stage}.json: metadata file
            - camera_angle_x, camera_angle_y
            - frames: a list of frame metadata
                - file_path: the path to the image file
                - transform_matrix: the camera pose (camera to world matrix)
        2. {stage}/: image files, depth files (optional), normal files (optional) ...

    The dataset should output:

        1. random sampled rays
        2. corresponding colors

    In training, we need to sample rays from all images.
    In validation, we need to sample all rays from one image.

    Args:
        root_dir (str): the root directory of the dataset
        stage (RunningStage): the stage of the training process
    """

    @configurable
    def __init__(self,
        root_dir: Path,
        stage: RunningStage,
        ray_generator: BaseRayGenerator,
        bg_color: torch.Tensor,
    ) -> None:
        super().__init__()

        self.root_dir = root_dir
        self.stage = stage
        self.ray_generator = ray_generator
        self.bg_color = bg_color

        meta = self.load_metadata(stage)
        self.colors, self.origins, self.directions = self.load_data(meta, stage)

    @classmethod
    def from_config(cls, cfg: DictConfig, stage: RunningStage) -> dict[str, Any]:
        return {
            "root_dir": Path(cfg.DATASET.ROOT),
            "stage": stage,
            "ray_generator": build_transform(cfg, stage),
            "bg_color": torch.Tensor(cfg.MODEL.BG_COLOR),
        }

    def load_metadata(self, stage: RunningStage) -> dict[str, Any]:
        meta_path = self.root_dir.joinpath(f"transforms_{stage.dataloader_prefix}.json")
        with meta_path.open("r") as f:
            meta = json.load(f)
        return meta

    def load_data(self, meta: dict[str, Any], stage: RunningStage):
        camera_angle_x = meta["camera_angle_x"]
        camera_angle_y = meta.get("camera_angle_y", camera_angle_x)

        poses, images, origins, directions = [], [], [], []
        for frame in tqdm.tqdm(meta["frames"], desc=f"Loading {stage.dataloader_prefix} images and poses"):

            pose = torch.Tensor(frame["transform_matrix"])
            poses.append(pose)

            image_path = self.root_dir.joinpath(frame["file_path"])
            if image_path.suffix == "":
                image_path = image_path.with_suffix(".png")
            image = load_image(image_path, self.bg_color) # [H, W, 3]
            images.append(image)

        poses = torch.stack(poses) # [N, 4, 4]
        images = torch.stack(images) # [N, H, W, 3]

        N, H, W, C = images.shape
        origins, directions = self.ray_generator(camera_angle_x, camera_angle_y, H, W, poses)

        if self.stage == RunningStage.TRAINING:
            images = images.reshape(-1, 3) # [N * H * W, 3]
            origins = origins.reshape(-1, 3) # [N * H * W, 3]
            directions = directions.reshape(-1, 3) # [N * H * W, 3]

        return images, origins, directions

    def __len__(self) -> int:
        return len(self.origins)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "origin": self.origins[index],
            "direction": self.directions[index],
            "color": self.colors[index]
        }

    @property
    def collate_fn(self) -> None:
        return None

    @property
    def sampler(self):
        return None
