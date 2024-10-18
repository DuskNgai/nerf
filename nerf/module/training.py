from collections import defaultdict
from typing import Any

from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import LRSchedulerTypeUnion
from rich import print
from timm.optim import optim_factory
from timm.scheduler import scheduler_factory
import torch

from coach_pl.configuration import configurable
from coach_pl.model import build_model, build_criterion, build_metric
from coach_pl.module import MODULE_REGISTRY
from nerf.dataset.utils.image_io import to_numpy

__all__ = ["NeuralRadianceFieldTrainingModule"]


@MODULE_REGISTRY.register()
class NeuralRadianceFieldTrainingModule(LightningModule):
    """
    Training module for Neural Radiance Field (NeRF) model, which includes:

        1. Model: Neural Radiance Field (NeRF)
        2. Criterion: Loss function for training
        3. Metric: Metrics for evaluation
    """

    @configurable
    def __init__(self,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        metric: torch.nn.Module,
        cfg: DictConfig,
    ) -> None:
        super().__init__()

        self.model = torch.compile(model) if cfg.MODULE.COMPILE else model
        self.criterion = torch.compile(criterion) if cfg.MODULE.COMPILE else criterion
        self.metric = torch.compile(metric) if cfg.MODULE.COMPILE else metric

        self.monitor = cfg.TRAINER.CHECKPOINT.MONITOR
        self.chunk_size = cfg.MODULE.CHUNK_SIZE
        self.batch_size = cfg.DATALOADER.TRAIN.BATCH_SIZE

        self.base_lr = cfg.MODULE.OPTIMIZER.BASE_LR
        self.optimizer_name = cfg.MODULE.OPTIMIZER.NAME
        self.optimizer_params = {k.lower(): v for k, v in cfg.MODULE.OPTIMIZER.PARAMS.items()}

        self.scheduler_name = cfg.MODULE.SCHEDULER.NAME
        self.scheduler_params = {k.lower(): v for k, v in cfg.MODULE.SCHEDULER.PARAMS.items()}
        self.scheduler_params["num_epochs"] = cfg.TRAINER.MAX_EPOCHS
        self.step_on_epochs = cfg.MODULE.SCHEDULER.STEP_ON_EPOCHS

        self.save_hyperparameters(ignore=["model", "criterion", "metric"])

    @classmethod
    def from_config(cls, cfg: DictConfig) -> dict[str, Any]:
        return {
            "model": build_model(cfg),
            "criterion": build_criterion(cfg),
            "metric": build_metric(cfg),
            "cfg": cfg,
        }

    def configure_optimizers(self) -> Any:
        total_batch_size = self.batch_size * self.trainer.accumulate_grad_batches * self.trainer.world_size
        lr = self.base_lr * total_batch_size / 4096
        print(f"Total training batch size ({total_batch_size}) = single batch size ({self.batch_size}) * accumulate ({self.trainer.accumulate_grad_batches}) * world size ({self.trainer.world_size}), actural learning rate: {lr}")

        optimizer = optim_factory.create_optimizer_v2(
            self.model,
            opt=self.optimizer_name,
            lr=lr,
            **self.optimizer_params            
        )

        scheduler, _ = scheduler_factory.create_scheduler_v2(
            optimizer,
            sched=self.scheduler_name,
            step_on_epochs=self.step_on_epochs,
            updates_per_epoch=int(len(self.trainer.datamodule.train_dataloader()) / (self.trainer.world_size * self.trainer.accumulate_grad_batches)),
            **self.scheduler_params
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch" if self.step_on_epochs else "step",
                "frequency": 1,
            }
        }

    def lr_scheduler_step(self, scheduler: LRSchedulerTypeUnion, metric: Any | None) -> None:
        if self.step_on_epochs:
            scheduler.step(self.current_epoch, metric)
        else:
            scheduler.step_update(self.global_step, metric)

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        output = self.model(batch)
        loss_dict = self.criterion(output, batch)

        self.log_dict({f"train/{k}": v for k, v in loss_dict.items()}, on_step=True, on_epoch=False, sync_dist=True, rank_zero_only=True)
        self.log("train_loss", loss_dict["loss"], prog_bar=True, logger=False, on_step=True, on_epoch=False, sync_dist=True, rank_zero_only=True)
        return loss_dict

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        origin = batch["origin"]
        direction = batch["direction"]
        color = batch["color"]

        B, H, W, C = color.shape

        batch["origin"] = origin.reshape(-1, 3)
        batch["direction"] = direction.reshape(-1, 3)

        output = defaultdict(list)
        for i in range(0, B * H * W, self.chunk_size):
            chunk = {k: v[i:i + self.chunk_size] for k, v in batch.items()}
            chunk_output = self.model(chunk)
            for k, v in chunk_output.items():
                output[k].append(v)

        output = {k: torch.cat(v, dim=0) for k, v in output.items()}
        output["color"] = output["color"].reshape(B, H, W, C).clamp(0.0, 1.0)
        output["transmittance"] = output["transmittance"].reshape(B, H, W, 1)

        loss_dict = self.criterion(output, batch)
        metric_dict = self.metric(output, batch)

        self.log_dict({f"val/{k}": v for k, v in loss_dict.items()}, on_step=False, on_epoch=True, sync_dist=True, rank_zero_only=True)
        self.log_dict({f"val/{k}": v for k, v in metric_dict.items()}, on_step=False, on_epoch=True, sync_dist=True, rank_zero_only=True)
        self.logger.experiment.add_image("val/color", to_numpy(output["color"].squeeze(0)), batch_idx)
        self.logger.experiment.add_image("val/transmittance", to_numpy(output["transmittance"].squeeze(0)), batch_idx)

        self.log(self.monitor, metric_dict[self.monitor], logger=False, on_step=False, on_epoch=True, sync_dist=True, rank_zero_only=True)

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        origin = batch["origin"]
        direction = batch["direction"]
        color = batch["color"]

        B, H, W, C = color.shape

        batch["origin"] = origin.reshape(-1, 3)
        batch["direction"] = direction.reshape(-1, 3)

        output = defaultdict(list)
        for i in range(0, B * H * W, self.chunk_size):
            chunk = {k: v[i:i + self.chunk_size] for k, v in batch.items()}
            chunk_output = self.model(chunk)
            for k, v in chunk_output.items():
                output[k].append(v)

        output = {k: torch.cat(v, dim=0) for k, v in output.items()}
        output["color"] = output["color"].reshape(B, H, W, C).clamp(0.0, 1.0)
        output["transmittance"] = output["transmittance"].reshape(B, H, W, 1)

        loss_dict = self.criterion(output, batch)
        metric_dict = self.metric(output, batch)

        self.log_dict({f"test/{k}": v for k, v in loss_dict.items()}, on_step=False, on_epoch=True, sync_dist=True, rank_zero_only=True)
        self.log_dict({f"test/{k}": v for k, v in metric_dict.items()}, on_step=False, on_epoch=True, sync_dist=True, rank_zero_only=True)
        self.logger.experiment.add_image("test/color", to_numpy(output["color"].squeeze(0)), batch_idx)
        self.logger.experiment.add_image("test/transmittance", to_numpy(output["transmittance"].squeeze(0)), batch_idx)
