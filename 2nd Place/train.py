import sys
import traceback
import warnings
from typing import Optional, Tuple, List

import catalyst
import torch
from catalyst.callbacks import EarlyStoppingCallback
from catalyst.core import Callback
from omegaconf import OmegaConf, DictConfig
from pytorch_toolbelt.datasets.wrappers import RandomSubsetDataset
from pytorch_toolbelt.utils.catalyst import (
    ShowPolarBatchesCallback,
    BestMetricCheckpointCallback,
    HyperParametersCallback,
)
from torch import nn
from torch.utils.data import Dataset, Sampler

from geopose import *


def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
    log = file if hasattr(file, "write") else sys.stderr
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message, category, filename, lineno, line))


warnings.showwarning = warn_with_traceback


class GeoposePipeline(Pipeline):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.dataset = OverheadGeoposeDataModule.load_dataset(cfg.dataset.data_dir)

    def build_datasets(self, config) -> Tuple[Dataset, Dataset, Optional[Sampler], List[Callback]]:
        train_df, valid_df = OverheadGeoposeDataModule.get_train_valid_split(
            self.dataset, fold_split=config.dataset.fold_split, fold=config.dataset.fold
        )

        train_ds, train_sampler = OverheadGeoposeDataModule.get_training_dataset(
            train_df,
            augmentations=config.train.augmentations,
            random_sized_crop=config.train.augmentations not in {"safe", "none", "d4_only"},
            train_image_size=config.train.train_image_size,
            downsample=config.train.downsample,
        )
        valid_ds = OverheadGeoposeDataModule.get_validation_dataset(
            valid_df, valid_image_size=config.train.valid_image_size, downsample=config.train.downsample
        )

        if config.train.fast:
            train_ds = RandomSubsetDataset(train_ds, 64)
            valid_ds = RandomSubsetDataset(valid_ds, 64)
        self.master_print("Train", "samples", len(train_df), "dataset", len(train_ds))
        self.master_print("Valid", "samples", len(valid_df), "dataset", len(valid_ds))
        self.master_print("  Downsample", config.train.downsample)
        return train_ds, valid_ds, train_sampler, []

    def build_metrics(self, config):
        show_batches = self.cfg["train"].get("show", False)

        callbacks = [
            R2MetricCallback(),
        ]

        if config["train"]["early_stopping"] > 0:
            callbacks.append(
                EarlyStoppingCallback(
                    metric="metrics/mean_r2",
                    minimize=False,
                    min_delta=1e-3,
                    patience=config["train"]["early_stopping"],
                )
            )

        if self.is_master:
            callbacks += [
                BestMetricCheckpointCallback(target_metric="loss", target_metric_minimize=True, save_n_best=3),
                BestMetricCheckpointCallback(target_metric="metrics/mean_r2", target_metric_minimize=False, save_n_best=3),
            ]

            if show_batches:
                callbacks.append(ShowPolarBatchesCallback(draw_predictions, metric="loss", minimize=True))

            callbacks.append(
                HyperParametersCallback(
                    hparam_dict=dict(
                        # Model stuff
                        model=str(self.cfg.model.config.slug),
                        optimizer=str(self.cfg.optimizer.name),
                        optimizer_lr=float(self.cfg.optimizer.params.lr),
                        optimizer_wd=float(self.cfg.optimizer.params.weight_decay),
                        optimizer_scheduler=str(self.cfg.scheduler.scheduler_name),
                        dataset=str(self.cfg.dataset.slug),
                        augmentations=str(self.cfg.train.augmentations),
                    )
                ),
            )

        return callbacks

    def get_model(self, config: DictConfig) -> nn.Module:
        model = get_geopose_model(config)
        return model

    def get_checkpoint_data(self):
        return {
            "config": OmegaConf.to_container(self.cfg, resolve=True),
        }


@hydra_dpp_friendly_main(config_path="configs", config_name="segmentation")
def main(config: DictConfig) -> None:
    torch.cuda.empty_cache()
    catalyst.utils.set_global_seed(int(config.seed))
    torch.set_anomaly_enabled(config.detect_anomaly)

    if config.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True

    if config.dataset.data_dir is None:
        raise ValueError("--data-dir must be set")

    GeoposePipeline(config).train()


if __name__ == "__main__":
    main()
