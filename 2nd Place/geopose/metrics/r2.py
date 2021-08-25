import math
from collections import defaultdict

import numpy as np
import torch
from catalyst.dl import IRunner, Callback, CallbackOrder
from pytorch_toolbelt.datasets import INPUT_IMAGE_ID_KEY
from pytorch_toolbelt.utils.distributed import all_gather

__all__ = [
    "R2MetricCallback",
]

from torch import Tensor

from geopose import (
    INPUT_PATCH_CITY_NAME,
    AGL_IGNORE_VALUE,
    compute_vflow_torch,
    TARGET_VFLOW_ANGLE,
    OUTPUT_VFLOW_SCALE,
    TARGET_VFLOW_SCALE,
    OUTPUT_VFLOW_ANGLE,
    OUTPUT_AGL_MASK,
    TARGET_AGL_MASK,
)


def scaling_error(gt, pred):
    return float(torch.abs(gt - pred) / gt)


def angular_error(gt, pred):
    gt_v = np.array([torch.sin(gt).item(), torch.cos(gt).item()])
    pred_v = np.array([torch.sin(pred).item(), torch.cos(pred).item()])
    dot = np.clip(gt_v.dot(pred_v), -1, 1)
    return float(np.rad2deg(math.acos(dot)))


def height_error(gt, pred, mask):
    mask = mask > 0
    if mask.any():
        return float(torch.sum(torch.abs((gt - pred) * mask)) / torch.count_nonzero(mask))
    else:
        return float("nan")


class R2MetricCallback(Callback):
    def __init__(
        self, prefix="metrics/mean_r2", agl_prefix="metrics/agl_r2", vflow_prefix="metrics/vflow_r2",
    ):
        super().__init__(CallbackOrder.Metric)
        self.prefix = prefix
        self.agl_prefix = agl_prefix
        self.vflow_prefix = vflow_prefix
        self.agl_scores = None
        self.vflow_scores = None

    def on_loader_start(self, state: IRunner):
        self.agl_scores = dict()
        self.vflow_scores = dict()

    @torch.no_grad()
    def on_batch_end(self, state: IRunner):

        average_angle_error = []
        average_scale_error = []
        average_agl_error = []

        for city_name, image_id, gt_height, gt_scale, gt_angle, pred_height, pred_scale, pred_angle, in zip(
            state.input[INPUT_PATCH_CITY_NAME],
            state.input[INPUT_IMAGE_ID_KEY],
            state.input[TARGET_AGL_MASK].detach().float(),
            state.input[TARGET_VFLOW_SCALE].detach().float(),
            state.input[TARGET_VFLOW_ANGLE].detach().float(),
            state.output[OUTPUT_AGL_MASK].detach().float(),
            state.output[OUTPUT_VFLOW_SCALE].detach().float(),
            state.output[OUTPUT_VFLOW_ANGLE].detach().float(),
        ):
            ignore_agl_mask: Tensor = gt_height.eq(AGL_IGNORE_VALUE)
            valid_agl_mask: Tensor = (~ignore_agl_mask).type_as(gt_height)

            # Convert height to centimeters
            gt_height = gt_height * 100
            pred_height = pred_height * 100

            masked_errors = (pred_height - gt_height) * valid_agl_mask
            masked_gt_agl = gt_height * valid_agl_mask

            if city_name not in self.agl_scores:
                self.agl_scores[city_name] = dict(
                    num_examples=torch.tensor(0.0, dtype=torch.float64),
                    sum_of_errors=torch.tensor(0.0, dtype=torch.float64),
                    y_sq_sum=torch.tensor(0.0, dtype=torch.float64),
                    y_sum=torch.tensor(0.0, dtype=torch.float64),
                )

            self.agl_scores[city_name]["num_examples"] += valid_agl_mask.sum().cpu()
            self.agl_scores[city_name]["sum_of_errors"] += torch.sum(masked_errors ** 2).cpu()
            self.agl_scores[city_name]["y_sum"] += torch.sum(masked_gt_agl).cpu()
            self.agl_scores[city_name]["y_sq_sum"] += torch.sum(masked_gt_agl ** 2).cpu()

            if city_name not in self.vflow_scores:
                self.vflow_scores[city_name] = dict(
                    num_examples=torch.tensor(0.0, dtype=torch.float64),
                    sum_of_errors=torch.tensor(0.0, dtype=torch.float64),
                    y_sq_sum=torch.tensor(0.0, dtype=torch.float64),
                    y_sum=torch.tensor(0.0, dtype=torch.float64),
                )

            gt_scale = gt_scale / 100.0  # Convert to pixels/cm
            pred_scale = pred_scale / 100.0  # Convert to pixels/cm

            gt_vflow, _, _, _ = compute_vflow_torch(gt_height, gt_scale, gt_angle)
            pred_vflow, _, _, _ = compute_vflow_torch(pred_height, pred_scale, pred_angle)

            masked_vflow_errors = (pred_vflow - gt_vflow) * valid_agl_mask.to(gt_vflow.device)
            masked_gt_vflow = gt_vflow * valid_agl_mask.to(gt_vflow.device)

            self.vflow_scores[city_name]["num_examples"] += valid_agl_mask.sum().cpu() * 2
            self.vflow_scores[city_name]["sum_of_errors"] += torch.sum(masked_vflow_errors ** 2).cpu()
            self.vflow_scores[city_name]["y_sum"] += torch.sum(masked_gt_vflow).cpu()
            self.vflow_scores[city_name]["y_sq_sum"] += torch.sum(masked_gt_vflow ** 2).cpu()

            average_scale_error.append(scaling_error(gt_scale, pred_scale))
            average_angle_error.append(angular_error(gt_angle, pred_angle))
            average_agl_error.append(height_error(gt_height, pred_height, valid_agl_mask))

        state.batch_metrics["details/average_scale_error"] = np.mean(average_scale_error)
        state.batch_metrics["details/average_angle_error"] = np.mean(average_angle_error)
        state.batch_metrics["details/average_agl_error"] = np.nanmean(average_agl_error)

    def gather_scores(self, r2_scores):
        gathered_scores = all_gather(r2_scores)
        all_scores_per_location = defaultdict(
            lambda: dict(
                num_examples=torch.tensor(0.0, dtype=torch.float64),
                sum_of_errors=torch.tensor(0.0, dtype=torch.float64),
                y_sq_sum=torch.tensor(0.0, dtype=torch.float64),
                y_sum=torch.tensor(0.0, dtype=torch.float64),
            )
        )

        for scores_per_location in gathered_scores:
            for location, accumulators in scores_per_location.items():
                all_scores_per_location[location]["num_examples"] += accumulators["num_examples"]
                all_scores_per_location[location]["sum_of_errors"] += accumulators["sum_of_errors"]
                all_scores_per_location[location]["y_sum"] += accumulators["y_sum"]
                all_scores_per_location[location]["y_sq_sum"] += accumulators["y_sq_sum"]

        r2_scores = {}
        for location, accumulators in all_scores_per_location.items():
            sum_of_errors = accumulators["sum_of_errors"]
            num_examples = accumulators["num_examples"]
            y_sum = accumulators["y_sum"]
            y_sq_sum = accumulators["y_sq_sum"]
            r2 = 1 - sum_of_errors / (y_sq_sum - (y_sum ** 2) / num_examples)
            r2_scores[location] = float(r2)

        return r2_scores

    @torch.no_grad()
    def on_loader_end(self, state: IRunner):
        r2_agl_per_city = self.gather_scores(self.agl_scores)
        r2_vflow_per_city = self.gather_scores(self.vflow_scores)

        agl_r2 = np.mean(list(r2_agl_per_city.values()))
        state.loader_metrics[self.agl_prefix] = agl_r2

        for city_name, r2 in r2_agl_per_city.items():
            state.loader_metrics[self.agl_prefix + "_" + city_name] = float(r2)

        flow_r2 = np.mean(list(r2_vflow_per_city.values()))
        state.loader_metrics[self.vflow_prefix] = flow_r2

        for city_name, r2 in r2_vflow_per_city.items():
            state.loader_metrics[self.vflow_prefix + "_" + city_name] = float(r2)

        state.loader_metrics[self.prefix] = np.mean([agl_r2, flow_r2])
