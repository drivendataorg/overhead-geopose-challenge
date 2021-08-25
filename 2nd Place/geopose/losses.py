from typing import Optional, List

import torch.nn.functional
from catalyst.core import Callback, CallbackOrder
from pytorch_toolbelt.utils.catalyst.callbacks.stop_if_nan import _any_is_nan
from torch import nn, Tensor

__all__ = ["LogCoshWithIgnore", "SmoothL1LossWithIgnore", "MSELossWithIgnore", "CosineSimilarityLoss", "HuberLossWithIgnore"]


class LogCoshWithIgnore(nn.Module):
    def __init__(self, ignore_value, fraction: float = 1.0):
        super().__init__()
        self.ignore_value = ignore_value
        self.fraction = fraction

    def forward(self, output, target):
        r = output - target
        log2 = 0.30102999566

        loss = torch.logaddexp(r, -r) - log2
        loss = torch.masked_fill(loss, target.eq(self.ignore_value), 0)

        if self.fraction < 1:
            loss = loss.reshape(loss.size(0), -1)
            M = loss.size(1)
            num_elements_to_keep = int(M * self.fraction)
            loss, _ = torch.topk(loss, k=num_elements_to_keep, dim=1, largest=False, sorted=False)

        # not_finite = torch.isnan(loss) | torch.isinf(loss)
        # if not_finite.any():
        #     inf_losses = to_numpy(loss[not_finite])
        #     preds = to_numpy(output[not_finite])
        #     targets = to_numpy(target[not_finite])
        #     np_loss = np.log(np.cosh(preds - targets))
        #     for p, t, np_l, t_l in zip(preds, targets, np_loss, inf_losses):
        #         print(p, t, np_l, t_l)
        return loss.mean()


class SmoothL1LossWithIgnore(nn.Module):
    def __init__(self, ignore_value: int, fraction: float = 1.0):
        super().__init__()
        self.ignore_value = ignore_value
        self.fraction = fraction

    def forward(self, output, target):
        loss = torch.nn.functional.smooth_l1_loss(output, target, reduction="none")
        loss = torch.masked_fill(loss, target.eq(self.ignore_value), 0)

        if self.fraction < 1:
            loss = loss.reshape(loss.size(0), -1)
            M = loss.size(1)
            num_elements_to_keep = int(M * self.fraction)
            loss, _ = torch.topk(loss, k=num_elements_to_keep, dim=1, largest=False, sorted=False)

        return loss.mean()


class HuberLossWithIgnore(nn.Module):
    def __init__(self, ignore_value: int, delta: float = 1, fraction: float = 1.0):
        super().__init__()
        self.ignore_value = ignore_value
        self.delta = delta
        self.fraction = fraction

    def forward(self, output, target):
        loss = torch.nn.functional.huber_loss(output, target, delta=self.delta, reduction="none")
        loss: Tensor = torch.masked_fill(loss, target.eq(self.ignore_value), 0)

        if self.fraction < 1:
            loss = loss.reshape(loss.size(0), -1)
            M = loss.size(1)
            num_elements_to_keep = int(M * self.fraction)
            loss, _ = torch.topk(loss, k=num_elements_to_keep, dim=1, largest=False, sorted=False)

        return loss.mean()


class MSELossWithIgnore(nn.Module):
    def __init__(self, ignore_value: int, fraction: float = 1.0):
        super().__init__()
        self.ignore_value = ignore_value
        self.fraction = fraction

    def forward(self, output, target):
        loss = torch.nn.functional.mse_loss(output, target, reduction="none")
        loss = torch.masked_fill(loss, target.eq(self.ignore_value), 0)

        if self.fraction < 1:
            loss = loss.reshape(loss.size(0), -1)
            M = loss.size(1)
            num_elements_to_keep = int(M * self.fraction)
            loss, _ = torch.topk(loss, k=num_elements_to_keep, dim=1, largest=False, sorted=False)

        return loss.mean()


class CosineSimilarityLoss(nn.Module):
    def __init__(self, gamma=1):
        super().__init__()
        self.gamma = gamma

    def forward(self, output, target):
        loss = 1.0 - torch.clamp(torch.nn.functional.cosine_similarity(output, target, dim=1, eps=1e-3), -1, +1)
        if self.gamma != 1:
            loss = torch.pow(loss, self.gamma)
        return loss.mean()


class WarnIfNanCallback(Callback):
    """
    Stop training process if NaN observed in batch_metrics
    """

    def __init__(self, metrics_to_monitor: Optional[List[str]] = None):
        super().__init__(CallbackOrder.Metric + 1)
        self.metrics_to_monitor = metrics_to_monitor

    def on_batch_end(self, runner):
        if self.metrics_to_monitor is not None:
            keys = self.metrics_to_monitor
        else:
            keys = runner.batch_metrics.keys()

        for key in keys:
            if _any_is_nan(runner.batch_metrics[key]):

                print(
                    f"Stopping training due to NaN presence in {key} metric at epoch {runner.global_epoch}."
                    f"batch_metrics={{{runner.batch_metrics}}}"
                )
                runner.need_early_stop = True
