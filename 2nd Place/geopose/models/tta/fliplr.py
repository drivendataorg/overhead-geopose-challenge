from functools import partial

import torch
import torch.nn.functional as F
from pytorch_toolbelt.datasets import INPUT_IMAGE_KEY
from pytorch_toolbelt.inference.tta import (
    _deaugment_averaging,
    fliplr_image_augment,
    fliplr_image_deaugment,
    fliplr_labels_deaugment,
    fliplr_labels_augment,
)
from torch import Tensor

from geopose.dataset import (
    INPUT_GROUND_SAMPLE_DISTANCE,
    OUTPUT_VFLOW_SCALE,
    OUTPUT_VFLOW_DIRECTION,
    OUTPUT_AGL_MASK,
)
from .geopose_tta import GeoposeTTAModel


def geopose_model_fliplr_tta(model, agl_reduction="mean"):
    return GeoposeTTAModel(
        model,
        augment_fn={INPUT_IMAGE_KEY: fliplr_image_augment, INPUT_GROUND_SAMPLE_DISTANCE: fliplr_labels_augment},
        deaugment_fn={
            OUTPUT_VFLOW_SCALE: fliplr_labels_deaugment,
            OUTPUT_VFLOW_DIRECTION: fliplr_direction_deaugment,
            OUTPUT_AGL_MASK: partial(fliplr_image_deaugment, reduction=agl_reduction),
        },
    )


def fliplr_direction_deaugment(direction: Tensor, reduction="mean") -> Tensor:
    """
    Deaugment input tensor (output of the model) assuming the input was D4-augmented image (See d4_augment).
    Args:
        direction: Tensor of [B * 2, 2] shape
        reduction: If True performs averaging of 2 outputs, otherwise - summation.

    Returns:
        Tensor of [B, 2] shape if reduction is not None or "none", otherwise returns de-augmented tensor of
        [2, B, 2] shape

    """
    if direction.size(0) % 2 != 0:
        raise RuntimeError("Batch size must be divisible by 2")

    b1, b2 = torch.chunk(direction, 2)

    fliplr = torch.tensor([[-1, +1]], dtype=direction.dtype, device=direction.device)
    x: Tensor = torch.stack(
        [F.normalize(b1), F.normalize(b2 * fliplr),]
    )
    return _deaugment_averaging(x, reduction=reduction)
