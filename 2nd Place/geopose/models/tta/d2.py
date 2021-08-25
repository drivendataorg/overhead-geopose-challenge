from functools import partial

import torch
import torch.nn.functional as F
from pytorch_toolbelt.datasets import INPUT_IMAGE_KEY
from pytorch_toolbelt.inference.tta import (
    _deaugment_averaging,
    d2_image_augment,
    d2_labels_augment,
    d2_image_deaugment,
    d2_labels_deaugment,
)
from torch import Tensor

from geopose.dataset import (
    INPUT_GROUND_SAMPLE_DISTANCE,
    OUTPUT_VFLOW_SCALE,
    OUTPUT_VFLOW_DIRECTION,
    OUTPUT_AGL_MASK,
)
from .geopose_tta import GeoposeTTAModel


def geopose_model_d2_tta(model, agl_reduction="mean"):
    return GeoposeTTAModel(
        model,
        augment_fn={INPUT_IMAGE_KEY: d2_image_augment, INPUT_GROUND_SAMPLE_DISTANCE: d2_labels_augment},
        deaugment_fn={
            OUTPUT_VFLOW_SCALE: d2_labels_deaugment,
            OUTPUT_VFLOW_DIRECTION: d2_direction_deaugment,
            OUTPUT_AGL_MASK: partial(d2_image_deaugment, reduction=agl_reduction),
        },
    )


def d2_direction_deaugment(direction: Tensor, reduction="mean") -> Tensor:
    if direction.size(0) % 4 != 0:
        raise RuntimeError("Batch size must be divisible by 4")

    b1, b2, b3, b4 = torch.chunk(direction, 4)

    rot180 = torch.tensor([[-1, -1]], dtype=direction.dtype, device=direction.device)
    fliplr = torch.tensor([[-1, +1]], dtype=direction.dtype, device=direction.device)
    flipud = torch.tensor([[-1, +1]], dtype=direction.dtype, device=direction.device)
    x: Tensor = torch.stack(
        [F.normalize(b1), F.normalize(b2 * rot180), F.normalize(b3 * fliplr), F.normalize(b4 * flipud),]
    )
    return _deaugment_averaging(x, reduction=reduction)
