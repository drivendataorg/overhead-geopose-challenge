from functools import partial

import torch
import torch.nn.functional as F
from pytorch_toolbelt.datasets import INPUT_IMAGE_KEY
from pytorch_toolbelt.inference.tta import (
    _deaugment_averaging,
    flips_image_augment,
    flips_image_deaugment,
    flips_labels_augment,
    flips_labels_deaugment,
)
from torch import Tensor

from geopose.dataset import (
    INPUT_GROUND_SAMPLE_DISTANCE,
    OUTPUT_VFLOW_SCALE,
    OUTPUT_VFLOW_DIRECTION,
    OUTPUT_AGL_MASK,
)
from .geopose_tta import GeoposeTTAModel


def geopose_model_flips_tta(model, agl_reduction="mean"):
    return GeoposeTTAModel(
        model,
        augment_fn={INPUT_IMAGE_KEY: flips_image_augment, INPUT_GROUND_SAMPLE_DISTANCE: flips_labels_augment},
        deaugment_fn={
            OUTPUT_VFLOW_SCALE: flips_labels_deaugment,
            OUTPUT_VFLOW_DIRECTION: flips_direction_deaugment,
            OUTPUT_AGL_MASK: partial(flips_image_deaugment, reduction=agl_reduction),
        },
    )


def flips_direction_deaugment(direction: Tensor, reduction="mean") -> Tensor:
    if direction.size(0) % 3 != 0:
        raise RuntimeError("Batch size must be divisible by 2")

    b1, b2, b3 = torch.chunk(direction, 3)

    fliplr = torch.tensor([[-1, +1]], dtype=direction.dtype, device=direction.device)
    flipud = torch.tensor([[+1, -1]], dtype=direction.dtype, device=direction.device)
    x: Tensor = torch.stack(
        [F.normalize(b1), F.normalize(b2 * fliplr), F.normalize(b3 * flipud),]
    )
    return _deaugment_averaging(x, reduction=reduction)
