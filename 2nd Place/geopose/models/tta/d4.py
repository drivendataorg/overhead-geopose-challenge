from functools import partial

import torch
import torch.nn.functional as F
from pytorch_toolbelt.datasets import INPUT_IMAGE_KEY
from pytorch_toolbelt.inference.tta import (
    _deaugment_averaging,
    d4_image_augment,
    d4_labels_augment,
    d4_image_deaugment,
    d4_labels_deaugment,
)
from torch import Tensor

from geopose.dataset import (
    INPUT_GROUND_SAMPLE_DISTANCE,
    OUTPUT_VFLOW_SCALE,
    OUTPUT_VFLOW_DIRECTION,
    OUTPUT_AGL_MASK,
)
from .geopose_tta import GeoposeTTAModel


def geopose_model_d4_tta(model, agl_reduction="mean"):
    return GeoposeTTAModel(
        model,
        augment_fn={INPUT_IMAGE_KEY: d4_image_augment, INPUT_GROUND_SAMPLE_DISTANCE: d4_labels_augment},
        deaugment_fn={
            OUTPUT_VFLOW_SCALE: d4_labels_deaugment,
            OUTPUT_VFLOW_DIRECTION: d4_direction_deaugment,
            OUTPUT_AGL_MASK: partial(d4_image_deaugment, reduction=agl_reduction),
        },
    )


def d4_direction_deaugment(direction: Tensor, reduction="mean") -> Tensor:
    if direction.size(0) % 8 != 0:
        raise RuntimeError("Batch size must be divisible by 8")

    b1, b2, b3, b4, b5, b6, b7, b8 = torch.chunk(direction, 8)

    rot090 = torch.tensor([[0, -1], [1, 0]], dtype=direction.dtype, device=direction.device)
    rot180 = torch.tensor([[-1, 0], [0, -1]], dtype=direction.dtype, device=direction.device)
    rot270 = torch.tensor([[0, 1], [-1, 0]], dtype=direction.dtype, device=direction.device)

    transpose = torch.tensor([[0, 1], [1, 0]], dtype=direction.dtype, device=direction.device)

    transpose_rot090 = torch.matmul(rot090, transpose)
    transpose_rot180 = torch.matmul(rot180, transpose)
    transpose_rot270 = torch.matmul(rot270, transpose)

    x: Tensor = torch.stack(
        [
            (b1),
            (torch.matmul(b2, rot090)),
            (torch.matmul(b3, rot180)),
            (torch.matmul(b4, rot270)),
            #
            (torch.matmul(b5, transpose)),
            (torch.matmul(b6, transpose_rot090)),
            (torch.matmul(b7, transpose_rot180)),
            (torch.matmul(b8, transpose_rot270)),
        ]
    )
    return _deaugment_averaging(F.normalize(x, dim=2), reduction=reduction)
