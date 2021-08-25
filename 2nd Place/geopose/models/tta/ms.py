from functools import partial
from typing import List

import torch
import torch.nn.functional as F
from pytorch_toolbelt.datasets import INPUT_IMAGE_KEY
from pytorch_toolbelt.inference.tta import (
    _deaugment_averaging,
    ms_image_augment,
    ms_image_deaugment,
)
from torch import Tensor

from geopose.dataset import (
    INPUT_GROUND_SAMPLE_DISTANCE,
    OUTPUT_VFLOW_SCALE,
    OUTPUT_VFLOW_DIRECTION,
    OUTPUT_AGL_MASK,
)
from .geopose_tta import GeoposeMultiscaleTTAModel


def geopose_model_multiscale_tta(model, size_offsets=(0, -512, +512), agl_reduction="mean"):
    return GeoposeMultiscaleTTAModel(
        model,
        size_offsets=size_offsets,
        augment_fn={INPUT_IMAGE_KEY: ms_image_augment, INPUT_GROUND_SAMPLE_DISTANCE: ms_gsd_augment},
        deaugment_fn={
            OUTPUT_VFLOW_SCALE: ms_vflow_scale_deaugment,
            OUTPUT_VFLOW_DIRECTION: ms_direction_deaugment,
            OUTPUT_AGL_MASK: partial(ms_image_deaugment, reduction=agl_reduction),
        },
    )


def ms_gsd_augment(gsd, size_offsets, size=2048) -> List[Tensor]:
    x = [gsd * (float(size) / float(size + offset)) for offset in size_offsets]
    return x


def ms_vflow_scale_deaugment(scales, size_offsets, size=2048, reduction="mean"):
    x = [scale * float(size + offset) / float(size) for scale, offset in zip(scales, size_offsets)]
    x = torch.stack(x)
    y = _deaugment_averaging(x, reduction=reduction)
    return y


def ms_direction_deaugment(directions, size_offsets, reduction="mean"):
    x = [F.normalize(x) for x in directions]
    x = torch.stack(x)
    return _deaugment_averaging(x, reduction=reduction)
