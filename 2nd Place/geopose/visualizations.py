from typing import Callable, Optional, List, Union, Dict

import cv2
import numpy as np
from pytorch_toolbelt.datasets import INPUT_IMAGE_KEY, INPUT_IMAGE_ID_KEY
from pytorch_toolbelt.utils import hstack_autopad, vstack_header
from pytorch_toolbelt.utils.torch_utils import rgb_image_from_tensor, to_numpy, mask_from_tensor

from .dataset import *

__all__ = ["draw_predictions", "agl_to_pseudocolor"]

from .dataset import TARGET_VFLOW_SCALE, OUTPUT_VFLOW_SCALE


def agl_to_pseudocolor(agl, max_value=None):
    agl_ignore = agl == AGL_IGNORE_VALUE
    if max_value is None:
        max_value = (agl[~agl_ignore]).max()

    print(max_value)
    agl = (255.0 * agl / (max_value + 0.01)).astype(np.uint8)
    agl_rgb = cv2.applyColorMap(agl, cv2.COLORMAP_VIRIDIS)[..., ::-1]
    agl_rgb[agl_ignore] = (255, 0, 0)
    return agl_rgb


def draw_predictions(
    input: Dict,
    output: Dict,
    image_key=INPUT_IMAGE_KEY,
    image_id_key: Optional[str] = INPUT_IMAGE_ID_KEY,
    mean=DATASET_MEAN,
    std=DATASET_STD,
    image_format: Union[str, Callable] = "bgr",
    max_images=None,
) -> List[np.ndarray]:
    images = []
    num_samples = len(input[image_key])
    if max_images is not None:
        num_samples = min(num_samples, max_images)

    for i in range(num_samples):
        image = rgb_image_from_tensor(input[image_key][i], mean, std)
        image_id = input[image_id_key][i]

        if image_format == "bgr":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif image_format == "gray":
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif hasattr(image_format, "__call__"):
            image = image_format(image)

        gt_angle = input[TARGET_VFLOW_ANGLE][i].item()
        gt_scale = input[TARGET_VFLOW_SCALE][i].item() / 100.0

        pred_angle = output[OUTPUT_VFLOW_ANGLE][i].item()
        pred_scale = output[OUTPUT_VFLOW_SCALE][i].item() / 100.0

        gt_agl = to_numpy(mask_from_tensor(input[TARGET_AGL_MASK][i], squeeze_single_channel=True))
        gt_agl_ignored_values = gt_agl == AGL_IGNORE_VALUE
        gt_agl[gt_agl_ignored_values] = 0

        pred_agl = to_numpy(mask_from_tensor(output[OUTPUT_AGL_MASK][i], squeeze_single_channel=True))

        gt_agl_max = gt_agl.max()
        pred_agl_max = pred_agl.max()
        max_agl = max(gt_agl_max, pred_agl_max)

        gt_agl = (255 * gt_agl / (max_agl + 0.01)).astype(np.uint8)
        pred_agl = (255 * pred_agl / (max_agl + 0.01)).astype(np.uint8)

        gt_agl = cv2.applyColorMap(gt_agl, cv2.COLORMAP_VIRIDIS)[..., ::-1]
        gt_agl[gt_agl_ignored_values] = (255, 0, 0)
        pred_agl = cv2.applyColorMap(pred_agl, cv2.COLORMAP_VIRIDIS)[..., ::-1]

        composition = hstack_autopad(
            [
                vstack_header(image, image_id),
                vstack_header(gt_agl, f"angle:{gt_angle:.3f} scale:{gt_scale:.5f} max:{gt_agl_max:.3f}"),
                vstack_header(pred_agl, f"angle:{pred_angle:.3f} scale:{pred_scale:.5f} max: {pred_agl_max:.3f}"),
            ]
        )
        images.append(composition)
    return images
