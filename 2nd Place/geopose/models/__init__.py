import collections
import os
from typing import Tuple, Dict, Union

import torch
import torch.nn.functional as F
from pytorch_toolbelt.datasets import INPUT_IMAGE_KEY
from torch import nn

__all__ = ["get_geopose_model", "model_from_checkpoint", "wrap_model_with_tta"]

from geopose import INPUT_GROUND_SAMPLE_DISTANCE, OUTPUT_AGL_MASK, OUTPUT_MAGNITUDE_MASK
from geopose.models.tta import (
    geopose_model_flips_tta,
    geopose_model_fliplr_tta,
    geopose_model_multiscale_tta,
    geopose_model_d2_tta,
    geopose_model_d4_tta,
)


def get_geopose_model(model_config):
    from hydra.utils import instantiate

    return instantiate(model_config, _recursive_=False)


class DownsampleUpsampleWrapper(nn.Module):
    def __init__(self, model, downsample):
        super().__init__()
        self.model = model
        self.downsample = downsample

    def forward(self, **kwargs):
        full_image_size = kwargs[INPUT_IMAGE_KEY].size()[2:]
        dst_image_size = full_image_size[0] // self.downsample, full_image_size[1] // self.downsample

        input = {
            INPUT_IMAGE_KEY: F.interpolate(kwargs[INPUT_IMAGE_KEY], size=dst_image_size, mode="bilinear", align_corners=False),
            INPUT_GROUND_SAMPLE_DISTANCE: kwargs[INPUT_GROUND_SAMPLE_DISTANCE],
        }
        output = self.model(**input)
        output[OUTPUT_AGL_MASK] = F.interpolate(
            output[OUTPUT_AGL_MASK], size=full_image_size, mode="bilinear", align_corners=False
        )
        output[OUTPUT_MAGNITUDE_MASK] = F.interpolate(
            output[OUTPUT_MAGNITUDE_MASK], size=full_image_size, mode="bilinear", align_corners=False
        )
        return output


def model_from_checkpoint(checkpoint_config: Union[str, Dict], **kwargs) -> Tuple[nn.Module, Dict]:
    if isinstance(checkpoint_config, collections.Mapping):
        downsample = checkpoint_config.get("downsample", 1)
        if "average_checkpoints" in checkpoint_config:
            checkpoint = average_checkpoints(checkpoint_config["average_checkpoints"])
        else:
            checkpoint_name = checkpoint_config["checkpoint"]
            if os.path.isfile(checkpoint_name):
                checkpoint = torch.load(checkpoint_name, map_location="cpu")
            else:
                checkpoint = torch.hub.load_state_dict_from_url(checkpoint_name)

        model_config = checkpoint["checkpoint_data"]["config"]["model"]
        if downsample != checkpoint["checkpoint_data"]["config"]["train"].get("downsample", 1):
            raise ValueError("Mismatch in downsample parameters in train/inference")
    else:
        checkpoint_name = checkpoint_config

        if os.path.isfile(checkpoint_name):
            checkpoint = torch.load(checkpoint_name, map_location="cpu")
        else:
            checkpoint = torch.hub.load_state_dict_from_url(checkpoint_name)

        train_config = checkpoint["checkpoint_data"]["config"]["train"]
        model_config = checkpoint["checkpoint_data"]["config"]["model"]
        downsample = train_config.get("downsample", 1)

    model_state_dict = checkpoint["model_state_dict"]

    model = get_geopose_model(model_config)
    model.load_state_dict(model_state_dict, strict=False)
    if downsample > 1:
        model = DownsampleUpsampleWrapper(model, downsample)

    return model.eval(), checkpoint


def wrap_model_with_tta(model, tta_mode, size_offsets=(0, -256, +256)):
    if tta_mode == "flips":
        model = geopose_model_flips_tta(model)
    elif tta_mode == "fliplr":
        model = geopose_model_fliplr_tta(model)
    elif tta_mode == "ms":
        model = geopose_model_multiscale_tta(model, size_offsets=size_offsets, agl_reduction="mean")
    elif tta_mode == "d4-ms":
        model = geopose_model_d4_tta(model)
        model = geopose_model_multiscale_tta(model, size_offsets=size_offsets, agl_reduction="mean")
    elif tta_mode == "d2":
        model = geopose_model_d2_tta(model)
    elif tta_mode == "d4":
        model = geopose_model_d4_tta(model)
    elif tta_mode == "flips":
        model = geopose_model_flips_tta(model, agl_reduction="gmean")
    elif tta_mode == "ms-gmean":
        model = geopose_model_multiscale_tta(model, size_offsets=size_offsets, agl_reduction="gmean")
    elif tta_mode == "fliplr-gmean":
        model = geopose_model_fliplr_tta(model, agl_reduction="gmean")
    elif tta_mode == "flips-gmean":
        model = geopose_model_flips_tta(model, agl_reduction="gmean")
    elif tta_mode == "d2-gmean":
        model = geopose_model_d2_tta(model, agl_reduction="gmean")
    elif tta_mode == "d4-gmean":
        model = geopose_model_d4_tta(model, agl_reduction="gmean")
    elif tta_mode is None:
        return None
    else:
        raise KeyError("Unusupported TTA mode '" + tta_mode + "'")

    return model


def average_checkpoints(inputs):
    """Loads checkpoints from inputs and returns a model with averaged weights. Original implementation taken from:
    https://github.com/pytorch/fairseq/blob/a48f235636557b8d3bc4922a6fa90f3a0fa57955/scripts/average_checkpoints.py#L16
    Args:
      inputs (List[str]): An iterable of string paths of checkpoints to load from.
    Returns:
      A dict of string keys mapping to various values. The 'model' key
      from the returned dict should correspond to an OrderedDict mapping
      string parameter names to torch Tensors.
    """
    params_dict = collections.OrderedDict()
    params_keys = None
    new_state = None
    num_models = len(inputs)
    for fpath in inputs:
        with open(fpath, "rb") as f:
            state = torch.load(
                f,
                map_location="cpu",
            )
        # Copies over the settings from the first checkpoint
        if new_state is None:
            new_state = state
        model_params = state["model_state_dict"]
        model_params_keys = list(model_params.keys())
        if params_keys is None:
            params_keys = model_params_keys
        elif params_keys != model_params_keys:
            raise KeyError(
                "For checkpoint {}, expected list of params: {}, " "but found: {}".format(f, params_keys, model_params_keys)
            )
        for k in params_keys:
            p = model_params[k]
            if isinstance(p, torch.HalfTensor):
                p = p.float()
            if k not in params_dict:
                params_dict[k] = p.clone()
                # NOTE: clone() is needed in case of p is a shared parameter
            else:
                params_dict[k] += p
    averaged_params = collections.OrderedDict()
    for k, v in params_dict.items():
        averaged_params[k] = v
        if averaged_params[k].is_floating_point():
            averaged_params[k].div_(num_models)
        else:
            averaged_params[k] //= num_models
    new_state["model_state_dict"] = averaged_params
    return new_state
