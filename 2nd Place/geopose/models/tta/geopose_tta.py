from typing import List, Dict, Callable
import torch.nn.functional as F
import torch
from pytorch_toolbelt.datasets import INPUT_IMAGE_KEY
from pytorch_toolbelt.inference.ensembling import Ensembler
from pytorch_toolbelt.inference.tta import GeneralizedTTA, _deaugment_averaging
from torch import nn

from geopose.dataset import (
    OUTPUT_VFLOW_DIRECTION,
    OUTPUT_VFLOW_ANGLE,
    tensor_vector2angle,
)

__all__ = ["GeoposeTTAModel", "GeoposeMultiscaleTTAModel", "GeoPoseEnsembler"]


class GeoposeTTAModel(GeneralizedTTA):
    def forward(self, **kwargs):
        augmented_inputs = dict((key, augment(kwargs[key])) for (key, augment) in self.augment_fn.items())
        num_inputs = augmented_inputs[INPUT_IMAGE_KEY].size(0)
        outputs = []
        for i in range(num_inputs):
            outputs_i = self.model(**dict((key, value[i : i + 1]) for (key, value) in augmented_inputs.items()))
            outputs.append(outputs_i)

        deaugmented_outputs = {}
        for key, deaugment_fn in self.deaugment_fn.items():
            key_outputs = torch.cat([outputs_i[key] for outputs_i in outputs], dim=0)
            # Ensure that angle orientation is always normalized
            if key == OUTPUT_VFLOW_DIRECTION:
                key_outputs = F.normalize(key_outputs, dim=1)  # [N,B,C]

            output = deaugment_fn(key_outputs)
            deaugmented_outputs[key] = output

        deaugmented_outputs[OUTPUT_VFLOW_ANGLE] = tensor_vector2angle(deaugmented_outputs[OUTPUT_VFLOW_DIRECTION])
        return deaugmented_outputs


class GeoposeMultiscaleTTAModel(nn.Module):
    def __init__(
        self, model: nn.Module, size_offsets: List[int], augment_fn: Dict[str, Callable], deaugment_fn: Dict[str, Callable]
    ):
        super().__init__()
        self.model = model
        self.size_offsets = size_offsets
        self.num_scales = len(size_offsets)
        self.augment_fn = augment_fn
        self.deaugment_fn = deaugment_fn

    def forward(self, **kwargs):
        augmented_inputs = dict(
            (key, augment(kwargs[key], size_offsets=self.size_offsets)) for (key, augment) in self.augment_fn.items()
        )
        outputs = []
        for i in range(self.num_scales):
            inputs_i = dict((key, items[i]) for (key, items) in augmented_inputs.items())
            outputs.append(self.model(**inputs_i))

        result = {}
        for key, deaugment_fn in self.deaugment_fn.items():
            ms_outputs = [output_i[key] for output_i in outputs]
            deaugmented_outputs = deaugment_fn(ms_outputs, size_offsets=self.size_offsets)
            result[key] = deaugmented_outputs

        result[OUTPUT_VFLOW_ANGLE] = tensor_vector2angle(result[OUTPUT_VFLOW_DIRECTION])
        return result


class GeoPoseEnsembler(Ensembler):
    def forward(self, *input, **kwargs):  # skipcq: PYL-W0221
        # for m in self.models:
        #     m.cuda()
        outputs = [model(*input, **kwargs) for model in self.models]
        # for m in self.models:
        #     m.cpu()
        if self.outputs:
            keys = self.outputs
        else:
            keys = outputs[0].keys()

        averaged_output = {}
        for key in keys:
            predictions = [output[key] for output in outputs]
            predictions = torch.stack(predictions)

            # Ensure that angle orientation is always normalized
            if key == OUTPUT_VFLOW_DIRECTION:
                predictions = F.normalize(predictions, dim=2)  # [N,B,C]

            predictions = _deaugment_averaging(predictions, self.reduction)
            averaged_output[key] = predictions

        averaged_output[OUTPUT_VFLOW_ANGLE] = tensor_vector2angle(averaged_output[OUTPUT_VFLOW_DIRECTION])
        return averaged_output
