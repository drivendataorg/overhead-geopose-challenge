from typing import List

import torch
from torch import Tensor, nn


def predict_tta(models: List[nn.Module], images: Tensor):
    results = []
    for model in models:
        output = model(images)
        pred = output['xydir'], output['height'], output['mag'], output[
            'scale']
        # hflip
        output = model(torch.flip(images, dims=[3]))
        output['xydir'][:, 0] *= -1
        output['height'] = torch.flip(output['height'], dims=[3])
        output['mag'] = torch.flip(output['mag'], dims=[3])

        predh = output['xydir'], output['height'], output['mag'], output[
            'scale']

        # vflip
        output = model(torch.flip(images, dims=[2]))
        output['xydir'][:, 1] *= -1
        output['height'] = torch.flip(output['height'], dims=[2])
        output['mag'] = torch.flip(output['mag'], dims=[2])

        predv = output['xydir'], output['height'], output['mag'], output[
            'scale']
        # hvflip
        output = model(torch.flip(images, dims=[2, 3]))
        output['xydir'] *= -1
        output['height'] = torch.flip(output['height'], dims=[2, 3])
        output['mag'] = torch.flip(output['mag'], dims=[2, 3])
        predhv = output['xydir'], output['height'], output['mag'], output[
            'scale']

        model_result = []
        allpreds = [pred, predh, predv, predhv]
        for i in range(len(pred)):
            model_result.append(sum([p[i] for p in allpreds]) / len(allpreds))
        results.append(model_result)
    result = []
    for i in range(4):
        result.append(sum([p[i] for p in results]) / len(results))

    return result
