from segmentation_models_pytorch.base import initialization as init

import torch.nn as nn
import torch
from segmentation_models_pytorch.base.modules import Flatten, Activation

from typing import Optional, Union, List
from segmentation_models_pytorch.unet.decoder import UnetDecoder
from segmentation_models_pytorch.encoders import get_encoder




class RegressionHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        conv2d = nn.Conv2d(
            in_channels, out_channels, kernel_size, padding=kernel_size // 2
        )
        identity = nn.Identity()
        activation = Activation(None)
        super().__init__(conv2d, identity, activation)


class EncoderRegressionHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        pool = nn.AdaptiveAvgPool2d(1)
        flatten = Flatten()
        dropout = nn.Dropout(p=0.5, inplace=True)
        linear = nn.Linear(in_channels, 2, bias=True)
        super().__init__(pool, flatten, dropout, linear)


class ScaleHead(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.flatten = torch.flatten
        self.dot = torch.dot

    def forward(self, mag, height):
        with torch.cuda.amp.autocast(enabled=False):
            curr_mag = self.flatten(mag, start_dim=1).float()
            curr_height = self.flatten(height, start_dim=1).float()
            batch_size = curr_mag.shape[0]
            length = curr_mag.shape[1]
            denom = (
                torch.squeeze(
                    torch.bmm(
                        curr_height.view(batch_size, 1, length),
                        curr_height.view(batch_size, length, 1),
                    )
                )
                + 0.01
            )
            pinv = curr_height / denom.view(batch_size, 1)
            scale = torch.squeeze(
                torch.bmm(
                    pinv.view(batch_size, 1, length), curr_mag.view(batch_size, length, 1)
                )
            )
        return scale
