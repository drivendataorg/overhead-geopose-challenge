from segmentation_models_pytorch.base import initialization as init

import torch.nn as nn
import torch
from segmentation_models_pytorch.base.modules import Flatten, Activation

from typing import Optional, Union, List
from segmentation_models_pytorch.fpn.decoder import FPNDecoder
from segmentation_models_pytorch.encoders import get_encoder


class FPNVFLOW(nn.Module):
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_pyramid_channels: int = 256,
        decoder_segmentation_channels: int = 128,
        decoder_merge_policy: str = "add",
        decoder_dropout: float = 0.2,
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[str] = None,
        upsampling: int = 4,
        aux_params: Optional[dict] = None,
    ):

        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )
        
        self.decoder = FPNDecoder(
            encoder_channels=self.encoder.out_channels,
            encoder_depth=encoder_depth,
            pyramid_channels=decoder_pyramid_channels,
            segmentation_channels=decoder_segmentation_channels,
            dropout=decoder_dropout,
            merge_policy=decoder_merge_policy,
        )

        self.xydir_head = EncoderRegressionHead(
            in_channels=self.encoder.out_channels[-1],
            out_channels=2,
        )

        self.height_head = RegressionHead(
            in_channels=self.decoder.out_channels,
            out_channels=1,
            kernel_size=3,
        )

        self.mag_head = RegressionHead(
            in_channels=self.decoder.out_channels,
            out_channels=1,
            kernel_size=3,
        )

        self.scale_head = ScaleHead()

        self.name = "u-{}".format(encoder_name)
        self.initialize()

    def initialize(self):
        init.initialize_decoder(self.decoder)
        init.initialize_head(self.xydir_head)
        init.initialize_head(self.height_head)
        init.initialize_head(self.mag_head)
        init.initialize_head(self.scale_head)

    def forward(self, x):

        features = self.encoder(x)
        decoder_output = self.decoder(*features)

        xydir = self.xydir_head(features[-1])
        height = self.height_head(decoder_output)
        mag = self.mag_head(decoder_output)
        scale = self.scale_head(mag, height)

        if scale.ndim == 0:
            scale = torch.unsqueeze(scale, axis=0)

        return xydir, height, mag, scale

    def predict(self, x):
        if self.training:
            self.eval()

        with torch.no_grad():
            x = self.forward(x)

        return x


class RegressionHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        conv2d = nn.Conv2d(
            in_channels, out_channels, kernel_size, padding=kernel_size // 2
        )
        identity = torch.nn.UpsamplingBilinear2d(scale_factor=4) # mod: 4x upsampling
        # identity = nn.Identity()
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
        curr_mag = self.flatten(mag, start_dim=1)
        curr_height = self.flatten(height, start_dim=1)
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
