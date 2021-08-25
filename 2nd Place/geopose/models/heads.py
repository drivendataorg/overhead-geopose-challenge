from typing import List

import torch
import torch.nn.functional as F
from pytorch_toolbelt.modules import (
    ACT_RELU,
    instantiate_activation_block,
    GlobalAvgPool2d,
    FPNFuseSum,
    ResidualDeconvolutionUpsample2d,
    conv1x1,
)
from torch import nn

from ..dataset import (
    OUTPUT_VFLOW_DIRECTION,
    tensor_vector2angle,
    OUTPUT_MAGNITUDE_MASK,
    OUTPUT_VFLOW_ANGLE,
    OUTPUT_AGL_MASK,
    OUTPUT_VFLOW_SCALE,
)

__all__ = ["SimpleAGLHead", "AGLHead"]


class RegressionHeadWithGSD(nn.Module):
    def __init__(self, in_channels, embedding_size, out_channels, activation=ACT_RELU):
        super().__init__()
        self.up = nn.Sequential(
            conv1x1(in_channels, (in_channels // 4) * 4), ResidualDeconvolutionUpsample2d((in_channels // 4) * 4)
        )

        self.conv1 = nn.Conv2d(in_channels // 4, embedding_size, kernel_size=3, padding=1)
        self.act1 = instantiate_activation_block(activation, inplace=True)
        self.conv2 = nn.Conv2d(embedding_size + 1, embedding_size, kernel_size=3, padding=1)
        self.act2 = instantiate_activation_block(activation, inplace=True)
        self.conv3 = nn.Conv2d(embedding_size, out_channels, kernel_size=3, padding=1)
        self.act3 = instantiate_activation_block(ACT_RELU, inplace=True)

        for layer in [self.conv1, self.conv2, self.conv3]:
            torch.nn.init.normal_(layer.weight, mean=0.0, std=0.1)
            # torch.nn.init.zeros_(layer.bias)
            torch.nn.init.ones_(layer.bias)

    def forward(self, x, gsd):
        x = self.up(x)
        x = self.act1(self.conv1(x))

        gsd = gsd.reshape(gsd.size(0), 1, 1, 1).expand((-1, -1, x.size(2), x.size(3)))
        x = torch.cat([x, gsd], dim=1)

        x = self.act2(self.conv2(x))
        x = self.act3(self.conv3(x))
        return x


class RegressionHead(nn.Module):
    def __init__(self, in_channels, embedding_size, out_channels, activation=ACT_RELU):
        super().__init__()
        # self.up = nn.Sequential(conv1x1(in_channels, in_channels * 4), nn.PixelShuffle(upscale_factor=2))
        self.up = nn.Sequential(
            conv1x1(in_channels, (in_channels // 4) * 4), ResidualDeconvolutionUpsample2d((in_channels // 4) * 4)
        )

        self.conv1 = nn.Conv2d(in_channels // 4, embedding_size, kernel_size=3, padding=1)
        self.act1 = instantiate_activation_block(activation, inplace=True)
        self.conv2 = nn.Conv2d(embedding_size, embedding_size, kernel_size=3, padding=1)
        self.act2 = instantiate_activation_block(activation, inplace=True)
        self.conv3 = nn.Conv2d(embedding_size, out_channels, kernel_size=3, padding=1)
        self.act3 = instantiate_activation_block(ACT_RELU, inplace=True)

        for layer in [self.conv1, self.conv2, self.conv3]:
            torch.nn.init.normal_(layer.weight, mean=0.0, std=0.1)
            # torch.nn.init.zeros_(layer.bias)
            torch.nn.init.ones_(layer.bias)

    def forward(self, x):
        x = self.up(x)
        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))
        x = self.act3(self.conv3(x))
        return x


class SimpleRegressionHead(nn.Module):
    def __init__(self, in_channels, embedding_size, out_channels, activation=ACT_RELU, dropout_rate=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, embedding_size, kernel_size=3, padding=1)
        self.drop = nn.Dropout2d(dropout_rate, inplace=False)
        self.act1 = instantiate_activation_block(activation, inplace=True)
        self.conv2 = nn.Conv2d(embedding_size, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.drop(x)
        x = self.act1(x)
        x = self.conv2(x)
        return x


class OnlyAGLHead(nn.Module):
    def __init__(
        self, encoder_channels: List[int], decoder_channels: List[int], embedding_size=64, dropout_rate=0.0, activation=ACT_RELU
    ):
        super().__init__()
        in_channels = decoder_channels[0]
        self.dropout = nn.Dropout2d(dropout_rate, inplace=False)
        self.height = RegressionHeadWithGSD(in_channels, embedding_size, 1, activation=activation)

    def forward(self, rgb, encoder_feature_maps, decoder_feature_maps, gsd, orientation_outputs):
        # Take the finest feature map from the decoder
        x = decoder_feature_maps[0]
        x = self.dropout(x)

        height = self.height(x, gsd)

        return {
            OUTPUT_AGL_MASK: F.interpolate(height, size=rgb.size()[2:], mode="bilinear", align_corners=True),
        }


class SimpleAGLHead(nn.Module):
    def __init__(
        self, encoder_channels: List[int], decoder_channels: List[int], embedding_size=64, dropout_rate=0.0, activation=ACT_RELU
    ):
        super().__init__()
        in_channels = decoder_channels[0]
        self.dropout = nn.Dropout2d(dropout_rate, inplace=False)
        self.height = RegressionHeadWithGSD(in_channels, embedding_size, 1, activation=activation)
        self.magnitude = RegressionHead(in_channels, embedding_size, 1, activation=activation)
        self.scale = ScaleHead()

    def forward(self, rgb, encoder_feature_maps, decoder_feature_maps, gsd, orientation_outputs):
        # Take the finest feature map from the decoder
        x = decoder_feature_maps[0]
        x = self.dropout(x)

        height = self.height(x, gsd)
        mag = self.magnitude(x)
        scale = self.scale(mag, height)

        return {
            OUTPUT_AGL_MASK: F.interpolate(height, size=rgb.size()[2:], mode="bilinear", align_corners=True),
            OUTPUT_MAGNITUDE_MASK: F.interpolate(mag, size=rgb.size()[2:], mode="bilinear", align_corners=True),
            OUTPUT_VFLOW_SCALE: scale,
        }


class SquareRoot(nn.Module):
    def forward(self, x):
        return x.sqrt()


class Exponent(nn.Module):
    def forward(self, x):
        return x.exp()


def instantiate_transformation(name):
    if name == "sqrt":
        return SquareRoot()
    elif name == "exp":
        return Exponent()
    elif name == "identity":
        return nn.Identity()
    else:
        raise KeyError(name)


class AGLHead(nn.Module):
    def __init__(
        self,
        encoder_channels: List[int],
        decoder_channels: List[int],
        embedding_size=64,
        dropout_rate=0.0,
        activation=ACT_RELU,
        num_upsample_blocks=1,
        agl_activation=ACT_RELU,
        agl_transformation="identity",
    ):
        super().__init__()
        in_channels = decoder_channels[0]
        self.dropout = nn.Dropout2d(dropout_rate, inplace=False)
        self.scale = ScaleHead()

        upsample_blocks = []
        for i in range(num_upsample_blocks):
            input_channels = (in_channels // 2) * 2
            upsampled_channels = input_channels // 4

            upsample_blocks += [
                conv1x1(in_channels, input_channels),
                ResidualDeconvolutionUpsample2d(input_channels, scale_factor=2),
                nn.Conv2d(upsampled_channels, upsampled_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(upsampled_channels),
                instantiate_activation_block(activation, inplace=True),
            ]
            in_channels = upsampled_channels

        self.upsample = nn.Sequential(*upsample_blocks)

        self.height = nn.Sequential(
            nn.Conv2d(upsampled_channels + 1, upsampled_channels, kernel_size=3, padding=1),
            instantiate_activation_block(activation, inplace=True),
            nn.Conv2d(upsampled_channels, 1, kernel_size=3, padding=1),
            instantiate_activation_block(agl_activation, inplace=True),
            instantiate_transformation(agl_transformation),
        )

        self.magnitude = nn.Sequential(
            nn.Conv2d(upsampled_channels, upsampled_channels, kernel_size=3, padding=1),
            instantiate_activation_block(activation, inplace=True),
            nn.Conv2d(upsampled_channels, 1, kernel_size=3, padding=1),
            instantiate_activation_block(agl_activation, inplace=True),
            instantiate_transformation(agl_transformation),
        )

    def forward(self, rgb, encoder_feature_maps, decoder_feature_maps, gsd, orientation_outputs):
        # Take the finest feature map from the decoder
        x = decoder_feature_maps[0]
        x = self.upsample(x)

        gsd = gsd.reshape(gsd.size(0), 1, 1, 1).expand((-1, -1, x.size(2), x.size(3)))

        height = self.height(torch.cat([x, gsd], dim=1))
        mag = self.magnitude(x)
        scale = self.scale(mag, height)

        return {
            OUTPUT_AGL_MASK: height,
            OUTPUT_MAGNITUDE_MASK: mag,
            OUTPUT_VFLOW_SCALE: scale,
        }


class HyperColumnAGLHead(nn.Module):
    def __init__(
        self, encoder_channels: List[int], decoder_channels: List[int], embedding_size=64, dropout_rate=0.0, activation=ACT_RELU
    ):
        super().__init__()
        self.height = nn.ModuleList(
            [
                SimpleRegressionHead(
                    in_channels=in_channels + 1,
                    out_channels=1,
                    dropout_rate=dropout_rate,
                    embedding_size=embedding_size,
                    activation=activation,
                )
                for in_channels in decoder_channels
            ]
        )
        self.magnitude = nn.ModuleList(
            [
                SimpleRegressionHead(
                    in_channels=in_channels,
                    out_channels=1,
                    dropout_rate=dropout_rate,
                    embedding_size=embedding_size,
                    activation=activation,
                )
                for in_channels in decoder_channels
            ]
        )
        self.fuse = FPNFuseSum(mode="bilinear", align_corners=True)
        self.scale = ScaleHead()

    def forward(self, rgb, encoder_feature_maps, decoder_feature_maps, gsd, orientation_outputs):
        heights = [
            height_layer(
                torch.cat(
                    [feature_map, gsd.reshape(gsd.size(0), 1, 1, 1).expand((-1, -1, feature_map.size(2), feature_map.size(3)))],
                    dim=1,
                )
            )
            for (feature_map, height_layer) in zip(decoder_feature_maps, self.height)
        ]
        magnitude = [mag_layer(feature_map) for (feature_map, mag_layer) in zip(decoder_feature_maps, self.magnitude)]

        height = F.relu(self.fuse(heights), inplace=True)
        mag = F.relu(self.fuse(magnitude), inplace=True)
        scale = self.scale(mag, height)

        return {
            OUTPUT_AGL_MASK: F.interpolate(height, size=rgb.size()[2:], mode="bilinear", align_corners=True),
            OUTPUT_MAGNITUDE_MASK: F.interpolate(mag, size=rgb.size()[2:], mode="bilinear", align_corners=True),
            OUTPUT_VFLOW_SCALE: scale,
        }


class BasicOrientationHead(nn.Module):
    def __init__(self, encoder_channels: List[int], decoder_channels: List[int], dropout_rate=0.0, activation=ACT_RELU):
        super().__init__()
        in_channels = encoder_channels[-1]
        self.pool = GlobalAvgPool2d(flatten=True)

        self.orientation = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=True),
            nn.Linear(in_channels, in_channels),
            instantiate_activation_block(activation, inplace=True),
            nn.Linear(in_channels, 2),
        )

    def forward(self, rgb, encoder_feature_maps, decoder_feature_maps, gsd):
        features = self.pool(encoder_feature_maps[-1])
        direction = self.orientation(features)
        angle = tensor_vector2angle(direction)
        return {OUTPUT_VFLOW_DIRECTION: direction, OUTPUT_VFLOW_ANGLE: angle}


class BasicOrientationScaleHead(nn.Module):
    def __init__(self, encoder_channels: List[int], decoder_channels: List[int], dropout_rate=0.0, activation=ACT_RELU):
        super().__init__()
        in_channels = encoder_channels[-1]
        self.pool = GlobalAvgPool2d(flatten=True)

        self.orientation = nn.Sequential(
            nn.Dropout2d(p=dropout_rate, inplace=False),
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            instantiate_activation_block(activation, inplace=True),
            nn.Conv2d(in_channels, 2, kernel_size=3),
        )

        self.scale = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=False),
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            instantiate_activation_block(activation, inplace=True),
            nn.Conv2d(in_channels, 1, kernel_size=3),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(
                    m.weight, a=0.01, mode="fan_out", nonlinearity="relu" if activation == ACT_RELU else "leaky_relu"
                )
                torch.nn.init.constant_(m.bias, 0.0001)

    def forward(self, rgb, encoder_feature_maps, decoder_feature_maps, gsd):
        features = encoder_feature_maps[-1]
        direction = self.pool(F.normalize(self.orientation(features)))
        scale = self.pool(F.relu(self.scale(features)))
        angle = tensor_vector2angle(direction)
        return {OUTPUT_VFLOW_DIRECTION: direction, OUTPUT_VFLOW_ANGLE: angle, OUTPUT_VFLOW_SCALE: scale}


class ScaleHead(nn.Module):
    def __init__(self):
        super().__init__()

    @torch.cuda.amp.autocast(False)
    def forward(self, mag, height):
        curr_mag = torch.flatten(mag, start_dim=1).float()
        curr_height = torch.flatten(height, start_dim=1).float()
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
        scale = torch.bmm(pinv.view(batch_size, 1, length), curr_mag.view(batch_size, length, 1))
        scale = torch.squeeze(scale, dim=2)
        return scale
