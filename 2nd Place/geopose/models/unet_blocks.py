import torch
from pytorch_toolbelt.modules import DepthwiseSeparableConv2d, ABN
from timm.models.efficientnet_blocks import InvertedResidual
from torch import nn

__all__ = ["ResidualUnetBlock", "InvertedResidual", "DenseNetUnetBlock", "AdditionalEncoderStage", "IRBlock", "DCUnetBlock"]


class ResidualUnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, abn_block=ABN):
        super().__init__()
        self.identity = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        self.conv1 = DepthwiseSeparableConv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False)
        self.abn1 = abn_block(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False)
        self.abn2 = abn_block(out_channels)
        self.conv3 = DepthwiseSeparableConv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False)
        self.abn3 = abn_block(out_channels)

    def forward(self, x):
        residual = self.identity(x)

        x = self.conv1(x)
        x = self.abn1(x)
        x = self.conv2(x)
        x = self.abn2(x)
        x = self.conv3(x)
        x = self.abn3(x)
        return x + residual


class DenseNetUnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, abn_block=ABN):
        super().__init__()
        self.conv1 = ResidualUnetBlock(in_channels, out_channels, abn_block=abn_block)
        self.conv2 = ResidualUnetBlock(in_channels + out_channels, out_channels, abn_block=abn_block)

    def forward(self, x):
        y = self.conv1(x)
        x = self.conv2(torch.cat([x, y], dim=1))
        return x


class AdditionalEncoderStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, act_layer=nn.ReLU):
        super().__init__()
        self.ir_block1 = InvertedResidual(in_channels, out_channels, act_layer=act_layer, stride=2)
        self.ir_block2 = InvertedResidual(out_channels, out_channels, act_layer=act_layer, dilation=2, se_ratio=0.25)

    def forward(self, x):
        x = self.ir_block1(x)
        return self.ir_block2(x)


class IRBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, act_block=nn.ReLU):
        super().__init__()
        self.ir_block1 = InvertedResidual(in_channels, out_channels, act_layer=act_block)
        self.ir_block2 = InvertedResidual(out_channels, out_channels, act_layer=act_block, se_ratio=0.25)

    def forward(self, x):
        x = self.ir_block1(x)
        x = self.ir_block2(x)
        return x


class DCUnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, abn_block=ABN):
        super().__init__()
        self.bottleneck = DepthwiseSeparableConv2d(in_channels, out_channels, kernel_size=3, padding=1)
        from mmcv.ops.deform_conv import DeformConv2dPack

        self.conv1 = DeformConv2dPack(out_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False)
        self.abn1 = abn_block(out_channels)

        self.conv2 = DeformConv2dPack(out_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False)
        self.abn2 = abn_block(out_channels)

    def forward(self, x):
        x = self.bottleneck(x)

        x = self.conv1(x)
        x = self.abn1(x)
        x = self.conv2(x)
        x = self.abn2(x)
        return x
