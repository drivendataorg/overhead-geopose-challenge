import typing

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1,
                 bias=True, padding_mode='zeros'):
        super(DepthwiseConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation,
                                              in_channels, bias, padding_mode)


class SeparableConv2d(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, bias=True, padding_mode='zeros'):
        super(SeparableConv2d, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,
                      bias=False, padding_mode=padding_mode),
            nn.Conv2d(in_channels, out_channels, 1, bias=bias)
        )


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True,
                 bn=True,
                 relu=True,
                 init_fn=None):
        super(ConvBlock, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                      padding, dilation, groups,
                      bias),
            nn.BatchNorm2d(out_channels) if bn else nn.Identity(),
            nn.ReLU(True) if relu else nn.Identity()
        )
        if init_fn:
            self.apply(init_fn)

    @staticmethod
    def same_padding(kernel_size, dilation):
        return dilation * (kernel_size - 1) // 2


class SeparableConvBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1,
                 bias=True,
                 bn=True,
                 relu=True,
                 init_fn=None):
        super(SeparableConvBlock, self).__init__(
            SeparableConv2d(in_channels, out_channels, kernel_size, stride,
                            padding, dilation,
                            bias),
            nn.BatchNorm2d(out_channels) if bn else nn.Identity(),
            nn.ReLU(True) if relu else nn.Identity()
        )
        if init_fn:
            self.apply(init_fn)

    @staticmethod
    def same_padding(kernel_size, dilation):
        return dilation * (kernel_size - 1) // 2


class PoolBlock(nn.Sequential):
    def __init__(self, output_size, in_channels, out_channels):
        super(PoolBlock, self).__init__(
            nn.AdaptiveAvgPool2d(output_size),
            ConvBlock(in_channels, out_channels, 1),
        )

    def forward(self, x: torch.Tensor):
        size = x.shape[-2:]
        for m in self:
            x = m(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ResidualBlock(nn.Sequential):
    def __init__(self, *args):
        super(ResidualBlock, self).__init__(*args)

    def forward(self, x):
        identity = x
        x = super(ResidualBlock, self).forward(x)
        x += identity
        return x


class ChannelReduction(nn.ModuleList):
    def __init__(self, in_channels_list, out_channels):
        super(ChannelReduction, self).__init__(
            [ConvBlock(in_channels, out_channels, 1, bn=True, relu=False) for in_channels in
             in_channels_list])

    def forward(self, features: typing.List[torch.Tensor]):
        return [m(feature) for m, feature in zip(self, features)]


class ChannelConcat(nn.Module):
    def forward(self, features: typing.List[torch.Tensor]):
        assert isinstance(features, (list, tuple))
        if len(features) == 1:
            return features[0]
        return torch.cat(features, dim=1)


class Sum(nn.Module):
    def forward(self, features: typing.List[torch.Tensor]):
        assert isinstance(features, (list, tuple))
        if len(features) == 1:
            return features[0]
        return sum(features)


class ListIndex(nn.Module):
    def __init__(self, *args):
        super(ListIndex, self).__init__()
        self.index = args

    def forward(self, features: typing.List[torch.Tensor]):
        if len(self.index) == 1:
            return features[self.index[0]]
        else:
            return [features[i] for i in self.index]


class ConvUpsampling(nn.Sequential):
    def __init__(self,
                 in_channels,
                 out_channels,
                 scale_factor,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1):
        super(ConvUpsampling, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation),
            nn.UpsamplingBilinear2d(scale_factor=scale_factor)
        )


class Squeeze(nn.Module):
    def __init__(self, dim):
        super(Squeeze, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor):
        return x.squeeze(dim=self.dim)
