import os

import timm
import torch.hub
from torch.nn import Dropout2d, UpsamplingBilinear2d, AdaptiveAvgPool2d
from torch.utils import model_zoo

from utilities.unet_vflow import EncoderRegressionHead, RegressionHead, ScaleHead

encoder_params = {
    "tf_efficientnetv2_l_in21k": {
        'last_upsample': 64,
        "decoder_filters": [64, 128, 192, 256],
        'url': None,
    },
}

default_decoder_filters = [48, 96, 176, 256]
default_last = 48

import torch
from torch import nn
import torch.nn.functional as F


class BasicConvAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, dilation=1, activation=nn.ReLU, bias=True):
        super().__init__()
        padding = int((kernel_size - 1) / 2) * dilation
        self.op = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation,
                            bias=bias)
        self.use_act = activation is not None
        if self.use_act:
            self.act = activation()

    def forward(self, x):
        x = self.op(x)
        if self.use_act:
            x = self.act(x)
        return x


class Conv1x1(BasicConvAct):
    def __init__(self, in_channels, out_channels, dilation=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size=1, dilation=dilation, activation=None, bias=bias)


class Conv3x3(BasicConvAct):
    def __init__(self, in_channels, out_channels, dilation=1):
        super().__init__(in_channels, out_channels, kernel_size=3, dilation=dilation, activation=None)


class ConvReLu1x1(BasicConvAct):
    def __init__(self, in_channels, out_channels, dilation=1):
        super().__init__(in_channels, out_channels, kernel_size=1, dilation=dilation, activation=nn.ReLU)


class ConvReLu3x3(BasicConvAct):
    def __init__(self, in_channels, out_channels, dilation=1):
        super().__init__(in_channels, out_channels, kernel_size=3, dilation=dilation, activation=nn.ReLU)


class BasicUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, activation=nn.ReLU, mode='nearest'):
        super().__init__()
        padding = int((kernel_size - 1) / 2) * 1
        self.op = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=1)
        self.use_act = activation is not None
        self.mode = mode
        if self.use_act:
            self.act = activation()

    def forward(self, x):
        x = F.upsample(x, scale_factor=2, mode=self.mode)
        x = self.op(x)
        if self.use_act:
            x = self.act(x)
        return x


class AbstractModel(nn.Module):
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                m.weight.data = nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def initialize_encoder(self, model, model_url, num_channels_changed=False):
        if os.path.isfile(model_url):
            pretrained_dict = torch.load(model_url)
        else:
            pretrained_dict = model_zoo.load_url(model_url)
        if 'state_dict' in pretrained_dict:
            pretrained_dict = pretrained_dict['state_dict']
            pretrained_dict = {k.replace("module.", ""): v for k, v in pretrained_dict.items()}
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        if num_channels_changed:
            model.state_dict()[self.first_layer_params_names[0] + '.weight'][:, :3, ...] = pretrained_dict[
                self.first_layer_params_names[0] + '.weight'].data
            skip_layers = self.first_layer_params_names
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                               not any(k.startswith(s) for s in skip_layers)}
        model.load_state_dict(pretrained_dict, strict=False)

    @property
    def first_layer_params_names(self):
        return ['conv1.conv']


class TimmUnet(AbstractModel):
    def __init__(self, encoder='resnet34', use_last_decoder=True, **kwargs):
        if not hasattr(self, 'first_layer_stride_two'):
            self.first_layer_stride_two = True
        if not hasattr(self, 'decoder_block'):
            self.decoder_block = UnetDecoderBlock
        if not hasattr(self, 'bottleneck_type'):
            self.bottleneck_type = ConvBottleneck

        backbone_arch = encoder
        backbone = timm.create_model(backbone_arch.replace("_fat", ""), features_only=True, pretrained=True, **kwargs)
        self.filters = [f["num_chs"] for f in backbone.feature_info]
        self.decoder_filters = default_decoder_filters
        self.last_upsample_filters = default_last
        if encoder in encoder_params:
            self.decoder_filters = encoder_params[encoder].get('decoder_filters', self.filters[:-1])
            self.last_upsample_filters = encoder_params[encoder].get('last_upsample', self.decoder_filters[0] // 2)

        super().__init__()

        self.bottlenecks = nn.ModuleList([self.bottleneck_type(self.filters[-i - 2] + f, f) for i, f in
                                          enumerate(reversed(self.decoder_filters[:]))])

        self.decoder_stages = nn.ModuleList([self.get_decoder(idx) for idx in range(0, len(self.decoder_filters))])

        if self.first_layer_stride_two:
            if use_last_decoder:
                self.last_upsample = UnetDecoderBlock2Conv(self.decoder_filters[0], self.last_upsample_filters,
                                                           self.last_upsample_filters)
            else:
                self.last_upsample = UpsamplingBilinear2d(scale_factor=2)
        self.xydir_head = EncoderRegressionHead(
            in_channels=self.filters[-1],
            out_channels=2,
        )

        self.height_head = RegressionHead(
            in_channels=self.last_upsample_filters,
            out_channels=1,
            kernel_size=3,
        )

        self.mag_head = RegressionHead(
            in_channels=self.last_upsample_filters,
            out_channels=1,
            kernel_size=3,
        )

        self.scale_head = ScaleHead()

        self.name = "u-{}".format(encoder)

        self._initialize_weights()
        self.dropout = Dropout2d(p=0.0)
        self.encoder = backbone

    # noinspection PyCallingNonCallable
    def forward(self, x, city=None, **kwargs):
        # Encoder
        x = x.contiguous(memory_format=torch.channels_last)
        enc_results = self.encoder(x)
        x = enc_results[-1]
        bottlenecks = self.bottlenecks
        for idx, bottleneck in enumerate(bottlenecks):
            rev_idx = - (idx + 1)
            x = self.decoder_stages[rev_idx](x)
            x = bottleneck(x, enc_results[rev_idx - 1])
        xydir = self.xydir_head(enc_results[-1])
        x = self.last_upsample(x)

        decoder_output = x
        height = self.height_head(decoder_output).contiguous(memory_format=torch.contiguous_format)
        mag = self.mag_head(decoder_output).contiguous(memory_format=torch.contiguous_format)
        scale = self.scale_head(mag, height)
        if scale.ndim == 0:
            scale = torch.unsqueeze(scale, axis=0)

        return {"xydir": xydir, "height": height, "mag": mag, "scale": scale}

    def get_decoder(self, layer):
        in_channels = self.filters[layer + 1] if layer + 1 == len(self.decoder_filters) else self.decoder_filters[
            layer + 1]
        return self.decoder_block(in_channels, self.decoder_filters[layer], self.decoder_filters[max(layer, 0)])

    def make_final_classifier(self, in_filters, num_classes):
        if isinstance(num_classes, int):
            return nn.Sequential(
                nn.Conv2d(in_filters, num_classes, 1, padding=0)
            )
        else:
            raise ValueError("unknown numclasses type: " + type(num_classes))

    def get_encoder(self, encoder, layer):
        raise NotImplementedError

    @property
    def first_layer_params(self):
        return _get_layers_params([self.encoder_stages[0]])

    @property
    def first_layer_params_names(self):
        raise NotImplementedError

    @property
    def layers_except_first_params(self):
        layers = get_slice(self.encoder_stages, 1, -1) + [self.bottlenecks, self.decoder_stages, self.final]
        return _get_layers_params(layers)


def _get_layers_params(layers):
    return sum((list(l.parameters()) for l in layers), [])


def get_slice(features, start, end):
    if end == -1:
        end = len(features)
    return [features[i] for i in range(start, end)]


class ConvBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, dec, enc):
        x = torch.cat([dec, enc], dim=1)
        return self.seq(x)


class UnetDecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layer(x)


class UnetDecoderBlock2Conv(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layer(x)

class TimmUnetFeat(TimmUnet):

    def __init__(self, encoder='resnet34', use_last_decoder=True, **kwargs):
        super().__init__(encoder, use_last_decoder, **kwargs)
        self.avg_pool = AdaptiveAvgPool2d((1, 1))

    def forward(self, x, city=None, **kwargs):
        enc_results = self.encoder(x)
        x = enc_results[-1]
        x = self.avg_pool(x).flatten(1)
        return x
