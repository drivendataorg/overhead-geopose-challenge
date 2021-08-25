from functools import partial
from typing import List, Dict, Any

import torch
from omegaconf import DictConfig
from pytorch_toolbelt.datasets import INPUT_IMAGE_KEY
from pytorch_toolbelt.modules import (
    decoders as D,
    UnetBlock,
    get_activation_block,
    ABN,
    EncoderModule,
    ResidualDeconvolutionUpsample2d,
)
from torch import nn, Tensor

__all__ = ["GeoposeUNetModel"]

from geopose import INPUT_GROUND_SAMPLE_DISTANCE, OUTPUT_AGL_MASK
from geopose.models.unet_blocks import IRBlock, DCUnetBlock, DenseNetUnetBlock, ResidualUnetBlock


class GeoposeUNetModel(nn.Module):
    def __init__(
        self,
        encoder: EncoderModule,
        decoder_features: List[int],
        decoder_block,
        agl_head: nn.Module,
        orientation_head: nn.Module,
        upsample_block=nn.UpsamplingBilinear2d,
    ):
        super().__init__()
        self.output_stride = encoder.strides[0]

        self.encoder = encoder
        self.decoder = D.UNetDecoder(encoder.channels, decoder_features, unet_block=decoder_block, upsample_block=upsample_block)
        self.agl_head = agl_head
        self.orientation_head = orientation_head

    def forward(self, **kwargs):
        rgb = kwargs[INPUT_IMAGE_KEY]
        gsd = kwargs[INPUT_GROUND_SAMPLE_DISTANCE]

        enc = self.encoder(rgb)
        dec = self.decoder(enc)

        outputs = self.orientation_head(rgb, enc, dec, gsd)

        # Returns OUTPUT_DENSE_AGL_MASK_METERS
        if self.agl_head:
            agl_outputs: Dict[str, Tensor] = self.agl_head(rgb, enc, dec, gsd, outputs)
            outputs.update(agl_outputs)
        else:
            outputs[OUTPUT_AGL_MASK] = torch.zeros(
                (rgb.size(0), 1, rgb.size(2), rgb.size(3)), device=rgb.device, dtype=torch.float16
            )

        return outputs

    @classmethod
    def from_config(cls, config: DictConfig):
        from hydra.utils import instantiate

        encoder = instantiate(config.encoder)
        orientation_head = instantiate(
            config.orientation_head, encoder_channels=encoder.channels, decoder_channels=config.decoder.channels
        )

        if config.agl_head:
            agl_head = instantiate(config.agl_head, encoder_channels=encoder.channels, decoder_channels=config.decoder.channels)
        else:
            agl_head = None

        return GeoposeUNetModel(
            encoder=encoder,
            decoder_features=config.decoder.channels,
            decoder_block=get_decoder_block(
                block_type=config.decoder.get("block_type", "unet"),
                activation=config.decoder.get("activation", config.activation),
            ),
            agl_head=agl_head,
            orientation_head=orientation_head,
            upsample_block=get_upsample_block(config.decoder.get("upsample", "bilinear")),
        )


#         self.extra_encoder_blocks = nn.ModuleList(
#             [extra_block(encoder.channels[-1], encoder.channels[-1]) for _ in range(num_extra_blocks)]
#         )


def get_upsample_block(upsample_type):
    if upsample_type == "bilinear":
        return nn.UpsamplingBilinear2d
    elif upsample_type == "rdtsc":
        return ResidualDeconvolutionUpsample2d
    else:
        raise KeyError(upsample_type)


def get_decoder_block(block_type, activation) -> Any:
    if block_type == "unet":
        abn_block = ABN
        abn_block = partial(abn_block, activation=activation)
        unet_block = partial(UnetBlock, abn_block=abn_block)
        return unet_block
    if block_type == "dense_unet":
        abn_block = ABN
        abn_block = partial(abn_block, activation=activation)
        unet_block = partial(DenseNetUnetBlock, abn_block=abn_block)
        return unet_block
    if block_type == "residual_unet":
        abn_block = ABN
        abn_block = partial(abn_block, activation=activation)
        unet_block = partial(ResidualUnetBlock, abn_block=abn_block)
        return unet_block
    if block_type == "ir_unet":
        unet_block = partial(IRBlock, act_block=get_activation_block(activation))
        return unet_block
    if block_type == "dc_unet":
        abn_block = ABN
        abn_block = partial(abn_block, activation=activation)
        unet_block = partial(DCUnetBlock, abn_block=abn_block)
        return unet_block
    else:
        raise KeyError(block_type)
