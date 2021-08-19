import torch
import torch.nn.functional as F
import torch.nn as nn
from segmentation_models_pytorch.base.modules import Flatten, Activation
from torch.cuda.amp import autocast


class RegressionHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        conv2d = nn.Conv2d(
            in_channels, out_channels, kernel_size, padding=kernel_size // 2
        )
        identity = nn.Identity()
        activation = Activation(None)
        super().__init__(conv2d, identity, activation)


class EncoderRegressionHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, dropout_ratio=0.5):
        pool = nn.AdaptiveAvgPool2d(1)
        flatten = Flatten()
        dropout = nn.Dropout(p=dropout_ratio, inplace=True)
        linear = nn.Linear(in_channels, out_channels, bias=True)
        super().__init__(pool, flatten, dropout, linear)


class ScaleHead(nn.Module):
    def __init__(
            self,
    ):
        super().__init__()
        self.flatten = torch.flatten
        self.dot = torch.dot

    @autocast(enabled=False)
    def forward(self, mag, height):
        mag = mag.float()
        height = height.float()

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


def resize(input, size=None, scale_factor=None):
    return F.interpolate(input.unsqueeze(dim=1),
                         size,
                         scale_factor,
                         mode='bilinear',
                         align_corners=True).squeeze(dim=1)


def nonan_mse(output, target, reduce='mean'):
    diff = torch.squeeze(output) - target
    not_nan = ~torch.isnan(diff)
    losses = diff.masked_select(not_nan) ** 2

    if reduce == 'mean':
        return losses.mean()
    elif reduce == 'none':
        return losses
    else:
        raise ValueError('Unknown reduce type.')


def r_square(y_pred, y_true):
    diff = torch.squeeze(y_pred) - y_true
    not_nan = ~torch.isnan(diff)
    rss = (diff.masked_select(not_nan) ** 2).sum()

    gt = y_true.masked_select(not_nan)
    tss = ((gt - torch.mean(gt)) ** 2).sum()
    r2 = torch.maximum(torch.zeros_like(rss), 1. - rss / tss)
    return r2


def make_vflow(mag, xydir):
    if mag.ndim == 3:
        mag = mag.unsqueeze(dim=1)
    # mag: [N, 1, H, W]
    # xydir: [N, 2]
    vflow = mag * xydir.view(xydir.size(0), xydir.size(1), 1, 1)
    return vflow


class OptimizationMixin(object):
    def baseline_loss(self,
                      xydir, agl, mag, scale,
                      gt_xy_dir, gt_agl, gt_mag, gt_scale,
                      loss_config):

        if loss_config.get('type', 'mse') == 'mse':
            return self.mse_loss(xydir, agl, mag, scale,
                                 gt_xy_dir, gt_agl, gt_mag, gt_scale,
                                 loss_config)

    def mse_loss(self,
                 xydir, agl, mag, scale,
                 gt_xy_dir, gt_agl, gt_mag, gt_scale,
                 loss_config):
        agl_weight = loss_config.get('agl_weight', 1.)
        mag_weight = loss_config.get('mag_weight', 2.)
        angle_weight = loss_config.get('angle_weight', 10.)
        scale_weight = loss_config.get('scale_weight', 10.)

        reduce = 'mean'

        loss_dict = {
            f'agl@{agl_weight}_loss': agl_weight * nonan_mse(agl, gt_agl, reduce),
            f'mag@{mag_weight}_loss': mag_weight * nonan_mse(mag, gt_mag, reduce),
            f'angle@{angle_weight}_loss': angle_weight * F.mse_loss(xydir, gt_xy_dir),
        }
        if scale is not None:
            loss_dict[f'scale@{scale_weight}_loss'] = scale_weight * F.mse_loss(scale, gt_scale)

        # print metric
        with torch.no_grad():
            agl_r2 = r_square(agl, gt_agl)
            loss_dict['agl_r2'] = agl_r2

            vflow = make_vflow(mag, xydir)
            gt_vflow = make_vflow(gt_mag, gt_xy_dir)
            vflow_r2 = r_square(vflow, gt_vflow)
            loss_dict['vflow_r2'] = vflow_r2

        return loss_dict
