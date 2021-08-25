from abc import ABC, abstractmethod

import torch
from torch.nn import MSELoss


class LossCalculator(ABC):

    @abstractmethod
    def calculate_loss(self, outputs, sample):
        pass


class MSEScaleLossCalculator(LossCalculator):

    def __init__(self, **kwargs):
        super().__init__()
        self.mse = MSELoss(**kwargs)

    def calculate_loss(self, outputs, sample):
        mask = sample["scale"].cuda().float()
        pred = outputs["scale"]
        return self.mse(pred, mask)


class MSEAngleLossCalculator(LossCalculator):

    def __init__(self, **kwargs):
        super().__init__()
        self.mse = MSELoss(**kwargs)

    def calculate_loss(self, outputs, sample):
        mask = sample["xydir"].cuda().float()
        pred = outputs["xydir"]
        return self.mse(pred, mask)


class AngleLossCalculator(LossCalculator):

    def __init__(self, **kwargs):
        super().__init__()

    def calculate_loss(self, outputs, sample):
        mask = sample["xydir"].cuda().float()
        pred = outputs["xydir"]

        def angle_error(dir_pred, dir_gt):
            cos = torch.dot(dir_pred, dir_gt) / (dir_pred.norm() * dir_gt.norm() + 1e-7)
            sin = torch.sqrt(1 - (cos * cos) + 1e-7)
            rad_diff = torch.atan2(sin, cos)
            return rad_diff

        error = 0
        for i in range(mask.size(0)):
            error += angle_error(pred[i], mask[i])
        return error / mask.size(0)


def r2_loss(output, target, target_mean):
    eps = 1e-8
    ss_tot = torch.sum((target - target_mean) ** 2)
    ss_res = torch.sum((target - output) ** 2)
    r2 = (ss_res + eps) / (ss_tot + eps)
    return r2


class NoNaNMSEAGLLossCalculator(LossCalculator):

    def __init__(self, **kwargs):
        super().__init__()

    def calculate_loss(self, outputs, sample):
        target = sample["agl"].cuda().float()
        output = outputs["height"]
        diff = torch.squeeze(output) - target
        not_nan = ~torch.isnan(diff)
        loss = torch.mean(diff.masked_select(not_nan) ** 2)
        return loss


class NoNaNR2AGLLossCalculator(LossCalculator):

    def __init__(self, **kwargs):
        super().__init__()

    def calculate_loss(self, outputs, sample):
        target = sample["agl"].cuda().float()
        target_mean = sample["agl_target_mean"].cuda().float()
        target_mean = target_mean.unsqueeze(1) + torch.zeros_like(target)

        output = outputs["height"]
        city_ohe = sample["city_ohe"].cuda()
        return r2_per_city(output, target, target_mean, city_ohe)


def weighted_focal_mse_loss(inputs, targets, activate='sigmoid', beta=.2, gamma=1, weights=None):
    loss = (inputs - targets) ** 2
    loss *= (torch.tanh(beta * torch.abs(inputs - targets))) ** gamma if activate == 'tanh' else \
        (2 * torch.sigmoid(beta * torch.abs(inputs - targets)) - 1) ** gamma
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss

class FocalAGLLossCalculator(LossCalculator):

    def __init__(self, **kwargs):
        super().__init__()

    def calculate_loss(self, outputs, sample):
        target = sample["agl"].cuda().float()
        output = outputs["height"]
        diff = torch.squeeze(output) - target
        not_nan = ~torch.isnan(diff)
        target = target.masked_select(not_nan)
        output = torch.squeeze(output).masked_select(not_nan)
        return weighted_focal_mse_loss(output, target)


class FocalMAGLossCalculator(LossCalculator):

    def __init__(self, **kwargs):
        super().__init__()

    def calculate_loss(self, outputs, sample):
        target = sample["mag"].cuda().float()
        output = outputs["mag"]
        diff = torch.squeeze(output) - target
        not_nan = ~torch.isnan(diff)
        target = target.masked_select(not_nan)
        output = torch.squeeze(output).masked_select(not_nan)
        return weighted_focal_mse_loss(output, target)


class NoNaNMSEMAGLossCalculator(LossCalculator):

    def __init__(self, **kwargs):
        super().__init__()

    def calculate_loss(self, outputs, sample):
        target = sample["mag"].cuda().float()
        output = outputs["mag"]
        diff = torch.squeeze(output) - target
        not_nan = ~torch.isnan(diff)
        loss = torch.mean(diff.masked_select(not_nan) ** 2)
        return loss


class NoNaNR2MAGLossCalculator(LossCalculator):

    def __init__(self, **kwargs):
        super().__init__()

    def calculate_loss(self, outputs, sample):
        target = sample["mag"].cuda().float()
        city_ohe = sample["city_ohe"].cuda()
        output = outputs["mag"]
        target_mean = sample["mag_target_mean"].cuda().float()
        target_mean = target_mean.unsqueeze(1) + torch.zeros_like(target)
        return r2_per_city(output, target, target_mean, city_ohe)


def r2_per_city(output_full, targets_full, target_mean_full, city_ohe):
    total_loss = 0
    num_cities = 0
    for c_i in range(4):
        city_idx = city_ohe[:, c_i] > 0
        if torch.sum(city_idx * 1.) == 0:
            continue
        output = output_full[city_idx]
        target = targets_full[city_idx]
        target_mean = target_mean_full[city_idx]
        diff = torch.squeeze(output) - target
        not_nan = ~torch.isnan(diff)
        target = target.masked_select(not_nan)
        output = torch.squeeze(output).masked_select(not_nan)
        target_mean = target_mean.masked_select(not_nan)
        total_loss += r2_loss(output, target, target_mean)
        num_cities += 1
    return total_loss / num_cities