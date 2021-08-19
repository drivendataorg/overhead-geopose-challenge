import itertools
import json
import os
import pickle
import random
import sys
from collections import OrderedDict
from pathlib import Path

import albumentations as A
import apex
import lmdb
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
import torch.distributed
from segmentation_models_pytorch.utils.meter import AverageValueMeter
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utilities.augmentation_vflow import augment_vflow, warp_agl
from utilities.misc_utils import (
    convert_and_compress_prediction_dir,
    get_angle_error,
    get_r2,
    get_r2_info,
    get_rms,
    load_image,
    load_vflow,
    save_image,
)
from utilities.unet_vflow import UnetVFLOW


def get_transforms(args, p=0.5):
    crop_size = 1024 if args.augmentation else 512
    crop_fn = A.RandomCrop(crop_size, crop_size)

    albu_train = None
    if args.albu:
        albu_train = A.Compose(
            args.augmentation * [A.RandomCrop(512, 512)] +
            [
                A.CoarseDropout(max_holes=32, max_height=32, max_width=32, p=p),
                A.OneOf(
                    [
                        A.Blur(p=1),
                        A.GlassBlur(p=1),
                        A.GaussianBlur(p=1),
                        A.MedianBlur(p=1),
                        A.MotionBlur(p=1),
                    ],
                    p=p,
                ),
                A.RandomBrightnessContrast(p=p),
                A.OneOf(
                    [
                        A.RandomGamma(p=1),
                        A.ColorJitter(p=1),
                        A.RandomToneCurve(p=1),
                    ],
                    p=p,
                ),
                A.OneOf(
                    [
                        A.GaussNoise(p=1),
                        A.MultiplicativeNoise(p=1),
                    ],
                    p=p,
                ),
                A.FancyPCA(p=0.2),
                A.RandomFog(p=0.2),
                A.RandomShadow(p=0.2),
                A.RandomSunFlare(src_radius=150, p=0.2),
            ]
        )

    return crop_fn, albu_train


class Dataset(BaseDataset):
    def __init__(
        self,
        df,
        args,
        is_val=False,
        cities=None,
        is_test=False,
    ):
        self.is_test = is_test
        self.is_val = is_val
        dataset_dir = Path(args.dataset_dir)
        rgb_paths = df.rgb.apply(
            lambda x: (dataset_dir / x).with_suffix(f".{args.rgb_suffix}")
        ).tolist()

        if not self.is_test:
            agl_paths = df.agl.apply(lambda x: dataset_dir / x).tolist()
            vflow_paths = df.json.apply(lambda x: dataset_dir / x).tolist()

        self.gsd = df.gsd.tolist()
        self.city = df.city.tolist()

        if self.is_test:
            self.paths_list = rgb_paths
        else:
            self.paths_list = [
                (rgb_paths[i], vflow_paths[i], agl_paths[i])
                for i in range(len(rgb_paths))
            ]

        self.preprocessing_fn = smp.encoders.get_preprocessing_fn(
            "senet154", "imagenet"
        )

        if args.lmdb is not None:
            self.env = lmdb.open(
                str(args.lmdb),
                readonly=True,
                lock=False,
                readahead=True,
                meminit=False,
            )

        self.args = args
        self.cities = cities

        crop_fn, albu_train = get_transforms(self.args)
        self.crop_fn = crop_fn
        self.albu_train = albu_train

    def __getitem__(self, i):
        gsd = None
        if self.cities is not None:
            city_str = self.city[i]
            city = np.zeros((len(self.cities), 1, 1), dtype="float32")
            city[self.cities.index(city_str)] = 1

            gsd = np.zeros((1, 1, 1), dtype="float32")
            gsd[0] = self.gsd[i]

        if self.is_test:
            rgb_path = self.paths_list[i]
            image = load_image(rgb_path, self.args, use_cv=True)
        else:
            rgb_path, vflow_path, agl_path = self.paths_list[i]
            if hasattr(self, "env"):
                with self.env.begin(write=False, buffers=True) as txn:
                    image = pickle.loads(txn.get(rgb_path.stem.encode()))
                    agl = pickle.loads(
                        txn.get(rgb_path.stem.replace("RGB", "AGL").encode())
                    )
            else:
                image = load_image(rgb_path, self.args)
                agl = load_image(agl_path, self.args)

            # max_agl = np.nanmax(agl)
            if (not self.is_test) and (not self.is_val):
                data = self.crop_fn(image=image, mask=agl)
                image = data["image"]
                agl = data["mask"]

            mag, xdir, ydir, vflow_data = load_vflow(vflow_path, agl, self.args)
            scale = vflow_data["scale"]
            if (not self.is_val) and self.args.augmentation:
                image, mag, xdir, ydir, agl, scale, gsd = augment_vflow(
                    image,
                    mag,
                    xdir,
                    ydir,
                    vflow_data["angle"],
                    vflow_data["scale"],
                    agl=agl,
                    rotate90_prob=0.5,
                    rotate_prob=0.3,
                    flip_prob=0.5,
                    scale_prob=0.5,
                    agl_prob=0.5,
                    gsd=gsd,
                    # max_agl=max_agl,
                )
            xdir = np.float32(xdir)
            ydir = np.float32(ydir)
            mag = mag.astype("float32")
            agl = agl.astype("float32")
            scale = np.float32(scale)

            xydir = np.array([xdir, ydir])

        image = image.astype("uint8")
        if self.args.albu and (not self.is_test) and (not self.is_val):
            data = self.albu_train(image=image, masks=[mag, agl])
            image = data["image"]
            mag, agl = data["masks"]

        image = self.preprocessing_fn(image).astype("float32")
        image = np.transpose(image, (2, 0, 1))

        if self.cities is not None:
            image = image, city, gsd

        if self.is_test:
            return image, str(rgb_path)
        else:
            return image, xydir, agl, mag, scale

    def __len__(self):
        return len(self.paths_list)


class DatasetPL(BaseDataset):
    def __init__(
        self,
        rgb_dir,
        pred_dir,
        args,
    ):
        # create all paths with respect to RGB path ordering to maintain alignment of samples
        dataset_dir = Path(args.dataset_dir) / rgb_dir
        rgb_paths = list(dataset_dir.glob(f"*_RGB.{args.rgb_suffix}"))
        pred_dir = Path(pred_dir)
        agl_paths = list(
            pred_dir
            / pth.with_name(pth.name.replace("_RGB", "_AGL")).with_suffix(".tif").name
            for pth in rgb_paths
        )
        vflow_paths = list(
            pred_dir
            / pth.with_name(pth.name.replace("_RGB", "_VFLOW"))
            .with_suffix(".json")
            .name
            for pth in rgb_paths
        )

        self.paths_list = [
            (rgb_paths[i], vflow_paths[i], agl_paths[i]) for i in range(len(rgb_paths))
        ]

        self.preprocessing_fn = smp.encoders.get_preprocessing_fn(
            "senet154", "imagenet"
        )

        import copy

        self.args = copy.deepcopy(args)
        self.args.unit = "m"
        self.max_building_agl = 200.0
        self.max_factor = 2.0

    def __getitem__(self, i):
        rgb_path, vflow_path, agl_path = self.paths_list[i]
        image = load_image(rgb_path, self.args, use_cv=True)
        agl = load_image(agl_path, self.args)

        mag, _, _, vflow_data = load_vflow(vflow_path, agl, self.args)

        max_agl = np.max(agl)
        assert max_agl > 0

        max_scale_agl = min(self.max_factor, (self.max_building_agl / max_agl))
        # scale_heights = [(1 + max_scale_agl) / 2, max_scale_agl]
        scale_heights = [max_scale_agl]
        images_aug = [image]
        images_aug.extend(
            [
                warp_agl(
                    image, mag, vflow_data["angle"], agl, scale_height, self.max_factor
                )[0]
                for scale_height in scale_heights
            ]
        )

        images_aug = [image.astype("uint8") for image in images_aug]
        images_aug = [
            self.preprocessing_fn(image).astype("float32") for image in images_aug
        ]
        images_aug = [np.transpose(image, (2, 0, 1)) for image in images_aug]

        return images_aug, str(rgb_path)

    def __len__(self):
        return len(self.paths_list)


class DatasetPseudoLabel(BaseDataset):
    def __init__(
        self,
        df,
        args,
        cities=None,
    ):
        rgb_paths = df.rgb.apply(lambda x: Path(args.test_rgb_path) / x).tolist()
        agl_paths = df.agl.apply(lambda x: Path(args.pl_dir) / x).tolist()
        vflow_paths = df.json.apply(lambda x: Path(args.pl_dir) / x).tolist()

        self.gsd = df.gsd.tolist()
        self.city = df.city.tolist()

        self.paths_list = [
            (rgb_paths[i], vflow_paths[i], agl_paths[i]) for i in range(len(rgb_paths))
        ]

        self.preprocessing_fn = smp.encoders.get_preprocessing_fn(
            "senet154", "imagenet"
        )

        self.args = args
        self.cities = cities

    def __getitem__(self, i):
        gsd = None
        if self.cities is not None:
            city_str = self.city[i]
            city = np.zeros((len(self.cities), 1, 1), dtype="float32")
            city[self.cities.index(city_str)] = 1

            gsd = np.zeros((1, 1, 1), dtype="float32")
            gsd[0] = self.gsd[i]

            rgb_path, vflow_path, agl_path = self.paths_list[i]
            image = load_image(rgb_path, self.args, use_cv=True)
            agl = load_image(agl_path, self.args)

            data = crop_fn(image=image, mask=agl)
            image = data["image"]
            agl = data["mask"]

            mag, xdir, ydir, vflow_data = load_vflow(vflow_path, agl, self.args)
            scale = vflow_data["scale"]
            if self.args.augmentation:
                image, mag, xdir, ydir, agl, scale, gsd = augment_vflow(
                    image,
                    mag,
                    xdir,
                    ydir,
                    vflow_data["angle"],
                    vflow_data["scale"],
                    agl=agl,
                    rotate90_prob=0.5,
                    rotate_prob=0.3,
                    flip_prob=0.5,
                    scale_prob=0.5,
                    agl_prob=0.5,
                    gsd=gsd,
                    # max_agl=max_agl,
                )
            xdir = np.float32(xdir)
            ydir = np.float32(ydir)
            mag = mag.astype("float32")
            agl = agl.astype("float32")
            scale = np.float32(scale)

            xydir = np.array([xdir, ydir])

        image = image.astype("uint8")
        data = albu_train(image=image, masks=[mag, agl])
        image = data["image"]
        mag, agl = data["masks"]

        image = self.preprocessing_fn(image).astype("float32")
        image = np.transpose(image, (2, 0, 1))

        if self.cities is not None:
            image = image, city, gsd

        return image, xydir, agl, mag, scale

    def __len__(self):
        return len(self.paths_list)


class PrefetchLoader:
    def __init__(self, loader):
        self.loader = loader

    def __iter__(self):
        stream = torch.cuda.Stream()
        first = True

        for next_image, next_xydir, next_agl, next_mag, next_scale in self.loader:
            with torch.cuda.stream(stream):
                next_image, next_xydir, next_agl, next_mag, next_scale = map(
                    lambda x: x.cuda(non_blocking=True),
                    (next_image, next_xydir, next_agl, next_mag, next_scale),
                )

            if not first:
                yield image, xydir, agl, mag, scale
            else:
                first = False

            torch.cuda.current_stream().wait_stream(stream)
            image, xydir, agl, mag, scale = (
                next_image,
                next_xydir,
                next_agl,
                next_mag,
                next_scale,
            )

        yield image, xydir, agl, mag, scale

    def __len__(self):
        return len(self.loader)

    @property
    def sampler(self):
        return self.loader.sampler

    @property
    def dataset(self):
        return self.loader.dataset


class Epoch:
    def __init__(
        self,
        model,
        args,
        dense_loss=None,
        angle_loss=None,
        scale_loss=None,
        stage_name=None,
        device="cpu",
        verbose=True,
        local_rank=0,
        channels_last=False,
        to_log=False,
    ):
        self.args = args
        self.model = model
        self.dense_loss = dense_loss
        self.angle_loss = angle_loss
        self.scale_loss = scale_loss
        self.stage_name = stage_name
        self.verbose = verbose
        self.device = device
        self.local_rank = local_rank
        self.channels_last = channels_last
        self.to_log = to_log

        self.loss_names = ["combined", "agl", "mag", "angle", "scale"]

        self._to_device()

    def _to_device(self):
        self.model.to(self.device)
        if self.stage_name != "valid":
            self.dense_loss.to(self.device)
            self.angle_loss.to(self.device)
            self.scale_loss.to(self.device)

    def _format_logs(self, logs):
        str_logs = ["{} - {:.4}".format(k, v) for k, v in logs.items()]
        s = ", ".join(str_logs)
        return s

    def batch_update(self, x, y):
        raise NotImplementedError

    def on_epoch_start(self):
        pass

    def run(self, dataloader, desc=""):

        self.on_epoch_start()

        logs = {}

        loss_meters = {}
        for loss_name in self.loss_names:
            loss_meters[loss_name] = AverageValueMeter()

        if self.stage_name == "valid":
            agl_count, agl_error_sum, agl_gt_sq_sum, agl_sum = 0, 0, 0, 0
            mag_count, mag_error_sum, mag_gt_sq_sum, mag_sum = 0, 0, 0, 0
            vflow_count, vflow_error_sum, vflow_gt_sq_sum, vflow_sum = 0, 0, 0, 0
            angle_errors = []
            agl_rms = []
            mag_rms = []
            vflow_rms = []
            scale_errors = []

        if self.local_rank == 0:
            iterator = tqdm(
                total=len(dataloader),
                desc=f"{self.stage_name} {desc}",
                file=sys.stdout,
                disable=not (self.verbose),
                mininterval=2,
                leave=False,
            )

        for itr_data in dataloader:
            image, xydir, agl, mag, scale = itr_data
            scale = torch.unsqueeze(scale, 1)

            use_city = isinstance(image, (tuple, list))
            if use_city:
                image, city, gsd = image
                city = city.to(self.device, non_blocking=True)
                gsd = gsd.to(self.device, non_blocking=True)

            image = image.to(self.device, non_blocking=True)

            if use_city:
                image = image, city, gsd

            if self.stage_name != "valid":
                xydir, agl, mag, scale = (
                    xydir.to(self.device, non_blocking=True),
                    agl.to(self.device, non_blocking=True),
                    mag.to(self.device, non_blocking=True),
                    scale.to(self.device, non_blocking=True),
                )
                y = [xydir, agl, mag, scale]

                loss, *_ = self.batch_update(image, y)

                loss_logs = {}

                for name in self.loss_names:
                    curr_loss = loss[name].cpu().detach().numpy()
                    if name == "scale":
                        curr_loss = np.mean(curr_loss)
                    loss_meters[name].add(curr_loss)
                    loss_logs[name] = loss_meters[name].mean

                logs.update(loss_logs)
            else:
                xydir_pred, agl_pred, mag_pred, scale_pred = self.batch_update(image)

                xydir = xydir.cpu().numpy()
                agl = agl.cpu().numpy()
                mag = mag.cpu().numpy()
                scale = scale.cpu().numpy()

                xydir_pred = xydir_pred.cpu().numpy()
                agl_pred = agl_pred.cpu().numpy()
                mag_pred = mag_pred.cpu().numpy()
                scale_pred = scale_pred.cpu().numpy()

                for batch_ind in range(agl.shape[0]):
                    count, error_sum, rms, data_sum, gt_sq_sum = get_r2_info(
                        agl[batch_ind], agl_pred[batch_ind]
                    )
                    agl_count += count
                    agl_error_sum += error_sum
                    agl_rms.append(rms)
                    agl_sum += data_sum
                    agl_gt_sq_sum += gt_sq_sum

                    count, error_sum, rms, data_sum, gt_sq_sum = get_r2_info(
                        mag[batch_ind], mag_pred[batch_ind]
                    )
                    mag_count += count
                    mag_error_sum += error_sum
                    mag_rms.append(rms)
                    mag_sum += data_sum
                    mag_gt_sq_sum += gt_sq_sum

                    vflow = np.zeros(
                        (
                            agl[batch_ind].squeeze().shape[0],
                            agl[batch_ind].squeeze().shape[1],
                            2,
                        )
                    )
                    vflow[..., 0] = mag[batch_ind].squeeze() * xydir[batch_ind, 0]
                    vflow[..., 1] = mag[batch_ind].squeeze() * xydir[batch_ind, 1]

                    vflow_pred = np.zeros_like(vflow)
                    vflow_pred[..., 0] = (
                        agl_pred[batch_ind].squeeze() * xydir_pred[batch_ind, 0]
                    )
                    vflow_pred[..., 1] = (
                        agl_pred[batch_ind].squeeze() * xydir_pred[batch_ind, 1]
                    )
                    vflow_pred = scale_pred[batch_ind] * vflow_pred

                    count, error_sum, rms, data_sum, gt_sq_sum = get_r2_info(
                        vflow, vflow_pred,
                    )
                    vflow_count += count
                    vflow_error_sum += error_sum
                    vflow_rms.append(rms)
                    vflow_sum += data_sum
                    vflow_gt_sq_sum += gt_sq_sum

                    dir_pred = xydir_pred[batch_ind, :]
                    dir_gt = xydir[batch_ind, :]

                    angle_error = get_angle_error(dir_pred, dir_gt)

                    angle_errors.append(angle_error)
                    scale_errors.append(
                        np.abs(scale[batch_ind] - scale_pred[batch_ind])
                    )

            torch.cuda.synchronize()

            if self.local_rank == 0 and self.verbose:
                s = self._format_logs(logs)
                iterator.set_postfix_str(s)
                iterator.update()

        if self.stage_name == "valid":
            logs.update(
                dict(
                    agl_error_sum=agl_error_sum,
                    agl_gt_sq_sum=agl_gt_sq_sum,
                    agl_sum=agl_sum,
                    agl_count=agl_count,
                    mag_error_sum=mag_error_sum,
                    mag_gt_sq_sum=mag_gt_sq_sum,
                    mag_sum=mag_sum,
                    mag_count=mag_count,
                    vflow_error_sum=vflow_error_sum,
                    vflow_gt_sq_sum=vflow_gt_sq_sum,
                    vflow_sum=vflow_sum,
                    vflow_count=vflow_count,
                    angle_errors=angle_errors,
                    scale_errors=scale_errors,
                    agl_rms=agl_rms,
                    mag_rms=mag_rms,
                    vflow_rms=vflow_rms,
                )
            )

        if self.local_rank == 0:
            iterator.close()

        return logs


class TrainEpoch(Epoch):
    def __init__(
        self,
        model,
        args,
        dense_loss,
        angle_loss,
        scale_loss,
        optimizer,
        scaler=None,
        device="cpu",
        verbose=True,
        local_rank=0,
        channels_last=False,
        to_log=False,
    ):
        super().__init__(
            model=model,
            args=args,
            dense_loss=dense_loss,
            angle_loss=angle_loss,
            scale_loss=scale_loss,
            stage_name="train",
            device=device,
            verbose=verbose,
            local_rank=local_rank,
            channels_last=channels_last,
            to_log=to_log,
        )
        self.optimizer = optimizer
        self.scaler = scaler

    def on_epoch_start(self):
        self.model.train()

    def batch_update(self, x, y):
        # if self.channels_last:
        #     x = x.contiguous(memory_format=torch.channels_last)

        self.optimizer.zero_grad()  # set_to_none=True)

        if self.scaler is not None:
            with torch.cuda.amp.autocast():
                xydir_pred, agl_pred, mag_pred, scale_pred = self.model.forward(x)
                scale_pred = torch.unsqueeze(scale_pred, 1)

                xydir, agl, mag, scale = y
                if self.to_log:
                    agl = torch.log1p(agl)
                    mag = torch.log1p(mag)

                loss_agl = self.dense_loss(agl_pred, agl)
                loss_mag = self.dense_loss(mag_pred, mag)
                loss_angle = self.angle_loss(xydir_pred, xydir)

                loss_scale = self.scale_loss(scale_pred, scale)

                loss_combined = (
                    self.args.agl_weight * loss_agl
                    # + self.args.mag_weight * loss_mag
                    + self.args.angle_weight * loss_angle
                    + self.args.scale_weight * loss_scale
                )
        else:
            xydir_pred, agl_pred, mag_pred, scale_pred = self.model.forward(x)

            scale_pred = torch.unsqueeze(scale_pred, 1)

            xydir, agl, mag, scale = y
            if self.to_log:
                agl = torch.log1p(agl)
                mag = torch.log1p(mag)

            loss_agl = self.dense_loss(agl_pred, agl)
            loss_mag = self.dense_loss(mag_pred, mag)
            loss_angle = self.angle_loss(xydir_pred, xydir)

            loss_scale = self.scale_loss(scale_pred, scale)

            loss_combined = (
                self.args.agl_weight * loss_agl
                # + self.args.mag_weight * loss_mag
                + self.args.angle_weight * loss_angle
                + self.args.scale_weight * loss_scale
            )

        loss = {
            "combined": loss_combined,
            "agl": loss_agl,
            "mag": loss_mag,
            "angle": loss_angle,
            "scale": loss_scale,
        }

        if self.scaler is None:
            loss_combined.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 2.0)
            self.optimizer.step()
        else:
            self.scaler.scale(loss_combined).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 2.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

        return loss, xydir_pred, agl_pred, mag_pred, scale_pred


class ValidEpoch(Epoch):
    def __init__(
        self, model, args, device="cpu", verbose=True, local_rank=0, channels_last=False, to_log=False,
    ):
        super().__init__(
            model=model,
            args=args,
            stage_name="valid",
            device=device,
            verbose=verbose,
            local_rank=local_rank,
            channels_last=channels_last,
            to_log=to_log,
        )

    def on_epoch_start(self):
        self.model.eval()

    def batch_update(self, x):
        # if self.channels_last:
        #     x = x.contiguous(memory_format=torch.channels_last)

        with torch.no_grad():
            xydir_pred, agl_pred, mag_pred, scale_pred = self.model.forward(x)
            if self.to_log:
                agl_pred = torch.expm1(agl_pred)
                mag_pred = torch.expm1(mag_pred)

            scale_pred = torch.unsqueeze(scale_pred, 1)

        return xydir_pred, agl_pred, mag_pred, scale_pred


class NoNaNMSE(smp.utils.base.Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mse = torch.nn.MSELoss()

    def forward(self, output, target):
        not_nan = ~torch.isnan(target)

        output = torch.squeeze(output)
        output = output.masked_select(not_nan)
        target = target.masked_select(not_nan)

        loss = self.mse(output, target)

        return loss


def train_dev_split(geopose, args):
    geopose["fold"] = None

    n_col = len(geopose.columns) - 1
    skf = StratifiedKFold(
        n_splits=args.n_folds, shuffle=True, random_state=args.random_state
    )
    for fold, (_, dev_index) in enumerate(skf.split(geopose, geopose.city)):
        geopose.iloc[dev_index, n_col] = fold

    train, dev = (
        geopose[geopose.fold != args.fold].copy(),
        geopose[geopose.fold == args.fold].copy(),
    )

    return train, dev


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def init_dist(args):
    # to autotune convolutions and other algorithms
    # to pick the best for current configuration
    torch.backends.cudnn.benchmark = True

    if args.deterministic:
        set_seed(args.random_state)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.set_printoptions(precision=10)

    args.distributed = False
    if "WORLD_SIZE" in os.environ:
        args.distributed = int(os.environ["WORLD_SIZE"]) > 1

    args.gpu = 0
    args.world_size = 1
    if args.distributed:
        args.gpu = args.local_rank
        torch.cuda.set_device(args.gpu)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        args.world_size = torch.distributed.get_world_size()

    assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."


def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)

    return [
        {"params": no_decay, "weight_decay": 0.0},
        {"params": decay, "weight_decay": weight_decay},
    ]


def save_jit(model, args, model_name):
    model.eval()
    if args.backbone.startswith("efficientnet"):
        model.module.encoder.set_swish(memory_efficient=False)

    inp = torch.rand(2, 3, 512, 512).cuda()
    if args.use_city:
        inp = (
            (
                inp,
                torch.zeros(2, 4, 1, 1).cuda(),
                torch.rand(2, 1, 1, 1).cuda(),
            ),
        )

    with torch.no_grad():
        traced_model = torch.jit.trace(model, inp)

    traced_model.save(os.path.join(args.checkpoint_dir, model_name))

    if args.backbone.startswith("efficientnet"):
        model.module.encoder.set_swish(memory_efficient=True)


def train(args):
    if args.distributed:
        init_dist(args)

    torch.backends.cudnn.benchmark = True

    summary_writer = None
    if args.local_rank == 0:
        summary_writer = SummaryWriter(Path(args.checkpoint_dir) / "logs")  # /exp_name

    model = build_model(args)
    model = model.cuda(args.gpu)
    if args.load:
        path_to_resume = Path(args.load).expanduser()
        if path_to_resume.is_file():
            print(f"=> loading resume checkpoint '{path_to_resume}'")
            checkpoint = torch.load(
                path_to_resume,
                map_location=lambda storage, loc: storage.cuda(args.gpu),
            )
            new_state_dict = OrderedDict()
            for k, v in checkpoint["state_dict"].items():
                name = k[7:] if k.startswith("module") else k
                new_state_dict[name] = v

            model.load_state_dict(new_state_dict)
            print(
                f"=> resume from checkpoint '{path_to_resume}' (epoch {checkpoint['epoch']})"
            )
        else:
            print(f"=> no checkpoint found at '{path_to_resume}'")

    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    weight_decay = args.weight_decay
    if weight_decay > 0:  # and filter_bias_and_bn:
        skip = {}
        if hasattr(model, "no_weight_decay"):
            skip = model.no_weight_decay()

        parameters = add_weight_decay(model, weight_decay, skip)
        weight_decay = 0.0
    else:
        parameters = model.parameters()

    optimizer = build_optimizer(parameters, args)

    if args.resume:
        optimizer.load_state_dict(checkpoint["opt_state_dict"])

    if args.distributed:
        # By default, apex.parallel.DistributedDataParallel overlaps communication with
        # computation in the backward pass.
        # model = DDP(model)
        # delay_allreduce delays all communication to the end of the backward pass.
        if args.syncbn:
            model = apex.parallel.convert_syncbn_model(model)

        model = apex.parallel.DistributedDataParallel(model, delay_allreduce=True)

    df = pd.read_csv(args.train_path_df)
    metadata = pd.read_csv(Path(args.train_path_df).with_name("metadata.csv"))
    df = pd.merge(df, metadata, on="id")

    train_df, dev_df = train_dev_split(df, args)

    CITIES = ["ARG", "ATL", "JAX", "OMA"]
    assert len(CITIES) == torch.distributed.get_world_size()

    if args.city is not None:
        CITIES = [args.city]
        train_df = train_df[train_df.city.isin(CITIES)].reset_index(drop=True)
        dev_df = dev_df[dev_df.city.isin(CITIES)].reset_index(drop=True)
    else:
        dev_df = dev_df[dev_df.city.isin([CITIES[args.local_rank]])].reset_index(drop=True)

    cities = ["ARG", "ATL", "JAX", "OMA"] if args.use_city else None
    train_dataset = Dataset(train_df, args=args, is_val=False, cities=cities)

    if args.pl_dir is not None:
        test_df = pd.read_csv(args.test_path_df)
        test_df["agl"] = test_df.rgb.str.replace("RGB.j2k", "AGL.tif")
        test_df["json"] = test_df.rgb.str.replace("RGB.j2k", "VFLOW.json")
        test_df = pd.merge(test_df, metadata, on="id")
        test_dataset = DatasetPseudoLabel(test_df, args, cities=cities)
        train_dataset = ConcatDataset([train_dataset, test_dataset])

    val_dataset = Dataset(dev_df, args=args, is_val=True, cities=cities)

    train_sampler = None
    val_sampler = None
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        # if args.city is not None:
        # If each rank has own city, so we need to use default sampler.
        # Otherwise distributed sampler devides dataset amount other workers.
        if args.city is not None:
            val_sampler = torch.utils.data.distributed.DistributedSampler(
                val_dataset,
                shuffle=args.city is None,
            )

    def worker_init_fn(worker_id):
        seed = (
            np.random.get_state()[1][0]
            + torch.distributed.get_world_size() * worker_id
            + args.local_rank
        )
        np.random.seed(seed)

    args.num_workers = min(args.batch_size, 16)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        collate_fn=None,  # train_dataset.fast_collate,
        num_workers=args.num_workers,
        pin_memory=False,
        persistent_workers=True,
        drop_last=True,
        worker_init_fn=worker_init_fn,
    )
    val_batch_size = max(args.batch_size // 4, 1)
    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        sampler=val_sampler,
        collate_fn=None,  # val_dataset.fast_collate,
        num_workers=val_batch_size,
        pin_memory=False,
        persistent_workers=True,
    )
    if args.prefetch:
        train_loader = PrefetchLoader(train_loader)
        val_loader = PrefetchLoader(val_loader)

    scheduler = build_scheduler(optimizer, args)

    scaler = None
    if args.fp16:
        scaler = torch.cuda.amp.GradScaler()

    dense_loss = NoNaNMSE()
    angle_loss = torch.nn.MSELoss()
    scale_loss = torch.nn.MSELoss()

    train_epoch = TrainEpoch(
        model,
        args=args,
        dense_loss=dense_loss,
        angle_loss=angle_loss,
        scale_loss=scale_loss,
        optimizer=optimizer,
        scaler=scaler,
        device="cuda",
        local_rank=args.local_rank,
        channels_last=args.channels_last,
        to_log=args.to_log,
    )

    val_epoch = ValidEpoch(
        model,
        args=args,
        device="cuda",
        local_rank=args.local_rank,
        channels_last=args.channels_last,
        to_log=args.to_log,
    )

    best_score = 0

    def saver(path):
        torch.save(
            {
                "epoch": i,
                "best_score": best_score,
                "state_dict": model.state_dict(),
                "opt_state_dict": optimizer.state_dict(),
                "sched_state_dict": scheduler.state_dict()
                if scheduler is not None
                else None,
                "scaler": scaler.state_dict() if scaler is not None else None,
                "args": args,
            },
            path,
        )

    IMAGES = ["agl", "mag", "vflow"]
    ERRORS = ["angle", "scale"]

    start_epoch = 0
    if args.resume:
        start_epoch = checkpoint["epoch"] + 1
        best_score = checkpoint["best_score"]
        if checkpoint["sched_state_dict"] is not None:
            scheduler.load_state_dict(checkpoint["sched_state_dict"])

        if checkpoint["scaler"] is not None:
            scaler.load_state_dict(checkpoint["scaler"])

    for i in range(start_epoch, args.num_epochs):
        if args.distributed:
            train_sampler.set_epoch(i)
            if args.city is None and val_sampler is not None:
                val_sampler.set_epoch(i)

        desc = f"{i}/{args.num_epochs}"
        train_logs = train_epoch.run(train_loader, desc=desc)
        # train_logs_out = [None for _ in range(torch.distributed.get_world_size())]
        # torch.distributed.all_gather_object(train_logs_out, train_logs)
        # train_lgs = {}
        # for name in list(train_logs):
        #     train_lgs[name] = sum(x[name] for x in train_logs_out) / len(train_logs_out)
        # train_logs = train_lgs
        for name in list(train_logs):
            train_logs_out = [None for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather_object(train_logs_out, train_logs[name])
            train_logs[name] = sum(train_logs_out) / len(train_logs_out)

        valid_logs = val_epoch.run(val_loader, desc=desc)
        # valid_logs_out = [None for _ in range(torch.distributed.get_world_size())]
        # torch.distributed.all_gather_object(valid_logs_out, valid_logs)
        # valid_lgs = {}
        # for name in list(valid_logs):
        #     valid_lgs[name] = list(itertools.chain(*[x[name] for x in valid_logs_out]))
        # valid_logs = valid_lgs
        for name in list(valid_logs):
            valid_logs_out = [None for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather_object(valid_logs_out, valid_logs[name])
            valid_logs[name] = valid_logs_out
            if args.city is not None:
                if isinstance(valid_logs[name][0], list):
                    valid_logs[name] = [list(itertools.chain(*valid_logs[name]))]
                else:
                    valid_logs[name] = [sum(valid_logs[name])]

        if args.local_rank == 0:
            saver(
                os.path.join(args.checkpoint_dir, "./model_last.pth"),
            )
            save_jit(model, args, "model_last.pt")

        if scheduler is not None:
            scheduler.step()

        if args.local_rank == 0:
            rms_per_city = {tpe: {} for tpe in IMAGES + ERRORS}
            r2_per_city = {tpe: {} for tpe in IMAGES}
            score_per_city = {}
            for city_index, city in enumerate(CITIES):
                for tpe in IMAGES:
                    error_sum = valid_logs[f"{tpe}_error_sum"][city_index]
                    gt_sq_sum = valid_logs[f"{tpe}_gt_sq_sum"][city_index]
                    sm = valid_logs[f"{tpe}_sum"][city_index]
                    count = valid_logs[f"{tpe}_count"][city_index]
                    r2 = get_r2(error_sum, gt_sq_sum, sm, count)
                    r2_per_city[tpe][city] = r2

                    rms = get_rms(valid_logs[f"{tpe}_rms"][city_index])
                    rms_per_city[tpe][city] = rms

                for tpe in ERRORS:
                    rms = get_rms(valid_logs[f"{tpe}_errors"][city_index])
                    rms_per_city[tpe][city] = rms

                score_per_city[city] = (
                    r2_per_city["agl"][city] + r2_per_city["vflow"][city]
                ) / 2

            score = sum(score_per_city.values()) / len(score_per_city)

            if i > 0:
                for idx, param_group in enumerate(optimizer.param_groups):
                    lr = param_group["lr"]
                    summary_writer.add_scalar(
                        "group{}/lr".format(idx), float(lr), global_step=i
                    )

                summary_writer.add_scalars("train_loss/mse", train_logs, global_step=i)

                for tpe in rms_per_city:
                    summary_writer.add_scalars(
                        f"val_rms/{tpe}", rms_per_city[tpe], global_step=i
                    )

                for tpe in r2_per_city:
                    summary_writer.add_scalars(
                        f"val_r2/{tpe}", r2_per_city[tpe], global_step=i
                    )

                summary_writer.add_scalars("val/scores", score_per_city, global_step=i)
                summary_writer.add_scalar("val/score", score, global_step=i)

            if score > best_score:
                best_score = score
                saver(os.path.join(args.checkpoint_dir, "./model_best.pth"))

                save_jit(model, args, "model_best.pt")


    if args.local_rank == 0:
        summary_writer.close()


def test(args):
    if args.distributed:
        init_dist(args)

    torch.backends.cudnn.benchmark = True

    models = [
        torch.jit.load(p, map_location=torch.device(f"cuda:{args.gpu}")).eval()
        for p in args.model_pt
    ]

    assert len(models) == len(args.use_cities)

    if args.distributed:
        # By default, apex.parallel.DistributedDataParallel overlaps communication with
        # computation in the backward pass.
        # model = DDP(model)
        # delay_allreduce delays all communication to the end of the backward pass.
        # model = apex.parallel.convert_syncbn_model(model)
        models = [
            apex.parallel.DistributedDataParallel(model, delay_allreduce=True)
            for model in models
        ]

    df = pd.read_csv(args.test_path_df)
    metadata = pd.read_csv(Path(args.test_path_df).with_name("metadata.csv"))
    df = pd.merge(df, metadata, on="id")

    CITIES = ["ARG", "ATL", "JAX", "OMA"]
    if args.city is not None:
        CITIES = [args.city]
        df = df[df.city.isin(CITIES)].reset_index(drop=True)

    cities = ["ARG", "ATL", "JAX", "OMA"] if args.use_city else None

    MAX_HEIGTS = {city: 200.0 for city in CITIES}
    MAX_HEIGTS["ARG"] = 100.0

    if args.pl_dir is not None:
        test_dataset = DatasetPL(
            rgb_dir=args.test_sub_dir, pred_dir=args.pl_dir, args=args
        )
    else:
        test_dataset = Dataset(df, args=args, is_val=True, cities=cities, is_test=True)

    test_sampler = None
    if args.distributed:
        test_sampler = torch.utils.data.distributed.DistributedSampler(
            test_dataset,
            shuffle=False,
        )

    args.num_workers = min(max(args.num_workers, args.batch_size), 16)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2, #args.num_workers,
        sampler=test_sampler,
        pin_memory=False,
        persistent_workers=False, # True,
    )

    predictions_dir = Path(args.predictions_dir)

    if args.local_rank == 0:
        iterator = tqdm(
            total=len(test_loader),
            mininterval=2,
        )

    with torch.no_grad():
        preds = [
            torch.zeros(
                (args.tta * len(models), args.batch_size, 2),
                dtype=torch.float32,
                device="cuda",
            ),
            torch.zeros(
                (args.tta * len(models), args.batch_size, 1, 2048, 2048),
                dtype=torch.float32,
                device="cuda",
            ),
            torch.zeros(
                (
                    args.tta * len(models),
                    args.batch_size,
                ),
                dtype=torch.float32,
                device="cuda",
            ),
        ]
        for images, rgb_paths in test_loader:
            if args.use_city:
                images, city, gsd = images
                city = city.to("cuda", non_blocking=True)
                gsd = gsd.to("cuda", non_blocking=True)

            images = images.to("cuda", non_blocking=True)
            bs = images.size(0)

            [pred.zero_() for pred in preds]
            for model_idx, (use_city, model) in enumerate(zip(args.use_cities, models)):
                if use_city:
                    pred = model((images, city, gsd))
                else:
                    pred = model(images)

                pred = list(pred)
                if args.to_log:
                    pred[1] = torch.expm1(pred[1])

                preds[0][model_idx * args.tta, :bs] = pred[0]
                preds[1][model_idx * args.tta, :bs] = pred[1]
                preds[2][model_idx * args.tta, :bs] = pred[3]

                if args.tta > 1:  # horizontal flip
                    if use_city:
                        pred_tta = model((torch.flip(images, dims=[-1]), city, gsd))
                    else:
                        pred_tta = model(torch.flip(images, dims=[-1]))
                    xydir_pred_tta, agl_pred_tta, _, scale_pred_tta = pred_tta
                    xydir_pred_tta[:, 0] *= -1
                    agl_pred_tta = torch.flip(agl_pred_tta, dims=[-1])
                    if args.to_log:
                        agl_pred_tta = torch.expm1(agl_pred_tta)

                    preds[0][model_idx * args.tta + 1, :bs] = xydir_pred_tta
                    preds[1][model_idx * args.tta + 1, :bs] = agl_pred_tta
                    preds[2][model_idx * args.tta + 1, :bs] = scale_pred_tta

                if args.tta > 2:  # vertical flip
                    if use_city:
                        pred_tta = model((torch.flip(images, dims=[-2]), city, gsd))
                    else:
                        pred_tta = model(torch.flip(images, dims=[-2]))
                    xydir_pred_tta, agl_pred_tta, _, scale_pred_tta = pred_tta
                    xydir_pred_tta[:, 1] *= -1
                    agl_pred_tta = torch.flip(agl_pred_tta, dims=[-2])
                    if args.to_log:
                        agl_pred_tta = torch.expm1(agl_pred_tta)

                    preds[0][model_idx * args.tta + 2, :bs] = xydir_pred_tta
                    preds[1][model_idx * args.tta + 2, :bs] = agl_pred_tta
                    preds[2][model_idx * args.tta + 2, :bs] = scale_pred_tta

                if args.tta > 3:  # vertical+horizontal flip
                    if use_city:
                        pred_tta = model((torch.flip(images, dims=[-1, -2]), city, gsd))
                    else:
                        pred_tta = model(torch.flip(images, dims=[-1, -2]))
                    xydir_pred_tta, agl_pred_tta, _, scale_pred_tta = pred_tta
                    xydir_pred_tta *= -1
                    agl_pred_tta = torch.flip(agl_pred_tta, dims=[-1, -2])
                    if args.to_log:
                        agl_pred_tta = torch.expm1(agl_pred_tta)

                    preds[0][model_idx * args.tta + 3, :bs] = xydir_pred_tta
                    preds[1][model_idx * args.tta + 3, :bs] = agl_pred_tta
                    preds[2][model_idx * args.tta + 3, :bs] = scale_pred_tta

                if args.tta > 7:  # rotate90
                    images_rot90 = torch.rot90(images, k=1, dims=[-2, -1])

                    if use_city:
                        pred_tta = model((images_rot90, city, gsd))
                    else:
                        pred_tta = model(images_rot90)
                    xydir_pred_tta, agl_pred_tta, _, scale_pred_tta = pred_tta
                    xydir_pred_tta = torch.stack(
                        [-xydir_pred_tta[:, 1], xydir_pred_tta[:, 0]], dim=1
                    )
                    agl_pred_tta = torch.rot90(agl_pred_tta, k=-1, dims=[-2, -1])

                    pred[0] += xydir_pred_tta
                    pred[1] += agl_pred_tta
                    pred[3] += scale_pred_tta

                    # vertical flip
                    if use_city:
                        pred_tta = model(
                            (torch.flip(images_rot90, dims=[-1]), city, gsd)
                        )
                    else:
                        pred_tta = model(torch.flip(images_rot90, dims=[-1]))
                    xydir_pred_tta, agl_pred_tta, _, scale_pred_tta = pred_tta
                    xydir_pred_tta[:, 0] *= -1
                    agl_pred_tta = torch.flip(agl_pred_tta, dims=[-1])
                    xydir_pred_tta = torch.stack(
                        [-xydir_pred_tta[:, 1], xydir_pred_tta[:, 0]], dim=1
                    )
                    agl_pred_tta = torch.rot90(agl_pred_tta, k=-1, dims=[-2, -1])

                    pred[0] += xydir_pred_tta
                    pred[1] += agl_pred_tta
                    pred[3] += scale_pred_tta

                    # horizontal flip
                    if use_city:
                        pred_tta = model(
                            (torch.flip(images_rot90, dims=[-2]), city, gsd)
                        )
                    else:
                        pred_tta = model(torch.flip(images_rot90, dims=[-2]))
                    xydir_pred_tta, agl_pred_tta, _, scale_pred_tta = pred_tta
                    xydir_pred_tta[:, 1] *= -1
                    agl_pred_tta = torch.flip(agl_pred_tta, dims=[-2])
                    xydir_pred_tta = torch.stack(
                        [-xydir_pred_tta[:, 1], xydir_pred_tta[:, 0]], dim=1
                    )
                    agl_pred_tta = torch.rot90(agl_pred_tta, k=-1, dims=[-2, -1])

                    pred[0] += xydir_pred_tta
                    pred[1] += agl_pred_tta
                    pred[3] += scale_pred_tta

                    # vertical+horizontal flip
                    if use_city:
                        pred_tta = model(
                            (torch.flip(images_rot90, dims=[-1, -2]), city, gsd)
                        )
                    else:
                        pred_tta = model(torch.flip(images_rot90, dims=[-1, -2]))
                    xydir_pred_tta, agl_pred_tta, _, scale_pred_tta = pred_tta
                    xydir_pred_tta *= -1
                    agl_pred_tta = torch.flip(agl_pred_tta, dims=[-1, -2])
                    xydir_pred_tta = torch.stack(
                        [-xydir_pred_tta[:, 1], xydir_pred_tta[:, 0]], dim=1
                    )
                    agl_pred_tta = torch.rot90(agl_pred_tta, k=-1, dims=[-2, -1])

                    pred[0] += xydir_pred_tta
                    pred[1] += agl_pred_tta
                    pred[3] += scale_pred_tta

            pred = [
                torch.mean(preds[0][:, :bs], dim=0),
                torch.mean(preds[1][:, :bs], dim=0),
                torch.mean(preds[2][:, :bs], dim=0),
            ]

            torch.cuda.synchronize()

            numpy_preds = []
            for i in range(len(pred)):
                numpy_preds.append(pred[i].cpu().numpy())

            xydir_pred, agl_pred, scale_pred = numpy_preds

            if scale_pred.ndim == 0:
                scale_pred = np.expand_dims(scale_pred, axis=0)

            for batch_ind in range(agl_pred.shape[0]):
                # vflow pred
                angle = np.arctan2(xydir_pred[batch_ind][0], xydir_pred[batch_ind][1])
                vflow_data = {
                    "scale": np.float64(scale_pred[batch_ind]),  # upsample
                    "angle": np.float64(angle),
                }

                rgb_path = predictions_dir / Path(rgb_paths[batch_ind]).name

                # agl pred
                curr_agl_pred = agl_pred[batch_ind, 0, :, :]
                # curr_agl_pred[curr_agl_pred < 0] = 0
                curr_agl_pred = np.clip(
                    curr_agl_pred, 0.0, MAX_HEIGTS[rgb_path.stem.split("_")[0]]
                )
                agl_resized = curr_agl_pred

                # save
                agl_path = rgb_path.with_name(
                    rgb_path.name.replace("_RGB", "_AGL")
                ).with_suffix(".tif")
                vflow_path = rgb_path.with_name(
                    rgb_path.name.replace("_RGB", "_VFLOW")
                ).with_suffix(".json")

                json.dump(vflow_data, vflow_path.open("w"))
                save_image(agl_path, agl_resized)  # save_image assumes units of meters

            if args.local_rank == 0:
                iterator.update()

    torch.distributed.barrier()

    # creates new dir predictions_dir_con
    if args.local_rank == 0:
        iterator.close()

    if args.convert_predictions_to_cm_and_compress:
        convert_and_compress_prediction_dir(
            predictions_dir=predictions_dir,
            local_rank=args.local_rank,
            n_ranks=torch.distributed.get_world_size(),
        )


def build_model(args):
    model = UnetVFLOW(
        args.backbone, encoder_weights=args.encoder_weights, use_city=args.use_city, to_log=args.to_log,
        model_type=args.model_type,
    )
    return model


def build_optimizer(parameters, args):
    if args.optim.lower() == "fusedadam":
        optimizer = apex.optimizers.FusedAdam(
            parameters,
            adam_w_mode=True,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
    elif args.optim.lower() == "fusedsgd":
        optimizer = apex.optimizers.FusedSGD(
            parameters,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
    elif args.optim.lower() == "sgd":
        optimizer = torch.optim.SGD(
            parameters,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
    else:
        raise NotImplementedError(f"not yet implemented {args.optim}")

    return optimizer


def build_scheduler(optimizer, args):
    scheduler = None

    if args.scheduler.lower() == "cosa":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.T_max, eta_min=max(args.learning_rate * 1e-2, 1e-7)
        )
    elif args.scheduler.lower() == "cosawr":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=5 * args.T_max,
            T_mult=2,  # 1.41421,
            eta_min=max(args.learning_rate * 1e-2, 1e-7),
        )
    else:
        print("No scheduler")

    return scheduler
