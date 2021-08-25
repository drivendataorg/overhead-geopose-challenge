import json
import math
import os.path
from typing import Optional, List, Callable, Tuple

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from pytorch_toolbelt.datasets import (
    mask_to_bce_target,
    INPUT_INDEX_KEY,
    INPUT_IMAGE_ID_KEY,
    INPUT_IMAGE_KEY,
    name_for_stride,
)
from pytorch_toolbelt.utils import fs, image_to_tensor
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from torch import Tensor
from torch.utils.data import Dataset, Sampler

__all__ = [
    "scalar_angle2vector",
    "AGL_IGNORE_VALUE",
    "DATASET_MEAN",
    "DATASET_STD",
    "read_bgr_opencv",
    "read_agl_opencv",
    "INPUT_GROUND_SAMPLE_DISTANCE",
    "INPUT_PATCH_CITY_NAME",
    "OUTPUT_AGL_MASK",
    "OUTPUT_DENSE_VFLOW_DIRECTION",
    "OUTPUT_DENSE_VFLOW_SCALE",
    "OUTPUT_VFLOW_ANGLE",
    "OUTPUT_VFLOW_DIRECTION",
    "OUTPUT_VFLOW_SCALE",
    "TARGET_VFLOW_SCALE",
    "OverheadGeoposeDataModule",
    "OverheadGeoposeDataset",
    "TARGET_AGL_MASK",
    "TARGET_VFLOW_ANGLE",
    "TARGET_VFLOW_DIRECTION",
    "OUTPUT_MAGNITUDE_MASK",
    "TARGET_MAGNITUDE_MASK",
    "read_agl_pil",
    "read_rgb_pil",
    "compute_vflow_np",
    "compute_vflow_torch",
    "tensor_angle2vector",
    "tensor_vector2angle",
]

from torch.utils.data.dataloader import default_collate

from geopose.augmentations import get_detector_augmentations, get_train_crops_transform, GeoDownsample

INPUT_PATCH_CITY_NAME = "INPUT_CITY_NAME"

# Ground sample distance (GSD) in meters per pixel. GSD is the average pixel size in meters. Units - meters/pixel
INPUT_GROUND_SAMPLE_DISTANCE = "INPUT_GROUND_SAMPLE_DISTANCE"

TARGET_AGL_MASK = "TARGET_AGL_MASK"  # Units in meters
TARGET_MAGNITUDE_MASK = "TARGET_MAGNITUDE_MASK"  # Units in pixels

TARGET_VFLOW_ANGLE = "TARGET_VFLOW_ANGLE"


TARGET_VFLOW_SCALE = "TARGET_VFLOW_SCALE"  # Scale is in pixels/meter
TARGET_VFLOW_DIRECTION = "TARGET_VFLOW_DIRECTION"


OUTPUT_VFLOW_SCALE = "OUTPUT_VFLOW_SCALE"  # Scale is in pixels/meter

# Angle is in radians, starting at 0 from the negative y axis and increasing counterclockwise.
OUTPUT_VFLOW_ANGLE = "OUTPUT_VFLOW_ANGLE"
OUTPUT_VFLOW_DIRECTION = "OUTPUT_VFLOW_DIRECTION"

OUTPUT_AGL_MASK = "OUTPUT_AGL_MASK"
OUTPUT_MAGNITUDE_MASK = "OUTPUT_MAGNITUDE_MASK"

OUTPUT_DENSE_VFLOW_DIRECTION = "OUTPUT_DENSE_VFLOW_DIRECTION"
OUTPUT_DENSE_VFLOW_SCALE = "OUTPUT_DENSE_VFLOW_SCALE"

DATASET_MEAN = (0.485, 0.456, 0.406)
DATASET_STD = (0.229, 0.224, 0.225)

# Note: Many AGL image arrays contain missing values.
# These pixels represent locations where the LiDAR that was used to assess true height did not get any data.
# In the training AGLs, 65535 is used as a placeholder for NaNs.
# You do not have to predict height for pixels with missing true height values -
# pixels that are missing in the ground truth AGLs will be excluded from performance evaluation.
AGL_IGNORE_VALUE = 65535


def read_agl_pil(fname: str):
    image = Image.open(fname)
    image = np.array(image)
    return image


def read_rgb_pil(image_fname) -> np.ndarray:
    image = Image.open(image_fname)
    image = np.array(image)
    return image


def read_agl_opencv(fname: str):
    return cv2.imread(fname, cv2.IMREAD_UNCHANGED)


def read_bgr_opencv(fname: str):
    return cv2.imread(fname)


def scalar_angle2vector(angle: float) -> Tuple[float, float]:
    # Order of sin/cos is correct and matches with coordinate system used in competition
    return math.sin(angle), math.cos(angle)


def scalar_vector2angle(direction: Tuple[float, float]) -> float:
    angle = math.atan2(direction[0], direction[1])
    if angle < 0:
        angle += math.pi * 2
    return angle


def tensor_angle2vector(angle) -> Tensor:
    # Order of sin/cos is correct and matches with coordinate system used in competition
    return torch.stack([torch.sin(angle), torch.cos(angle)], dim=-1)


def tensor_vector2angle(direction: Tensor) -> Tensor:
    angle = torch.atan2(direction[:, 0:1], direction[:, 1:2])
    mask = angle < 0
    angle[mask] = (angle + 2 * math.pi)[mask]
    return angle


def compute_vflow_np(
    agl: np.ndarray,
    scale: float,
    angle: float,
):
    agl_ignore = agl == AGL_IGNORE_VALUE

    # Order of sin/cos is correct and matches with coordinate system used in competition
    xdir, ydir = np.sin(angle), np.cos(angle)

    mag = (agl * scale).astype(np.float32)
    mag[agl_ignore] = AGL_IGNORE_VALUE

    vflow = np.zeros((agl.shape[0], agl.shape[1], 2), dtype=np.float32)
    vflow[:, :, 0] = mag * xdir
    vflow[:, :, 1] = mag * ydir
    return vflow, mag, xdir, ydir


def compute_vflow_torch(
    agl: Tensor,
    scale: Tensor,
    angle: Tensor,
):
    agl = agl.float()
    agl_ignore = agl == AGL_IGNORE_VALUE
    # Order of sin/cos is correct and matches with coordinate system used in competition
    xdir, ydir = torch.sin(angle), torch.cos(angle)

    mag = (agl * scale).float()
    mag[agl_ignore] = AGL_IGNORE_VALUE

    vflow = torch.zeros((2, agl.shape[1], agl.shape[2]), dtype=torch.float32)
    vflow[0, :, :] = mag * xdir
    vflow[1, :, :] = mag * ydir
    return vflow, mag, xdir, ydir


class OverheadGeoposeDataset(Dataset):
    def __init__(
        self,
        image_filenames: List[str],
        gsd: List[float],
        target_agl_filenames: Optional[List[str]],
        target_vflow_scale: Optional[List[float]],
        target_vflow_angle: Optional[List[float]],
        transform: A.Compose,
        read_image_fn: Callable = read_bgr_opencv,
        read_mask_fn: Callable = read_agl_opencv,
        need_weight_mask=False,
        need_supervision_masks=False,
        make_mask_target_fn: Callable = mask_to_bce_target,
        city_ids: Optional[List[str]] = None,
        downsample=1,
    ):
        if target_agl_filenames is not None:
            if (
                len(image_filenames) != len(target_agl_filenames)
                or len(target_vflow_angle) != len(target_agl_filenames)
                or len(target_vflow_scale) != len(target_agl_filenames)
            ):
                raise ValueError("Number of images does not corresponds to number of targets")

        if city_ids is None:
            self.city_ids = [fs.id_from_fname(fname) for fname in image_filenames]
        else:
            self.city_ids = city_ids

        self.need_weight_mask = need_weight_mask
        self.need_supervision_masks = need_supervision_masks

        self.images = image_filenames
        self.masks = target_agl_filenames
        self.gsd = gsd
        self.vflow_scales = target_vflow_scale
        self.vflow_angles = target_vflow_angle
        self.read_image = read_image_fn
        self.read_mask = read_mask_fn
        self.downsample = downsample

        self.transform = transform
        self.make_target = make_mask_target_fn

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.read_image(self.images[index])
        data = {"image": image, "gsd": self.gsd[index]}

        has_targets = self.masks is not None
        if has_targets:
            mask_centimeters = self.read_mask(self.masks[index])
            mask_ignore = mask_centimeters == AGL_IGNORE_VALUE

            mask = mask_centimeters.astype(np.float32)
            mask[mask_ignore] = float("nan")

            data["mask"] = mask
            data["vflow_scale"] = self.vflow_scales[index]
            data["vflow_angle"] = self.vflow_angles[index]

        data = self.transform(**data)

        sample = {
            INPUT_INDEX_KEY: index,
            INPUT_IMAGE_ID_KEY: fs.id_from_fname(self.images[index]),
            INPUT_IMAGE_KEY: image_to_tensor(data["image"]),
            INPUT_PATCH_CITY_NAME: self.city_ids[index],
            INPUT_GROUND_SAMPLE_DISTANCE: torch.tensor([data["gsd"]], dtype=torch.float32),
        }

        if has_targets:
            agl_centimeters = data["mask"]
            valid_mask = np.isfinite(agl_centimeters)
            ignore_mask = ~valid_mask

            agl_meters = agl_centimeters / 100.0
            agl_meters[ignore_mask] = AGL_IGNORE_VALUE

            scale_pixels_per_cm = data["vflow_scale"]
            scale_pixels_per_meter = scale_pixels_per_cm * 100

            angle_in_radians = data["vflow_angle"]

            mag = (agl_centimeters * scale_pixels_per_cm).astype(np.float32)
            mag[ignore_mask] = AGL_IGNORE_VALUE

            sample[TARGET_AGL_MASK] = self.make_target(agl_meters).float()
            sample[TARGET_MAGNITUDE_MASK] = self.make_target(mag).float()
            sample[TARGET_VFLOW_SCALE] = torch.tensor([scale_pixels_per_meter], dtype=torch.float32)
            sample[TARGET_VFLOW_ANGLE] = torch.tensor([angle_in_radians], dtype=torch.float32)
            sample[TARGET_VFLOW_DIRECTION] = np.array(scalar_angle2vector(angle_in_radians), dtype=np.float32)
            if self.need_supervision_masks:
                for i in range(1, 6):
                    stride = 2 ** i
                    agl_meters = cv2.resize(
                        agl_meters, dsize=(agl_meters.shape[1] // 2, agl_meters.shape[0] // 2), interpolation=cv2.INTER_CUBIC
                    )
                    sample[name_for_stride(TARGET_AGL_MASK, stride)] = self.make_target(agl_meters)

        return sample

    def get_collate_fn(self):
        return default_collate


class OverheadGeoposeDataModule:
    @classmethod
    def load_dataset(cls, data_dir) -> pd.DataFrame:
        metadata = pd.read_csv(os.path.join(data_dir, "metadata.csv"))

        dataset = pd.read_csv(os.path.join(data_dir, "geopose_train.csv"))
        dataset = pd.merge(left=dataset, right=metadata, on="id", how="left")

        # id,agl,json,rgb
        dataset["agl_fname"] = dataset["agl"].apply(lambda x: os.path.join(data_dir, "train", x))
        dataset["rgb_fname"] = dataset["rgb"].apply(lambda x: fs.change_extension(os.path.join(data_dir, "train", x), ".png"))
        dataset["json_fname"] = dataset["json"].apply(lambda x: os.path.join(data_dir, "train", x))
        dataset["json_data"] = dataset["json_fname"].apply(lambda x: json.load(open(x, "r")))
        dataset["scale"] = dataset["json_data"].apply(lambda x: x["scale"])
        dataset["angle"] = dataset["json_data"].apply(lambda x: x["angle"])

        del dataset["json_data"]
        return dataset

    @classmethod
    def get_train_valid_split(cls, dataset: pd.DataFrame, fold_split: str, fold: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if fold_split == "region":
            dataset["fold"] = LabelEncoder().fit_transform(dataset["city"])
        elif fold_split == "region_stratified":
            dataset["fold"] = -1
            kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
            for fold_index, (train_index, valid_index) in enumerate(kfold.split(dataset, dataset.city)):
                dataset.loc[valid_index, "fold"] = fold_index

        return dataset[dataset.fold != fold].copy(), dataset[dataset.fold == fold].copy()

    @classmethod
    def get_normalization(cls) -> A.Normalize:
        return A.Normalize(DATASET_MEAN, DATASET_STD)

    @classmethod
    def get_training_dataset(
        cls, train_df: pd.DataFrame, train_image_size: Tuple[int, int], random_sized_crop: bool, augmentations: str, downsample=1
    ) -> Tuple[OverheadGeoposeDataset, Optional[Sampler]]:

        # If we are using downsampling, we should make effective crop size N times larger
        train_image_size_wrt_downsample = (train_image_size[0] * downsample, train_image_size[1] * downsample)

        train_crops = get_train_crops_transform(
            train_image_size=train_image_size_wrt_downsample, random_sized_crop=random_sized_crop
        )

        train_augmentations = get_detector_augmentations(
            train_image_size=train_image_size_wrt_downsample, level=augmentations, ignore_label=float("nan")
        )

        train_sampler = None
        train_transform = train_crops + train_augmentations + [cls.get_normalization()]
        if downsample > 1:
            train_transform.append(GeoDownsample(downsample))

        train_ds = OverheadGeoposeDataset(
            image_filenames=train_df.rgb_fname.tolist(),
            gsd=train_df.gsd.tolist(),
            target_agl_filenames=train_df.agl_fname.tolist(),
            target_vflow_angle=train_df.angle.tolist(),
            target_vflow_scale=train_df.scale.tolist(),
            city_ids=train_df.city.tolist(),
            transform=A.Compose(train_transform),
        )

        return train_ds, train_sampler

    @classmethod
    def get_validation_dataset(
        cls, valid_df: pd.DataFrame, valid_image_size: Tuple[int, int] = None, downsample=1
    ) -> OverheadGeoposeDataset:
        valid_transform = [cls.get_normalization()]
        if downsample > 1:
            valid_transform.append(GeoDownsample(downsample))

        valid_ds = OverheadGeoposeDataset(
            image_filenames=valid_df.rgb_fname.tolist(),
            gsd=valid_df.gsd.tolist(),
            target_agl_filenames=valid_df.agl_fname.tolist(),
            target_vflow_angle=valid_df.angle.tolist(),
            target_vflow_scale=valid_df.scale.tolist(),
            transform=A.Compose(valid_transform),
            city_ids=valid_df.city.tolist(),
        )
        return valid_ds

    @classmethod
    def get_test_dataset(cls, data_dir, downsample=1) -> OverheadGeoposeDataset:
        dataset = pd.read_csv(os.path.join(data_dir, "geopose_test.csv"))
        metadata = pd.read_csv(os.path.join(data_dir, "metadata.csv"))
        dataset = pd.merge(left=dataset, right=metadata, on="id", how="left")
        dataset["rgb_fname"] = dataset["rgb"].apply(lambda x: fs.change_extension(os.path.join(data_dir, "test", x), ".png"))

        valid_transform = [cls.get_normalization()]
        if downsample > 1:
            valid_transform.append(GeoDownsample(downsample))

        valid_ds = OverheadGeoposeDataset(
            image_filenames=dataset.rgb_fname.tolist(),
            gsd=dataset.gsd.tolist(),
            target_agl_filenames=None,
            target_vflow_angle=None,
            target_vflow_scale=None,
            transform=A.Compose(valid_transform),
            city_ids=dataset.city.tolist(),
        )
        return valid_ds
