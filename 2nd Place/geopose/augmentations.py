import math
from typing import Tuple, List

import albumentations as A
import cv2
import numpy as np

__all__ = [
    "get_light_augmentations",
    "get_medium_augmentations",
    "get_safe_augmentations",
    "get_train_crops_transform",
    "get_hard_augmentations",
    "get_detector_augmentations",
    "GeoRandomSizedCrop",
    "GeoHorizontalFlip",
    "GeoVerticalFlip",
    "GeoShiftScaleRotate",
    "GeoTranspose",
    "GeoRandomRotate90",
    "GeoDownsample",
]


class GeoDownsample(A.DualTransform):
    def __init__(self, downsample: int = 1, image_interpolation=cv2.INTER_CUBIC, mask_interpolation=cv2.INTER_LINEAR):
        super().__init__(p=1)
        self.downsample = downsample
        self.image_interpolation = image_interpolation
        self.mask_interpolation = mask_interpolation

    def apply(self, img, **kwargs):
        dst_size = img.shape[1] // self.downsample, img.shape[0] // self.downsample
        return cv2.resize(img, dsize=dst_size, interpolation=self.image_interpolation)

    def apply_to_mask(self, img, **params):
        dst_size = img.shape[1] // self.downsample, img.shape[0] // self.downsample
        return cv2.resize(img, dsize=dst_size, interpolation=self.mask_interpolation)


class GeoHorizontalFlip(A.HorizontalFlip):
    @property
    def targets(self):
        return {
            "image": self.apply,
            "mask": self.apply_to_mask,
            "masks": self.apply_to_masks,
            "bboxes": self.apply_to_bboxes,
            "keypoints": self.apply_to_keypoints,
            "vflow_angle": self.apply_to_vflow_angle,
        }

    def apply_to_vflow_angle(self, vflow_angle, factor=0, **params):
        direction = np.sin(vflow_angle), np.cos(vflow_angle)

        vflow_angle_transposed = math.atan2(-direction[0], direction[1])
        if vflow_angle_transposed < 0:
            vflow_angle_transposed += math.pi * 2
        return vflow_angle_transposed


class GeoVerticalFlip(A.VerticalFlip):
    @property
    def targets(self):
        return {
            "image": self.apply,
            "mask": self.apply_to_mask,
            "masks": self.apply_to_masks,
            "bboxes": self.apply_to_bboxes,
            "keypoints": self.apply_to_keypoints,
            "vflow_angle": self.apply_to_vflow_angle,
        }

    def apply_to_vflow_angle(self, vflow_angle, factor=0, **params):
        direction = np.sin(vflow_angle), np.cos(vflow_angle)
        vflow_angle_transposed = math.atan2(direction[0], -direction[1])
        if vflow_angle_transposed < 0:
            vflow_angle_transposed += math.pi * 2
        return vflow_angle_transposed


class GeoRandomRotate90(A.RandomRotate90):
    @property
    def targets(self):
        return {
            "image": self.apply,
            "mask": self.apply_to_mask,
            "masks": self.apply_to_masks,
            "bboxes": self.apply_to_bboxes,
            "keypoints": self.apply_to_keypoints,
            "vflow_angle": self.apply_to_vflow_angle,
        }

    def apply_to_vflow_angle(self, vflow_angle, factor=0, **params):
        vflow_angle = vflow_angle + math.pi * factor / 2.0  # + pi/2 * N times
        if vflow_angle > math.pi * 2:
            vflow_angle -= math.pi * 2
        if vflow_angle < 0:
            vflow_angle += math.pi * 2
        return vflow_angle


class GeoRandomSizedCrop(A.RandomSizedCrop):
    mask_interpolation = cv2.INTER_LINEAR

    @property
    def targets(self):
        return {
            "image": self.apply,
            "mask": self.apply_to_mask,
            "masks": self.apply_to_masks,
            "bboxes": self.apply_to_bboxes,
            "keypoints": self.apply_to_keypoints,
            "gsd": self.apply_to_gsd,
            "vflow_scale": self.apply_to_vflow_scale,
        }

    def apply_to_mask(self, img, crop_height=0, crop_width=0, h_start=0, w_start=0, interpolation=cv2.INTER_LINEAR, **params):
        crop = A.random_crop(img, crop_height, crop_width, h_start, w_start)
        return A.resize(crop, self.height, self.width, interpolation=self.mask_interpolation)

    def apply_to_gsd(self, gsd, crop_height=0, crop_width=0, h_start=0, w_start=0, rows=0, cols=0, **params):
        """
        Ground sample distance (GSD) in meters per pixel.
        GSD is the average pixel size in meters. Units - meters/pixel
        """
        scale_x = crop_width / self.width
        scale_y = crop_height / self.height
        assert math.fabs(scale_y - scale_x) < 1e-5

        scale = (scale_y + scale_x) * 0.5
        # When scale is < 1, crop patch is smaller than destination patch, so number of meters in pixel decreases
        # When scale is > 1, crop patch is larger than destination patch, so number of meters in pixel increases
        return gsd * scale

    def apply_to_vflow_scale(self, vflow_scale, crop_height=0, crop_width=0, h_start=0, w_start=0, rows=0, cols=0, **params):
        """
        Scale is in pixels/meter
        """

        # Here we compute scale in inverted way
        scale_x = self.width / crop_width
        scale_y = self.height / crop_height
        scale = (scale_y + scale_x) * 0.5

        assert math.fabs(scale_y - scale_x) < 1e-5
        return vflow_scale * scale


class GeoShiftScaleRotate(A.ShiftScaleRotate):
    mask_interpolation = cv2.INTER_LINEAR

    @property
    def targets(self):
        return {
            "image": self.apply,
            "mask": self.apply_to_mask,
            "masks": self.apply_to_masks,
            "bboxes": self.apply_to_bboxes,
            "keypoints": self.apply_to_keypoints,
            "gsd": self.apply_to_gsd,
            "vflow_scale": self.apply_to_vflow_scale,
            "vflow_angle": self.apply_to_vflow_angle,
        }

    def apply_to_mask(self, img, angle=0, scale=0, dx=0, dy=0, **params):
        return A.shift_scale_rotate(img, angle, scale, dx, dy, self.mask_interpolation, self.border_mode, self.mask_value)

    def apply_to_gsd(self, gsd, scale=0, **params):
        """
        Ground sample distance (GSD) in meters per pixel.
        GSD is the average pixel size in meters. Units - meters/pixel
        """

        # When scale is < 1, crop patch is smaller than destination patch, so number of meters in pixel decreases
        # When scale is > 1, crop patch is larger than destination patch, so number of meters in pixel increases
        return gsd * scale

    def apply_to_vflow_scale(self, vflow_scale, scale=0, **params):
        return vflow_scale / scale

    def apply_to_vflow_angle(self, vflow_angle, angle=0, **params):
        angle_rads = np.radians(angle)
        vflow_angle = vflow_angle + angle_rads

        if vflow_angle > math.pi * 2:
            vflow_angle -= math.pi * 2
        if vflow_angle < 0:
            vflow_angle += math.pi * 2
        return vflow_angle


class GeoTranspose(A.Transpose):
    @property
    def targets(self):
        return {
            "image": self.apply,
            "mask": self.apply_to_mask,
            "masks": self.apply_to_masks,
            "bboxes": self.apply_to_bboxes,
            "keypoints": self.apply_to_keypoints,
            "vflow_angle": self.apply_to_vflow_angle,
        }

    def apply_to_vflow_angle(self, vflow_angle, **params):
        x = math.cos(vflow_angle)
        y = math.sin(vflow_angle)
        vflow_angle_transposed = math.atan2(x, y)  # transpose
        if vflow_angle_transposed < 0:
            vflow_angle_transposed += math.pi * 2
        return vflow_angle_transposed


def get_train_crops_transform(train_image_size: Tuple[int, int], random_sized_crop=True) -> List[A.BasicTransform]:
    dst_height, dst_width = train_image_size

    if random_sized_crop:
        return [
            GeoRandomSizedCrop(
                min_max_height=(int(dst_height * 0.9), int(dst_height / 0.9)),
                width=dst_width,
                height=dst_height,
            )
        ]
    else:
        return [A.RandomCrop(dst_height, dst_width)]


def get_none_augmentations(train_image_size: Tuple[int, int], ignore_label: int) -> List[A.DualTransform]:
    return []


def get_d4_only_augmentations(train_image_size: Tuple[int, int], ignore_label: int) -> List[A.DualTransform]:
    return [
        GeoRandomRotate90(p=1),
        GeoTranspose(p=0.5),
    ]


def get_safe_augmentations(train_image_size: Tuple[int, int], ignore_label: int) -> List[A.DualTransform]:
    return [
        GeoRandomRotate90(p=1),
        GeoTranspose(p=0.5),
        #
        A.RandomGamma(),
        A.RandomToneCurve(),
        A.RandomBrightnessContrast(),
    ]


def get_light_augmentations(train_image_size: Tuple[int, int], ignore_label: int) -> List[A.DualTransform]:
    return [
        GeoRandomRotate90(p=1),
        GeoTranspose(p=0.5),
        #
        GeoShiftScaleRotate(
            scale_limit=0.1, rotate_limit=22, value=0, mask_value=ignore_label, border_mode=cv2.BORDER_CONSTANT, p=0.5
        ),
        A.OneOf(
            [
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1),
                A.CLAHE(),
                A.RandomToneCurve(),
                A.RandomGamma(),
            ],
            p=0.5,
        ),
        # More color changes
        A.OneOf([A.ISONoise(), A.GaussNoise(), A.MultiplicativeNoise()], p=0.1),
        A.OneOf(
            [
                A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10),
                A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=5, val_shift_limit=5),
                A.FancyPCA(),
            ],
            p=0.1,
        ),
    ]


def get_medium_augmentations(train_image_size: Tuple[int, int], ignore_label: int) -> List[A.DualTransform]:
    return [
        GeoRandomRotate90(p=1),
        GeoTranspose(p=0.5),
        #
        GeoShiftScaleRotate(
            scale_limit=0.1, rotate_limit=22, value=0, mask_value=ignore_label, border_mode=cv2.BORDER_CONSTANT, p=0.5
        ),
        A.OneOf(
            [
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1),
                A.CLAHE(),
                A.RandomToneCurve(),
                A.RandomGamma(),
            ],
            p=0.5,
        ),
        # More color changes
        A.OneOf(
            [A.ISONoise(), A.GaussNoise(), A.MultiplicativeNoise(), A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.5)], p=0.1
        ),
        A.OneOf(
            [
                A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10),
                A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=5, val_shift_limit=5),
                A.FancyPCA(),
            ],
            p=0.1,
        ),
        A.CoarseDropout(
            max_holes=2,
            min_height=8,
            max_height=128,
            min_width=8,
            max_width=128,
            mask_fill_value=ignore_label,
            fill_value=0,
            p=0.1,
        ),
    ]


def get_hard_augmentations(train_image_size: Tuple[int, int], ignore_label: int) -> List[A.DualTransform]:
    return [
        # D4
        A.RandomRotate90(p=1.0),
        A.Transpose(p=0.5),
        # Small color changes
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3),
        A.CLAHE(),
        A.RandomToneCurve(),
        # More color changes
        A.OneOf([A.ISONoise(), A.GaussNoise(), A.MultiplicativeNoise()]),
        A.OneOf([A.RGBShift(), A.HueSaturationValue(), A.FancyPCA()]),
        # Some affine
        A.OneOf(
            [
                A.ShiftScaleRotate(scale_limit=(-0.5, 0), rotate_limit=5, border_mode=cv2.BORDER_CONSTANT),
                A.ShiftScaleRotate(scale_limit=(0, 0.5), rotate_limit=5, border_mode=cv2.BORDER_CONSTANT),
                A.ShiftScaleRotate(scale_limit=(-0.75, 0), rotate_limit=5, border_mode=cv2.BORDER_CONSTANT),
                A.ShiftScaleRotate(scale_limit=(0, 1), rotate_limit=5, border_mode=cv2.BORDER_CONSTANT),
            ]
        ),
        # Image compression
        A.OneOf(
            [
                A.ImageCompression(quality_lower=70),
                A.Downscale(scale_min=0.5, scale_max=0.75, interpolation=cv2.INTER_NEAREST),
                A.Downscale(interpolation=cv2.INTER_LINEAR),
            ],
            p=0.3,
        ),
        A.OneOf([A.ISONoise(), A.GaussNoise(), A.MultiplicativeNoise()], p=0.3),
        A.OneOf([A.MotionBlur(blur_limit=(3, 5)), A.GaussianBlur(blur_limit=(3, 5)), A.Sharpen()], p=0.3),
    ]


def get_detector_augmentations(level: str, train_image_size: Tuple[int, int], ignore_label) -> List[A.DualTransform]:
    LEVELS = {
        "none": get_none_augmentations,
        "d4_only": get_d4_only_augmentations,
        "safe": get_safe_augmentations,
        "light": get_light_augmentations,
        "medium": get_medium_augmentations,
        "hard": get_hard_augmentations,
    }

    return LEVELS[level](train_image_size=train_image_size, ignore_label=ignore_label)
