import json
import os
import random
import traceback
from pathlib import Path

import albumentations as A
import numpy as np
import pandas as pd
from osgeo import gdal
from torch.utils.data import Dataset

from utilities.augmentation_vflow import augment_vflow

RNG = np.random.RandomState(777)

CITIES = ["ARG", "ATL", "JAX", "OMA"]

UNITS_PER_METER_CONVERSION_FACTORS = {"cm": 100.0, "m": 1.0}

agl_mean_by_city = {'ARG': 3.86079083107182, 'ATL': 8.43988072861438, 'JAX': 4.429840689418792,
                    'OMA': 2.2688709544387238}
mag_mean_by_city = {'ARG': 4.069477072669801, 'ATL': 5.453358976309836, 'JAX': 4.344437881497266,
                    'OMA': 2.27878907341509}

transforms = [
    A.RandomBrightnessContrast(p=0.5),
    A.RandomGamma(p=0.3)
]
transforms =  A.Compose(transforms)

def load_image(
        image_path,
        nan_placeholder: int,
        unit: str,
        dtype_out="float32",
        units_per_meter_conversion_factors=UNITS_PER_METER_CONVERSION_FACTORS,
):
    image_path = Path(image_path)
    if not image_path.exists():
        return None
    image = gdal.Open(str(image_path))
    image = image.ReadAsArray()

    # convert AGL units and fill nan placeholder with nan
    if "AGL" in image_path.name:
        image = image.astype(dtype_out)
        np.putmask(image, image == nan_placeholder, np.nan)
        # e.g., (cm) / (cm / m) = m
        units_per_meter = units_per_meter_conversion_factors[unit]
        image = (image / units_per_meter).astype(dtype_out)

    # transpose if RGB
    if len(image.shape) == 3:
        image = np.transpose(image, [1, 2, 0])

    return image


def load_vflow(
        vflow_path,
        agl,
        unit: str,
        dtype_out="float32",
        units_per_meter_conversion_factors=UNITS_PER_METER_CONVERSION_FACTORS,
        return_vflow_pred_mat=False,
):
    vflow_path = Path(vflow_path)
    vflow_data = json.load(vflow_path.open("r"))

    # e.g., (pixels / cm) * (cm / m) = (pixels / m)
    units_per_meter = units_per_meter_conversion_factors[unit]
    vflow_data["scale"] = vflow_data["scale"] * units_per_meter

    xdir, ydir = np.sin(vflow_data["angle"]), np.cos(vflow_data["angle"])
    mag = agl * vflow_data["scale"]

    vflow_items = [mag.astype(dtype_out), xdir.astype(dtype_out), ydir.astype(dtype_out), vflow_data]

    if return_vflow_pred_mat:
        vflow = np.zeros((agl.shape[0], agl.shape[1], 2))
        vflow[:, :, 0] = mag * xdir
        vflow[:, :, 1] = mag * ydir
        vflow_items.insert(0, vflow)

    return vflow_items


class GeoposeDataset(Dataset):
    def __init__(
            self,
            mode: str,
            dataset_dir: str,
            folds_csv: str,
            fold: int,
            crop_size=640,
            unit: str = "m",
            nan_placeholder: int = 65535,
            multiplier=1,
            rng=RNG,
    ):
        self.dataset_dir = dataset_dir
        self.crop_size = crop_size
        self.is_val = mode == "val"
        self.is_train = mode == "train"
        self.rng = rng
        self.unit = unit
        self.nan_placeholder = nan_placeholder
        df = pd.read_csv(folds_csv)
        if self.is_val:
            df = df[df.fold == fold]
        else:
            df = df[df.fold != fold]

        self.names = df[["city", "id"]].values.tolist()
        if self.is_train:
            self.names = self.names * multiplier
        self.all_names = df[["city", "id"]].values
        self.cities = df.city.values
        self.weights = np.ones((len(self.names, )))

    def __getitem__(self, i):
        city, file_id = self.names[i]
        try:
            rgb_path = os.path.join(self.dataset_dir, f"{city}_{file_id}_RGB.tif")
            agl_path = os.path.join(self.dataset_dir, f"{city}_{file_id}_AGL.tif")
            vflow_path = os.path.join(self.dataset_dir, f"{city}_{file_id}_VFLOW.json")
            image = load_image(rgb_path, self.nan_placeholder, self.unit)
            agl = load_image(agl_path, self.nan_placeholder, self.unit)
            mag, xdir, ydir, vflow_data = load_vflow(vflow_path, agl, unit=self.unit)
            scale = vflow_data["scale"]
            if self.is_train:
                image = transforms(image=image)["image"]
                image, mag, xdir, ydir, agl, scale = augment_vflow(
                    image,
                    mag,
                    xdir,
                    ydir,
                    vflow_data["angle"],
                    vflow_data["scale"],
                    agl=agl,
                )
            xdir = np.float32(xdir)
            ydir = np.float32(ydir)
            mag = mag.astype("float32")
            agl = agl.astype("float32")
            scale = np.float32(scale)

            xydir = np.array([xdir, ydir])

            crop_size = self.crop_size
            height = image.shape[0]
            tries = 0
            while tries < 10:
                y = random.randint(0, height - crop_size - 1)
                x = random.randint(0, height - crop_size - 1)
                if not self.is_val:
                    tries += 1
                    nan_mask = np.isnan(agl[y: y + crop_size, x:x + crop_size])
                    if np.count_nonzero(nan_mask[~nan_mask]) < 1024 and tries < 9:
                        continue

                    agl = agl[y: y + crop_size, x:x + crop_size]
                    image = image[y: y + crop_size, x:x + crop_size]
                    mag = mag[y: y + crop_size, x:x + crop_size]
                    break
                else:
                    break

            image = (np.transpose(image, (2, 0, 1)) / 255. - 0.5) * 2

            city_ohe = np.zeros((4,))
            city_ohe[CITIES.index(city)] = 1
            mag_target_mean = np.array([mag_mean_by_city[city]])
            agl_target_mean = np.array([agl_mean_by_city[city]])
            return {"image": image, "xydir": xydir, "agl": agl, "mag": mag, "scale": scale, "city_ohe": city_ohe,
                    "city": city, "mag_target_mean": mag_target_mean, "agl_target_mean": agl_target_mean,
                    "name": file_id}
        except:
            traceback.print_exc()
            return self.__getitem__(random.randint(0, len(self.names) - 1))

    def __len__(self):
        return len(self.names)

    def set_city(self, city: str):
        self.names = self.all_names[self.cities == city]