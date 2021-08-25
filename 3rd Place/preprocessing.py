from pathlib import Path
import argparse
import json

from sklearn.model_selection import train_test_split
import lightgbm as lgb
import numpy as np
import pandas as pd

from utilities.downsample_images import downsample_images
from utils import timer


TRAIN_DIR = "data/train"
CONVERTED_DIR = "data/train-orig-resolution"

FN_GEOPOSE_TST = "data/geopose_test.csv"
FN_GEOPOSE_TRN = "data/geopose_train.csv"
FN_METADATA = "data/metadata.csv"

OUTPUT_LGBM_SCALE = "data/working/lgbm_scale.csv"


def lgbm_scale():
    df = pd.read_csv(FN_METADATA)

    rows = []
    for p in Path("data/train/").glob("*_VFLOW.json"):
        data = json.load(p.open("r"))
        r = data
        r["id"] = p.name.split("_")[1]
        # r["city"] = p.name.split("_")[0]
        rows.append(r)

    df = df.merge(pd.DataFrame(rows)[["id", "scale"]], how="left", on="id")
    df["city_lbl"] = df["city"].factorize(sort=True)[0]
    df["is_test"] = df["scale"].isna()

    X = df[~df["is_test"]][["city_lbl", "gsd"]].values
    y = df[~df["is_test"]]["scale"].values
    X_test = df[df["is_test"]][["city_lbl", "gsd"]].values
    clf = lgb.LGBMRegressor()
    clf.fit(X, y)
    df.loc[df[df["is_test"]].index, "scale"] = clf.predict(X_test)
    df[["id", "city", "gsd", "scale", "city_lbl", "is_test"]].to_csv(
        OUTPUT_LGBM_SCALE,
        index=False)


def filecheck():
    assert Path(FN_GEOPOSE_TRN).exists()
    assert Path(FN_GEOPOSE_TST).exists()
    assert Path(FN_METADATA).exists()
    assert Path("data/train/ARG_ACffgQ_AGL.tif").exists()
    assert Path("data/test/ARG_AHTGbG_RGB.j2k").exists()

    Path("data/working/logs").mkdir(exist_ok=True, parents=True)
    Path("data/working/models").mkdir(exist_ok=True, parents=True)
    Path("data/working/preds").mkdir(exist_ok=True, parents=True)


def convert_images():
    with timer("convert images"):
        downsample_images(argparse.Namespace(
            nan_placeholder=65535,
            indir=TRAIN_DIR,
            outdir=CONVERTED_DIR,
            unit="cm",
            rgb_suffix="j2k",
        ), downsample=1)


def split_train_data():
    train_dir = Path(CONVERTED_DIR) / "train"
    valid_dir = Path(CONVERTED_DIR) / "valid"
    trainval_dir = Path(CONVERTED_DIR) / "trainval"
    train_dir.mkdir()
    valid_dir.mkdir()
    trainval_dir.mkdir()
    image_dir = Path(CONVERTED_DIR)

    # symlink RGB.tif, VFLOW.json and AGL.tif
    city_names = ["ARG", "ATL", "JAX", "OMA"]
    for city_name in city_names:
        image_path_list = list(sorted(image_dir.glob(f"{city_name}_*_RGB.tif")))
        assert len(image_path_list) > 0
        np.random.seed(seed=11)
        np.random.shuffle(image_path_list)

        # Select 4 files for validation.
        for target_path in image_path_list[:4]:
            rgb_filename = target_path.name
            vflow_filename = target_path.name.replace("_RGB.tif", "_VFLOW.json")
            agl_filename = target_path.name.replace("_RGB.tif", "_AGL.tif")

            for filename in [rgb_filename, vflow_filename, agl_filename]:
                (valid_dir / filename).symlink_to(Path("../") / filename)

        for target_path in image_path_list[4:]:
            rgb_filename = target_path.name
            vflow_filename = target_path.name.replace("_RGB.tif", "_VFLOW.json")
            agl_filename = target_path.name.replace("_RGB.tif", "_AGL.tif")

            for filename in [rgb_filename, vflow_filename, agl_filename]:
                (train_dir / filename).symlink_to(Path("../") / filename)

        for target_path in image_path_list:
            rgb_filename = target_path.name
            vflow_filename = target_path.name.replace("_RGB.tif", "_VFLOW.json")
            agl_filename = target_path.name.replace("_RGB.tif", "_AGL.tif")

            for filename in [rgb_filename, vflow_filename, agl_filename]:
                (trainval_dir / filename).symlink_to(Path("../") / filename)


if __name__ == "__main__":
    filecheck()
    convert_images()
    split_train_data()
    lgbm_scale()
