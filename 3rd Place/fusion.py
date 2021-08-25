from pathlib import Path
import json
import math
import shutil
import tarfile

import numpy as np
import pandas as pd
import click
import tqdm
import cv2
from PIL import Image

from utils import timer
from imagesearch import (
    extract_train_features,
    extract_test_features,
    search_test_train_rev3,
)
from imagematch import (
    load_image,
    verify_test_train_rev3,
    match_test_train_rev3,
)

@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx):
    pass


@cli.command()
def search():
    # Extract features
    with timer("extract test features"):
        Path("data/working/search/test").mkdir(parents=True, exist_ok=True)
        extract_test_features()
    with timer("extract train features"):
        Path("data/working/search/train").mkdir(parents=True, exist_ok=True)
        extract_train_features()

    # Train-test image search
    with timer("search top100"):
        search_test_train_rev3()
        # Expected output files:
        assert Path("data/working/search/test_train_search_result_rev3.csv").exists()


@cli.command()
def match():
    with timer("quick match"):
        # Required working files:
        assert Path("data/working/search/test_train_search_result_rev3.csv").exists()
        verify_test_train_rev3()
        # Expected output files:
        assert Path("data/working/search/test_train_verify_result_rev3.csv").exists()

    with timer("match"):
        # Required working files:
        assert Path("data/working/search/test_train_verify_result_rev3.csv").exists()
        match_test_train_rev3()


@cli.command()
def fusion():
    # (check) Required working files:
    fusion_out_dir = Path("data/working/preds/fusion_sub/")
    sub_out_dir = Path("data/working/preds/ensemble_sub/")
    match_out_dir = Path("data/working/search/match")
    assert (sub_out_dir / "ATL_ABbbIw_VFLOW.json").exists()
    assert (match_out_dir / "ATL_ABbbIw_ATL_FKEOMU_matchMeta.json").exists()
    fusion_out_dir.mkdir(parents=True, exist_ok=True)

    df_match = get_matched_dataframe(match_out_dir, sub_out_dir)
    update_mapping = get_angle_mapping(df_match)

    files = list(sub_out_dir.glob("*_AGL.tif"))
    for path in tqdm.tqdm(files, total=len(files)):
        test_image_id = path.stem[:10]
        out_path = fusion_out_dir / f"{test_image_id}_AGL.tif"
        if out_path.exists():
            continue

        # AGL fusion
        agl, rgb_test, visible_mask_fusioned, visible_mask_nearangle_fusioned = fusion_agl(
            test_image_id,
            sub_out_dir,
            df_match,
            min_inlier=2000,
            min_height=40,
        )
        if agl is None:
            shutil.copy(path, out_path)
        else:
            imarray = np.round(agl * 100.0)
            assert imarray.min() >= 0
            im = Image.fromarray(imarray.astype("uint16"))
            im.save(str(out_path), "TIFF", compression="tiff_adobe_deflate")

        # Angle fusion
        if test_image_id not in update_mapping:
            orig_json_data = json.load(
                Path(str(path).replace("AGL.tif", "VFLOW.json")).open("r")
            )
            orig_json_data["angle"] = (orig_json_data["angle"] + math.pi * 2) % (
                math.pi * 2
            )
            json.dump(
                orig_json_data,
                Path(str(out_path).replace("AGL.tif", "VFLOW.json")).open("w")
            )
        else:
            orig_json_data = json.load(
                Path(str(path).replace("AGL.tif", "VFLOW.json")).open("r")
            )
            orig_json_data["angle"] = update_mapping[test_image_id]
            json.dump(
                orig_json_data,
                Path(str(out_path).replace("AGL.tif", "VFLOW.json")).open("w")
            )

    # Generate tar file
    submission_tar_path = "fusion_sub.tar.gz"
    with tarfile.open(submission_tar_path, "w:gz") as tar:
        files = list(fusion_out_dir.glob("*"))
        for file in tqdm.tqdm(files, total=len(files)):
            tar.add(file, arcname=file.name)


def get_matched_dataframe(match_out_dir, sub_out_dir):
    rows = []
    for p in match_out_dir.glob("*_matchMeta.json"):
        test_image_id = p.stem[:10]
        train_image_id = p.stem[11:21]
        test_json_data = json.load((sub_out_dir / Path(f"{test_image_id}_VFLOW.json")).open("r"))
        matched_json_data = json.load(Path(f"data/train/{train_image_id}_VFLOW.json").open("r"))

        json_data = json.load(p.open("r"))
        json_data.update({
            "path": str(p),
            "query": p.stem[:10],
            "hit": p.stem[11:21],
            "test_image_id": p.stem[:10],
            "train_image_id": p.stem[11:21],
            "inlier_count": json_data["inlier_count"],
            "pix_rmse": json_data["pix_rmse"],
            "test_angle": (test_json_data["angle"] + math.pi * 2) % (math.pi * 2),
            "test_scale": test_json_data["scale"],
            "matched_angle": matched_json_data["angle"],
            "matched_scale": matched_json_data["scale"],
        })
        rows.append(json_data)

    df_match = pd.DataFrame(rows).drop_duplicates(subset=["query", "hit"])[[
        "path",
        "query",
        "hit",
        "test_image_id",
        "train_image_id",
        "inlier_count",
        "pix_rmse",
        "test_angle",
        "test_scale",
        "matched_angle",
        "matched_scale",
    ]]
    return df_match


def get_angle_mapping(df):
    df["angle_diff"] = np.abs(df["test_angle"] - df["matched_angle"])
    df_agg = df[df["pix_rmse"] < 7.0].groupby("test_image_id").agg(
        inlier_count=("inlier_count", "first"),
        test_angle=("test_angle", "first"),
        test_scale=("test_scale", "first"),
        matched_angle_mean=("matched_angle", "mean"),
        matched_angle_max=("matched_angle", "max"),
        matched_angle_min=("matched_angle", "min"),
        matched_scale_min=("matched_scale", "min"),
        matched_scale_max=("matched_scale", "max"),
        matched_count=("matched_angle", "count"),
    ).reset_index()
    update_mapping = {
        r["test_image_id"]: r["matched_angle_mean"]
        for _, r in df_agg[["test_image_id", "matched_angle_mean"]].iterrows()
    }

    return update_mapping


def fusion_agl(test_image_id, sub_out_dir, df_match, min_inlier=2000, min_height=20):
    df_part = pd.concat([
        df_match[(df_match["query"] == test_image_id) & (df_match["inlier_count"] > min_inlier) & (df_match["pix_rmse"] > 7.0)].sort_values(by="inlier_count", ascending=True),
        df_match[(df_match["query"] == test_image_id) & (df_match["inlier_count"] > min_inlier) & (df_match["pix_rmse"] <= 7.0)].sort_values(by="pix_rmse", ascending=False),
    ], sort=False)
    if len(df_part) == 0:
        return None, None, None, None

    rgb2 = load_image(Path(f"data/test/{test_image_id}_RGB.j2k")).astype(np.uint8)

    agl = load_image(sub_out_dir / f"{test_image_id}_AGL.tif")
    agl_orig = agl.copy()
    agl_mask = (agl > min_height).astype(np.uint8) * 255
    agl_mask = cv2.dilate(agl_mask, np.ones((17, 17), np.uint8), iterations=3)
    visible_mask_fusioned = np.zeros((agl_mask.shape[:2]))
    visible_mask_nearangle_fusioned = np.zeros((agl_mask.shape[:2]))

    rgb2_masked = rgb2.copy()
    rgb2_masked[agl_mask > 0, :] = 0

    for _, r in df_part.iterrows():
        train_image_id = r["hit"]

        rgb1 = load_image(Path(f"data/train-orig-resolution/trainval/{train_image_id}_RGB.tif")).astype(np.uint8)
        agl_trn = load_image(Path(f"data/train-orig-resolution/trainval/{train_image_id}_AGL.tif")) * 100
        agl_trn_mask = (agl_trn > min_height).astype(np.uint8) * 255
        agl_trn_mask = cv2.dilate(agl_trn_mask, np.ones((19, 19), np.uint8), iterations=5)

        rgb1_masked = rgb1.copy()
        rgb1_masked[agl_trn_mask > 0, :] = 0

        json_data = json.load(Path(r["path"]).open("r"))
        # print(json_data)
        cmpH = np.load(r["path"].replace("matchMeta.json", "matchH.npy"))

        if json_data["pix_rmse"] > 7.0:
            trn_removel_mask_warped = np.maximum(cv2.warpPerspective(agl_trn_mask, cmpH, agl_trn_mask.shape[:2]), agl_mask)
            agl_trn_warped = cv2.warpPerspective(agl_trn, cmpH, agl_trn_mask.shape[:2], agl_trn_mask)
            visible_mask_warped = cv2.warpPerspective(np.ones(agl_trn_mask.shape[:2]), cmpH, agl_trn_mask.shape[:2], agl_trn_mask)
            visible_mask_warped[trn_removel_mask_warped > 0] = 0

            # 細かいノイズのような形状をマッチさせるのを防ぐ
            visible_mask_warped = cv2.erode(visible_mask_warped, np.ones((3, 3), np.uint8), iterations=1)

            visible_mask_fusioned = np.maximum(visible_mask_fusioned, visible_mask_warped)

            # Update AGL
            agl_trn_warped[np.isnan(agl_trn_warped)] = agl[np.isnan(agl_trn_warped)]
            agl[visible_mask_warped == 1] = agl_trn_warped[visible_mask_warped == 1]
        else:
            # Near angle update
            agl_trn_warped = cv2.warpPerspective(agl_trn, cmpH, agl_trn.shape[:2])
            visible_mask_warped = cv2.warpPerspective(np.ones(agl_trn.shape[:2]), cmpH, agl_trn.shape[:2])
            visible_mask_fusioned = np.maximum(visible_mask_fusioned, visible_mask_warped)
            visible_mask_nearangle_fusioned = np.maximum(visible_mask_nearangle_fusioned, visible_mask_warped)
            # visible_mask_warped = cv2.erode(visible_mask_warped, np.ones((3, 3), np.uint8), iterations=1)

            # Update AGL
            agl_trn_warped[np.isnan(agl_trn_warped)] = agl[np.isnan(agl_trn_warped)]
            agl[visible_mask_warped == 1] = agl_trn_warped[visible_mask_warped == 1]

    return agl, rgb2, visible_mask_fusioned, visible_mask_nearangle_fusioned


if __name__ == "__main__":
    cli()
