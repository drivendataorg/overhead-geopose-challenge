import json
import tarfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


def ensemble(
    input_dirs,
    output_dir,
    compression_type="tiff_adobe_deflate",
    folder_search="*_AGL*.tif*",
    replace=True,
    add_jsons=True,
    conversion_factor=100,
    dtype="uint16",
):
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    input_dir = input_dirs[0]

    tifs = list(input_dir.glob(folder_search))
    for tif_path in tqdm(tifs, total=len(tifs)):
        path = output_dir / tif_path.name
        if replace or not path.exists():
            # pred0
            imarray = np.array(Image.open(tif_path))
            # pred1
            tif_path2 = input_dirs[1] / tif_path.name
            imarray2 = np.array(Image.open(tif_path2))
            # pred2
            tif_path3 = input_dirs[2] / tif_path.name
            imarray3 = np.array(Image.open(tif_path3))
            # pred3
            tif_path4 = input_dirs[3] / tif_path.name
            imarray4 = np.array(Image.open(tif_path4))

            im = Image.fromarray(
                np.round(
                    ((imarray * conversion_factor) + (
                        imarray2 * conversion_factor) + (
                        imarray3 * conversion_factor) + (
                        imarray4 * conversion_factor))
                    / 4.0
                ).astype(dtype)
            )

            im.save(str(path), "TIFF", compression=compression_type)

    df_scale = pd.read_csv("data/working/lgbm_scale.csv")
    scale_dict = {}
    for _, r in df_scale.iterrows():
        scale_dict[r["id"]] = r["scale"]

    if add_jsons:
        for json_path in input_dir.glob("*.json"):
            if replace or not (output_dir / json_path.name).exists():
                data1 = json.load(json_path.open("r"))
                data2 = json.load((input_dirs[1] / json_path.name).open("r"))
                if data1["angle"] < 0 and data2["angle"] > 0:
                    data1["angle"] = -1 * data1["angle"]
                elif data1["angle"] > 0 and data2["angle"] < 0:
                    data2["angle"] = -1 * data2["angle"]

                data3 = json.load((input_dirs[2] / json_path.name).open("r"))
                if data1["angle"] < 0 and data3["angle"] > 0:
                    data3["angle"] = -1 * data3["angle"]
                elif data1["angle"] > 0 and data3["angle"] < 0:
                    data3["angle"] = -1 * data3["angle"]

                data4 = json.load((input_dirs[3] / json_path.name).open("r"))
                if data1["angle"] < 0 and data4["angle"] > 0:
                    data4["angle"] = -1 * data4["angle"]
                elif data1["angle"] > 0 and data4["angle"] < 0:
                    data4["angle"] = -1 * data4["angle"]

                mean_angle = np.mean([
                    data1["angle"],
                    data2["angle"],
                    data3["angle"],
                    data4["angle"],
                ])

                id_str = json_path.name.split("_")[1]
                scale_value = scale_dict[id_str]
                vflow = {
                    "scale": scale_value,
                    "angle": mean_angle,
                }
                new_json_path = output_dir / json_path.name
                json.dump(vflow, new_json_path.open("w"))

    return output_dir


def main():
    preds_0_dir = Path("data/working/preds/v12_rs101e/")
    preds_1_dir = Path("data/working/preds/v13_regnety120/")
    preds_2_dir = Path("data/working/preds/v25_rs101e/")
    preds_3_dir = Path("data/working/preds/v40_rs101e/")

    sub_out_dir = Path("data/working/preds/ensemble_sub")
    sub_out_dir.mkdir(parents=True, exist_ok=True)

    submission_format_dir = Path("data/submission_format")
    my_preds_files = [pth.name for pth in preds_0_dir.iterdir()]
    submission_format_files = [pth.name for pth in submission_format_dir.iterdir()]
    assert not set(my_preds_files).symmetric_difference(submission_format_files)

    # compress our submission folder
    ensemble([
        preds_0_dir,
        preds_1_dir,
        preds_2_dir,
        preds_3_dir,
    ], sub_out_dir)


if __name__ == "__main__":
    main()
