import argparse
import json
import os
import tarfile
from pathlib import Path
from typing import List, Dict, Optional

import cv2
import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf
from pytorch_toolbelt.datasets import (
    INPUT_IMAGE_KEY,
    INPUT_IMAGE_ID_KEY,
)
from pytorch_toolbelt.utils import mask_from_tensor, rgb_image_from_tensor, hstack_autopad, vstack_header
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from geopose import *
from geopose.models import model_from_checkpoint, wrap_model_with_tta
from geopose.models.tta.geopose_tta import GeoPoseEnsembler


def ensemble_from_checkpoints(
    checkpoint_fnames: List[str], strict: bool = True, tta: Optional[str] = None, tta_mode="ensemble"  # model|ensemble
):
    if tta_mode not in {"model", "ensemble"}:
        raise KeyError(tta_mode)

    models = []
    checkpoints = []

    for ck in checkpoint_fnames:
        model, checkpoint = model_from_checkpoint(ck, strict=strict)

        if tta_mode == "model" and tta not in {None, "None"}:
            model = wrap_model_with_tta(model, tta)
            print("Wrapping individual model with TTA", tta)

        models.append(model)
        checkpoints.append(checkpoint)

    if len(models) > 1:
        model = GeoPoseEnsembler(
            models, outputs=[OUTPUT_AGL_MASK, OUTPUT_VFLOW_DIRECTION, OUTPUT_VFLOW_SCALE, OUTPUT_VFLOW_ANGLE]
        )

    else:
        assert len(models) == 1
        model = models[0]

    if tta_mode == "ensemble" and tta not in {None, "None"}:
        model = wrap_model_with_tta(model, tta)
        print("Wrapping ensemble with TTA", tta)

    return model.eval(), checkpoints


@torch.no_grad()
def compute_predictions(
    model: nn.Module,
    dataset: OverheadGeoposeDataset,
    output_dir: str,
    batch_size=1,
    num_workers=1,
    fp16=True,
    visualize=False,
    desc=None,
) -> None:
    model = model.cuda().eval()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    submission_dir = os.path.join(output_dir, "submission")
    os.makedirs(submission_dir, exist_ok=True)

    visualization_dir = os.path.join(output_dir, "visualization")
    if visualize:
        os.makedirs(visualization_dir, exist_ok=True)

    for batch in tqdm(loader, desc=desc):
        image = batch[INPUT_IMAGE_KEY]
        gsd = batch[INPUT_GROUND_SAMPLE_DISTANCE]
        if fp16:
            image = image.half()
            gsd = gsd.half()

        with torch.cuda.amp.autocast(fp16):
            with torch.inference_mode():
                outputs = model(
                    **{INPUT_IMAGE_KEY: image.cuda(non_blocking=True), INPUT_GROUND_SAMPLE_DISTANCE: gsd.cuda(non_blocking=True)}
                )

        for (image_id, city_abbreviation, image, agl_meters, vflow_angle, vflow_scale) in zip(
            batch[INPUT_IMAGE_ID_KEY],
            batch[INPUT_PATCH_CITY_NAME],
            batch[INPUT_IMAGE_KEY],
            outputs[OUTPUT_AGL_MASK],
            outputs[OUTPUT_VFLOW_ANGLE],
            outputs[OUTPUT_VFLOW_SCALE],
        ):
            agl_fname = os.path.join(submission_dir, image_id.replace("_RGB", "_AGL") + ".tif")
            vfl_fname = os.path.join(submission_dir, image_id.replace("_RGB", "_VFLOW") + ".json")
            viz_fname = os.path.join(visualization_dir, image_id.replace("_RGB", "_VIZ") + ".jpg")

            agl_cm = np.round(mask_from_tensor(agl_meters * 100, squeeze_single_channel=True)).astype(np.uint16)
            Image.fromarray(agl_cm).save(agl_fname, "TIFF", compression="tiff_adobe_deflate")

            vflow_scale_cm_per_pixel = vflow_scale / 100

            vflow = dict(scale=float(vflow_scale_cm_per_pixel), angle=float(vflow_angle))
            json.dump(vflow, open(vfl_fname, "w"))

            # Visualizations
            if visualize:
                img_bgr = rgb_image_from_tensor(image, DATASET_MEAN, DATASET_STD)
                max_agl = agl_cm.max()
                pred_agl = (255.0 * agl_cm / max_agl).astype(np.uint8)
                pred_agl = cv2.applyColorMap(pred_agl, cv2.COLORMAP_VIRIDIS)

                composition = hstack_autopad(
                    [
                        vstack_header(img_bgr, image_id),
                        vstack_header(
                            pred_agl,
                            f"angle:{float(vflow_angle):.3f} scale:{float(vflow_scale_cm_per_pixel):.3f} max: {max_agl:.3f}",
                        ),
                    ]
                )
                cv2.imwrite(viz_fname, composition)

    submission_tar_path = os.path.join(output_dir, "submission_compressed.tar.gz")
    with tarfile.open(submission_tar_path, "w:gz") as tar:
        # iterate over files and add each to the TAR
        files = list(Path(submission_dir).glob("*"))
        for file in tqdm(files, total=len(files)):
            tar.add(file, arcname=file.name)


@torch.no_grad()
def run_predict(config_fname: str, dataset: OverheadGeoposeDataset, force=False, visualize=False):
    if isinstance(config_fname, str):
        config: Dict = OmegaConf.load(config_fname)
    else:
        config: Dict = config_fname

    tta = config["ensemble"]["tta"]
    tta_mode = config["ensemble"]["tta_mode"]
    models = config["ensemble"]["models"]

    batch_size = config["inference"]["batch_size"]
    num_workers = config["inference"]["num_workers"]
    fp16 = config["inference"]["fp16"]

    submission_dir = config["submission_dir"]
    os.makedirs(submission_dir, exist_ok=True)

    detect_anomaly = config.get("detect_anomaly", False)
    torch.set_anomaly_enabled(detect_anomaly)
    print("Detect anomaly", detect_anomaly)

    model, checkpoint = ensemble_from_checkpoints(checkpoint_fnames=models, tta=tta, tta_mode=tta_mode)
    compute_predictions(
        model,
        dataset=dataset,
        output_dir=submission_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        fp16=fp16,
        visualize=visualize,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, nargs="+", help="Configuration file for inference")
    parser.add_argument(
        "-dd",
        "--data-dir",
        type=str,
        default=os.environ.get("DRIVENDATA_OVERHEAD_GEOPOSE", "d:/datasets/overhead-geopose-challenge"),
    )
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()
    print(args.config)

    for config in args.config:
        test_dataset = OverheadGeoposeDataModule.get_test_dataset(args.data_dir)
        run_predict(config, test_dataset, visualize=args.visualize)


if __name__ == "__main__":
    # Give no chance to randomness
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    main()
