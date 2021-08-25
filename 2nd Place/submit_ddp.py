import argparse
import json
import os
import tarfile
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from PIL import Image
from omegaconf import OmegaConf
from pytorch_toolbelt.datasets import (
    INPUT_IMAGE_KEY,
    INPUT_IMAGE_ID_KEY,
)
from pytorch_toolbelt.utils import mask_from_tensor, rgb_image_from_tensor, hstack_autopad, vstack_header
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from geopose import *
from submit import ensemble_from_checkpoints


@torch.no_grad()
def compute_predictions(
    model: nn.Module,
    dataset: OverheadGeoposeDataset,
    output_dir: str,
    batch_size=1,
    num_workers=1,
    fp16=True,
    desc=None,
    visualize=False,
) -> None:
    model = DistributedDataParallel(model.cuda()).eval()
    world_size = torch.distributed.get_world_size()
    local_rank = torch.distributed.get_rank()

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        sampler=DistributedSampler(dataset, world_size, local_rank),
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

    torch.distributed.barrier()

    if local_rank == 0:
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


def run(rank, size):
    # Give no chance to randomness
    torch.manual_seed(0)
    np.random.seed(0)
    torch.cuda.set_device(rank)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

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
        torch.distributed.barrier()
        test_dataset = OverheadGeoposeDataModule.get_test_dataset(args.data_dir)
        run_predict(config, test_dataset, visualize=args.visualize)
        torch.distributed.barrier()


def init_process(rank, size, fn, backend="nccl"):
    """Initialize the distributed environment."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


if __name__ == "__main__":
    size = 2  # NB: You may want to tune this value to match the number of GPUs available.
    processes = []
    mp.set_start_method("spawn")
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
