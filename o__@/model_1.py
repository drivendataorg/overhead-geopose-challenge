# v12:
# backbone: ResNeSt101e
# CosineAnnealingLR(1e-4, 1e-6).
import sys

import argparse
import json
import os
import shutil
import tarfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from utilities.misc_utils import get_r2, get_r2_info
from utilities.ml_utils_v2 import Dataset, build_model, test, train_v2

np.random.seed(11)


def run_train(gpus):
    model_name = "v12_rs101e"

    args = argparse.Namespace(
        pretraining=None,
        checkpoint_dir=f"data/working/models/{model_name}",
        dataset_dir="data/train-orig-resolution/",
        predictions_dir=f"data/working/preds/{model_name}",
        train_sub_dir="train",
        valid_sub_dir="valid",
        test_sub_dir="test",
        rgb_suffix="tif",
        random_crop=True,
        model_path=None,
        gpus=gpus,
        augmentation=True,
        batch_size=4,
        num_workers=4,
        num_epochs=200,
        save_best=True,
        train=True,
        test=False,
        unit="m",
        convert_prediction_to_cm_and_compress=True,
        nan_placeholder=65535,
        sample_size=None,
        angle_weight=10.0,
        scale_weight=10.0,
        mag_weight=2.0,
        agl_weight=1.0,
        backbone="timm-resnest101e",
        learning_rate=1e-4,
        save_period=2,
        val_period=1,
        downsample=1,
    )

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    train_v2(args)


def run_eval():
    args = argparse.Namespace(
        pretraining="data/models/v12_rs101e/model_183.pth",
        valid_sub_dir="valid",
        test_sub_dir="test",
        dataset_dir="data/train-orig-resolution",
        rgb_suffix="tif",
        backbone="timm-resnest101e",
        nan_placeholder=65535,
        train=False,
        test=True,
        unit="m",
        sample_size=None,
        random_crop=False,
        downsample=1,
        augmentation=False,
        gpus="1",
    )
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    model = build_model(args)
    model.load_state_dict(torch.load(args.pretraining))
    model.to("cuda")
    model.eval()

    val_dataset = Dataset(sub_dir=args.valid_sub_dir, args=args)
    assert len(val_dataset) > 0

    with torch.no_grad():
        agl_count, agl_error_sum, agl_gt_sq_sum, agl_sum = 0, 0, 0, 0
        agl_rms = []

        for idx in range(len(val_dataset)):
            img, xydir, agl, _, scale = val_dataset[idx]
            X = torch.from_numpy(img)
            X = X.unsqueeze(0)
            X = X.to("cuda")

            out = model.forward(X.float())
            _, agl_pred, _, _ = out
            agl_pred = agl_pred.squeeze().cpu().detach().numpy()
            # AGL+gt_angle+gt_scale => R2
            count, error_sum, rms, data_sum, gt_sq_sum = get_r2_info(agl, agl_pred)
            agl_count += count
            agl_error_sum += error_sum
            agl_rms.append(rms)
            agl_sum += data_sum
            agl_gt_sq_sum += gt_sq_sum

        r2_agl = get_r2(agl_error_sum, agl_gt_sq_sum, agl_sum, agl_count)
        print(f"R2-AGL: {r2_agl:.6f}")


def run_test(gpus):
    model_name = "v12_rs101e"

    args = argparse.Namespace(
        pretraining=None,
        checkpoint_dir=f"data/working/models/{model_name}",
        dataset_dir="data/test/",
        predictions_dir=f"data/working/preds/{model_name}",
        train_sub_dir="train",
        valid_sub_dir="valid",
        test_sub_dir="",
        rgb_suffix="j2k",
        random_crop=False,
        model_path=f"data/working/models/{model_name}/model_183.pth",
        gpus=gpus,
        augmentation=False,
        batch_size=1,
        num_workers=1,
        num_epochs=180,
        save_best=True,
        train=False,
        test=True,
        unit="m",
        convert_predictions_to_cm_and_compress=False,
        nan_placeholder=65535,
        sample_size=None,
        angle_weight=10.0,
        scale_weight=10.0,
        mag_weight=2.0,
        agl_weight=1.0,
        backbone="timm-resnest101e",
        learning_rate=1e-4,
        save_period=2,
        val_period=1,
        downsample=1,
    )

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    if args.test:
        Path(args.predictions_dir).mkdir(parents=True, exist_ok=True)
        test(args)


if __name__ == "__main__":
    # run_train("1")
    # run_test("1")

    # run_eval()
    pass
