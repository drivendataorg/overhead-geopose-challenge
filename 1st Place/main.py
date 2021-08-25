import argparse
import os

import torch

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import cv2
import torch.distributed as dist

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="train model")
    parser.add_argument("--test", action="store_true", help="generate test predictions")
    parser.add_argument("--gpus", type=str, default="0")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=2)
    parser.add_argument(
        "--num-workers", type=int, help="number of data loader workers", default=1
    )
    parser.add_argument("--batch-size", type=int, help="batch size", default=4)
    parser.add_argument(
        "--downsample",
        type=int,
        help="factor for downsampling image at test time",
        default=1,
    )
    parser.add_argument(
        "--model-path",
        nargs='+',
        help="Default is most recent in checkpoint dir"
    )
    parser.add_argument(
        "--dataset-dir", type=str, help="dataset directory", default="./dataset"
    )

    parser.add_argument(
        "--test-sub-dir", type=str, help="test folder within datset-dir", default="test"
    )
    parser.add_argument(
        "--valid-sub-dir",
        type=str,
        help="validation folder within datset-dir",
        default="valid",
    )
    parser.add_argument("--predictions-dir", type=str, default="./predictions")

    parser.add_argument(
        "--sample-size",
        type=int,
        help="number of images to randomly sample for training",
        default=None,
    )
    parser.add_argument(
        "--rgb-suffix", type=str, help="suffix for rgb files", default="j2k"
    )
    parser.add_argument(
        "--nan-placeholder",
        type=int,
        help="placeholder value for nans. use 0 for no placeholder",
        default=65535,
    )
    parser.add_argument(
        "--unit",
        type=str,
        help="unit of AGLS (m, cm) -- converted inputs are in cm, downsampled data is in m",
        default="cm",
    )
    parser.add_argument(
        "--convert-predictions-to-cm-and-compress",
        type=bool,
        help="Whether to process predictions by converting to cm and compressing",
        default=True,
    )

    args = parser.parse_args()

    dist.init_process_group(backend="nccl",
                                      rank=args.local_rank,
                                      world_size=args.world_size)
    torch.cuda.set_device(args.local_rank)

    if args.test:
        os.makedirs(args.predictions_dir, exist_ok=True)
        from utilities.inference import test
        test(args)
    else:
        raise ValueError("Training is not supported, use train_segmentor instead")
