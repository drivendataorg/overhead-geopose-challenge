import argparse
import os

from utilities.test import test


def str2bool(v):
    if v.lower().startswith("t"):
        return True
    elif v.lower().startswith("f"):
        return False

    raise argparse.ArgumentTypeError("Bool value expected")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--augmentation", action="store_true", help="whether or not to use augmentation"
    )
    parser.add_argument("--train", action="store_true", help="train model")
    parser.add_argument("--test", action="store_true", help="generate test predictions")
    parser.add_argument(
        "--num-workers", type=int, help="number of data loader workers", default=8
    )
    parser.add_argument(
        "--num-epochs", type=int, help="number of epochs to train", default=1005
    )
    parser.add_argument("--batch-size", type=int, help="batch size", default=4)
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints")
    parser.add_argument("--predictions-dir", type=str, default="./predictions")
    parser.add_argument(
        "--dataset-dir", type=str, help="dataset directory", default="./dataset"
    )
    parser.add_argument(
        "--train-path-df",
        type=str,
        help="path to train df",
        default="geopose_train.csv",
    )
    parser.add_argument(
        "--test-path-df",
        type=str,
        help="path to test df",
        default="geopose_test.csv",
    )
    parser.add_argument("--model-type", type=str, default="unet")
    parser.add_argument("--backbone", type=str, default="sene154")
    parser.add_argument("--encoder-weights", type=str, default="imagenet")
    parser.add_argument("--optim", type=str, default="fusedadam", help="optimizer name")
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--scheduler", type=str, default="cosa", help="scheduler name")
    parser.add_argument("--T-max", type=int, default=5)
    parser.add_argument("--agl-weight", type=float, help="agl loss weight", default=1)
    parser.add_argument("--mag-weight", type=float, help="mag loss weight", default=1)
    parser.add_argument(
        "--angle-weight", type=float, help="angle loss weight", default=50
    )
    parser.add_argument(
        "--scale-weight", type=float, help="scale loss weight", default=50
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
    parser.add_argument(
        "--random-state",
        type=int,
        help="random seed",
        default=314159,
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        help="number of folds",
        default=10,
    )
    parser.add_argument(
        "--fold",
        type=int,
        help="fold",
        default=0,
    )
    parser.add_argument(
        "--tta",
        type=int,
        help="test time augmentation",
        default=1,
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        help="fold",
        default=0,
    )
    parser.add_argument(
        "--gpu",
        type=int,
        help="gpu",
        default=0,
    )
    parser.add_argument(
        "--distributed", action="store_true", help="distributed training"
    )
    parser.add_argument("--syncbn", action="store_true", help="sync batchnorm")
    parser.add_argument(
        "--deterministic", action="store_true", help="deterministic training"
    )
    parser.add_argument(
        "--load", type=str, default="", help="path to pretrained model weights"
    )
    parser.add_argument("--model-pt", nargs="+", type=str)
    parser.add_argument("--use-cities", nargs="+", type=str2bool)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="path to pretrained model to resume training",
    )
    parser.add_argument("--lmdb", type=str, default=None, help="path to lmdb")
    parser.add_argument(
        "--channels-last", action="store_true", help="Use channels_last memory layout"
    )
    parser.add_argument("--prefetch", action="store_true", help="Use prefetching")
    parser.add_argument(
        "--test-rgb-path", type=str, default=None, help="path to rgb test files"
    )
    parser.add_argument(
        "--pl-dir", type=str, default=None, help="path to predicted test files"
    )
    parser.add_argument("--city", type=str, default=None, help="city name")
    parser.add_argument(
        "--use-city", action="store_true", help="Use city ohe in decoder"
    )
    parser.add_argument("--fp16", action="store_true", help="fp16 training")
    parser.add_argument("--to-log", action="store_true", help="use log heights")
    parser.add_argument("--albu", action="store_true", help="use albu augs")

    args = parser.parse_args()
    if args.local_rank == 0:
        print(args)

    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"

    if args.test:
        if args.local_rank == 0:
            os.makedirs(args.predictions_dir, exist_ok=True)

        test(args)
