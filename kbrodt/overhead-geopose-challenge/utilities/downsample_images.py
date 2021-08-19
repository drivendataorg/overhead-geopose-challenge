import argparse
import json
import multiprocessing
import pickle
from pathlib import Path

import lmdb
from misc_utils import load_image, load_vflow, save_image
from tqdm import tqdm


def load(rgb_path):
    agl_path = rgb_path.with_name(rgb_path.name.replace("_RGB", "_AGL")).with_suffix(
        ".tif"
    )
    vflow_path = rgb_path.with_name(
        rgb_path.name.replace("_RGB", "_VFLOW")
    ).with_suffix(".json")
    rgb = load_image(
        rgb_path, args, use_cv=True
    )  # args.unit used to convert units on load
    assert len(rgb.shape) == 3
    agl = load_image(agl_path, args)  # args.unit used to convert units on load
    _, _, _, vflow_data = load_vflow(
        vflow_path, agl, args
    )  # arg.unit used to convert units on load

    return (rgb, agl, vflow_data), (rgb_path, agl_path, vflow_path)


def load_save(rgb_path):
    outdir = Path(args.outdir)

    (rgb, agl, vflow_data), (rgb_path, agl_path, vflow_path) = load(rgb_path)

    # save
    # units are NOT converted back here, so are in m
    #    save_image((outdir / rgb_path.name), rgb)
    save_image(
        (outdir / rgb_path.name.replace(args.rgb_suffix, "tif")), rgb
    )  # save as tif to be consistent with old code
    save_image((outdir / agl_path.name), agl)
    with open((outdir / vflow_path.name), "w") as outfile:
        json.dump(vflow_data, outfile)


def downsample_images(args):
    indir = Path(args.indir)
    outdir = Path(args.outdir)

    outdir.mkdir(exist_ok=True)
    rgb_paths = list(indir.glob(f"*_RGB.{args.rgb_suffix}"))
    if rgb_paths == []:
        rgb_paths = list(indir.glob(f"*_RGB*.{args.rgb_suffix}"))  # original file names

    with multiprocessing.Pool(args.n_jobs) as p:
        _ = list(p.imap_unordered(func=load_save, iterable=tqdm(rgb_paths)))


def save_to_db(args):
    indir = Path(args.indir)

    rgb_paths = list(indir.glob(f"*_RGB.{args.rgb_suffix}"))
    if rgb_paths == []:
        rgb_paths = list(indir.glob(f"*_RGB*.{args.rgb_suffix}"))  # original file names

    with multiprocessing.Pool(args.n_jobs) as p:
        items = list(p.imap_unordered(func=load, iterable=tqdm(rgb_paths)))

    map_size = 0
    for item in items:
        map_size += sum(x.nbytes for x in item[0][:-1])
        map_size += 2 * 8
    map_size = int(1.1 * map_size)
    print(map_size // 1024 // 1024)

    env = lmdb.open(
        str(args.lmdb),
        map_size=map_size,
    )

    with env.begin(write=True) as txn:
        with tqdm(items) as pbar:
            for item in pbar:
                (rgb, agl, vflow_data), (rgb_path, agl_path, vflow_path) = item
                vflow_data = (vflow_data["angle"], vflow_data["scale"])
                txn.put(rgb_path.stem.encode(), pickle.dumps(rgb))
                txn.put(agl_path.stem.encode(), pickle.dumps(agl))
                txn.put(vflow_path.stem.encode(), pickle.dumps(vflow_data))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--indir", type=str, help="input directory", default=None)
    parser.add_argument("--outdir", type=str, help="output directory", default=None)
    parser.add_argument("--lmdb", type=str, help="output directory", default=None)
    parser.add_argument(
        "--nan-placeholder", type=int, help="placeholder value for nans", default=65535
    )
    parser.add_argument(
        "--unit", type=str, help="unit of AGLS (m, cm, or dm)", default="cm"
    )
    parser.add_argument(
        "--rgb-suffix",
        type=str,
        help="file extension for RGB data, e.g., tif or j2k",
        default="j2k",
    )
    parser.add_argument("--n-jobs", type=int, help="downsample image", default=20)
    args = parser.parse_args()
    print(args)
    if args.lmdb is None:
        downsample_images(args)
    else:
        save_to_db(args)
