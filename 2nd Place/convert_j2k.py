from multiprocessing import Pool

import cv2
from fire import Fire
from pytorch_toolbelt.utils import fs
import os

from tqdm import tqdm
from geopose import read_rgb_pil


def convert_image(image_fname):
    output_fname = fs.change_extension(image_fname, ".png")
    image = read_rgb_pil(image_fname)[..., ::-1]
    cv2.imwrite(output_fname, image)


def convert(images, workers):
    with Pool(workers) as wp:
        for _ in tqdm(wp.imap_unordered(convert_image, images), total=len(images)):
            pass


def main(data_dir=os.environ.get("DRIVENDATA_OVERHEAD_GEOPOSE"), workers=6):
    train = fs.find_in_dir_with_ext(os.path.join(data_dir, "train"), ".j2k")
    convert(train, workers=workers)

    test = fs.find_in_dir_with_ext(os.path.join(data_dir, "test"), ".j2k")
    convert(test, workers=workers)


if __name__ == "__main__":
    Fire(main)
