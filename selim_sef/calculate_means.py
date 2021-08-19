import glob
from multiprocessing import Pool
import numpy as np
import tifffile
from osgeo import gdal
from tqdm import tqdm

from training.geopose_dataset import load_vflow, load_image

CITIES = ["ARG", "ATL", "JAX", "OMA"]


def calc_mean(file_path):
    img = load_image(file_path, 65535, unit="m")
    agl_sum = np.sum(img[~np.isnan(img)])
    agl_count = np.count_nonzero(~np.isnan(img))
    mag = load_vflow(file_path.replace("AGL.tif", "VFLOW.json"), img, "m")[0]
    mag_sum = np.sum(mag[~np.isnan(mag)])
    mag_count = np.count_nonzero(~np.isnan(mag))
    return agl_sum, agl_count, mag_sum, mag_count

pool = Pool(10)
result_agl = {
}
result_mag = {
}
for city in CITIES:
    print(city)
    agl_files = list(glob.glob(f"/home/selim/data/train_tiff/{city}*AGL*"))[:20]

    agl_sum = 0
    agl_count = 0
    mag_sum = 0
    mag_count = 0
    with tqdm(total=len(agl_files)) as pbar:
        for agl_sum_img, agl_count_img, mag_sum_img, mag_count_img  in pool.imap_unordered(calc_mean, agl_files):
            pbar.update()
            agl_sum += agl_sum_img
            agl_count += agl_count_img
            mag_sum += mag_sum_img
            mag_count += mag_count_img


    agl_mean = agl_sum / agl_count
    mag_mean = mag_sum / mag_count
    print(f"AGL AVG {agl_mean :.4f}")
    print(f"MAG AVG  {mag_mean :.4f}")
    result_agl[city] = agl_mean
    result_mag[city] = mag_mean
print("MAG", result_mag)
print("AGL", result_agl)