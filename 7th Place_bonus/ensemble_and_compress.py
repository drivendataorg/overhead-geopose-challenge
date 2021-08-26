#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

import os
import glob
import json
import shutil
import sys
import numpy as np
import pandas as pd
from scipy.stats import circmean
from pathlib import Path
from PIL import Image
import multiprocessing as mp
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--in_dir', type=str, default='output', help='Directory where predictions are stored in separate subdirectories per model')
parser.add_argument('--out_dir', type=str, default='output/ensemble_preds', help='Directory to save ensembled predictions')
parser.add_argument('--n_proc', type=int, default=16, help='Number of processes to use for multiproc ensembling and compression')
args = parser.parse_args()

#------------------------------------------------------------------------------
# Settings
#------------------------------------------------------------------------------

coef_0 = 0.25
coef_1 = 0.15
coef_2 = 0.15
coef_3 = 0.15
coef_4 = 0.30

dirs = [
    'run-20210608-2131-unet-resnet34-e500',
    'run-20210626-0051-unet-inceptionv4-e290',
    'run-20210704-1650-unet-seresnext50-e120',
    'run-20210713-0048-unet-regnetx064-e80',
    'run-20210713-0107-unet-resnext50swsl-e230',
]

coef_arg = 1.20
coef_jax = 1.05
coef_oma = 1.10

#------------------------------------------------------------------------------
# Ensembling
#------------------------------------------------------------------------------

if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir)

files_tif  = sorted(glob.glob(os.path.join(args.in_dir, dirs[0], 'preds_f0/*.tif')))
files_json = sorted(glob.glob(os.path.join(args.in_dir, dirs[0], 'preds_f0/*.json')))
print(files_tif[0])
print(files_json[0])
print(len(files_tif))  # 1025
print(len(files_json)) # 1025

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def worker(file):
    file = file.split('/')[-1]
    arrays = []
    for d in dirs:
        arrays_per_model = []
        for fold_id in range(5):
            path = os.path.join(args.in_dir, d, 'preds_f%d' % fold_id, file)
            # print(path)
            arrays_per_model.append(np.array(Image.open(path)).astype('float32'))
        arrays.append(arrays_per_model)
    #
    # print(len(arrays)) # 5
    # print(len(arrays[0])) # 5
    # print(arrays[0][0].shape) # (2048, 2048), float32
    #
    ens = (
           coef_0 * np.mean(arrays[0], axis=0) + 
           coef_1 * np.mean(arrays[1], axis=0) +
           coef_2 * np.mean(arrays[2], axis=0) +
           coef_3 * np.mean(arrays[3], axis=0) +
           coef_4 * np.mean(arrays[4], axis=0)
          )
    #
    # Multiply by bias coefs
    if 'ARG_' in file:
        # print('Mult file:', coef_arg, file)
        ens = ens * coef_arg
    if 'JAX_' in file:
        # print('Mult file:', coef_jax, file)
        ens = ens * coef_jax
    if 'OMA_' in file:
        # print('Mult file:', coef_oma, file)
        ens = ens * coef_oma
    #
    # Save
    img = Image.fromarray(ens)
    img.save(os.path.join(args.out_dir, file), 'TIFF')
    #
    return 0


def master(worker, data, n_processes=None, print_each=100):
    print('N processes requested:', n_processes)
    print('N cores available:    ', mp.cpu_count())
    #
    print('Create Pool...')
    pool = mp.Pool(n_processes)
    #
    print('Start ensembling...')
    for i, res in enumerate(pool.imap_unordered(worker, data)):
        if not i % print_each:
            print(i)
    #
    pool.close()
    pool.join()
    #
    return i+1

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

# Run TIFF
examples_count = master(worker, files_tif, n_processes=args.n_proc)

# Run JSON
for counter, file in enumerate(files_json):
    file = file.split('/')[-1]
    angles = []
    scales = []
    for di in dirs:
        scales_per_model = []
        for fold_id in range(5):
            path = os.path.join(args.in_dir, di, 'preds_f%d' % fold_id, file)
            with open(path, 'r') as f:
                d = json.load(f)
                angles.append(d['angle'])
                scales_per_model.append(d['scale'])
        scales.append(scales_per_model)
    #
    angle_ens = circmean(angles) # <---- CIRCMEAN
    #
    scale_ens = (
                 coef_0 * np.mean(scales[0]) + 
                 coef_1 * np.mean(scales[1]) +
                 coef_2 * np.mean(scales[2]) +
                 coef_3 * np.mean(scales[3]) +
                 coef_4 * np.mean(scales[4])
                )
    #
    d_ens = {'scale': scale_ens, 'angle': angle_ens}
    with open(os.path.join(args.out_dir, file), 'w') as f:
        json.dump(d_ens, f)
    #
    if counter % 100 == 0:
        print(counter)

#------------------------------------------------------------------------------
# Compression
#------------------------------------------------------------------------------

input_dir=Path(args.out_dir)
output_dir=Path(args.out_dir + '_converted_cm_compressed_uint16')
compression_type="tiff_adobe_deflate"
folder_search="*_AGL*.tif*"
replace=True
add_jsons=True
conversion_factor=100
dtype='uint16'

tifs = list(input_dir.glob(folder_search))

if not output_dir.exists():
    output_dir.mkdir(parents=True)


def worker(tif_path):
    path = output_dir / tif_path.name
    if replace or not path.exists():
        imarray = np.array(Image.open(tif_path))
        imarray = np.round(imarray * conversion_factor).astype(dtype)
        im = Image.fromarray(imarray)
        im.save(str(path), "TIFF", compression=compression_type)
    #
    return 0


def master(worker, data, n_processes=None, print_each=100):
    print('N processes requested:', n_processes)
    print('N cores available:    ', mp.cpu_count())
    #
    print('Create Pool...')
    pool = mp.Pool(n_processes)
    #
    print('Start compressing...')
    for i, res in enumerate(pool.imap_unordered(worker, data)):
        if not i % print_each:
            print(i)
    #
    pool.close()
    pool.join()
    #
    return i+1

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

# Run TIFF
examples_count = master(worker, tifs, n_processes=args.n_proc)

# Run JSON
if add_jsons:
    for json_path in input_dir.glob("*.json"):
        if replace or not (output_dir / json_path.name).exists():
            vflow = json.load(json_path.open('r'))
            vflow['scale'] = vflow['scale'] / conversion_factor
            new_json_path = output_dir / json_path.name
            json.dump(vflow, new_json_path.open('w'))

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------


