import json
import os
import random
import re

from utilities.augmentation_vflow import augment_vflow

import cv2
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

from utilities.tta import predict_tta

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

from pathlib import Path

from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset

from utilities.misc_utils import (
    convert_and_compress_prediction_dir,
    load_image,
    load_vflow,
    save_image,
)
from utilities.unet import TimmUnet

RNG = np.random.RandomState(4321)

CITIES = ["ARG", "ATL", "JAX", "OMA"]


class Dataset(BaseDataset):
    def __init__(
            self,
            sub_dir,
            args,
            crop_size=640,
            rng=RNG,
    ):
        self.crop_size = crop_size
        self.is_test = sub_dir == args.test_sub_dir
        self.is_val = sub_dir == args.valid_sub_dir
        self.rng = rng

        # create all paths with respect to RGB path ordering to maintain alignment of samples
        dataset_dir = Path(args.dataset_dir) / sub_dir
        print(dataset_dir)
        rgb_paths = list(dataset_dir.glob(f"*_RGB.{args.rgb_suffix}"))
        if rgb_paths == []: rgb_paths = list(dataset_dir.glob(f"*_RGB*.{args.rgb_suffix}"))  # original file names
        agl_paths = list(
            pth.with_name(pth.name.replace("_RGB", "_AGL")).with_suffix(".tif")
            for pth in rgb_paths
        )
        vflow_paths = list(
            pth.with_name(pth.name.replace("_RGB", "_VFLOW")).with_suffix(".json")
            for pth in rgb_paths
        )

        if self.is_test:
            self.paths_list = rgb_paths
        else:
            self.paths_list = [
                (rgb_paths[i], vflow_paths[i], agl_paths[i])
                for i in range(len(rgb_paths))
            ]

            self.paths_list = [
                self.paths_list[ind]
                for ind in self.rng.permutation(len(self.paths_list))
            ]
            if args.sample_size is not None:
                self.paths_list = self.paths_list[: args.sample_size]

        self.args = args
        self.sub_dir = sub_dir

    def __getitem__(self, i):
        try:
            if self.is_test:
                rgb_path = self.paths_list[i]
                image = load_image(rgb_path, self.args)
            else:
                rgb_path, vflow_path, agl_path = self.paths_list[i]
                image = load_image(rgb_path, self.args)
                agl = load_image(agl_path, self.args)
                mag, xdir, ydir, vflow_data = load_vflow(vflow_path, agl, self.args)
                scale = vflow_data["scale"]
                if self.args.augmentation and not self.is_val:
                    image, mag, xdir, ydir, agl, scale = augment_vflow(
                        image,
                        mag,
                        xdir,
                        ydir,
                        vflow_data["angle"],
                        vflow_data["scale"],
                        agl=agl,
                    )
                xdir = np.float32(xdir)
                ydir = np.float32(ydir)
                mag = mag.astype("float32")
                agl = agl.astype("float32")
                scale = np.float32(scale)

                xydir = np.array([xdir, ydir])

            if self.is_test and self.args.downsample > 1:
                image = cv2.resize(
                    image,
                    (
                        int(image.shape[0] / self.args.downsample),
                        int(image.shape[1] / self.args.downsample),
                    ),
                    interpolation=cv2.INTER_NEAREST,
                )
            crop_size = self.crop_size
            height = image.shape[0]
            y = random.randint(0, height - crop_size - 1)
            x = random.randint(0, height - crop_size - 1)
            if not self.is_val and not self.is_test:
                image = image[y: y + crop_size, x:x + crop_size]

            image = (np.transpose(image, (2, 0, 1)) / 255. - 0.5) * 2
            city_name = os.path.basename(rgb_path).split("_")[0]
            city = np.zeros((4, 1, 1))
            city[CITIES.index(city_name), 0, 0] = 1
            if self.is_test:
                return image, str(rgb_path)
            else:
                if not self.is_val:
                    agl = agl[y: y + crop_size, x:x + crop_size]
                    mag = mag[y: y + crop_size, x:x + crop_size]
                return {"image": image, "xydir": xydir, "agl": agl, "mag": mag, "scale": scale, "city": city}
        except:
            return self.__getitem__(random.randint(0, len(self.paths_list) - 1))

    def __len__(self):
        return len(self.paths_list)


def test(args):
    model_paths = args.model_path
    models = []
    torch.backends.cudnn.benchmark = True

    for model_path in model_paths:
        model = TimmUnet("tf_efficientnetv2_l_in21k")
        print(f" loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location="cpu")
        state_dict = checkpoint['state_dict']
        print(f" Epoch {checkpoint['epoch']} metrics: {checkpoint['metrics']}")
        state_dict = {re.sub("^module.", "", k): w for k, w in state_dict.items()}
        model.load_state_dict(state_dict)
        model.cuda()
        model = DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank,
                                        find_unused_parameters=True)
        model.eval()
        models.append(model)

    with torch.no_grad():

        test_dataset = Dataset(sub_dir=args.test_sub_dir, args=args)
        sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False)
        test_loader = DataLoader(
            test_dataset, batch_size=args.batch_size, sampler=sampler, shuffle=False, num_workers=4, pin_memory=True
        )
        predictions_dir = Path(args.predictions_dir)
        for images, rgb_paths in tqdm(test_loader):
            images = images.float().cuda()
            pred = predict_tta(models, images)

            numpy_preds = []
            for i in range(len(pred)):
                numpy_preds.append(pred[i].detach().cpu().numpy())

            xydir_pred, agl_pred, mag_pred, scale_pred = numpy_preds

            if scale_pred.ndim == 0:
                scale_pred = np.expand_dims(scale_pred, axis=0)

            for batch_ind in range(agl_pred.shape[0]):
                # vflow pred
                angle = np.arctan2(xydir_pred[batch_ind][0], xydir_pred[batch_ind][1])
                vflow_data = {
                    "scale": np.float64(
                        scale_pred[batch_ind] * args.downsample
                    ),  # upsample
                    "angle": np.float64(angle),
                }

                # agl pred
                curr_agl_pred = agl_pred[batch_ind, 0, :, :]
                curr_agl_pred[curr_agl_pred < 0] = 0
                agl_resized = cv2.resize(
                    curr_agl_pred,
                    (
                        curr_agl_pred.shape[0] * args.downsample,  # upsample
                        curr_agl_pred.shape[1] * args.downsample,  # upsample
                    ),
                    interpolation=cv2.INTER_NEAREST,
                )

                # save
                rgb_path = predictions_dir / Path(rgb_paths[batch_ind]).name
                agl_path = rgb_path.with_name(
                    rgb_path.name.replace("_RGB", "_AGL")
                ).with_suffix(".tif")
                vflow_path = rgb_path.with_name(
                    rgb_path.name.replace("_RGB", "_VFLOW")
                ).with_suffix(".json")

                json.dump(vflow_data, vflow_path.open("w"))
                save_image(agl_path, agl_resized)  # save_image assumes units of meters
    print(f"Rank {args.local_rank} finished")
    dist.barrier()
    print("postprocessing")

    if args.local_rank == 0:
        # creates new dir predictions_dir_con
        if args.convert_predictions_to_cm_and_compress:
            convert_and_compress_prediction_dir(predictions_dir=predictions_dir)
