import json
from pathlib import Path

import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
from tqdm import tqdm

from utilities.misc_utils import (
    convert_and_compress_prediction_dir,
    load_image,
    save_image,
)


class Dataset(BaseDataset):
    def __init__(
        self,
        df,
        args,
        cities=None,
    ):
        dataset_dir = Path(args.dataset_dir)
        rgb_paths = df.rgb.apply(
            lambda x: (dataset_dir / x).with_suffix(f".{args.rgb_suffix}")
        ).tolist()

        self.gsd = df.gsd.tolist()
        self.city = df.city.tolist()

        self.paths_list = rgb_paths

        self.preprocessing_fn = smp.encoders.get_preprocessing_fn(
            "senet154", "imagenet"
        )

        self.args = args
        self.cities = cities

    def __getitem__(self, i):
        gsd = None
        if self.cities is not None:
            city_str = self.city[i]
            city = np.zeros((len(self.cities), 1, 1), dtype="float32")
            city[self.cities.index(city_str)] = 1

            gsd = np.zeros((1, 1, 1), dtype="float32")
            gsd[0] = self.gsd[i]

        rgb_path = self.paths_list[i]
        image = load_image(rgb_path, self.args, use_cv=True)

        image = image.astype("uint8")

        image = self.preprocessing_fn(image).astype("float32")
        image = np.transpose(image, (2, 0, 1))

        if self.cities is not None:
            image = image, city, gsd

        return image, str(rgb_path)

    def __len__(self):
        return len(self.paths_list)


def test(args):
    models = [
        torch.jit.load(p, map_location=torch.device(f"cuda:{args.gpu}")).eval()
        for p in args.model_pt
    ]

    assert len(models) == len(args.use_cities)

    df = pd.read_csv(args.test_path_df)
    metadata = pd.read_csv(Path(args.test_path_df).with_name("metadata.csv"))
    df = pd.merge(df, metadata, on="id")

    CITIES = ["ARG", "ATL", "JAX", "OMA"]
    if args.city is not None:
        CITIES = [args.city]
        df = df[df.city.isin(CITIES)].reset_index(drop=True)

    cities = ["ARG", "ATL", "JAX", "OMA"] if args.use_city else None

    MAX_HEIGTS = {city: 200.0 for city in CITIES}
    MAX_HEIGTS["ARG"] = 100.0

    test_dataset = Dataset(df, args=args, cities=cities)

    test_sampler = None

    args.num_workers = min(max(args.num_workers, args.batch_size), 16)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2, #args.num_workers,
        sampler=test_sampler,
        pin_memory=False,
        persistent_workers=False, # True,
    )

    predictions_dir = Path(args.predictions_dir)

    iterator = tqdm(
        total=len(test_loader),
        mininterval=2,
    )

    with torch.no_grad():
        preds = [
            torch.zeros(
                (args.tta * len(models), args.batch_size, 2),
                dtype=torch.float32,
                device="cuda",
            ),
            torch.zeros(
                (args.tta * len(models), args.batch_size, 1, 2048, 2048),
                dtype=torch.float32,
                device="cuda",
            ),
            torch.zeros(
                (
                    args.tta * len(models),
                    args.batch_size,
                ),
                dtype=torch.float32,
                device="cuda",
            ),
        ]
        for images, rgb_paths in test_loader:
            if args.use_city:
                images, city, gsd = images
                city = city.to("cuda", non_blocking=True)
                gsd = gsd.to("cuda", non_blocking=True)

            images = images.to("cuda", non_blocking=True)
            bs = images.size(0)

            [pred.zero_() for pred in preds]
            for model_idx, (use_city, model) in enumerate(zip(args.use_cities, models)):
                if use_city:
                    pred = model((images, city, gsd))
                else:
                    pred = model(images)

                pred = list(pred)
                if args.to_log:
                    pred[1] = torch.expm1(pred[1])

                preds[0][model_idx * args.tta, :bs] = pred[0]
                preds[1][model_idx * args.tta, :bs] = pred[1]
                preds[2][model_idx * args.tta, :bs] = pred[3]

                if args.tta > 1:  # horizontal flip
                    if use_city:
                        pred_tta = model((torch.flip(images, dims=[-1]), city, gsd))
                    else:
                        pred_tta = model(torch.flip(images, dims=[-1]))
                    xydir_pred_tta, agl_pred_tta, _, scale_pred_tta = pred_tta
                    xydir_pred_tta[:, 0] *= -1
                    agl_pred_tta = torch.flip(agl_pred_tta, dims=[-1])
                    if args.to_log:
                        agl_pred_tta = torch.expm1(agl_pred_tta)

                    preds[0][model_idx * args.tta + 1, :bs] = xydir_pred_tta
                    preds[1][model_idx * args.tta + 1, :bs] = agl_pred_tta
                    preds[2][model_idx * args.tta + 1, :bs] = scale_pred_tta

                if args.tta > 2:  # vertical flip
                    if use_city:
                        pred_tta = model((torch.flip(images, dims=[-2]), city, gsd))
                    else:
                        pred_tta = model(torch.flip(images, dims=[-2]))
                    xydir_pred_tta, agl_pred_tta, _, scale_pred_tta = pred_tta
                    xydir_pred_tta[:, 1] *= -1
                    agl_pred_tta = torch.flip(agl_pred_tta, dims=[-2])
                    if args.to_log:
                        agl_pred_tta = torch.expm1(agl_pred_tta)

                    preds[0][model_idx * args.tta + 2, :bs] = xydir_pred_tta
                    preds[1][model_idx * args.tta + 2, :bs] = agl_pred_tta
                    preds[2][model_idx * args.tta + 2, :bs] = scale_pred_tta

                if args.tta > 3:  # vertical+horizontal flip
                    if use_city:
                        pred_tta = model((torch.flip(images, dims=[-1, -2]), city, gsd))
                    else:
                        pred_tta = model(torch.flip(images, dims=[-1, -2]))
                    xydir_pred_tta, agl_pred_tta, _, scale_pred_tta = pred_tta
                    xydir_pred_tta *= -1
                    agl_pred_tta = torch.flip(agl_pred_tta, dims=[-1, -2])
                    if args.to_log:
                        agl_pred_tta = torch.expm1(agl_pred_tta)

                    preds[0][model_idx * args.tta + 3, :bs] = xydir_pred_tta
                    preds[1][model_idx * args.tta + 3, :bs] = agl_pred_tta
                    preds[2][model_idx * args.tta + 3, :bs] = scale_pred_tta

                if args.tta > 7:  # rotate90
                    images_rot90 = torch.rot90(images, k=1, dims=[-2, -1])

                    if use_city:
                        pred_tta = model((images_rot90, city, gsd))
                    else:
                        pred_tta = model(images_rot90)
                    xydir_pred_tta, agl_pred_tta, _, scale_pred_tta = pred_tta
                    xydir_pred_tta = torch.stack(
                        [-xydir_pred_tta[:, 1], xydir_pred_tta[:, 0]], dim=1
                    )
                    agl_pred_tta = torch.rot90(agl_pred_tta, k=-1, dims=[-2, -1])

                    pred[0] += xydir_pred_tta
                    pred[1] += agl_pred_tta
                    pred[3] += scale_pred_tta

                    # vertical flip
                    if use_city:
                        pred_tta = model(
                            (torch.flip(images_rot90, dims=[-1]), city, gsd)
                        )
                    else:
                        pred_tta = model(torch.flip(images_rot90, dims=[-1]))
                    xydir_pred_tta, agl_pred_tta, _, scale_pred_tta = pred_tta
                    xydir_pred_tta[:, 0] *= -1
                    agl_pred_tta = torch.flip(agl_pred_tta, dims=[-1])
                    xydir_pred_tta = torch.stack(
                        [-xydir_pred_tta[:, 1], xydir_pred_tta[:, 0]], dim=1
                    )
                    agl_pred_tta = torch.rot90(agl_pred_tta, k=-1, dims=[-2, -1])

                    pred[0] += xydir_pred_tta
                    pred[1] += agl_pred_tta
                    pred[3] += scale_pred_tta

                    # horizontal flip
                    if use_city:
                        pred_tta = model(
                            (torch.flip(images_rot90, dims=[-2]), city, gsd)
                        )
                    else:
                        pred_tta = model(torch.flip(images_rot90, dims=[-2]))
                    xydir_pred_tta, agl_pred_tta, _, scale_pred_tta = pred_tta
                    xydir_pred_tta[:, 1] *= -1
                    agl_pred_tta = torch.flip(agl_pred_tta, dims=[-2])
                    xydir_pred_tta = torch.stack(
                        [-xydir_pred_tta[:, 1], xydir_pred_tta[:, 0]], dim=1
                    )
                    agl_pred_tta = torch.rot90(agl_pred_tta, k=-1, dims=[-2, -1])

                    pred[0] += xydir_pred_tta
                    pred[1] += agl_pred_tta
                    pred[3] += scale_pred_tta

                    # vertical+horizontal flip
                    if use_city:
                        pred_tta = model(
                            (torch.flip(images_rot90, dims=[-1, -2]), city, gsd)
                        )
                    else:
                        pred_tta = model(torch.flip(images_rot90, dims=[-1, -2]))
                    xydir_pred_tta, agl_pred_tta, _, scale_pred_tta = pred_tta
                    xydir_pred_tta *= -1
                    agl_pred_tta = torch.flip(agl_pred_tta, dims=[-1, -2])
                    xydir_pred_tta = torch.stack(
                        [-xydir_pred_tta[:, 1], xydir_pred_tta[:, 0]], dim=1
                    )
                    agl_pred_tta = torch.rot90(agl_pred_tta, k=-1, dims=[-2, -1])

                    pred[0] += xydir_pred_tta
                    pred[1] += agl_pred_tta
                    pred[3] += scale_pred_tta

            pred = [
                torch.mean(preds[0][:, :bs], dim=0),
                torch.mean(preds[1][:, :bs], dim=0),
                torch.mean(preds[2][:, :bs], dim=0),
            ]

            torch.cuda.synchronize()

            numpy_preds = []
            for i in range(len(pred)):
                numpy_preds.append(pred[i].cpu().numpy())

            xydir_pred, agl_pred, scale_pred = numpy_preds

            if scale_pred.ndim == 0:
                scale_pred = np.expand_dims(scale_pred, axis=0)

            for batch_ind in range(agl_pred.shape[0]):
                # vflow pred
                angle = np.arctan2(xydir_pred[batch_ind][0], xydir_pred[batch_ind][1])
                vflow_data = {
                    "scale": np.float64(scale_pred[batch_ind]),  # upsample
                    "angle": np.float64(angle),
                }

                rgb_path = predictions_dir / Path(rgb_paths[batch_ind]).name

                # agl pred
                curr_agl_pred = agl_pred[batch_ind, 0, :, :]
                # curr_agl_pred[curr_agl_pred < 0] = 0
                curr_agl_pred = np.clip(
                    curr_agl_pred, 0.0, MAX_HEIGTS[rgb_path.stem.split("_")[0]]
                )
                agl_resized = curr_agl_pred

                # save
                agl_path = rgb_path.with_name(
                    rgb_path.name.replace("_RGB", "_AGL")
                ).with_suffix(".tif")
                vflow_path = rgb_path.with_name(
                    rgb_path.name.replace("_RGB", "_VFLOW")
                ).with_suffix(".json")

                json.dump(vflow_data, vflow_path.open("w"))
                save_image(agl_path, agl_resized)  # save_image assumes units of meters

            iterator.update()

    # creates new dir predictions_dir_con
    iterator.close()

    if args.convert_predictions_to_cm_and_compress:
        convert_and_compress_prediction_dir(
            predictions_dir=predictions_dir,
        )
