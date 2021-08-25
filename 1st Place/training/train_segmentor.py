import argparse
import json
import os
import traceback
from typing import Dict


os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

from utilities.evaluate import create_arg_parser, evaluate_r2
import cv2

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

from tqdm import tqdm

import torch.distributed as dist

from training.geopose_dataset import GeoposeDataset
from utilities.misc_utils import save_image
import numpy as np



import cv2

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)
from training.trainer import TrainConfiguration, PytorchTrainer, Evaluator

from torch.utils.data import DataLoader
import torch.distributed



class GeoposeEvaluator(Evaluator):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args

    def init_metrics(self) -> Dict:
        return {"r2": 0}

    def validate(self, dataloader: DataLoader, model: torch.nn.Module, distributed: bool = False, local_rank: int = 0,
                 snapshot_name: str = "") -> Dict:

        for sample in tqdm(dataloader):
                image = sample["image"]
                city_ohe = sample["city_ohe"]
                agl = sample["agl"]
                cities = sample['city']
                names = sample['name']

                image = image.cuda().float()
                city_ohe = city_ohe.cuda().float()
                with torch.no_grad():
                    output = model.forward(image.float(), city_ohe.float())
                    xydir_pred, agl_pred, mag_pred, scale_pred = output['xydir'], output['height'], output['mag'], output[
                        'scale']
                    scale_pred = torch.unsqueeze(scale_pred, 1)
                agl = agl.cpu().detach().numpy()
                xydir_pred = xydir_pred.cpu().detach().numpy()
                agl_pred = agl_pred.cpu().detach().numpy()
                scale_pred = scale_pred.cpu().detach().numpy()

                for batch_ind in range(agl.shape[0]):
                    city = cities[batch_ind]
                    name = names[batch_ind]
                    # vflow pred
                    angle = np.arctan2(xydir_pred[batch_ind][0], xydir_pred[batch_ind][1])
                    vflow_data = {
                        "scale": np.float64(
                            scale_pred[batch_ind]
                        ),  # upsample
                        "angle": np.float64(angle),
                    }

                    # agl pred
                    curr_agl_pred = agl_pred[batch_ind, 0, :, :]
                    curr_agl_pred[curr_agl_pred < 0] = 0
                    agl_resized = curr_agl_pred

                    # save
                    os.makedirs("preds", exist_ok=True)
                    agl_path = os.path.join("preds", f"{city}_{name}_AGL.tif")
                    vflow_path = os.path.join("preds", f"{city}_{name}_VFLOW.json")

                    with open(vflow_path, "w") as f:
                        json.dump(vflow_data, f)
                    save_image(agl_path, agl_resized)
        torch.cuda.synchronize()
        res = {}
        if local_rank == 0:
            e_args = create_arg_parser().parse_args('')
            e_args.truth_dir = self.args.train_dir
            e_args.predictions_dir = "preds"
            e_args.output_dir = "output"
            e_args.num_processes = os.cpu_count()
            res = {"r2": evaluate_r2(e_args)}
        dist.barrier()
        torch.cuda.empty_cache()
        return res

    def get_improved_metrics(self, prev_metrics: Dict, current_metrics: Dict) -> Dict:
        improved = {}
        if current_metrics["r2"] > prev_metrics["r2"]:
            print("R2 AGL improved from {:.4f} to {:.4f}".format(prev_metrics["r2"], current_metrics["r2"]))
            improved["r2"] = current_metrics["r2"]
        else:
            print("Best R2 AGL {:.4f} current {:.4f}".format(prev_metrics["r2"], current_metrics["r2"]))
        return improved


def parse_args():
    parser = argparse.ArgumentParser(" Segmentor Pipeline")
    arg = parser.add_argument
    arg('--config', metavar='CONFIG_FILE', help='path to configuration file', default="configs/b3.json")
    arg('--workers', type=int, default=16, help='number of cpu threads to use')
    arg('--gpu', type=str, default='0', help='List of GPUs for parallel training, e.g. 0,1,2,3')
    arg('--output-dir', type=str, default='weights/')
    arg('--resume', type=str, default='')
    arg('--fold', type=int, default=0)
    arg('--prefix', type=str, default='')
    arg('--city', type=str, default='')
    arg('--data-dir', type=str, default="/wdata/train/")
    arg('--train-dir', type=str, default="/mnt/sota/datasets/geopose/train/")
    arg('--preds-dir', type=str, default="/preds/")
    arg('--folds-csv', type=str, default='folds.csv')
    arg('--logdir', type=str, default='logs')
    arg('--zero-score', action='store_true', default=False)
    arg('--from-zero', action='store_true', default=False)
    arg('--distributed', action='store_true', default=False)
    arg("--local_rank", default=0, type=int)
    arg("--world-size", default=1, type=int)
    arg("--test_every", type=int, default=1)
    arg('--freeze-epochs', type=int, default=0)
    arg('--crop', type=int, default=1024)
    arg('--multiplier', type=int, default=1)

    args = parser.parse_args()

    return args


def create_data_datasets(args):
    train_dataset = GeoposeDataset(mode="train", dataset_dir=args.data_dir, folds_csv=args.folds_csv, fold=args.fold,
                                   crop_size=args.crop, multiplier=args.multiplier)
    val_dataset = GeoposeDataset(mode="val", dataset_dir=args.data_dir, folds_csv=args.folds_csv, fold=args.fold,
                                 crop_size=args.crop)
    if args.city:
        train_dataset.set_city(args.city)
        val_dataset.set_city(args.city)
    return train_dataset, val_dataset


def main():
    args = parse_args()
    trainer_config = TrainConfiguration(
        config_path=args.config,
        gpu=args.gpu,
        resume_checkpoint=args.resume,
        prefix=args.prefix,
        world_size=args.world_size,
        test_every=args.test_every,
        local_rank=args.local_rank,
        distributed=args.distributed,
        freeze_epochs=args.freeze_epochs,
        log_dir=args.logdir,
        output_dir=args.output_dir,
        workers=args.workers,
        from_zero=args.from_zero,
        zero_score=args.zero_score
    )

    data_train, data_val = create_data_datasets(args)
    seg_evaluator = GeoposeEvaluator(args)
    trainer = PytorchTrainer(train_config=trainer_config, evaluator=seg_evaluator, fold=args.fold,
                             train_data=data_train, val_data=data_val)
    try:
        trainer.fit()
    except:
        traceback.print_exc()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
