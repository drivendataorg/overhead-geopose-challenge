import os.path
import re
from pathlib import Path

import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset as BaseDataset, DataLoader
import numpy as np
from tqdm import tqdm

from utilities.unet import TimmUnetFeat


class Dataset(BaseDataset):
    def __init__(
            self,
            dataset_dir,
            rgb_suffix
    ):
        dataset_dir = Path(dataset_dir)
        print(dataset_dir)
        rgb_paths = [f for f in list(dataset_dir.glob(f"*_RGB.{rgb_suffix}"))]

        self.paths_list = rgb_paths

    def __getitem__(self, i):
        rgb_path = self.paths_list[i]
        image = cv2.imread(str(rgb_path.absolute()), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (800, 800))
        image = (np.transpose(image, (2, 0, 1)) / 255. - 0.5) * 2
        return image, str(rgb_path)

    def __len__(self):
        return len(self.paths_list)


modelf6 = TimmUnetFeat("tf_efficientnetv2_l_in21k")
# checkpoint = torch.load("weights/folds_TimmUnet_tf_efficientnetv2_l_in21k_6_r2", map_location="cpu")
# state_dict = checkpoint['state_dict']
# state_dict = {re.sub("^module.", "", k): w for k, w in state_dict.items()}
#modelf6.load_state_dict(state_dict)
modelf6.cuda()
modelf6.eval()

data = []
indices = []
with torch.no_grad():

    test_dataset = Dataset(dataset_dir="/home/selim/data/trainpng", rgb_suffix="png")
    test_loader = DataLoader(
        test_dataset, batch_size=1, sampler=None, shuffle=False, num_workers=4, pin_memory=True
    )
    for i, (images, rgb_paths) in enumerate(tqdm(test_loader)):
        images = images.float().cuda()
        feat = modelf6(images).cpu().numpy()[0]
        fid = os.path.basename(rgb_paths[0]).replace("_RGB.png", "")
        data.append(feat)
        indices.append([fid, i])
pd.DataFrame(indices, columns=["fid", "index"]).to_csv("feature_indices.csv", index=False)
np.save("features", np.asarray(data))

a = np.load("features.npy")
print(a.shape)