from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F
import faiss

from utilities.misc_utils import load_image
from utilities.unet_vflow import UnetVFLOW


def l2norm_numpy(x):
    return x / np.linalg.norm(x, ord=2, axis=1, keepdims=True)


def build_model(backbone):
    model = UnetVFLOW(
        backbone,
        encoder_weights="imagenet",
        decoder_use_batchnorm=True,
        decoder_channels=[256, 128, 64, 32, 16],
    )
    return model


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM,self).__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)


def extract_train_features():
    traindir = "data/train-orig-resolution/trainval/"
    checkpoint = "data/working/models/v25_rs101e/model_155.pth"
    backbone = "timm-resnest101e"

    pooling = GeM()
    pooling.to("cuda")
    pooling.eval()

    model = build_model(backbone)
    model.load_state_dict(torch.load(checkpoint))
    model.to("cuda")
    model.eval()
    preprocessing_fn = smp.encoders.get_preprocessing_fn(backbone, "imagenet")
    args = argparse.Namespace(unit="m", nan_placeholder=65535)

    path_list = list(Path(traindir).glob("*_RGB.tif"))
    for idx, rgb_path in enumerate(path_list):
        if idx % 50 == 0:
            print("Processing {} of {}...".format(idx, len(path_list)))

        rows = []
        imgs = []
        for i in range(4):
            for j in range(4):
                img = load_image(str(rgb_path), args)
                img = preprocessing_fn(img).astype("float32")
                img = np.transpose(img, (2, 0, 1))
                img = img[:, 512*i:512*(i+1), 512*j:512*(j+1)]
                imgs.append(img)
                rows.append(dict(x0=512 * i, y0=512 * j))

        imgs = np.stack(imgs, axis=0)
        X = torch.from_numpy(imgs)
        X = X.to("cuda")

        with torch.no_grad():
            out = model.encoder.forward(X)
            feats = pooling(out[-1]).squeeze().data.cpu().numpy()
            feats = l2norm_numpy(feats.astype(np.float32))

        name = rgb_path.stem
        feat_path = f"data/working/search/train/{name}.npy"
        meta_path = f"data/working/search/train/{name}.csv"
        pd.DataFrame(rows).to_csv(meta_path, index=False)
        np.save(feat_path, feats)

        del X, feats, out


def extract_test_features():
    testdir = "data/test/"
    checkpoint = "data/working/models/v25_rs101e/model_155.pth"
    backbone = "timm-resnest101e"

    pooling = GeM()
    pooling.to("cuda")
    pooling.eval()

    model = build_model(backbone)
    model.load_state_dict(torch.load(checkpoint))
    model.to("cuda")
    model.eval()
    preprocessing_fn = smp.encoders.get_preprocessing_fn(backbone, "imagenet")
    args = argparse.Namespace(unit="m", nan_placeholder=65535)

    path_list = list(Path(testdir).glob("*_RGB.j2k"))
    for idx, rgb_path in enumerate(path_list):
        if idx % 50 == 0:
            print("Processing {} of {}...".format(idx, len(path_list)))

        rows = []
        imgs = []
        for i in range(4):
            for j in range(4):
                img = load_image(str(rgb_path), args)
                img = preprocessing_fn(img).astype("float32")
                img = np.transpose(img, (2, 0, 1))
                img = img[:, 512*i:512*(i+1), 512*j:512*(j+1)]
                imgs.append(img)
                rows.append(dict(x0=512*i, y0=512*j))

        imgs = np.stack(imgs, axis=0)
        X = torch.from_numpy(imgs)
        X = X.to("cuda")

        with torch.no_grad():
            out = model.encoder.forward(X)
            feats = pooling(out[-1]).squeeze().data.cpu().numpy()
            feats = l2norm_numpy(feats.astype(np.float32))

        name = rgb_path.stem
        feat_path = f"data/working/search/test/{name}.npy"
        meta_path = f"data/working/search/test/{name}.csv"
        pd.DataFrame(rows).to_csv(meta_path, index=False)
        np.save(feat_path, feats)

        del X, feats, out


def search_with_faiss_cpu(feat_test, feat_index, topk=5):
    n_dim = feat_index.shape[1]
    cpu_index = faiss.IndexFlatIP(n_dim)
    cpu_index.add(feat_index)

    dists, topk_idx = cpu_index.search(x=feat_test, k=topk)

    cpu_index.reset()
    del cpu_index

    return dists, topk_idx



def load_feats_names():
    feats = []
    dfs = []
    for npy_file in Path("data/working/search/test/").glob("*.npy"):
        feat = np.load(npy_file)
        df = pd.read_csv(str(npy_file).replace(".npy", ".csv"))
        df["name"] = npy_file.stem

        feats.append(feat)
        dfs.append(df)

    return np.vstack(feats), pd.concat(dfs, sort=False)


def load_feats_names_train():
    feats = []
    dfs = []
    for npy_file in Path("data/working/search/train/").glob("*.npy"):
        feat = np.load(npy_file)
        df = pd.read_csv(str(npy_file).replace(".npy", ".csv"))
        df["name"] = npy_file.stem

        feats.append(feat)
        dfs.append(df)

    return np.vstack(feats), pd.concat(dfs, sort=False)


def search_test_train_rev3():
    print("loading feats")
    feat_train, df_train = load_feats_names_train()
    assert len(df_train) == 94768
    assert feat_train.shape == (94768, 2048)

    print("loading feats")
    feat_test, df_test = load_feats_names()
    assert len(df_test) == 16400
    assert feat_test.shape == (16400, 2048)

    print("search_with_faiss_cpu")
    dists, topk_idx = search_with_faiss_cpu(feat_test, feat_train, topk=100)

    print("dump")
    rows = []
    for idx, (_, r) in enumerate(df_test.iterrows()):
        for j in range(100):
            sim_idx = topk_idx[idx, j]
            r2 = df_train.iloc[sim_idx]
            rows.append({
                "query": r["name"],
                "query_x0": r["x0"],
                "query_y0": r["y0"],
                "hit": r2["name"],
                "hit_rank": j,
                "hit_score": dists[idx, j],
                "hit_x0": r2["x0"],
                "hit_y0": r2["y0"],
            })

    search_result_path = f"data/working/search/test_train_search_result_rev3.csv"
    pd.DataFrame(rows).to_csv(search_result_path, index=False)


if __name__ == "__main__":
    # extract_train_features()
    # extract_test_features()
    # search_test_train_rev3()
    pass
