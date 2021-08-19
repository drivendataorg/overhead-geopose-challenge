import argparse
import os

import pandas as pd
from numpy.random import RandomState
from sklearn.model_selection import StratifiedKFold

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Gen folds")
    arg = parser.add_argument
    arg('--data-dir', type=str, default="/home/selim/data", help='path to root dataset folder')
    arg('--folds', type=int, default=10)
    args = parser.parse_args()

    df = pd.read_csv(os.path.join(args.data_dir, "geopose_train.csv"))
    x = []
    y = []
    for i, row in df.iterrows():
        x.append(row.id)
        y.append(row.rgb.split("_")[0])
    kfold = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=RandomState(seed=777))
    data = []
    for fold, (_, test_idx) in enumerate(kfold.split(x, y)):
        for i in test_idx:
            data.append([x[i], y[i], fold])
    df = pd.DataFrame(data, columns=["id", "city", "fold"])
    df.to_csv("folds.csv", index=False)
    print(len(df[df.fold != 0]))
    print(len(df[df.fold == 1]))