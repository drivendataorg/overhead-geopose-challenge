import json
from collections import defaultdict
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold
import numpy as np
CITIES = ["ARG", "ATL", "JAX", "OMA"]

if __name__ == '__main__':
    with open("clusters.json", "r") as f:
        all_clusters = json.load(f)
    data = []
    for city in CITIES:
        clusters  = np.array([cl for cl, ids in all_clusters.items() if ids[0].startswith(city)])

        kfold = KFold(n_splits=10)
        for i, (train_idx, test_idx) in enumerate(kfold.split(clusters)):
            for cl in clusters[test_idx]:
                data.append([city, cl, i])
    pd.DataFrame(data, columns=["city", "cluster", "fold"]).to_csv("folds_clusters.csv", index=False)


