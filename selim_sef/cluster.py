import json
from collections import defaultdict

import  pandas as pd
import numpy as np
from scipy.spatial.distance import cosine
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm

df = pd.read_csv("feature_indices.csv")
feats = np.load("features.npy")
feats = feats[df.index.values]
indices = df.index.values
fids = df.fid.values

cl = AgglomerativeClustering(distance_threshold=0.05, linkage="complete", n_clusters=None, affinity ="cosine")
model = cl.fit(feats)

clusters = defaultdict(list)
for i, lbl in enumerate(model.labels_):
    clusters[str(lbl)].append(fids[i])

with open("clusters.json", "w") as f:
    json.dump(clusters, f)