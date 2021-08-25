import os
import cv2

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

import pandas as pd
import pydegensac
import time
from copy import deepcopy
from multiprocessing import Pool
import tqdm
import json
from pathlib import Path

import numpy as np
from osgeo import gdal

UNITS_PER_METER_CONVERSION_FACTORS = {"cm": 100.0, "m": 1.0}


def load_image(path: Path) -> np.ndarray:
    nan_placeholder: int = 65535
    assert path.exists()
    raster = gdal.Open(str(path))
    im_array = raster.ReadAsArray().astype("float32")

    # AGL
    if path.stem.endswith("_AGL"):
        np.putmask(im_array, im_array == nan_placeholder, np.nan)
        units_per_meter = UNITS_PER_METER_CONVERSION_FACTORS["cm"]
        im_array = im_array / units_per_meter

    # RGB
    if len(im_array.shape) == 3:
        im_array = np.transpose(im_array, [1, 2, 0])

    return im_array


def verify_pydegensac(kps1, kps2, tentatives, th = 4.0,  n_iter = 2000):
    src_pts = np.float32([ kps1[m.queryIdx].pt for m in tentatives ]).reshape(-1,2)
    dst_pts = np.float32([ kps2[m.trainIdx].pt for m in tentatives ]).reshape(-1,2)
    H, mask = pydegensac.findHomography(src_pts, dst_pts, th, 0.99, n_iter)
    return H, mask


def stereo_matching(rgb_rct1, rgb_rct2):
    det = cv2.AKAZE_create(descriptor_type = 3, threshold=0.00001)
    kps1, descs1 = det.detectAndCompute(rgb_rct1, None)
    kps2, descs2 = det.detectAndCompute(rgb_rct2, None)

    for i in range(len(kps1)):
        kps1[i].size = 5*kps1[i].size
    for i in range(len(kps2)):
        kps2[i].size = 5*kps2[i].size

    vis_img1, vis_img2 = None,None
    vis_img1 = cv2.drawKeypoints(cv2.cvtColor(rgb_rct1,cv2.COLOR_RGB2GRAY),kps1,vis_img1,
                                 flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    vis_img2 = cv2.drawKeypoints(cv2.cvtColor(rgb_rct2,cv2.COLOR_RGB2GRAY),kps2,vis_img2,
                                 flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descs2, descs1, k=2)

    matchesMask = [False for i in range(len(matches))]

    for i,(m,n) in enumerate(matches):
        if m.distance < 0.9*n.distance:
            matchesMask[i]=True
    tentatives = [m[0] for i, m in enumerate(matches) if matchesMask[i] ]

    return kps1, kps2, tentatives


def count_inliers(src_rgb, dst_rgb):
    rgb_rct1 = load_image(Path(src_rgb)).astype(np.uint8)
    rgb_rct1 = cv2.resize(rgb_rct1, (1024, 1024))
    rgb_rct2 = load_image(Path(dst_rgb)).astype(np.uint8)
    rgb_rct2 = cv2.resize(rgb_rct2, (1024, 1024))

    det = cv2.AKAZE_create(descriptor_type = 3, threshold=0.00001)
    kps1, descs1 = det.detectAndCompute(rgb_rct1, None)
    kps2, descs2 = det.detectAndCompute(rgb_rct2, None)

    # AKAZE features output "detection" scale, which is 6x less than one used for descriptor
    # For correct drawing, we increase it manually
    for i in range(len(kps1)):
        kps1[i].size = 5*kps1[i].size
    for i in range(len(kps2)):
        kps2[i].size = 5*kps2[i].size

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descs2, descs1, k=2)

    # Need to draw only good matches, so create a mask
    matchesMask = [False for i in range(len(matches))]

    # SNN ratio test
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.9*n.distance:
            matchesMask[i]=True
    tentatives = [m[0] for i, m in enumerate(matches) if matchesMask[i] ]

    _, mask = verify_pydegensac(kps2, kps1, tentatives, th=4.0, n_iter=2000)
    return int(deepcopy(mask).astype(np.float32).sum())


def proc(image_pair):
    src_rgb, dst_rgb = image_pair
    inlier_count = count_inliers(src_rgb, dst_rgb)
    return (src_rgb, dst_rgb, inlier_count)


def match_test_train_rev3():
    verify_result_path = f"data/working/search/test_train_verify_result_rev3.csv"
    assert Path(verify_result_path).exists()

    df = pd.read_csv(verify_result_path)
    df["query"] = df["query"].str[:10]
    df["hit"] = df["hit"].str[:10]

    df = df.sort_values(by="inlier_count", ascending=False)
    df = df[df["inlier_count"] > 500]
    rows = [(r["query"], r["hit"]) for _, r in df.iterrows()]

    th = 4.0
    n_iter = 20000

    Path("data/working/search/match").mkdir(parents=True, exist_ok=True)
    for _, (query_image_id, hit_image_id) in tqdm.tqdm(enumerate(rows), total=len(rows)):
        fn_cmp_h = f"data/working/search/match/{query_image_id}_{hit_image_id}_matchH.npy"
        fn_match_res = f"data/working/search/match/{query_image_id}_{hit_image_id}_matchMeta.json"
        if Path(fn_match_res).exists():
            continue

        rgb1 = load_image(Path(f"data/test/{query_image_id}_RGB.j2k"))
        rgb2 = load_image(Path(f"data/train-orig-resolution/trainval/{hit_image_id}_RGB.tif"))

        kps1, kps2, tentatives = stereo_matching(rgb1.astype(np.uint8), rgb2.astype(np.uint8))
        cmp_H, cmp_mask = verify_pydegensac(kps2, kps1, tentatives, th, n_iter)
        inlier_count = int(deepcopy(cmp_mask).astype(np.float32).sum())

        rgb2_transformed = cv2.warpPerspective(rgb2, cmp_H, rgb2.shape[:2])
        mask = cv2.warpPerspective(np.ones(rgb2.shape), cmp_H, rgb2.shape[:2])
        pix_rmse = np.abs(rgb1[mask == 1] - rgb2_transformed[mask == 1]).mean()

        if False:
            print(json.dumps({
                "inlier_count": inlier_count,
                "pix_rmse": float(pix_rmse),
            }, indent=4))

        np.save(str(Path(fn_cmp_h)), cmp_H)
        json.dump({
            "inlier_count": inlier_count,
            "pix_rmse": float(pix_rmse),
        }, Path(fn_match_res).open("w"))


def verify_test_train_rev3():
    search_result_path = f"data/working/search/test_train_search_result_rev3.csv"
    verify_result_path = f"data/working/search/test_train_verify_result_rev3.csv"

    df = pd.read_csv(search_result_path)
    df = df[df["hit_rank"] < 80]
    df = df[df["query"] != df["hit"]]
    df = df[df["query"].str[:3] == df["hit"].str[:3]]
    df = df.sort_values(by="hit_rank", ascending=True)
    df = df.groupby(["query", "hit"], as_index=False).agg("first")
    df = df.sort_values(by="hit_rank", ascending=True)

    ret_list = []
    image_pairs = []
    for i, (_, r) in enumerate(df.iterrows()):
        src_rgb = "data/test/{}.j2k".format(r["query"])
        dst_rgb = "data/train-orig-resolution/trainval/{}.tif".format(r["hit"])
        image_pairs.append((src_rgb, dst_rgb))

    print(len(df), len(ret_list), len(image_pairs))
    image_pairs = list(reversed(sorted(image_pairs)))

    ttl_cnt = len(image_pairs)
    with Pool(processes=45) as pool:
        for i, res in enumerate(tqdm.tqdm(pool.imap_unordered(proc, image_pairs), total=ttl_cnt)):
            src_rgb, dst_rgb, inlier_count = res
            ret_list.append({
                "idx": i,
                "query": Path(src_rgb).stem,
                "hit": Path(dst_rgb).stem,
                "inlier_count": inlier_count,
            })
            if i % 300 == 0:
                pd.DataFrame(ret_list).to_csv(verify_result_path, index=False)

    pd.DataFrame(ret_list).to_csv(verify_result_path, index=False)
