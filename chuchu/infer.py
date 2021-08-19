import fire
import ever as er
import os
from data.geopose import Geopose
from torch.utils.data import DataLoader
import albumentations as A
import cv2
from tqdm import tqdm
import torch
import torch.nn.functional as F
import numpy as np
from utilities.misc_utils import save_image, convert_and_compress_prediction_dir
import json

IMG_SIZE = 2048


def rotate_xydir(xdir, ydir, rotate_angle):
    base_angle = torch.rad2deg(torch.atan2(xdir, ydir))
    xdir = torch.sin(torch.deg2rad(base_angle + rotate_angle))
    ydir = torch.cos(torch.deg2rad(base_angle + rotate_angle))
    return xdir, ydir


def keep_positive_agl(agl):
    return F.relu(agl, inplace=True)


def flip_predict(model, image, dim, info):
    if dim == "x":
        image = torch.flip(image, dims=[3])
        xydir, agl, mag, scale = model(image, info)
        agl = torch.flip(agl, dims=[3])
        mag = torch.flip(mag, dims=[3])
        xydir[:, 0] *= -1

    elif dim == "y":
        image = torch.flip(image, dims=[2])
        xydir, agl, mag, scale = model(image, info)
        agl = torch.flip(agl, dims=[2])
        mag = torch.flip(mag, dims=[2])
        xydir[:, 1] *= -1

    return xydir, agl, mag, scale


def rotate90_predict(model, image, k, info):
    assert k in [1, 2, 3]
    image = torch.rot90(image, k, [2, 3])
    xydir, agl, mag, scale = model(image, info)

    agl = torch.rot90(agl, 4 - k, [2, 3])
    mag = torch.rot90(mag, 4 - k, [2, 3])

    xydir = torch.stack(rotate_xydir(xydir[:, 0], xydir[:, 1], 90 * (4 - k)), dim=1)
    return xydir, agl, mag, scale


def tta_predict(model, image, info):
    xydir0, agl0, mag0, scale0 = model(image, info)
    xydir1, agl1, mag1, scale1 = flip_predict(model, image, dim='x', info=info)
    xydir2, agl2, mag2, scale2 = flip_predict(model, image, dim='y', info=info)
    xydir3, agl3, mag3, scale3 = rotate90_predict(model, image, k=1, info=info)
    xydir4, agl4, mag4, scale4 = rotate90_predict(model, image, k=2, info=info)
    xydir5, agl5, mag5, scale5 = rotate90_predict(model, image, k=3, info=info)

    xydir = (xydir0 + xydir1 + xydir2 + xydir3 + xydir4 + xydir5) / 6.
    agl = (agl0 + agl1 + agl2 + agl3 + agl4 + agl5) / 6.
    mag = (mag0 + mag1 + mag2 + mag3 + mag4 + mag5) / 6.
    scale = (scale0 + scale1 + scale2 + scale3 + scale4 + scale5) / 6.

    return xydir, agl, mag, scale


def single_scale_predict(model, image, size=None, scale=None):
    org_h, org_w = image.size(2), image.size(3)
    s_img = F.interpolate(image, size=size, scale_factor=scale, mode='bilinear',
                          align_corners=True)
    _, s_agl, _, _ = model(s_img)
    agl = F.interpolate(s_agl, size=(org_h, org_w), mode='bilinear', align_corners=True)
    return agl


def ms_predict(model, image, sizes=(768, 1280)):
    xydir, agl, mag, scale = model(image)
    agls = [single_scale_predict(model, image, size=s) for s in sizes] + [agl]

    m_agl = sum(agls) / (1 + len(sizes))
    return xydir, m_agl, mag, scale


def competition(model_dir,
                test_data_dir,
                test_csv_file,
                output_dir,
                downsample=2,
                convert_predictions_to_cm_and_compress=True,
                tta=False,
                city='all',
                batch_size=4,
                mst=False,
                ):
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_name = ''
    for name in os.listdir(model_dir):
        if name.endswith('.pth'):
            checkpoint_name = name
            break

    model, _ = er.infer_tool.build_from_model_dir(model_dir, checkpoint_name)
    model.to(er.auto_device())

    dataset = Geopose(test_data_dir, test_csv_file, rgb_ext='j2k', city=city)
    dataset.transforms = A.Compose([
        A.Resize(IMG_SIZE // downsample, IMG_SIZE // downsample, interpolation=cv2.INTER_NEAREST),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                    max_pixel_value=255.0),
        er.preprocess.albu.ToTensor()
    ])

    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=2, pin_memory=True)

    with torch.no_grad():
        for img, info in tqdm(dataloader, desc=f'tta = {tta}, mst = {mst}'):
            img = img.to(er.auto_device())
            if tta:
                xydir, agl, mag, scale = tta_predict(model, img, info)
            elif mst:
                xydir, agl, mag, scale = ms_predict(model, img, mst)
            else:
                xydir, agl, mag, scale = model(img, info)

            xydir = xydir.cpu().numpy()

            angle = np.arctan2(xydir[:, 0], xydir[:, 1])
            scale = (downsample * scale).cpu().numpy()

            agl = keep_positive_agl(agl)
            agl = F.interpolate(agl, size=(downsample * agl.size(2),
                                           downsample * agl.size(3)),
                                mode='nearest').cpu().squeeze(dim=1).numpy()

            for i, rgb_name in enumerate(info['rgb_name']):
                vflow_data = {
                    'scale': np.float64(scale[i]),
                    'angle': np.float64(angle[i])
                }
                # save
                flow_name = rgb_name.replace('RGB', 'VFLOW').replace('j2k', 'json')
                with open(os.path.join(output_dir, flow_name), 'w') as f:
                    json.dump(vflow_data, f)
                agl_name = rgb_name.replace('RGB', 'AGL').replace('j2k', 'tif')
                save_image(os.path.join(output_dir, agl_name), agl[i])

    if convert_predictions_to_cm_and_compress:
        convert_and_compress_prediction_dir(output_dir)


if __name__ == '__main__':
    er.registry.register_modules()
    fire.Fire(dict(
        competition=competition,
    ))
