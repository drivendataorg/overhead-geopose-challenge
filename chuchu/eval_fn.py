import ever as er
import torch
import os
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
import json
from utilities.misc_utils import save_image


class Evaluator(object):
    def __init__(self):
        self.angle_errors = []
        self.scale_errors = []
        self.mag_errors = []
        self.agl_errors = []
        self.endpoint_errors = []
        self.mag_rmss = []
        self.endpoint_rmss = []
        self.agl_rmss = []

        self.agl_count = 0.
        self.agl_sse = 0.
        self.agl_gt_sum = 0.
        self.vflow_count = 0.
        self.vflow_gt_sum = 0.
        self.vflow_sse = 0.

        self.gt_agls = []
        self.gt_vflows = []

    def comput_angle_error(self, pr_xydir, gt_xydir):
        pr_xydir /= np.linalg.norm(pr_xydir)
        gt_xydir /= np.linalg.norm(gt_xydir)

        cos_ang = np.dot(pr_xydir, gt_xydir)
        sin_ang = np.linalg.norm(np.cross(pr_xydir, gt_xydir))
        rad_diff = np.arctan2(sin_ang, cos_ang)
        angle_error = np.degrees(rad_diff)

        self.angle_errors.append(angle_error)
        return angle_error

    def comput_scale_error(self, pr_scale, gt_scale):
        scale_error = np.abs(pr_scale - gt_scale)

        self.scale_errors.append(scale_error)
        return scale_error

    def comput_mag_error(self, pr_mag, gt_mag, valid_mask=None):
        ae = np.abs(pr_mag - gt_mag)
        se = np.square(pr_mag - gt_mag)
        if valid_mask is not None:
            ae = ae.ravel()[valid_mask.ravel()]
            se = se.ravel()[valid_mask.ravel()]

        mag_error = np.nanmean(ae)
        mag_rms = np.sqrt(np.nanmean(se))

        self.mag_errors.append(mag_error)
        self.mag_rmss.append(mag_rms)
        return mag_error, mag_rms

    def comput_agl_error(self, pr_agl, gt_agl, valid_mask=None):
        ae = np.abs(pr_agl - gt_agl)
        se = np.square(pr_agl - gt_agl)
        if valid_mask is not None:
            ae = ae.ravel()[valid_mask.ravel()]
            se = se.ravel()[valid_mask.ravel()]

        agl_error = np.nanmean(ae)
        agl_rms = np.sqrt(np.nanmean(se))

        self.agl_errors.append(agl_error)
        self.agl_rmss.append(agl_rms)

        agl_count = np.sum(np.isfinite(gt_agl))
        agl_sse = np.nansum(se)
        agl_gt_sum = np.nansum(gt_agl)

        self.agl_count += agl_count
        self.agl_sse += agl_sse
        self.agl_gt_sum += agl_gt_sum

        self.gt_agls.append(gt_agl)
        return agl_error, agl_rms, agl_count, agl_sse, agl_gt_sum

    def comput_endpoint_error(self,
                              pr_xydir, pr_mag,
                              gt_xydir, gt_mag,
                              valid_mask=None):
        pr_vflow = self.generate_vflow(pr_xydir, pr_mag)
        gt_vflow = self.generate_vflow(gt_xydir, gt_mag)

        se = np.square(pr_vflow - gt_vflow)
        sse = np.sum(se, axis=2)
        if valid_mask is not None:
            se = se.reshape((-1, 2))[valid_mask.ravel(), :]
            sse = sse.ravel()[valid_mask.ravel()]

        epe = np.nanmean(np.sqrt(sse))
        epe_rms = np.sqrt(np.nanmean(sse))

        self.endpoint_errors.append(epe)
        self.endpoint_rmss.append(epe_rms)

        vflow_count = np.sum(np.isfinite(gt_vflow))
        vflow_gt_sum = np.nansum(gt_vflow)
        vflow_sse = np.nansum(se)

        self.vflow_count += vflow_count
        self.vflow_gt_sum += vflow_gt_sum
        self.vflow_sse += vflow_sse

        self.gt_vflows.append(gt_vflow)

    def generate_vflow(self, xy_dir, mag):
        vflow = np.zeros((mag.shape[0], mag.shape[1], 2))
        vflow[:, :, 0] = mag * xy_dir[0]
        vflow[:, :, 1] = mag * xy_dir[1]
        return vflow

    def forward(self,
                pr_xydir, pr_agl, pr_mag, pr_scale,
                gt_xydir, gt_agl, gt_mag, gt_scale,
                valid_mask=None):
        self.comput_angle_error(pr_xydir, gt_xydir)
        if pr_scale is not None:
            self.comput_scale_error(pr_scale, gt_scale)
        self.comput_mag_error(pr_mag, gt_mag, valid_mask=valid_mask)
        self.comput_agl_error(pr_agl, gt_agl, valid_mask=valid_mask)
        self.comput_endpoint_error(pr_xydir, pr_mag, gt_xydir, gt_mag, valid_mask=valid_mask)

    def summary(self, logger):
        mean_angle_error = np.nanmean(self.angle_errors)
        rms_angle_error = np.sqrt(np.nanmean(np.square(self.angle_errors)))
        if len(self.scale_errors) > 0:
            mean_scale_error = np.nanmean(self.scale_errors)
            rms_scale_error = np.sqrt(np.nanmean(np.square(self.scale_errors)))
        mean_mag_error = np.nanmean(self.mag_errors)
        rms_mag_error = np.sqrt(np.nanmean(np.square(self.mag_rmss)))
        mean_epe = np.nanmean(self.endpoint_errors)
        rms_epe = np.sqrt(np.nanmean(np.square(self.endpoint_rmss)))
        mean_agl_error = np.nanmean(self.agl_errors)
        rms_agl_error = np.sqrt(np.nanmean(np.square(self.agl_rmss)))

        agl_gt_mean = self.agl_gt_sum / (self.agl_count + 0.0001)
        vflow_gt_mean = self.vflow_gt_sum / (self.vflow_count + 0.0001)

        agl_denoms = []
        vflow_denoms = []
        for gt_agl_i, gt_vflow_i in zip(self.gt_agls, self.gt_vflows):
            agl_denom = np.nansum(np.square(gt_agl_i - agl_gt_mean))
            agl_denoms.append(agl_denom)
            vflow_denom = np.nansum(np.square(gt_vflow_i - vflow_gt_mean))
            vflow_denoms.append(vflow_denom)

        agl_denom = np.sum(agl_denoms)
        vflow_denom = np.sum(vflow_denoms)

        agl_R2 = 1.0 - (self.agl_sse / (agl_denom + 0.0001))
        vflow_R2 = 1.0 - (self.vflow_sse / (vflow_denom + 0.0001))

        logger.info("Angle error: %f" % mean_angle_error)
        if len(self.scale_errors) > 0:
            logger.info("Scale error: %f" % mean_scale_error)
        logger.info("Mag error: %f" % mean_mag_error)
        logger.info("EPE: %f" % mean_epe)
        logger.info("AGL error: %f" % mean_agl_error)
        logger.info("ROOT MEAN SQUARE ERROR")
        logger.info("Angle error: %f" % rms_angle_error)
        if len(self.scale_errors) > 0:
            logger.info("Scale error: %f" % rms_scale_error)
        logger.info("Mag error: %f" % rms_mag_error)
        logger.info("EPE error: %f" % rms_epe)
        logger.info("AGL error: %f" % rms_agl_error)
        logger.info(f"AGL R-square: {agl_R2}")
        logger.info(f"VFLOW R-square: {vflow_R2}")
        return (vflow_R2 + agl_R2) / 2.0


def keep_positive_agl(agl):
    return F.relu(agl, inplace=True)


def geopose_evaluate_fn(self, test_dataloader, config=None):
    self.model.eval()
    self.model.to(er.auto_device())

    pred_output_dir = os.path.join(self.model_dir, 'eval_output')
    os.makedirs(pred_output_dir, exist_ok=True)
    # downsample = 2

    evaluators = {
        'JAX': Evaluator(),
        'ATL': Evaluator(),
        'OMA': Evaluator(),
        'ARG': Evaluator(),
    }
    with torch.no_grad():
        for x, y in tqdm(test_dataloader, desc='EVAL'):
            xydir, agl, mag, scale = self.model(x.to(er.auto_device()))
            xydir = xydir.cpu().numpy()
            mag = mag.cpu().squeeze(dim=1).numpy()
            angle = np.arctan2(xydir[:, 0], xydir[:, 1])

            if scale is not None:
                scale = scale.cpu().numpy()
            else:
                scale = [None] * xydir.shape[0]

            agl = keep_positive_agl(agl)
            agl = agl.cpu().squeeze(dim=1).numpy()

            for i, rgb_name in enumerate(y['rgb_name']):
                vflow_data = {
                    'angle': np.float64(angle[i])
                }
                if scale[i] is not None:
                    vflow_data['scale'] = np.float64(scale[i])
                # save
                flow_name = rgb_name.replace('RGB', 'VFLOW').replace('j2k', 'json')
                with open(os.path.join(pred_output_dir, flow_name), 'w') as f:
                    json.dump(vflow_data, f)
                agl_name = rgb_name.replace('RGB', 'AGL').replace('j2k', 'tif')
                save_image(os.path.join(pred_output_dir, agl_name), agl[i])
                for city in ['ATL', 'ARG', 'OMA', 'JAX']:
                    if city in rgb_name:
                        evaluators[city].forward(xydir[i], agl[i], mag[i], scale[i],
                                                 y['xydir'][i].numpy(),
                                                 y['agl'][i].numpy(),
                                                 y['mag'][i].numpy(),
                                                 y['scale'][i].numpy())
                        break

        # city-wise
        scores = {}
        for k, v in evaluators.items():
            self.logger.info(f'City: {k}')
            scores[k] = v.summary(self.logger)
        # overall
        self.logger.info('Overall')
        final_score = sum(list(scores.values())) / 4.
        for k, v in scores.items():
            self.logger.info(f'{k} R-square: {v}')

        self.logger.info(f'FINAL R-square: {final_score}')


def us3d_evaluate_fn(self, test_dataloader, config=None):
    self.model.eval()
    self.model.to(er.auto_device())

    pred_output_dir = os.path.join(self.model_dir, 'eval_output')
    os.makedirs(pred_output_dir, exist_ok=True)

    evaluator = Evaluator()
    fg_evaluator = Evaluator()
    with torch.no_grad():
        for x, y in tqdm(test_dataloader, desc='EVAL'):
            xydir, agl, mag, scale = self.model(x.to(er.auto_device()))
            xydir = xydir.cpu().numpy()
            mag = mag.cpu().squeeze(dim=1).numpy()
            angle = np.arctan2(xydir[:, 0], xydir[:, 1])

            if scale is not None:
                scale = scale.cpu().numpy()
            else:
                scale = [None] * xydir.shape[0]

            agl = keep_positive_agl(agl)
            agl = agl.cpu().squeeze(dim=1).numpy()

            for i, rgb_name in enumerate(y['rgb_name']):
                vflow_data = {
                    'angle': np.float64(angle[i])
                }
                if scale[i] is not None:
                    vflow_data['scale'] = np.float64(scale[i])
                # save
                flow_name = rgb_name.replace('RGB', 'VFLOW').replace('tif', 'json')
                with open(os.path.join(pred_output_dir, flow_name), 'w') as f:
                    json.dump(vflow_data, f)

                agl_name = rgb_name.replace('RGB', 'AGL')
                save_image(os.path.join(pred_output_dir, agl_name), agl[i])

                pr_agl = agl[i]
                pr_angle = xydir[i]
                pr_scale = scale[i]

                if 'use_gt_angle' in config and config.use_gt_angle:
                    pr_angle = y['xydir'][i].numpy()
                if 'use_gt_agl' in config and config.use_gt_agl:
                    pr_agl = y['agl'][i].numpy()
                if 'use_gt_scale' in config and config.use_gt_scale:
                    pr_scale = y['scale'][i].numpy()
                if pr_scale is not None:
                    pr_mag = pr_agl * pr_scale
                else:
                    pr_mag = mag[i]

                evaluator.forward(pr_angle, pr_agl, pr_mag, pr_scale,
                                  y['xydir'][i].numpy(),
                                  y['agl'][i].numpy(),
                                  y['mag'][i].numpy(),
                                  y['scale'][i].numpy())

                gt_agl = y['agl'][i].numpy()
                valid = gt_agl > 0.5
                fg_evaluator.forward(pr_angle, pr_agl, pr_mag, pr_scale,
                                     y['xydir'][i].numpy(),
                                     y['agl'][i].numpy(),
                                     y['mag'][i].numpy(),
                                     y['scale'][i].numpy(),
                                     valid_mask=valid)

    self.logger.info('------Overall Region Eval------')
    final_score = evaluator.summary(self.logger)
    self.logger.info(f'FINAL R-square: {final_score}')

    self.logger.info('------Above Ground Region Eval------')
    final_score = fg_evaluator.summary(self.logger)
    self.logger.info(f'Above Ground FINAL R-square: {final_score}')
