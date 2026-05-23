import argparse
import os
import random
import time
import warnings
from datetime import datetime

import imageio
import numpy as np
import torch
import torch.multiprocessing
import torch.nn.functional as F
import torchvision.transforms
import yaml
from shapely.errors import ShapelyDeprecationWarning
from tabulate import tabulate
from tensorboardX import SummaryWriter

import nuscenes_data 

#import torch_tensorrt
#from  nuscenes_data import EgoPose_Class

# os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import saverloader
import utils.basic
import utils.geom
import utils.improc
import utils.misc
import utils.vox


from nets.segnet_transformer_lift_fuse_new_decoders import (
    SegnetTransformerLiftFuse,
)

import csv
import pandas as pd
import math
import scipy.spatial.transform


'''
depth
'''
import imageio
import cv2  
import matplotlib.pyplot as plt
import math

scene_number_data = None
scene_sample_data = None


# Suppress deprecation warnings from shapely regarding the nuscenes map api
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning, module="nuscenes.map_expansion.map_api")

torch.multiprocessing.set_sharing_strategy('file_system')

# Global perf knobs (no accuracy impact for inference):
#  - cuDNN auto-tuner: fixed input shapes (B=1, 6 cams, 448x896) ⇒ benchmark wins.
#  - TF32 on matmul/conv: trivial speedup for any remaining fp32 GEMMs (e.g. fp32-forced MSDA).
#  - 'high' matmul precision activates TF32 in tf32-capable kernels.
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_float32_matmul_precision('high')
except Exception:
    pass

random.seed(125)
np.random.seed(125)

# the scene centroid is defined w.r.t. a reference camera
# which is usually random
scene_centroid_x = 0.0
scene_centroid_y = 1.0
scene_centroid_z = 0.0

scene_centroid_py = np.array([scene_centroid_x,
                              scene_centroid_y,
                              scene_centroid_z]).reshape([1, 3])
scene_centroid = torch.from_numpy(scene_centroid_py).float()

XMIN, XMAX = -50, 50
ZMIN, ZMAX = -50, 50
YMIN, YMAX = -5, 5
bounds = (XMIN, XMAX, YMIN, YMAX, ZMIN, ZMAX)

Z, Y, X = 200, 8, 200

val_day_len = 111   # 4449 (samples) -> 111 (scenes)
val_rain_len = 24   # 968 (samples) ->  24 (scenes)
val_night_len = 15  # 602 (samples) ->  15 (scenes)


def update_metrics(metric_prefix: str, condition_metrics_dict: dict,  metrics_model: dict) -> None:
    intersections_key = f'{metric_prefix}_intersections'
    unions_key = f'{metric_prefix}_unions'
    iou_key = f'{metric_prefix}_iou'
    condition_metrics_dict[intersections_key] += metrics_model[intersections_key]
    condition_metrics_dict[unions_key] += metrics_model[unions_key]
    condition_metrics_dict[iou_key] = 100 * condition_metrics_dict[intersections_key] /\
        condition_metrics_dict[unions_key]


def update_range_metrics(metric_prefix: str, range_metric_dict: dict, metrics_model: dict) -> None:
    for range_suffix in ['0_20', '20_35', '35_50']:
        update_metrics(f'{metric_prefix}_{range_suffix}', range_metric_dict, metrics_model)


def update_and_calculate_map_metrics(eval_status: str, metrics: dict, map_metrics: dict, iou_labels: list[str]) \
        -> tuple[dict, float]:
    for key in map_metrics.keys():
        if key == 'map_seg_thresholds':
            map_metrics[key] = metrics[key]
        else:
            map_metrics[key] += metrics[key]
            
    map_ious = {f'{label}': 100 * map_metrics['map_masks_intersections'][i] /
                map_metrics['map_masks_unions'][i] for i, label in enumerate(iou_labels)}
    mean_map_iou = 100 * (map_metrics['map_masks_intersections'] / map_metrics['map_masks_unions'])
    mean_map_iou = mean_map_iou.sum() / torch.count_nonzero(mean_map_iou)
    return map_ious, mean_map_iou


def calculate_best_map_ious_and_thresholds(intersections: torch.Tensor, unions: torch.Tensor, thresholds: torch.Tensor):
    multi_map_ious = intersections / unions
    best_map_ious, best_threshold_index = torch.max(multi_map_ious, dim=1)
    best_thresholds = thresholds[best_threshold_index]
    best_map_mean_iou = best_map_ious.sum(dim=0) / torch.count_nonzero(best_map_ious, dim=0)
    return best_map_ious, best_thresholds, best_map_mean_iou


def format_value(value):
    if isinstance(value, torch.Tensor):
        return f"{value.item():.3f}"
    return f"{float(value):.3f}"


def display_final_results(train_task, dset, obj_metrics, day_metrics, rain_metrics, night_metrics,
                          map_metrics, day_map_metrics, rain_map_metrics, night_map_metrics,
                          mean_map_iou, map_ious, day_mean_map_iou, day_map_ious,
                          rain_mean_map_iou, rain_map_ious, night_mean_map_iou, night_map_ious, do_drn_val_split):

    print("##################   FINAL RESULTS   ###################")
    print("##################   OBJ IOUs  ###################")
    if train_task == 'both' or train_task == 'object':
        obj_data = [
            ["ALL", format_value(obj_metrics['obj_iou']), format_value(obj_metrics['obj_0_20_iou']),
             format_value(obj_metrics['obj_20_35_iou']), format_value(obj_metrics['obj_35_50_iou'])],

            ["DAY", format_value(day_metrics['obj_iou']), format_value(day_metrics['obj_0_20_iou']),
             format_value(day_metrics['obj_20_35_iou']),
             format_value(day_metrics['obj_35_50_iou'])] if do_drn_val_split else ["DAY", "-", "-", "-", "-"],

            ["RAIN", format_value(rain_metrics['obj_iou']), format_value(rain_metrics['obj_0_20_iou']),
             format_value(rain_metrics['obj_20_35_iou']),
             format_value(rain_metrics['obj_35_50_iou'])] if do_drn_val_split else ["RAIN", "-", "-", "-", "-"],

            ["NIGHT", format_value(night_metrics['obj_iou']), format_value(night_metrics['obj_0_20_iou']),
             format_value(night_metrics['obj_20_35_iou']),
             format_value(night_metrics['obj_35_50_iou'])] if do_drn_val_split else ["NIGHT", "-", "-", "-", "-"]
        ]

        headers = ["", "mean obj_IoU", "0-20m obj_IoU", "20-35m obj_IoU", "35-50m obj_IoU"]
        print(tabulate(obj_data, headers=headers, tablefmt="pretty"))
        print('##############################################################')

    if train_task == 'both' or train_task == 'map':
        print("##################   MAP IOUs (UNIFORM THRESHOLD = 40%) ###################")
        map_data = [
            ["ALL", format_value(mean_map_iou), format_value(map_ious['drivable_iou'].item()),
             format_value(map_ious['carpark_iou'].item()), format_value(map_ious['ped_cross_iou'].item()),
             format_value(map_ious['walkway_iou'].item())],

            ["DAY", format_value(day_mean_map_iou), format_value(day_map_ious['drivable_iou'].item()),
             format_value(day_map_ious['carpark_iou'].item()), format_value(day_map_ious['ped_cross_iou'].item()),
             format_value(day_map_ious['walkway_iou'].item()), format_value(day_map_ious['stop_line_iou'].item()),
             format_value(day_map_ious['road_divider_iou'].item()),
             format_value(day_map_ious['lane_divider_iou'].item())] if do_drn_val_split
            else ["DAY", "-", "-", "-", "-", "-", "-", "-", "-"],

            ["RAIN", format_value(rain_mean_map_iou), format_value(rain_map_ious['drivable_iou'].item()),
             format_value(rain_map_ious['carpark_iou'].item()), format_value(rain_map_ious['ped_cross_iou'].item()),
             format_value(rain_map_ious['walkway_iou'].item()), format_value(rain_map_ious['stop_line_iou'].item()),
             format_value(rain_map_ious['road_divider_iou'].item()),
             format_value(rain_map_ious['lane_divider_iou'].item())] if do_drn_val_split
            else ["RAIN", "-", "-", "-", "-", "-", "-", "-", "-"],

            ["NIGHT", format_value(night_mean_map_iou), format_value(night_map_ious['drivable_iou'].item()),
             format_value(night_map_ious['carpark_iou'].item()), format_value(night_map_ious['ped_cross_iou'].item()),
             format_value(night_map_ious['walkway_iou'].item()), format_value(night_map_ious['stop_line_iou'].item()),
             format_value(night_map_ious['road_divider_iou'].item()),
             format_value(night_map_ious['lane_divider_iou'].item())] if do_drn_val_split
            else ["NIGHT", "-", "-", "-", "-", "-", "-", "-", "-"]
        ]

        headers = ["", "mean map_IoU", "drivable_IoU", "carpark_IoU", "ped_cross_IoU", "walkway_IoU"]
        
        print(tabulate(map_data, headers=headers, tablefmt="pretty"))

        print("##################   BEST MAP IOUs (CLASS-SPECIFIC THRESHOLD)  ###################")
        best_map_ious, best_thresholds, best_map_mean_iou = calculate_best_map_ious_and_thresholds(
            intersections=map_metrics['map_masks_multi_ious_intersections'],
            unions=map_metrics['map_masks_multi_ious_unions'],
            thresholds=map_metrics['map_seg_thresholds'])

        day_best_map_ious, day_best_thresholds, day_best_map_mean_iou = calculate_best_map_ious_and_thresholds(
            intersections=day_map_metrics['map_masks_multi_ious_intersections'],
            unions=day_map_metrics['map_masks_multi_ious_unions'],
            thresholds=day_map_metrics['map_seg_thresholds'])

        rain_best_map_ious, rain_best_thresholds, rain_best_map_mean_iou = calculate_best_map_ious_and_thresholds(
            intersections=rain_map_metrics['map_masks_multi_ious_intersections'],
            unions=rain_map_metrics['map_masks_multi_ious_unions'],
            thresholds=rain_map_metrics['map_seg_thresholds'])

        night_best_map_ious, night_best_thresholds, night_best_map_mean_iou = calculate_best_map_ious_and_thresholds(
            intersections=night_map_metrics['map_masks_multi_ious_intersections'],
            unions=night_map_metrics['map_masks_multi_ious_unions'],
            thresholds=night_map_metrics['map_seg_thresholds'])

        best_data = [
            ["ALL", format_value(best_map_mean_iou*100), *[f"{x * 100:.3f}" for x in best_map_ious]],
            ["DAY", format_value(day_best_map_mean_iou*100), *[f"{x * 100:.3f}" for x in day_best_map_ious]]
            if do_drn_val_split else ["DAY", "-", "-", "-", "-", "-", "-", "-"],
            ["RAIN", format_value(rain_best_map_mean_iou*100), *[f"{x * 100:.3f}" for x in rain_best_map_ious]]
            if do_drn_val_split else ["RAIN", "-", "-", "-", "-", "-", "-", "-"],
            ["NIGHT", format_value(night_best_map_mean_iou*100), *[f"{x * 100:.3f}" for x in night_best_map_ious]]
            if do_drn_val_split else ["NIGHT", "-", "-", "-", "-", "-", "-", "-"]
        ]
        # [f"{x * 100:.3f}" for x in best_map_ious]  (torch.round(best_map_ious*100000)/1000)
        headers = ["", "best map_IoU", "drivable_IoU", "carpark_IoU", "ped_cross_IoU", "walkway_IoU", "stop_line_IoU",
                   "road_divider_IoU", "lane_divider_IoU"]
        
        print(tabulate(best_data, headers=headers, tablefmt="pretty"))

        print("##################   BEST CLASS-SPECIFIC THRESHOLD ###################")
        thresholds_data = [
            ["ALL", *(torch.round(best_thresholds*100))],
            ["DAY", *(torch.round(day_best_thresholds*100))] if do_drn_val_split
            else ["DAY", "-", "-", "-", "-", "-", "-", "-"],
            ["RAIN", *(torch.round(rain_best_thresholds*100))] if do_drn_val_split
            else ["RAIN", "-", "-", "-", "-", "-", "-", "-"],
            ["NIGHT", *(torch.round(night_best_thresholds*100))] if do_drn_val_split
            else ["NIGHT", "-", "-", "-", "-", "-", "-", "-"]
        ]

        headers = ["", "drivable_th", "carpark_th", "ped_cross_th", "walkway_th", "stop_line_th", "road_divider_th",
                   "lane_divider_th"]
        
        print(tabulate(thresholds_data, headers=headers, tablefmt="pretty"))


def requires_grad(parameters: iter, flag: bool = True) -> None:
    """
        Sets the `requires_grad` attribute of the given parameters.
        Args:
            parameters (iterable): An iterable of parameter tensors whose `requires_grad` attribute will be set.
            flag (bool, optional): If True, sets `requires_grad` to True. If False, sets it to False.
                Default is True.

        Returns:
            None
        """
    for p in parameters:
        p.requires_grad = flag


class SimpleLoss(torch.nn.Module):
    """
    SimpleLoss module that computes the binary cross-entropy loss.

    Args:
        pos_weight (float): Positive class weight for the binary cross-entropy loss.

    Methods:
        forward(ypred: torch.Tensor, ytgt: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
            Forward pass that computes the binary cross-entropy loss.
    """
    def __init__(self, pos_weight: float):
        """Initializes the SimpleLoss module with the specified positive class weight."""
        super(SimpleLoss, self).__init__()
        self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([pos_weight]), reduction='none')

    def forward(self, ypred: torch.Tensor, ytgt: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that computes the binary cross-entropy loss.

        Args:
            ypred (torch.Tensor): Predicted logits.
            ytgt (torch.Tensor): Target tensor.
            valid (torch.Tensor): Mask indicating valid elements.

        Returns:
            torch.Tensor: The computed loss.
        """
        loss = self.loss_fn(ypred, ytgt)
        loss = utils.basic.reduce_masked_mean(loss, valid)
        return loss


class SigmoidFocalLoss(torch.nn.Module):
    """
        Computes the sigmoid of the model output to get values between 0 and 1, then applies the Focal Loss.
    """
    def __init__(self, alpha: float = -1.0, gamma: int = 2, reduction: str = "mean"):
        """
        Args:
            alpha (float, optional): Balances the importance of positive/negative examples. Default is -1.0.
            gamma (int, optional): If >= 0, reduces the loss contribution from easy examples
            and extends the range in which an example receives low loss. Default is 2.
            reduction (str, optional): Specifies the reduction to apply to the output. Options are 'mean', 'sum',
            and 'sum_of_class_means'. Default is 'mean'.
        """
        super(SigmoidFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
        '''added for sampled scene by kjh 241021'''
        self.scene_num :int = 0

    def forward(self, map_seg_e: torch.Tensor, map_seg_gt: torch.Tensor):
        """
        Forward pass that computes the sigmoid focal loss.

        Args:
            map_seg_e (torch.Tensor): Predicted logits.
            map_seg_gt (torch.Tensor): Target tensor.

        Returns:
                torch.Tensor: The computed loss.
        """
        # get predictions between 0 and 1
        p = torch.sigmoid(map_seg_e)
        # BCE with logits
        ce_loss = F.binary_cross_entropy_with_logits(input=map_seg_e, target=map_seg_gt, reduction="none")
        p_t = p * map_seg_gt + (1 - p) * (1 - map_seg_gt)
        f_loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * map_seg_gt + (1 - self.alpha) * (1 - map_seg_gt)
            f_loss = alpha_t * f_loss
        else:
            f_loss = f_loss

        if self.reduction == "mean":  # get mean over all classes
            f_loss = f_loss.mean()
        elif self.reduction == "sum":
            f_loss = f_loss.sum()
        elif self.reduction == "sum_of_class_means":
            '''
            f_loss = f_loss.mean(dim=[2,3])  # mean over bev map space
            f_loss = f_loss.mean(dim=0)  # mean over batch dim -> results in sum over class errors
            f_loss = f_loss.sum()
            '''
            # mean over B and bev grid -> then sum avg class error
            f_loss = f_loss.mean(dim=[0, 2, 3]).sum()
        return f_loss


def run_model(loader, index, model, loss_fn, map_seg_loss_fn, d, img_dir, device='cuda:2', use_pseudolidar_encoder=None,
              pseudolidar_encoder_type=None, train_task='both', use_shallow_metadata=True,
              use_obj_layer_only_on_map=False, model_name=None , eval_status="All", ncams=None,
              iou_csv_dir=None):
    
    metrics = {
        'map_masks_intersections': torch.zeros(7, device=device),
        'map_masks_unions': torch.zeros(7, device=device),
        'map_masks_multi_ious_intersections': torch.zeros((7, 12), device=device),
        'map_masks_multi_ious_unions': torch.zeros((7, 12), device=device),
        'map_seg_thresholds': torch.zeros(12, device=device),
        'obj_intersections': 0, 'obj_unions': 0, 'obj_0_20_intersections': 0, 'obj_0_20_unions': 0,
        'obj_20_35_intersections': 0, 'obj_20_35_unions': 0, 'obj_35_50_intersections': 0, 'obj_35_50_unions': 0
    }
    
    total_scene_loss = torch.tensor(0.0, requires_grad=False).to(device)

    voxel_input_feature_buffer_all = None
    voxel_coordinate_buffer_all = None
    number_of_occupied_voxels_all = None
    in_occ_mem0 = None
    plus_drivalbe_iou = 0.0
    plus_lane_iou = 0.0
    
    valid_drivable_cnt = 0

    

    if pseudolidar_encoder_type == "voxel_net":
        # voxelnet
        imgs_all, rots_all, trans_all, intrins_all, seg_bev_g_all, valid_bev_g_all, \
            pseudolidar_data_all, bev_map_mask_g_all, bev_map_g_all, egocar_bev_all, \
            voxel_input_feature_buffer_all, voxel_coordinate_buffer_all, number_of_occupied_voxels_all, all_transformate, all_rotation, all_depth_img= d 
            
            
    else:
        imgs_all, rots_all, trans_all, intrins_all, seg_bev_g_all, valid_bev_g_all, \
            pseudolidar_data_all, bev_map_mask_g_all, bev_map_g_all, egocar_bev_all,all_transformate, all_rotation, all_depth_img= d 
    
    T = imgs_all.shape[1]  # problem: if T is 39,40,OR 41 --> not consistent --> check
    
    
    folder_name = os.path.join(img_dir, model_name + "_scene_%03d" % index)
    os.makedirs(folder_name, exist_ok=True)
    metrics_name = os.path.join(folder_name, "000_metrics_scene_%03d.txt" % index)
    with open(file=metrics_name, mode='w') as f:
        f.write('####### Metrics for: ' + model_name + '  SCENE: ' + str(index) + ' ####### \n\n')

    # ALL SCENES
    
    scene_obj_intersections = 0
    scene_obj_unions = 0
    # 0 - 20 m
    scene_obj_0_20_intersections = 0
    scene_obj_0_20_unions = 0
    # 20 - 35 m
    scene_obj_20_35_intersections = 0
    scene_obj_20_35_unions = 0
    # 35 - 50 m
    scene_obj_35_50_intersections = 0
    scene_obj_35_50_unions = 0

    scene_map_intersections = torch.zeros(7, requires_grad=False, device=device)
    scene_map_unions = torch.zeros(7, requires_grad=False, device=device)

    # 추론 시간 누적 (paper-style: torch.cuda.synchronize 양쪽 + time.perf_counter)
    scene_inference_total_t = 0.0
    scene_inference_count = 0

    # warmup: torch.compile/cudnn benchmark 첫 forward는 매우 느림 — 측정에서 제외
    # 환경변수 VIS_EVAL_WARMUP_FRAMES=N (default 0)으로 제어
    try:
        warmup_frames = int(os.environ.get("VIS_EVAL_WARMUP_FRAMES", "1"))
    except ValueError:
        warmup_frames = 50
    if warmup_frames > 0:
        print(f"[warmup] excluding first {warmup_frames} frame(s) from inference timing")

    for t in range(T):
        
        print("Sample: " + str(t))
        total_loss = torch.tensor(0.0, requires_grad=False).to(device)
        sample_obj_iou = 0
        sample_obj_0_20_iou = 0
        sample_obj_20_35_iou = 0
        sample_obj_35_50_iou = 0
        sample_map_iou = 0

        voxel_input_feature_buffer = None
        voxel_coordinate_buffer = None
        number_of_occupied_voxels = None

        # eliminate the time dimension
        transformation = all_transformate[:, t]
        rotation = all_rotation[:, t]
        
        imgs = imgs_all[:, t]
        rots = rots_all[:, t]
        trans = trans_all[:, t]
        #print(f'transtranstrans {len(trans[0][0])}')
        #print(f'transtranstrans {trans_all}')
        #print(f'trans {trans}')
        
        #print(f'rots {rots}')
        #print(f'trans_all {len(trans_all[0])}')
        #print(f'rotsrotsrots {rots[0]}')
        #print(f'rotsrotsrots {len(rots[0][0])}')
        #print(f'trans { trans[0]}')
        intrins = intrins_all[:, t]
        seg_bev_g = seg_bev_g_all[:, t]
        valid_bev_g = valid_bev_g_all[:, t]
        pseudolidar_data = pseudolidar_data_all[:, t]

        '''depth'''
        depth_img = all_depth_img[:, t]

        # for own map from mask
        bev_map_mask_g = bev_map_mask_g_all[:, t]
        if use_obj_layer_only_on_map:
            bev_map_mask_g = bev_map_mask_g[:, :-1]  # remove attached object class
        bev_map_g = bev_map_g_all[:, t]
        
        # added egocar in bev plane
        egocar_bev = egocar_bev_all[:, t]

        if pseudolidar_encoder_type == "voxel_net":
            voxel_input_feature_buffer = voxel_input_feature_buffer_all[:, t]
            voxel_coordinate_buffer = voxel_coordinate_buffer_all[:, t]
            number_of_occupied_voxels = number_of_occupied_voxels_all[:, t]
            voxel_input_feature_buffer = voxel_input_feature_buffer.to(device)
            voxel_coordinate_buffer = voxel_coordinate_buffer.to(device)
            number_of_occupied_voxels = number_of_occupied_voxels.to(device)

        rgb_camXs = imgs.float().to(device)
        rgb_camXs = rgb_camXs - 0.5  # go to -0.5, 0.5


        '''depth'''
        depth_camXs = depth_img.float().to(device)
        depth_camXs = depth_camXs - 0.5  # go to -0.5, 0.5

        seg_bev_g = seg_bev_g.to(device)
        obj_seg_bev_e = torch.zeros_like(seg_bev_g)
        valid_bev_g = valid_bev_g.to(device)
        # added bev_map_gt
        bev_map_mask_g = bev_map_mask_g.to(device)
        bev_map_g = bev_map_g.to(device)
        bev_map_e = torch.zeros_like(bev_map_g)
        # added egocar in bev plane
        egocar_bev = egocar_bev.to(device)

        # create ego car color plane
        ego_plane = torch.zeros_like(bev_map_g).to(device)
        ego_plane[:, [0, 2]] = 0.0
        ego_plane[:, 1] = 1.0
        
        
        
        # combine ego car and map
        ego_car_on_map_g = bev_map_g * (1 - egocar_bev) + ego_plane * egocar_bev
        # get ego car on plane
        ego_car_on_plane = ego_plane * egocar_bev

        # create other cars plane
        other_cars_plane = torch.zeros_like(bev_map_g).to(device)
        other_cars_plane[:, [0, 1]] = 0.0
        other_cars_plane[:, 2] = 1.0
        
        # combine ego car other cars and map
        ego_other_cars_on_map_g = ego_car_on_map_g * (1 - seg_bev_g) + other_cars_plane * seg_bev_g
        ego_other_cars_on_map_e = torch.zeros_like(ego_other_cars_on_map_g)
        # combine ego car with other cars -> no map
        ego_other_cars_g = ego_car_on_plane * (1 - seg_bev_g) + other_cars_plane * seg_bev_g

        rad_data = pseudolidar_data.to(device).permute(0, 2, 1)  # B, R, data
        xyz_rad = rad_data[:, :, :3]
        meta_rad = rad_data[:, :, 3:]
        shallow_meta_rad = rad_data[:, :, 5:8]

        B, S, C, H, W = rgb_camXs.shape

        def __p(x):
            # Wrapper function: e.g. unites B,S dim to B*S
            return utils.basic.pack_seqdim(x, B)

        def __u(x):
            # Wrapper function: e.g. splits B*S dim into B,S
            return utils.basic.unpack_seqdim(x, B)

        intrins_ = __p(intrins)
        pix_T_cams_ = utils.geom.merge_intrinsics(*utils.geom.split_intrinsics(intrins_)).to(device)
        pix_T_cams = __u(pix_T_cams_)

        velo_T_cams = utils.geom.merge_rtlist(rots, trans).to(device)
        cams_T_velo = __u(utils.geom.safe_inverse(__p(velo_T_cams)))

        cam0_T_camXs = utils.geom.get_camM_T_camXs(velo_T_cams, ind=0)
        rad_xyz_cam0 = utils.geom.apply_4x4(cams_T_velo[:, 0], xyz_rad)

        vox_util = utils.vox.Vox_util(
            Z, Y, X,
            scene_centroid=scene_centroid.to(device),
            bounds=bounds,
            assert_cube=False)

        if not model.module.use_pseudolidar:
            in_occ_mem0 = None
        elif model.module.use_pseudolidar and (model.module.use_metapseudolidar or use_shallow_metadata):
            # rad_occ_mem0 for vis only
            rad_occ_mem0 = vox_util.voxelize_xyz(rad_xyz_cam0, Z, Y, X, assert_cube=False)
            if use_pseudolidar_encoder and pseudolidar_encoder_type == 'voxel_net':
                voxelnet_feats_mem0 = voxel_input_feature_buffer, voxel_coordinate_buffer, number_of_occupied_voxels
                in_occ_mem0 = voxelnet_feats_mem0
            elif use_shallow_metadata:
                shallow_metarad_occ_mem0 = vox_util.voxelize_xyz_and_feats(rad_xyz_cam0, shallow_meta_rad, Z, Y, X,
                                                                           assert_cube=False)
                in_occ_mem0 = shallow_metarad_occ_mem0
            else:  # use_metapseudolidar
                metarad_occ_mem0 = vox_util.voxelize_xyz_and_feats(rad_xyz_cam0, meta_rad, Z, Y, X, assert_cube=False)
                in_occ_mem0 = metarad_occ_mem0
        elif model.module.use_pseudolidar:
            rad_occ_mem0 = vox_util.voxelize_xyz(rad_xyz_cam0, Z, Y, X, assert_cube=False)
            in_occ_mem0 = rad_occ_mem0
        elif model.module.use_metapseudolidar or use_shallow_metadata:
            assert False  # cannot use_metapseudolidar without use_pseudolidar
    
    # for  lighting added
    #with torch.amp.autocast('cuda', enabled=True):

        # paper-style forward pass timing (SegMam Table 5 caption 기준):
        # - torch.cuda.synchronize() 로 forward 양쪽을 명시적으로 감싸 GPU async 보정
        # - time.perf_counter() (monotonic, high-res) 사용
        # - 모델 내부 per-stage cuda.Event/_maybe_sync 는 main() 에서
        #   SEGMAM_DISABLE_PROFILING=1 로 강제 비활성화되어 no-op
        torch.cuda.synchronize()
        start_inference_t = time.perf_counter()
        # 안전한 가속: half-precision autocast (frozen DINOv2 + voxel_net 모두 안정).
        # VIS_EVAL_AUTOCAST_DTYPE ∈ {"fp16","bf16","none"} — default fp16.
        # bf16 은 Blackwell 에서 동급 속도이고 dynamic range 넓어서 가끔 더 안정.
        _ac_dtype = os.environ.get("VIS_EVAL_AUTOCAST_DTYPE", "fp16").lower()
        # torch.inference_mode(): 평가 단계에서 autograd 추적/version counter 를 완전히
        # 끄고, 모델 내부의 torch.utils.checkpoint(...) 호출(예: lifting loop 의
        # `self_mamba_attn_layers_fuser`)을 사실상 no-op 으로 만든다. eval.no_grad
        # 보다 더 가벼움. 그래프 노드 생성/저장이 없으므로 메모리·런치 오버헤드 큰 감소.
        with torch.inference_mode():
            if _ac_dtype == "none":
                model_out = model(
                    rgb_camXs=rgb_camXs,
                    pix_T_cams=pix_T_cams,
                    cam0_T_camXs=cam0_T_camXs,
                    vox_util=vox_util,
                    rad_occ_mem0=in_occ_mem0)
            else:
                _ac_torch_dtype = torch.bfloat16 if _ac_dtype == "bf16" else torch.float16
                with torch.amp.autocast('cuda', dtype=_ac_torch_dtype):
                    model_out = model(
                        rgb_camXs=rgb_camXs,
                        pix_T_cams=pix_T_cams,
                        cam0_T_camXs=cam0_T_camXs,
                        vox_util=vox_util,
                        rad_occ_mem0=in_occ_mem0)
        torch.cuda.synchronize()
        inference_t = time.perf_counter() - start_inference_t

        # 모델은 (seg_e, timings) 튜플을 반환.
        # VIS_EVAL_DEBUG_STAGES=1 일 때만 stage timing 을 한 번 출력 (진단용).
        # 이 옵션은 SEGMAM_DISABLE_PROFILING=0 일 때 의미 있다.
        _stages_dict = None
        if isinstance(model_out, tuple):
            seg_e = model_out[0]
            if len(model_out) >= 2 and isinstance(model_out[1], dict):
                _stages_dict = model_out[1]
        else:
            seg_e = model_out
        # 후속 metric 계산은 fp32로 (autocast 외부에서 sigmoid/threshold 진행)
        if seg_e is not None and isinstance(seg_e, torch.Tensor) and seg_e.dtype != torch.float32:
            seg_e = seg_e.float()

        # warmup frame은 timing 누적에서 제외 (torch.compile graph build 등이 포함되므로)
        is_warmup = (t < warmup_frames)
        if not is_warmup:
            scene_inference_total_t += inference_t
            scene_inference_count += 1
            # VIS_EVAL_DEBUG_STAGES=1 일 때 stage timing 1회 출력 (warmup 이후 첫 정상 frame).
            # SEGMAM_DISABLE_PROFILING=0 이 필요하므로 같이 켜져야 의미 있음.
            if (os.environ.get("VIS_EVAL_DEBUG_STAGES", "0") == "1"
                    and _stages_dict is not None
                    and any(v > 0 for v in _stages_dict.values())
                    and scene_inference_count == 1):
                print(f"[stage-timing frame {t}] " + ", ".join(
                    f"{k}={v:.1f}" for k, v in _stages_dict.items()))
        else:
            print(f"[warmup] frame {t}: {inference_t*1000:.1f} ms (excluded)")

        # calc metrics
        if train_task == 'both' or train_task == 'map':

            if train_task == 'both':
                bev_map_mask_e = seg_e[:, :-1]
                obj_seg_bev_e = seg_e[:, -1].unsqueeze(dim=1)
                obj_seg_bev = torch.sigmoid(obj_seg_bev_e)

                bev_map_only_mask_g = bev_map_mask_g

            else:
                bev_map_mask_e = seg_e
                obj_seg_bev = seg_bev_g  # add gt vehicles on map (optional)
                bev_map_only_mask_g = bev_map_mask_g

            map_seg_threshold = 0.4
            bev_map_e = nuscenes_data.get_rgba_map_from_mask2_on_batch(
                torch.sigmoid(bev_map_mask_e).detach().cpu().numpy(),
                threshold=map_seg_threshold, a=1.0).to(device)  # a=0.4

            # combine ego car and bev_map_e
            ego_car_on_map_e = bev_map_e * (1 - egocar_bev) + ego_plane * egocar_bev  # check dims

            # create other cars estimate plane
            other_cars_plane_e = torch.zeros_like(bev_map_e).to(device)
            other_cars_plane_e[:, [0, 1]] = 0.0
            other_cars_plane_e[:, 2] = 1.0

            # combine ego car other cars and map
            obj_seg_bev_round = obj_seg_bev.round()
            ego_other_cars_on_map_e = ego_car_on_map_e * (1 - obj_seg_bev_round) + other_cars_plane_e * obj_seg_bev_round

            # loss calculation
            map_seg_fc_loss = map_seg_loss_fn(bev_map_mask_e, bev_map_only_mask_g)
            #   map
            fc_map_factor = 1 / torch.exp(model.module.fc_map_weight)
            map_seg_fc_loss = 20.0 * map_seg_fc_loss * fc_map_factor  # 20.0
            # add to total loss
            total_loss += map_seg_fc_loss


            tp = ((torch.sigmoid(bev_map_mask_e) >= map_seg_threshold).bool() & bev_map_mask_g.bool()).sum(dim=[2, 3])
            tn = (~(torch.sigmoid(bev_map_mask_e) >= map_seg_threshold).bool() & ~bev_map_mask_g.bool()).sum(dim=[2, 3])
            fp = ((torch.sigmoid(bev_map_mask_e) >= map_seg_threshold).bool() & ~bev_map_mask_g.bool()).sum(dim=[2, 3])
            fn = (~(torch.sigmoid(bev_map_mask_e) >= map_seg_threshold).bool() & bev_map_mask_g.bool()).sum(dim=[2, 3])
            pa = tp / (tp + fp + fn)
                  
            map_ious, mean_map_iou = update_and_calculate_map_metrics(eval_status='ALL', metrics=metrics,
                                                                      map_metrics=map_metrics,
                                                                      iou_labels=iou_labels)
            
            #print(f"map_ious { map_ious}")
            
            # map_ious['drivable_iou'].item(),
            #           map_ious['carpark_iou'].item(), map_ious['ped_cross_iou'].item(), map_ious['walkway_iou'].item())
            
            # print(f"map_ious['drivable_iou'].item(), {map_ious['drivable_iou'].item()} ")
            # print(f"map_ious['carpark_iou'].item(), {map_ious['carpark_iou'].item()} ")
            # print(f"map_ious['ped_cross_iou'].item(), {map_ious['ped_cross_iou'].item()} ")
            # print(f"map_ious['walkway_iou'].item(), {map_ious['walkway_iou'].item()} ")


            # #-----------------------------start of iou tp tn fn fp save---------------------------------
            # if index == 2:
            #     m_iou = [round(float(map_ious['drivable_iou'].item()),1), round(float(map_ious['carpark_iou'].item()),1), \
            #             round(float(map_ious['ped_cross_iou'].item()),1),round(float(map_ious['walkway_iou'].item()),1)]
                
            #     #l_m_iou = [round(float(map_ious['drivable_iou'].item()),1), round(float(map_ious['carpark_iou'].item()),1), \
            #      #       round(float(map_ious['ped_cross_iou'].item()),1),round(float(map_ious['walkway_iou'].item()),1)]
                
            #     l_tp = tp.flatten().tolist()
            #     l_tn = tn.flatten().tolist()
            #     l_fp = fp.flatten().tolist()
            #     l_fn = fn.flatten().tolist()
            #     l_pa = pa.flatten().tolist()

            #     #l_m_iou = [0.0 if math.isnan(x) else round(x,2)for x in l_m_iou]
            #     l_tp = [0.0 if math.isnan(x) else round(x,2)for x in l_tp]
            #     l_tn = [0.0 if math.isnan(x) else round(x,2)for x in l_tn]
            #     l_fp = [0.0 if math.isnan(x) else round(x,2)for x in l_fp]
            #     l_fn = [0.0 if math.isnan(x) else round(x,2)for x in l_fn]
            #     l_pa = [0.0 if math.isnan(x) else round(x,2)for x in l_pa]
        

            #     columns = ["da", "cp", "pc", "ww"]
                
            #     data = {
            #         "iou": m_iou,
            #         "pa": l_pa,
            #         "tp": l_tp,
            #         "tn": l_tn,
            #         "fp": l_fp,
            #         "fn": l_fn,
            #     }

            #     df = pd.DataFrame(data, index=columns).transpose()
            #     print(f"df {df}")
            #     scenes_name = 'scene_0590'
            #     output_path = f"/root/data/dataset/pdcms/ogm/{scenes_name}/{scenes_name}_iou.xlsx"

            #     file_exists = os.path.isfile(output_path)

            #     with open(output_path, mode='a', newline='', encoding='utf-8') as file:
            #         if not file_exists:
            #             df.to_csv(file, header=True)
            #         else:
            #             df.to_csv(file, header=False)

            # #-----------------------------end of iou tp tn fn fp save---------------------------------


            map_intersections_per_class = tp.sum(dim=0)  # sum over batch --> 7 intersection values
            map_unions_per_class = (
                    tp.sum(dim=0) + fp.sum(dim=0) + fn.sum(dim=0) + 1e-4)  # sum over batch --> 7 union value
            

            # --------------- trainvalidation 150개 씬에대해서 dataset Drviabble Area 평균 계산 평가 지표 산출을 위해서다. Start ---------

            # drivable_iou = [round(float(map_ious['drivable_iou'].item()),1)]
            # print(f"round(float(map_ious['drivable_iou'].item()),1){ round(float(map_ious['drivable_iou'].item()),1)}")
            # plus_drivalbe_iou = plus_drivalbe_iou + round(float(map_ious['drivable_iou'].item()),1)
            
            # is_last = (t == T - 1)
            
            # if  is_last : 
                
            #     print(f"plus_drivalbe_iou {plus_drivalbe_iou}")
            #     print(f"drivable_iou {drivable_iou}")


            #     plus_drivalbe_iou = [plus_drivalbe_iou / T]
            #     columns = ["drivalble"]
                
            #     data = {
            #         "iou": plus_drivalbe_iou
            #     }

            #     df = pd.DataFrame(data, index=columns).transpose()
            #     output_path = f"/SegMam/model_iou/camsix_iou/drivable_iou_.xlsx"
            #     #/root/data/nuscenes_datasets/nuscenes_datasets/nuscenes
            #     file_exists = os.path.isfile(output_path)

            #     with open(output_path, mode='a', newline='', encoding='utf-8') as file:
            #         if not file_exists:
            #             df.to_csv(file, header=True)
            #         else:
            #             df.to_csv(file, header=False)


            
            #drivable_iou = [round(float(map_ious['drivable_iou'].item()),1)]
            #print(f"round(float(map_ious['drivable_iou'].item()),1){ round(float(map_ious['drivable_iou'].item()),1)}")
            # plus_drivalbe_iou = plus_drivalbe_iou + round(float(map_ious['drivable_iou'].item()),1)
            # plus_lane_iou = plus_lane_iou + round(float(map_ious['lane_divider_iou'].item()),1)
            # is_last = (t == T - 1)
            
            # if  is_last : 
                
            #     # print(f"plus_drivalbe_iou {plus_drivalbe_iou}")
            #     # print(f"drivable_iou {drivable_iou}")


            #     plus_drivalbe_iou = [plus_drivalbe_iou / T]
            #     plus_lane_iou =  [plus_lane_iou / T]
            #     columns = ["drivalble", "lane"]
                
            #     data = {
            #         "drivable": plus_drivalbe_iou,
            #         "lane": plus_lane_iou
            #     }

            #     df = pd.DataFrame(data, index=columns).transpose()
            #     output_path = f"/SegMam/model_iou/camsix_iou/drivable_iou_.xlsx"
            #     #/root/data/nuscenes_datasets/nuscenes_datasets/nuscenes
            #     file_exists = os.path.isfile(output_path)

            #     with open(output_path, mode='a', newline='', encoding='utf-8') as file:
            #         if not file_exists:
            #             df.to_csv(file, header=True)
            #         else:
            #             df.to_csv(file, header=False)
            

            # ################# NEW MULTI-IOU CALCULATION #####################

            # ######
            map_seg_thresholds = torch.Tensor([0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]).to(
                device)
            # map_seg_thresholds = torch.Tensor([0.25, 0.3, 0.35, 0.4, 0.45]).to(
            #     device)
            sig_map_bev_e_new = torch.sigmoid(bev_map_mask_e)[:, :, :, :, None] >= map_seg_thresholds
            bev_map_mask_g_new = bev_map_only_mask_g[:, :, :, :, None]

            tps = (sig_map_bev_e_new.bool() & bev_map_mask_g_new.bool()).sum(dim=[2, 3])  # (B,7,12)
            fps = (sig_map_bev_e_new.bool() & ~bev_map_mask_g_new.bool()).sum(dim=[2, 3])
            fns = (~sig_map_bev_e_new.bool() & bev_map_mask_g_new.bool()).sum(dim=[2, 3])

            # best i's and u's
            map_masks_multi_ious_intersections = tps.sum(0)
            map_masks_multi_ious_unions = (tps.sum(0) + fps.sum(0) + fns.sum(0) + 1e-4)

            # metric
            # single threshold IoUs (t=0.4)
            # metrics['map_masks_intersections'] =4
            # metrics['map_masks_unions'] =4
            # num_non_zero = torch.nonzero(map_intersections_per_class).size(0)
            # sample_map_iou = (100*(map_intersections_per_class/map_unions_per_class).sum()/num_non_zero).detach()

            # scene_map_intersections =4
            # scene_map_unions =4
            # # multi threshold IoUs
            # metrics['map_masks_multi_ious_intersections'] =4
            # metrics['map_masks_multi_ious_unions'] =4
            # metrics['map_seg_thresholds'] = map_seg_thresholds
            

            # single threshold IoUs (t=0.4)
            metrics['map_masks_intersections'] += map_intersections_per_class
            metrics['map_masks_unions'] += map_unions_per_class
            num_non_zero = torch.nonzero(map_intersections_per_class).size(0)
            

            scene_map_intersections += map_intersections_per_class.detach()
            scene_map_unions += map_unions_per_class.detach()
            
            # multi threshold IoUs
            metrics['map_masks_multi_ious_intersections'] += map_masks_multi_ious_intersections
            metrics['map_masks_multi_ious_unions'] += map_masks_multi_ious_unions
            metrics['map_seg_thresholds'] = map_seg_thresholds

        # object seg task
        if train_task == 'both' or train_task == 'object':
            if train_task == 'both':
                obj_seg_bev_e = seg_e[:, -1].unsqueeze(dim=1)
            else:  # 'object'
                obj_seg_bev_e = seg_e
                obj_seg_bev_e_sigmoid = torch.sigmoid(obj_seg_bev_e)
                ego_other_cars_on_map_e = ego_car_on_map_g * (1 - obj_seg_bev_e_sigmoid) + \
                    other_cars_plane * obj_seg_bev_e_sigmoid
            # clc loss
            ce_loss = loss_fn(obj_seg_bev_e, seg_bev_g, valid_bev_g)
            # obj
            ce_factor = 1 / torch.exp(model.module.ce_weight)
            ce_loss = 10.0 * ce_loss * ce_factor  # 10.0
            total_loss += ce_loss

            # object IoUs
            obj_seg_bev_e_round = torch.sigmoid(obj_seg_bev_e).round()  # --> thresh = 0.5
            # overall intersection and unions
            obj_intersection = (obj_seg_bev_e_round * seg_bev_g * valid_bev_g).sum(dim=[1, 2, 3])
            obj_union = ((obj_seg_bev_e_round + seg_bev_g) * valid_bev_g).clamp(0, 1).sum(dim=[1, 2, 3])
            obj_intersections = obj_intersection.sum()
            obj_unions = obj_union.sum()

            # distance based IoU calc
            # 0 - 20 m
            bev_0_20_mask = torch.zeros_like(obj_seg_bev_e_round)  # init with zeros
            _, _, mask_h, mask_w = bev_0_20_mask.shape
            start_20 = (mask_h // 2) - 40
            end_20 = (mask_h // 2) + 40
            bev_0_20_mask[:, :, start_20:end_20, start_20:end_20] = 1.0
            # bev_0_20_mask_np = bev_0_20_mask.detach().cpu().numpy()  # debug only -> better visualization of the masks

            obj_0_20_intersection = (obj_seg_bev_e_round * seg_bev_g * valid_bev_g * bev_0_20_mask).sum(
                    dim=[1, 2, 3])
            obj_0_20_union = ((obj_seg_bev_e_round + seg_bev_g) * valid_bev_g * bev_0_20_mask).clamp(0, 1).sum(
                    dim=[1, 2, 3])
            obj_0_20_intersections = obj_0_20_intersection.sum()
            obj_0_20_unions = obj_0_20_union.sum()

            # 20 - 35 m
            bev_20_35_mask = torch.zeros_like(obj_seg_bev_e_round)  # init with zeros
            start_0_35 = (mask_h // 2) - 70
            end_0_35 = (mask_h // 2) + 70
            bev_20_35_mask[:, :, start_0_35:end_0_35, start_0_35:end_0_35] = 1.0
            # set the inner (0-20) mask to zero
            bev_20_35_mask[:, :, start_20:end_20, start_20:end_20] = 0.

            obj_20_35_intersection = (obj_seg_bev_e_round * seg_bev_g * valid_bev_g * bev_20_35_mask).sum(
                    dim=[1, 2, 3])
            obj_20_35_union = ((obj_seg_bev_e_round + seg_bev_g) * valid_bev_g * bev_20_35_mask).clamp(0, 1).sum(
                    dim=[1, 2, 3])
            obj_20_35_intersections = obj_20_35_intersection.sum()
            obj_20_35_unions = obj_20_35_union.sum()

            # 35 - 50 m
            bev_35_50_mask = torch.ones_like(obj_seg_bev_e_round)  # init with ones
            # set the inner (0-35) mask to zero
            bev_35_50_mask[:, :, start_0_35:end_0_35, start_0_35:end_0_35] = 0.0

            obj_35_50_intersection = (obj_seg_bev_e_round * seg_bev_g * valid_bev_g * bev_35_50_mask).sum(
                    dim=[1, 2, 3])
            obj_35_50_union = ((obj_seg_bev_e_round + seg_bev_g) * valid_bev_g * bev_35_50_mask).clamp(0, 1).sum(
                    dim=[1, 2, 3])
            obj_35_50_intersections = obj_35_50_intersection.sum()
            obj_35_50_unions = obj_35_50_union.sum()

            metrics['ce_loss'] = ce_loss
            metrics['ce_weight'] = model.module.ce_weight.item()
            metrics['obj_intersections'] += obj_intersections
            metrics['obj_unions'] += obj_unions
            sample_obj_iou = (100 * obj_intersections / (obj_unions + 1e-4)).detach()
            scene_obj_intersections += obj_intersections.detach()
            scene_obj_unions += obj_unions.detach()

            # 0 - 20 m
            metrics['obj_0_20_intersections'] += obj_0_20_intersections
            metrics['obj_0_20_unions'] += obj_0_20_unions
            sample_obj_0_20_iou = (100 * obj_0_20_intersections / (obj_0_20_unions + 1e-4)).detach()
            scene_obj_0_20_intersections += obj_0_20_intersections.detach()
            scene_obj_0_20_unions += obj_0_20_unions.detach()
            # 20 - 35 m
            metrics['obj_20_35_intersections'] += obj_20_35_intersections
            metrics['obj_20_35_unions'] += obj_20_35_unions
            sample_obj_20_35_iou = (100 * obj_20_35_intersections / (obj_20_35_unions + 1e-4)).detach()
            scene_obj_20_35_intersections += obj_20_35_intersections.detach()
            scene_obj_20_35_unions += obj_20_35_unions.detach()
            # 35 - 50 m
            metrics['obj_35_50_intersections'] += obj_35_50_intersections
            metrics['obj_35_50_unions'] += obj_35_50_unions
            sample_obj_35_50_iou = (100 * obj_35_50_intersections / (obj_35_50_unions + 1e-4)).detach()
            scene_obj_35_50_intersections += obj_35_50_intersections.detach()
            scene_obj_35_50_unions += obj_35_50_unions.detach()



            # --------------- trainvalidation 150개 씬에대해서 dataset Drviabble Area 평균 계산 평가 지표 산출을 위해서다. end ---------

            '''---- obj iou calculation per scene  start ----'''
            scene_obj_iou = 100 * scene_obj_intersections / (scene_obj_unions + 1e-4)
            scene_obj_0_20_iou = 100 * scene_obj_0_20_intersections / (scene_obj_0_20_unions + 1e-4)
            scene_obj_20_35_iou = 100 * scene_obj_20_35_intersections / (scene_obj_20_35_unions + 1e-4)
            scene_obj_35_50_iou = 100 * scene_obj_35_50_intersections / (scene_obj_35_50_unions  + 1e-4)
            '''---- obj iou calculation per scene  end ----'''
        
            sample_map_iou = (100*(map_intersections_per_class/map_unions_per_class).sum()/num_non_zero).detach()
            
            drivable_iou = float(map_ious['drivable_iou'].item())  
            
            # all_data_check  = [scene_obj_iou, scene_obj_0_20_iou, scene_obj_20_35_iou, scene_obj_35_50_iou, sample_map_iou ,drivable_iou] 
    
            # #print(f"map_ious {map_ious.items()}")
            
            # for i in all_data_check: 
            #     if not (math.isnan(all_data_check[i]) or math.isinf(all_data_check[i])):
            #         all_data_check[i] += all_data_check[i]
            #         valid_data_cnt += 1

            # is_last = (t == T - 1)
            
            # if is_last:
                
            #     for i in range (len(all_data_check)):
            #         denom = max(valid_data_cnt, 1)
            #         all_data_check[i] = all_data_check[i] / denom

            #     # print(f"valid_drivable_cnt {valid_drivable_cnt} / T {T}")
            #     # print(f"mean_drivable_iou {mean_drivable_iou:.3f}")

            #     df = pd.DataFrame([{
            #         "scene_obj_io" : round(all_data_check[0],1),
            #         "scene_obj_0_20_iou" : round(all_data_check[1],1),
            #         "scene_obj_20_35_iou": round(all_data_check[2],1),
            #         "scene_obj_35_50_iou": round(all_data_check[3],1),
            #         "sample_map_iou" : round(all_data_check[4],1),
            #         "drivable_iou" : round(all_data_check[5], 1),
            #         "valid_frames": valid_drivable_cnt,
            #     }])

            #     output_path = "/SegMam/model_iou/camsix_iou/drivable_iou_mean_1.csv"
            #     os.makedirs(os.path.dirname(output_path), exist_ok=True)
            #     df.to_csv(output_path, mode="a", header=not os.path.exists(output_path), index=False)

            # --------------- trainvalidation 150개 씬에대해서 dataset Drviabble Area 평균 계산 평가 지표 산출을 위해서다. end ---------
            
            
            # # sci start appnedix kjh 26.01.05
            if eval_status == "All":
                frame_vals = [
                    float(scene_obj_iou.detach().cpu()),
                    float(scene_obj_0_20_iou.detach().cpu()),
                    float(scene_obj_20_35_iou.detach().cpu()),
                    float(scene_obj_35_50_iou.detach().cpu()),
                    float(sample_map_iou.detach().cpu()),
                    float(map_ious['drivable_iou'].detach().cpu()),
                    float(map_ious['carpark_iou'].detach().cpu()),
                    float(map_ious['drivable_iou'].detach().cpu()),
                    float(map_ious['ped_cross_iou'].detach().cpu()),
                    float(map_ious['walkway_iou'].detach().cpu()),
                    float(map_ious['stop_line_iou'].detach().cpu()),
                    float(map_ious['road_divider_iou'].detach().cpu()),
                    float(map_ious['lane_divider_iou'].detach().cpu())
                ]
                            
                sum_data = [0.0] * len(frame_vals)
                valid_cnt = [0] * len(frame_vals)

                # 프레임 루프 내부(지표 계산한 직후)에 아래로 교체:


                for k, v in enumerate(frame_vals):
                    if not (math.isnan(v) or math.isinf(v)):
                        sum_data[k] += v
                        valid_cnt[k] += 1

                is_last = (t == T - 1)
                
                if is_last:
                    
                    avg = [sum_data[k] / max(valid_cnt[k], 1) for k in range(len(frame_vals))]

                    df = pd.DataFrame([{
                        "scene_obj_iou": round(avg[0], 1),
                        "scene_obj_0_20_iou": round(avg[1], 1),
                        "scene_obj_20_35_iou": round(avg[2], 1),
                        "scene_obj_35_50_iou": round(avg[3], 1),
                        "sample_map_iou": round(avg[4], 1),
                        "drivable_iou": round(avg[5], 1),
                        "carpark_iou": round(avg[6], 1),
                        "ped_cross_iou": round(avg[7], 1),
                        "walkway_iou": round(avg[8], 1),
                        "stop_line_iou": round(avg[9], 1),
                        "road_divider_iou": round(avg[10], 1),
                        "lane_divider_iou": round(avg[11], 1),
                        "valid_frames": valid_cnt[5],
                    }])

                    csv_dir = iou_csv_dir if iou_csv_dir else "/SegMam/model_iou/camsix_iou"
                    output_path = os.path.join(csv_dir, f"obj_map_iou_{eval_status}_{ncams}.csv")
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)

                    write_header = (not os.path.exists(output_path)) or (os.path.getsize(output_path) == 0)
                    df.to_csv(output_path, mode="a", header=write_header, index=False)
            else:
                
                frame_vals = [
                    float(scene_obj_iou.detach().cpu()),
                    float(scene_obj_0_20_iou.detach().cpu()),
                    float(scene_obj_20_35_iou.detach().cpu()),
                    float(scene_obj_35_50_iou.detach().cpu()),
                    float(sample_map_iou.detach().cpu()),
                    float(map_ious['drivable_iou'].detach().cpu()),
                    float(map_ious['carpark_iou'].detach().cpu()),
                    float(map_ious['drivable_iou'].detach().cpu()),
                    float(map_ious['ped_cross_iou'].detach().cpu()),
                    float(map_ious['walkway_iou'].detach().cpu()),
                    float(map_ious['stop_line_iou'].detach().cpu()),
                    float(map_ious['road_divider_iou'].detach().cpu()),
                    float(map_ious['lane_divider_iou'].detach().cpu())
                ]
                            
                sum_data = [0.0] * len(frame_vals)
                valid_cnt = [0] * len(frame_vals)

                # 프레임 루프 내부(지표 계산한 직후)에 아래로 교체:


                for k, v in enumerate(frame_vals):
                    if not (math.isnan(v) or math.isinf(v)):
                        sum_data[k] += v
                        valid_cnt[k] += 1

                is_last = (t == T - 1)
                
                if is_last:
                    
                    avg = [sum_data[k] / max(valid_cnt[k], 1) for k in range(len(frame_vals))]

                    df = pd.DataFrame([{
                        "scene_obj_iou": round(avg[0], 1),
                        "scene_obj_0_20_iou": round(avg[1], 1),
                        "scene_obj_20_35_iou": round(avg[2], 1),
                        "scene_obj_35_50_iou": round(avg[3], 1),
                        "sample_map_iou": round(avg[4], 1),
                        "drivable_iou": round(avg[5], 1),
                        "carpark_iou": round(avg[6], 1),
                        "ped_cross_iou": round(avg[7], 1),
                        "walkway_iou": round(avg[8], 1),
                        "stop_line_iou": round(avg[9], 1),
                        "road_divider_iou": round(avg[10], 1),
                        "lane_divider_iou": round(avg[11], 1),
                        "valid_frames": valid_cnt[5],
                    }])

                    csv_dir = iou_csv_dir if iou_csv_dir else "/SegMam/model_iou/camsix_iou"
                    output_path = os.path.join(csv_dir, f"obj_map_iou_{eval_status}_{ncams}.csv")
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)

                    write_header = (not os.path.exists(output_path)) or (os.path.getsize(output_path) == 0)
                    df.to_csv(output_path, mode="a", header=write_header, index=False)
                

            
                # sci end appnedix kjh 26.01.05


        # save own map from g masks
        # bev_map_g_img = bev_map_g.squeeze().permute(1, 2, 0).detach().cpu().numpy() * 255
        # bev_map_g_img_name = os.path.join(folder_name, "own_map_from_g_masks_%03d.png" % t)
        # imageio.imwrite(bev_map_g_img_name, bev_map_g_img.astype(np.uint8))

        # save own map from e masks
        # bev_map_e_img = bev_map_e.squeeze().permute(1, 2, 0).detach().cpu().numpy() * 255
        # bev_map_e_img_name = os.path.join(folder_name, "own_map_from_e_masks_%03d.png" % t)
        # imageio.imwrite(bev_map_e_img_name, bev_map_e_img.astype(np.uint8))

        # save all cars and ego car on map
        #ego_other_cars_on_map_g_img = ego_other_cars_on_map_g.squeeze().permute(1, 2, 0).detach().cpu().numpy() * 255
        # ego_other_cars_on_map_g_img_name = os.path.join(folder_name, "ego_other_cars_on_map_g_img_%03d.png" % t)
        # imageio.imwrite(ego_other_cars_on_map_g_img_name, ego_other_cars_on_map_g_img.astype(np.uint8))

        # save all cars and ego car on map --> estimate
        #ego_other_cars_on_map_e_img = ego_other_cars_on_map_e.squeeze().permute(1, 2, 0).detach().cpu().numpy() * 255
        # ego_other_cars_on_map_e_img_name = os.path.join(folder_name, "ego_other_cars_on_map_e_img_%03d.png" % t)
        # imageio.imwrite(ego_other_cars_on_map_e_img_name, ego_other_cars_on_map_e_img.astype(np.uint8))

        # store flipped version such that the car "drives" from bottom to top
        # flipped_ego_other_cars_on_map_g_img = np.flip(ego_other_cars_on_map_g_img, axis=0)
        # flipped_ego_other_cars_on_map_g_img_name = os.path.join(folder_name,
        #                                                         "flipped_ego_other_cars_on_map_g_img_%03d.png" % t)
        # imageio.imwrite(flipped_ego_other_cars_on_map_g_img_name,
        #                 flipped_ego_other_cars_on_map_g_img.astype(np.uint8))
        
        
        # # store flipped version such that the car "drives" from bottom to top


        # '''flipped ego car save start'''
        # ego_other_cars_on_map_e_img = ego_other_cars_on_map_e.squeeze().permute(1, 2, 0).detach().cpu().numpy() * 255
        # flipped_ego_other_cars_on_map_e_img = np.flip(ego_other_cars_on_map_e_img, axis=0)
        # flipped_ego_other_cars_on_map_e_img_name = os.path.join(folder_name,
        #                                                         "flipped_ego_other_cars_on_map_e_img_%03d.png" % t)
        # imageio.imwrite(flipped_ego_other_cars_on_map_e_img_name,
        #                 flipped_ego_other_cars_on_map_e_img.astype(np.uint8))
        # '''flipped ego car save end'''

        # ''' map pred img save start'''
        # map_pred_img = bev_map_e.squeeze().permute(1, 2, 0).detach().cpu().numpy() * 255
        # flipped_map_pred_img = np.flip(map_pred_img, axis=0)
        # # scene_number = "0590"
        # # folder_name = f"/SegMam/inference/{scene_number}"
        # flipped_map_pred_img_name = os.path.join(folder_name, "map_pred_flipped_img_%03d.png" % t)
        # imageio.imwrite(flipped_map_pred_img_name, flipped_map_pred_img.astype(np.uint8))
        # ''' map pred img save end'''

        # '''index 3 predeic img start'''
        # if index ==3:
        #     map_pred_img = bev_map_e.squeeze().permute(1, 2, 0).detach().cpu().numpy() * 255
        #     flipped_map_pred_img = np.flip(map_pred_img, axis=0)
        #     folder_name = "/root/data/dataset/pdcms/ogm/scenes/SCENE_001/"
        #     flipped_map_pred_img_name = os.path.join(folder_name, "map_pred_flipped_img_%03d.png" % t)
        #     imageio.imwrite(flipped_map_pred_img_name, flipped_map_pred_img.astype(np.uint8))
        # '''index 3 predeic img end'''

        
        '''make egopose excel data save into excel file kjh ''' 

        """start  rotation tarnslation"""
        # rotation = scipy.spatial.transform.Rotation.from_quat(rotation.numpy())
        # rotation = rotation.as_matrix()
        # translation = transformation.squeeze().tolist()
        # rotation = rotation.squeeze().tolist()

        # scne_name = 'scene_0653'        

        # if index == 2:

        #     rot_file_path = f'/root/data/dataset/pdcms/ogm/{scne_name}/pose/rotation.xlsx'

        #     if os.path.exists(rot_file_path):
        #         existing_data = pd.read_excel(rot_file_path, sheet_name='rotation')
        #     else:
        #         existing_data = pd.DataFrame()

        #     if t  in range(T):
        #         data_list = []
        #         data_list.append(rotation)
                                
        #         rots = f'rots{t}'
        #         # rota  = f'rot{t}'

        #         if existing_data.empty:
        #             existing_data = pd.DataFrame({rots : rotation})
        #         else:
        #             if len(existing_data) != len(data_list):
        #                 print("must be same data length")
        #                 min_length = min(len(existing_data), len(data_list))
        #                 existing_data = existing_data.iloc[:min_length]
        #                 data_list = data_list[:min_length]

        #             existing_data[rots] = data_list
        #     existing_data.to_excel(rot_file_path, sheet_name='rotation', index=False)


        #     trans_file_path = f'/root/data/dataset/pdcms/ogm/{scne_name}/pose/translation.xlsx'

        #     if os.path.exists(trans_file_path):
        #         existing_data = pd.read_excel(trans_file_path, sheet_name='translation')
        #     else:
        #         existing_data = pd.DataFrame()

        #     if t  in range(T):
        #         data_list = []
        #         data_list.append(translation)
                                
        #         rots = f'trans{t}'
        #         # rota  = f'rot{t}'

        #         if existing_data.empty:
        #             existing_data = pd.DataFrame({rots : translation})
        #         else:
        #             if len(existing_data) != len(data_list):
        #                 print("must be same data length")
        #                 min_length = min(len(existing_data), len(data_list))
        #                 existing_data = existing_data.iloc[:min_length]
        #                 data_list = data_list[:min_length]

        #             existing_data[rots] = data_list
        #     existing_data.to_excel(trans_file_path, sheet_name='translation', index=False)


        # # 엑셀 파일 불러오기
        # file_path = "/root/data/dataset/pdcms/ogm/scene_0653/changepose/rotation.xlsx"
        # df = pd.read_excel(file_path, sheet_name="rotation")  # 시트 이름 지정

        # # 행 → 열 변환 (Transpose)
        # df_transposed = df.T

        # # 변환된 데이터를 새로운 엑셀 파일에 저장
        # output_path = "/root/data/dataset/pdcms/ogm/scene_0653/changepose/rotation_transposed.xlsx"
        # df_transposed.to_excel(output_path, index=False)

        
        


       
                
        # print(f"trnasofrmaation shape {len(transformation)}")
        # # print(f"rotation shape{len(rotation)}")

        # #  #-----------------------------start of iou tp tn fn fp save---------------------------------
        # if index == 2:
            
        #     # translation = [0.0 if math.isnan(x) else round(x,2)for x in translation]
        #     # rotation = [0.0 if math.isnan(x) else round(x,2)for x in rotation]
    

        #     columns = ["trans", "ro"]
            
        #     data = {
        #         "trans": translation,
        #         "rot": rotation,
        #     }

        #     df = pd.DataFrame(data, index=columns).transpose()
            
        #     scenes_name = 'scene_0989'
        #     output_path = f"/root/data/dataset/pdcms/ogm/{scenes_name}/pose/pose.xlsx"

        #     file_exists = os.path.isfile(output_path)

        #     with open(output_path, mode='a', newline='', encoding='utf-8') as file:
        #         if not file_exists:
        #             df.to_csv(file, header=True)
        #         else:
        #             df.to_csv(file, header=False)

        #     #-----------------------------end of iou tp tn fn fp save---------------------------------


        
        
        # if index == 2:
            
        #     file_path_rot = '/root/data/dataset/pdcms/ogm/scene_0989/rotation.xlsx'

        #     if os.path.exists(file_path_rot):
        #         existing_data = pd.read_excel(file_path_rot, sheet_name='rotation')
        #     else:
        #         existing_data = pd.DataFrame()

        #     if t  in range(T):
        #         data_list = []
        #         data_list.append(rotation)
                                
        #         rots = f'rots{t}'
        #         # rota  = f'rot{t}'

        #         if existing_data.empty:
        #             existing_data = pd.DataFrame({rots : rotation})
        #         else:
        #             if len(existing_data) != len(data_list):
        #                 print("must be smae data length rots")
        #                 min_length = min(len(existing_data), len(data_list))
        #                 existing_data = existing_data.iloc[:min_length]
        #                 data_list = data_list[:min_length]

        #             existing_data[rots] = data_list
        #     existing_data.to_excel(file_path_rot, sheet_name='rotation', index=False)

                    
            
        # if index == 2:
                
        #     file_path_translation = '/root/data/dataset/pdcms/ogm/scene_0989/translation.xlsx'

        #     if os.path.exists(file_path_translation):
        #         existing_data = pd.read_excel(file_path_translation, sheet_name='translation')
        #     else:
        #         existing_data = pd.DataFrame()

        #     if t  in range(T):
        #         data_list = []
        #         data_list.append(rotation)
                                
        #         tran = f'tran{t}'
        #         # rota  = f'rot{t}'

        #         if existing_data.empty:
        #             existing_data = pd.DataFrame({tran : transformation})
        #         else:
        #             if len(existing_data) != len(data_list):
        #                 print("must be smae data length tran")
        #                 min_length = min(len(existing_data), len(data_list))
        #                 existing_data = existing_data.iloc[:min_length]
        #                 data_list = data_list[:min_length]

        #             existing_data[tran] = data_list
        #     existing_data.to_excel(file_path_translation, sheet_name='translation', index=False)


        #print(f'trans_all_floats{len(trans_all_floats)}')
        """end  rotation tarnslation"""
        
        '''best iou scene 3 index scene rgb data save into excel file kjh ''' 
        # if index == 2:
        #     flipped_map_pred_img_name = os.path.join("/root/data/pdcms/nuscene_data/SCENE_0001/SEGMENTATION/", "map_pred_flipped_img_%03d.png" % t)
        #     file_path = '/root/data/pdcms/ogm/egopose.xlsx'

        #     if os.path.exists(file_path):
        #         existing_data = pd.read_excel(file_path, sheet_name='Sheet1')
        #     else:
        #         existing_data = pd.DataFrame()

        #     if t  in range(T):
        #         global 
        #          = True
        #         data_list = []
        #         for i in range(index):
        #             #data_list.append(trans_use[i])
        #         column_name = f'tran{t}'

        #         if existing_data.empty:
        #             existing_data = pd.DataFrame({column_name: data_list})
        #         else:
        #             if len(existing_data) != len(data_list):
        #                 print("must be smae data length")
        #                 min_length = min(len(existing_data), len(data_list))
        #                 existing_data = existing_data.iloc[:min_length]
        #                 data_list = data_list[:min_length]

        #             existing_data[column_name] = data_list
        #     existing_data.to_excel(file_path, sheet_name='Sheet1', index=False)


        
        

        # """"rgb pixel start """

        '''best iou scene 3 index scene rgb data save into excel file kjh ''' 
        # if index == 2:
            
        #     scenes_number = 'scene_0590'
        #     file_path = f'/root/data/dataset/pdcms/ogm/{scenes_number}/{scenes_number}_rgb.xlsx'

        #     if os.path.exists(file_path):
        #         existing_data = pd.read_excel(file_path, sheet_name='pixel')
        #     else:
        #         existing_data = pd.DataFrame()

        #     if t  in range(T):
        #         data_list = []
        #         for i in range(len(flipped_map_pred_img)):
        #             for j in range(len(flipped_map_pred_img)):
        #                 #print(f"vis eval line 780 sample RGB Save Progress i: {i}")
        #                 #print(f"vis eval line 780 sample RGB Save Progress j: {j}")
        #                 data_list.append(flipped_map_pred_img[i][j])
        #         column_name = f'DC_{t}'

        #         if existing_data.empty:
        #             existing_data = pd.DataFrame({column_name: data_list})
        #         else:
        #             if len(existing_data) != len(data_list):
        #                 print("must be smae data length pixel")
        #                 min_length = min(len(existing_data), len(data_list))
        #                 existing_data = existing_data.iloc[:min_length]
        #                 data_list = data_list[:min_length]

        #             existing_data[column_name] = data_list
        #     existing_data.to_excel(file_path, sheet_name='pixel', index=False)
        
        # """"rgb pixel end """
        
        #imageio.imwrite(flipped_map_pred_img_name, flipped_map_pred_img.astype(np.uint8))
        
        # ''''start map gt start'''
        # # if index ==3:
        # map_gt_img = bev_map_g.squeeze().permute(1, 2, 0).detach().cpu().numpy() * 255
        # flipped_map_gt_img = np.flip(map_gt_img, axis=0)
        # # scene_number = "0590"
        # # folder_name = f"/root/data/dataset/pdcms/ogm/scene_{scene_number}/gt/"
        # flipped_map_gt_img_name = os.path.join(folder_name, "map_gt_flipped_img_%03d.png" % t)
        # imageio.imwrite(flipped_map_gt_img_name, flipped_map_gt_img.astype(np.uint8))
        # ''' end map gt'''

        # '''obj predic detected car start '''
        # #detected cars
        # obj_pred = other_cars_plane_e * obj_seg_bev_round
        # obj_pred_img = obj_pred.squeeze().permute(1, 2, 0).detach().cpu().numpy() * 255
        # flipped_obj_pred_img = np.flip(obj_pred_img, axis=0)
        # #print(f" flipped_obj_pred_img {flipped_obj_pred_img} ")
        # flipped_obj_pred_img_name = os.path.join(folder_name, "obj_pred_flipped_img_%03d.png" % t)
        # imageio.imwrite(flipped_obj_pred_img_name, flipped_obj_pred_img.astype(np.uint8))
        # '''obj predic detected car end '''
        
        # '''pesudo lidar  predic  start '''
        # if model.module.use_pseudolidar:
        #     pseudolidar_t_vis = torch.sum(rad_occ_mem0[0], 2).clamp(0, 1)  # (1, 200, 200)
        #     pseudolidar_t_vis = utils.improc.back2color(pseudolidar_t_vis.repeat(3, 1, 1) - 0.5).cpu().numpy().transpose(1, 2,
        #                                                                                                      0)
        #     # flip pseudolidar correctly
        #     pseudolidar_t_vis = np.flip(pseudolidar_t_vis, axis=0)
        #     pseudolidar_t_vis_name = os.path.join(folder_name, "pesudo_lidar%03d.png" % t)
        #     imageio.imwrite(pseudolidar_t_vis_name, pseudolidar_t_vis.astype(np.uint8))
        # '''pesudo lidar  predic  start '''
        
        
        # n_cam = rgb_camXs.shape[1]
        # for cam_id in range(n_cam):
        #     # resize image to original ratio:
        #     single_img = rgb_camXs[0, cam_id:cam_id + 1]
        #     reshaped_img = torchvision.transforms.functional.resize(single_img, (450, 800))
        #     camX_t_vis = utils.improc.back2color(reshaped_img).cpu().numpy()[0].transpose(1, 2, 0)
        #     camX_t_vis_name = os.path.join(folder_name, "cam" + str(cam_id) + "_rgb_%03d.png" % t)
        #     imageio.imwrite(camX_t_vis_name, camX_t_vis.astype(np.uint8))

        
        
        '''depth predic detected car start '''
        
        # n_cam = depth_camXs.shape[1]

        # for cam_id in range(n_cam):
        #     cam_forler_name = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT','CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
        #     #scenes_number = "0553"
        #     for j in range(len(cam_forler_name)):
        #         if  cam_id==j:
        #             if not os.path.exists(f"{folder_name}/{cam_forler_name[j]}/"):
        #                 os.mkdir(f"{folder_name}/{cam_forler_name[j]}/")
        #             single_img = depth_camXs[0, cam_id:cam_id + 1]
        #             single_img_cpu = single_img.squeeze().detach().cpu().numpy()

        #             save_dir = f"{folder_name}/{cam_forler_name[j]}/"
        #             os.makedirs(save_dir, exist_ok=True)

        #             # 저장할 경로 + 파일명 설정
        #             save_path = os.path.join(save_dir, f"cam" + str(cam_id) + "_rgb_%03d.png" % t)
                    
        #             if single_img_cpu.ndim == 3:
                        
        #                 if single_img_cpu.shape[0] == 3:
        #                         # RGB (C,H,W) → (H,W,C)
        #                         single_img_cpu = single_img_cpu.transpose(1, 2, 0)
        #                         plt.imshow(single_img_cpu)
        #                 else:
        #                         # Depth (1,H,W) → (H,W)
        #                         single_img_cpu = single_img_cpu[0]
        #                         plt.imshow(single_img_cpu, cmap='magma', vmax=0.741)
        #             else:
        #                     # (H,W)
        #                 plt.imshow(single_img_cpu, cmap='magma', vmax=0.741)

        #             plt.axis('off')
        #             plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        #             plt.close()


        '''each cam image save kjh  default start'''
        # folder_name = '/root/data/Nuscenes_datasets/nuscenes_original_junhan/pdcms/ogm/camsix/scene_0653/rgb_img'
        # n_cam = rgb_camXs.shape[1]
        # for cam_id in range(n_cam):
        #     # resize image to original ratio:
        #     single_img = rgb_camXs[0, cam_id:cam_id + 1]
        #     reshaped_img = torchvision.transforms.functional.resize(single_img, (450, 800), antialias=True)
        #     camX_t_vis = utils.improc.back2color(reshaped_img).cpu().numpy()[0].transpose(1, 2, 0)
        #     camX_t_vis_name = os.path.join(folder_name, "cam" + str(cam_id) + "_rgb_%03d.png" % t)
        #     imageio.imwrite(camX_t_vis_name, camX_t_vis.astype(np.uint8))
        # imageio.imwrite(camX_t_vis_name, camX_t_vis.astype(np.uint8))
        '''each cam image save kjh end'''

        '''each cam image save kjh  default start'''
        # scene_number = '0653'
        # cam_forler_name = ["CAM_FRONT","CAM_FRONT_LEFT","CAM_FRONT_RIGHT","CAM_BACK_LEFT","CAM_BACK","CAM_BACK_RIGHT"]
        # n_cam = rgb_camXs.shape[1]

        # for cam_name in  range (cam_forler_name):
        #     for cam_id in range(n_cam):
        #         # resize image to original ratio:
        #         if cam_name == cma_id:
        #             folder_name = f'/root/data/Nuscenes_datasets/nuscenes_original_junhan/pdcms/ogm/camsix/scene_{scene_number}/rgb_img/{cam_forler_name}'
        #             single_img = rgb_camXs[0, cam_id:cam_id + 1]
        #             reshaped_img = torchvision.transforms.functional.resize(single_img, (450, 800), antialias=True)
        #             camX_t_vis = utils.improc.back2color(reshaped_img).cpu().numpy()[0].transpose(1, 2, 0)
        #             camX_t_vis_name = os.path.join(folder_name, "cam" + str(cam_id) + "_rgb_%03d.png" % t)
        #             imageio.imwrite(camX_t_vis_name, camX_t_vis.astype(np.uint8))
        #         imageio.imwrite(camX_t_vis_name, camX_t_vis.astype(np.uint8))
        '''each cam image save kjh end'''

        '''each cam image save kjh  default start'''
        # scene_number = '0653'
        # cam_forler_name = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"]
        # n_cam = rgb_camXs.shape[1]

        # for cam_id, cam_name in enumerate(cam_forler_name):
        #     folder_name = f'/root/data/Nuscenes_datasets/nuscenes_original_junhan/pdcms/ogm/camsix/scene_{scene_number}/rgb_img/{cam_name}'
        #     os.makedirs(folder_name, exist_ok=True)

        #     single_img = rgb_camXs[0, cam_id:cam_id + 1]
        #     reshaped_img = torchvision.transforms.functional.resize(single_img, (450, 800), antialias=True)
        #     camX_t_vis = utils.improc.back2color(reshaped_img).cpu().numpy()[0].transpose(1, 2, 0)
        #     camX_t_vis_name = os.path.join(folder_name, f"cam{cam_id}_rgb_{t:03d}.png")
        #     imageio.imwrite(camX_t_vis_name, camX_t_vis.astype(np.uint8))

        '''each cam image save kjh end'''


                
        '''for scene img data save in nas2 hub kjh start'''
        # if index==1:
        #     n_cam = rgb_camXs.shape[1]
        #     for cam_id in range(n_cam):
        #         cam_forler_name = ["CAM_FRONT","CAM_FRONT_LEFT","CAM_FRONT_RIGHT","CAM_BACK_LEFT","CAM_BACK","CAM_BACK_RIGHT"]
        #         scenes_number = "0653"
        #         for j in range(len(cam_forler_name)):
        #             if  cam_id==j:
        #                 if not os.path.exists(f"/root/data/Nuscenes_datasets/nuscenes_original_junhan/pdcms/ogm/camsix/scene_{scenes_number}/rgb_img/{cam_forler_name[j]}/"):
        #                     os.mkdir(f"/root/data/Nuscenes_datasets/nuscenes_original_junhan/pdcms/ogm/camsix/scene_{scenes_number}/rgb_img/{cam_forler_name[j]}/")
        #                     single_img = rgb_camXs[0, cam_id:cam_id + 1]
        #                     simngle img shape 448 896
        #                     reshaped_img = torchvision.transforms.functional.resize(single_img, (450, 800), antialias=True)
        #                     camX_t_vis = utils.improc.back2color(reshaped_img).cpu().numpy()[0].transpose(1, 2, 0)
        #                     camX_t_vis_name = os.path.join(f"/root/data/Nuscenes_datasets/nuscenes_original_junhan/pdcms/ogm/camsix/scene_{scenes_number}/rgb_img/{cam_forler_name[j]}/", "cam" + str(cam_id) + "_rgb_%03d.png" % t)            
        #                     imageio.imwrite(camX_t_vis_name, camX_t_vis.astype(np.uint8))


        '''for scene img data save in nas2 hub kjh end'''

        # if index==1:
        #     n_cam = rgb_camXs.shape[1]
        #     for cam_id in range(n_cam):
        #         cam_forler_name = ["CAM_FRONT","CAM_FRONT_LEFT","CAM_FRONT_RIGHT","CAM_BACK_LEFT","CAM_BACK","CAM_BACK_RIGHT"]
        #         scenes_number = "0653"
        #         for j in range(len(cam_forler_name)):
        #             if  cam_id==j:
        #                 if not os.path.exists(f"/root/data/Nuscenes_datasets/nuscenes_original_junhan/pdcms/ogm/camsix/scene_{scenes_number}/rgb_img/{cam_forler_name[j]}/"):
        #                     os.mkdir(f"/root/data/Nuscenes_datasets/nuscenes_original_junhan/pdcms/ogm/camsix/scene_{scenes_number}/rgb_img/{cam_forler_name[j]}/")
        #                     single_img = rgb_camXs[0, cam_id:cam_id + 1]
        #                     #simngle img shape 448 896
        #                     reshaped_img = torchvision.transforms.functional.resize(single_img, (450, 800), antialias=True)
        #                     camX_t_vis = utils.improc.back2color(reshaped_img).cpu().numpy()[0].transpose(1, 2, 0)
        #                     camX_t_vis_name = os.path.join(f"/root/data/Nuscenes_datasets/nuscenes_original_junhan/pdcms/ogm/camsix/scene_{scenes_number}/rgb_img/{cam_forler_name[j]}/", "cam" + str(cam_id) + "_rgb_%03d.png" % t)            
        #                     imageio.imwrite(camX_t_vis_name, camX_t_vis.astype(np.uint8))

        
        '''depth save start'''

        # if index==1:
        #     n_cam = depth_camXs.shape[1]

        #     for cam_id in range(n_cam):
        #         cam_forler_name = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT','CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
        #         scenes_number = "0553"
        #         for j in range(len(cam_forler_name)):
        #             if  cam_id==j:
        #                 if not os.path.exists(f"/root/data/nuscenes_original_junhan/pdcms/ogm/camsix/scene_{scenes_number}/depth_img/{cam_forler_name[j]}/"):
        #                     os.mkdir(f"/root/data/nuscenes_original_junhan/pdcms/ogm/camsix/scene_{scenes_number}/depth_img/{cam_forler_name[j]}/")
        #                 single_img = depth_camXs[0, cam_id:cam_id + 1]
        #                 single_img_cpu = single_img.squeeze().detach().cpu().numpy()

        #                 save_dir = f"/root/data/nuscenes_original_junhan/pdcms/ogm/camsix/scene_{scenes_number}/depth_img/{cam_forler_name[j]}/"
        #                 os.makedirs(save_dir, exist_ok=True)

        #                 # 저장할 경로 + 파일명 설정
        #                 save_path = os.path.join(save_dir, f"cam" + str(cam_id) + "_rgb_%03d.png" % t)

        #                 plt.imshow(single_img_cpu, cmap='magma', vmax=0.741)
        #                 plt.axis('off')
        #                 plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        #                 plt.close()

        '''depth save end'''

        
        '''pseudolidar data save start '''
        # if model.module.use_pseudolidar:
        #     pseudolidar_t_vis = torch.sum(rad_occ_mem0[0], 2).clamp(0, 1)  # (1, 200, 200)
        #     pseudolidar_t_vis = utils.improc.back2color(pseudolidar_t_vis.repeat(3, 1, 1) - 0.5).cpu().numpy().transpose(1, 2,
        #                                                                                                      0)
        #     # flip pseudolidar correctly
        #     pseudolidar_t_vis = np.flip(pseudolidar_t_vis, axis=0)
        #     pseudolidar_t_vis_name = os.path.join(folder_name, "pseudolidar_%03d.png" % t)
        #     imageio.imwrite(pseudolidar_t_vis_name, pseudolidar_t_vis.astype(np.uint8))
        '''pseudolidar data save end '''
        


        '''inference mean time save start '''
        # csv 파일 하나씩 생성
        # if 'timings' in locals():
        #     csv_file_path = os.path.join(folder_name, f"inference_time_{t:03d}.csv")
        #     fieldnames = ['scene', 'sample'] + list(timings.keys())

        #     with open(csv_file_path, mode='a', newline='') as csv_file:
        #         writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        #         # if t == 0 and index == 0:
        #         if os.stat(csv_file_path).st_size == 0:
        #             writer.writeheader()

        #         row = {'scene': index, 'sample': t}
        #         row.update(timings)
        #         writer.writerow(row)
        
        

        #csv파일 하나에 행추가
        # csv_file_path = os.path.join(folder_name, "inference_time.csv")
        # if not os.path.exists(csv_file_path):
        #     mode = 'w'
        # else:
        #     mode = 'a'

        # with open(csv_file_path, mode=mode, newline='') as csv_file:
            
        #     #print(f"timings.keys() {timings.values()}")
        #     fieldnames = ['scene', 'sample'] + list(timings.keys())
        #     writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        #     # if index == 0 and t == 0:
        #     if mode == 'w':
        #         writer.writeheader()

        #     row = {'scene': index, 'sample': t}
        #     row.update(timings)
        #     writer.writerow(row)

        # df = pd.read_csv(csv_file_path)
        # fieldnames = ['scene', 'sample'] + list(df.columns[2:])

        # mean_value = df.iloc[:, 2:].mean() * 1000 # s에서 ms로 변환
        # mean_value = mean_value.round(3) # 소숫점 3자리까지 반올림

        # mean_row = pd.DataFrame([mean_value], columns=df.columns[2:])
        # mean_csv_file_path = os.path.join(folder_name, "Scene_%d_mean_time.csv" % index)
        # mean_row.to_csv(mean_csv_file_path, index=False)
        
        
        # csv_file_path = os.path.join(folder_name, "inference_time.csv")
        # if not os.path.exists(csv_file_path):
        #     mode = 'w'
        # else:
        #     mode = 'a'

        # with open(csv_file_path, mode=mode, newline='') as csv_file:
            
        #     #print(f"timings.keys() {timings.values()}")

                
        #     fieldnames = ['scene', 'sample'] + list(timings.keys())
        #     writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        #     # if index == 0 and t == 0:
        #     if mode == 'w':
        #         writer.writeheader()

        #     row = {'scene': index, 'sample': t}
        #     row.update(timings)
        #     writer.writerow(row)


        #csv파일 씬에 평균 간단 저장 start
        # inferenctime_file_path = "/root/data/nuscenes_junhan/pdcms/outputexcel/inference_time"
        # csv_file_path = os.path.join(inferenctime_file_path, "inference_time.csv")
        # if not os.path.exists(csv_file_path):
        #     mode = 'w'
        # else:
        #     mode = 'a'

        # with open(csv_file_path, mode=mode, newline='') as csv_file:
            
        #     #print(f"timings.keys() {timings.values()}")
        #     fieldnames = ['scene', 'sample'] + list(timings.keys())
        #     writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        #     # if index == 0 and t == 0:
        #     if mode == 'w':
        #         writer.writeheader()

        #     row = {'scene': index, 'sample': t}
        #     row.update(timings)
        #     writer.writerow(row)

        # df = pd.read_csv(csv_file_path)
        # fieldnames = ['scene', 'sample'] + list(df.columns[2:])

        # mean_value = df.iloc[:, 2:].mean() #* 1000 # s에서 ms로 변환
        # mean_value = mean_value.round(3) # 소숫점 3자리까지 반올림

        # mean_row = pd.DataFrame([mean_value], columns=df.columns[2:])
        # mean_csv_file_path = os.path.join(inferenctime_file_path, "Scene_%d_mean_time.csv" % index)
        # mean_row.to_csv(mean_csv_file_path, index=False)


        '''inference mean time save end '''
        
        ''' obj iou cal start '''
        
        # scene calc:
        # scene_obj_iou = 100 * scene_obj_intersections / (scene_obj_unions + 1e-4)
        # scene_obj_0_20_iou = 100 * scene_obj_0_20_intersections / (scene_obj_0_20_unions + 1e-4)
        # scene_obj_20_35_iou = 100 * scene_obj_20_35_intersections / (scene_obj_20_35_unions + 1e-4)
        # scene_obj_35_50_iou = 100 * scene_obj_35_50_intersections / (scene_obj_35_50_unions  + 1e-4)
        
        # print('ALL SCENE:     Obj IoU: ' + str(scene_obj_iou.item()) + "\n")
        # print('ALL SCENE:     Obj 0-20 IoU: ' + str(scene_obj_0_20_iou.item()) + "\n")
        # print('ALL SCENE:     Obj 20-35 IoU: ' + str(scene_obj_20_35_iou.item()) + "\n")
        # print('ALL SCENE:     Obj 35-50 IoU: ' + str(scene_obj_35_50_iou.item()) + "\n")
        
        ''' obj iou cal end'''
        
        

        total_scene_loss += total_loss

        with open(file=metrics_name, mode='a') as f:
            f.write(str(t) + ':     Obj IoU: ' + str(sample_obj_iou.item()) + "\n")
            f.write(str(t) + ':     Obj 0-20 IoU: ' + str(sample_obj_0_20_iou.item()) + "\n")
            f.write(str(t) + ':     Obj 20-35 IoU: ' + str(sample_obj_20_35_iou.item()) + "\n")
            f.write(str(t) + ':     Obj 35-50 IoU: ' + str(sample_obj_35_50_iou.item()) + "\n")
            f.write(str(t) + ':     Map IoU: ' + str(sample_map_iou.item()) + "\n")

        del seg_e



    # check for nonzero classes:
    num_non_zero = torch.nonzero(scene_map_intersections).size(0)
    scene_map_iou = (100 * scene_map_intersections / scene_map_unions).sum() / num_non_zero

    with open(file=metrics_name, mode='a') as f:
        f.write("###########################################################\n")
        f.write('ALL SCENE:     Obj IoU: ' + str(scene_obj_iou.item()) + "\n")
        f.write('ALL SCENE:     Obj 0-20 IoU: ' + str(scene_obj_0_20_iou.item()) + "\n")
        f.write('ALL SCENE:     Obj 20-35 IoU: ' + str(scene_obj_20_35_iou.item()) + "\n")
        f.write('ALL SCENE:     Obj 35-50 IoU: ' + str(scene_obj_35_50_iou.item()) + "\n")
        f.write('ALL SCENE:     Map IoU: ' + str(scene_map_iou.item()) + "\n")

    return (total_scene_loss/T, metrics, scene_inference_total_t, scene_inference_count)


def main(
        exp_name='vis_eval',
        # eval
        log_freq=100,
        dset='trainval',
        batch_size=1,  # batch size = 1 only
        timesteps=40,  # a sequence is typically 40 frames (20s * 2fps)
        vis_full_scenes=True,   # to allow different scene lengths
        nworkers=8,  # 12
        # data/log/save/load directories
        # data_dir='../../../nuscenes/nuscenes/',  # local
        data_dir='/home/shared/segmam/data/nuscenes',  # server
        custom_dataroot='../../custom_nuscenes/scaled_images',  # server
        log_dir='logs_nuscenes_segmam',
        img_dir='/root/SegMam/vis',
        init_dir='checkpoints/segmam',
        ignore_load=None,
        # data
        final_dim=[448, 896],  # to match //8, //14, //16 and //32 in Vit
        ncams=6,
        nsweeps=5,
        # model
        encoder_type='dino_v2',
        pseudolidar_encoder_type='voxel_net',
        train_task='both',
        use_pseudolidar=True,
        use_pseudolidar_filters=False,
        use_metapseudolidar=False,
        use_shallow_metadata=True,
        use_pre_scaled_imgs=False,
        use_obj_layer_only_on_map=True,
        init_query_with_image_feats=True,
        use_pseudolidar_encoder=True,
        do_rgbcompress=False,
        use_multi_scale_img_feats=True,
        num_layers=6,
        # cuda
        device_ids=[1],  # 1 device only for now
        combine_feat_init_w_learned_q=True,
        load_step=None,
        model_type='transformer',
        do_drn_val_split=True,
        use_rpn_pseudolidar=False,
        use_pseudolidar_occupancy_map=False,
        freeze_dino=True,
        do_feat_enc_dec=True,
        learnable_fuse_query=True,
        
):
    assert model_type == 'transformer', "SegMam release only supports model_type='transformer'"
    B = batch_size
    assert (B % len(device_ids) == 0)  # batch size must be divisible by number of gpus

    device = 'cuda:%d' % device_ids[0]
    torch.cuda.set_device(device)  # CUDA_VISIBLE_DEVICES=1이 설정되었기 때문에 내부적으로는 0번    
    # 
    # print(f"device id {device}")
    # torch.cuda.set_device(device)
    

    model_name = str(load_step) + '_' + exp_name
    print('model_name', model_name)

    # paper-style forward pass timing 을 위해 모델 내부 per-stage cuda.Event +
    # _maybe_sync (= torch.cuda.synchronize) 8개를 강제 비활성화.
    # forward 안에 있던 stage sync barrier 가 사라지면서 kernel overlap 이 가능해지고,
    # SegMam Table 5 caption 의 "Runtime of a forward pass" 정의와 동일한 측정이 됨.
    # 단, 외부에서 진단용으로 SEGMAM_DISABLE_PROFILING=0 명시한 경우는 존중.
    if os.environ.get("SEGMAM_DISABLE_PROFILING") != "0":
        os.environ["SEGMAM_DISABLE_PROFILING"] = "1"

    # IoU CSV 폴더 (세션 식별을 위해 시각 포함 유지)
    run_ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    iou_csv_dir = os.path.join("/SegMam/model_iou", run_ts)
    os.makedirs(iou_csv_dir, exist_ok=True)
    print(f"[IoU CSV] saving per-scene IoU CSVs to: {iou_csv_dir}")

    # 추론 시간 CSV 폴더 — 사용자 요청에 따라 "오늘 날짜" 만 사용
    today_str = datetime.now().strftime("%Y-%m-%d")
    inference_time_dir = os.path.join("/SegMam/inferencetime", today_str)
    os.makedirs(inference_time_dir, exist_ok=True)
    per_scene_time_csv = os.path.join(inference_time_dir, "per_scene_inference_time.csv")
    # 같은 날짜 폴더에 이전 실행 결과가 누적되지 않도록 시작 시 비움
    if os.path.exists(per_scene_time_csv):
        os.remove(per_scene_time_csv)
    print(f"[InferenceTime CSV] saving paper-style forward-pass timing CSV to: {per_scene_time_csv}")

    # 전체 누적용 (최종 평균 계산)
    total_inference_time_s = 0.0
    total_inference_frames = 0
    total_inference_scenes = 0

    # set up logging
    os.makedirs(img_dir, exist_ok=True)
    writer = SummaryWriter(os.path.join(log_dir, model_name), max_queue=10, flush_secs=60)

    print('resolution:', final_dim)

    resize_lim = [1.0, 1.0]
    crop_offset = 0

    data_aug_conf = {
        'crop_offset': crop_offset,
        'resize_lim': resize_lim,
        'final_dim': final_dim,
        'H': 900, 'W': 1600,
        'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
        'ncams': ncams,
    }

    _, dataloader = nuscenes_data.compile_data(
        dset,
        data_dir,
        data_aug_conf=data_aug_conf,
        centroid=scene_centroid_py,
        bounds=bounds,
        res_3d=(Z, Y, X),
        bsz=B,
        nworkers=nworkers,
        shuffle=False,
        use_pseudolidar_filters=use_pseudolidar_filters,
        seqlen=timesteps,
        nsweeps=nsweeps,
        do_shuffle_cams=False,
        get_tids=True,
        pseudolidar_encoder_type=pseudolidar_encoder_type,
        use_shallow_metadata=use_shallow_metadata,
        use_pre_scaled_imgs=use_pre_scaled_imgs,
        custom_dataroot=custom_dataroot,
        use_obj_layer_only_on_map=use_obj_layer_only_on_map,
        vis_full_scenes=vis_full_scenes,
        do_drn_val_split=do_drn_val_split
    )
    
    
    iterloader = iter(dataloader)
    max_iters = len(dataloader)  # determine iters by length of dataset
    # Debug/profile escape hatch: VIS_EVAL_MAX_SCENES=N 으로 N개 씬만 돌릴 수 있음
    _max_env = os.environ.get("VIS_EVAL_MAX_SCENES")
    if _max_env is not None:
        try:
            _n = int(_max_env)
            if _n > 0:
                max_iters = min(max_iters, _n)
                print(f"[VIS_EVAL_MAX_SCENES] limiting max_iters to {max_iters}")
        except ValueError:
            pass
    print(f"max_iter {max_iters}")
    

    # set up model & seg loss
    seg_loss_fn = SimpleLoss(2.13).to(device)  # value from lift-splat
    map_seg_loss_fn = SigmoidFocalLoss(alpha=0.25, gamma=3, reduction="sum_of_class_means").to(
        device)  # for map segmentation head

    model = SegnetTransformerLiftFuse(Z_cam=200, Y_cam=8, X_cam=200, Z_rad=Z, Y_rad=Y, X_rad=X, vox_util=None,
                                      use_pseudolidar=use_pseudolidar, use_metapseudolidar=use_metapseudolidar,
                                      use_shallow_metadata=use_shallow_metadata,
                                      use_pseudolidar_encoder=use_pseudolidar_encoder,
                                      do_rgbcompress=do_rgbcompress, encoder_type=encoder_type,
                                      pseudolidar_encoder_type=pseudolidar_encoder_type, rand_flip=False, train_task=train_task,
                                      init_query_with_image_feats=init_query_with_image_feats,
                                      use_obj_layer_only_on_map=use_obj_layer_only_on_map,
                                      do_feat_enc_dec=do_feat_enc_dec,
                                      use_multi_scale_img_feats=use_multi_scale_img_feats, num_layers=num_layers,
                                      combine_feat_init_w_learned_q=combine_feat_init_w_learned_q,
                                      use_rpn_pseudolidar=use_rpn_pseudolidar, use_pseudolidar_occupancy_map=use_pseudolidar_occupancy_map,
                                      freeze_dino=freeze_dino, learnable_fuse_query=learnable_fuse_query)

    model = model.to(device)
    model = torch.nn.DataParallel(model, device_ids=device_ids)
    parameters = list(model.parameters())

    # Counting trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable parameters: {trainable_params}')
    # Counting non-trainable parameters
    non_trainable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f'Non-trainable parameters: {non_trainable_params}')
    # Overall parameters
    total_params = trainable_params + non_trainable_params
    print('Total parameters (trainable + fixed)', total_params)

    # load checkpoint
    global_step = 0
    if init_dir:
        _ = saverloader.load(init_dir, model.module, ignore_load=ignore_load, is_DP=True, step=load_step)
        global_step = 0
        print('checkpoint loaded...')
    requires_grad(parameters, False)
    model.eval()
    print('model set to eval mode...')

    # Permanent half-precision weights for the frozen image encoder.
    # Eliminates the per-forward fp32→half weight cast that autocast otherwise does
    # on every call. Safe because freeze_dino=True and we are in eval-only mode.
    # Must run BEFORE torch.compile so the compiled graph traces half weights.
    # Must match VIS_EVAL_AUTOCAST_DTYPE (fp16/bf16) — otherwise activations and weights
    # diverge (e.g. batch_norm fails on fp16 weights + bf16 activations).
    if os.environ.get("VIS_EVAL_ENCODER_HALF", "1") == "1" and hasattr(model.module, 'encoder'):
        _enc_ac = os.environ.get("VIS_EVAL_AUTOCAST_DTYPE", "fp16").lower()
        if _enc_ac == "bf16":
            model.module.encoder.to(dtype=torch.bfloat16)
            print('[bf16] model.module.encoder cast to bf16 (frozen + eval only)')
        elif _enc_ac == "none":
            print('[fp32] autocast disabled — encoder stays fp32')
        else:
            model.module.encoder.half()
            print('[fp16] model.module.encoder cast to fp16 (frozen + eval only)')

    # Optional: torch.compile() — reduces graph overhead.
    # The model's forward contains per-stage torch.cuda.Event profiling. With
    # SEGMAM_DISABLE_PROFILING=1 those profiling calls become no-ops, which lets
    # Dynamo trace cleanly and lets reduce-overhead (cudagraph) capture work.
    # Targets via VIS_EVAL_COMPILE_TARGET ∈ {"encoder","all"}; default "encoder".
    if os.environ.get("VIS_EVAL_TORCH_COMPILE", "0") == "1":
        try:
            compile_mode = os.environ.get("VIS_EVAL_COMPILE_MODE", "default")
            target = os.environ.get("VIS_EVAL_COMPILE_TARGET", "encoder")

            # If the user wants whole-model compile, profiling MUST be disabled —
            # otherwise cuda.Event.elapsed_time() collides with the captured graph.
            if target == "all" and os.environ.get("SEGMAM_DISABLE_PROFILING", "0") != "1":
                print("[torch.compile] auto-setting SEGMAM_DISABLE_PROFILING=1 for whole-model compile.")
                os.environ["SEGMAM_DISABLE_PROFILING"] = "1"

            if target == "encoder" and hasattr(model.module, 'encoder'):
                print(f"[torch.compile] compiling model.module.encoder with mode={compile_mode!r} ...")
                model.module.encoder = torch.compile(model.module.encoder, mode=compile_mode)
            elif target == "encoder_mamba":
                # Compile the image encoder AND each mamba self-attn layer.
                # 진단 결과 lifting 단계의 63ms/layer dominant cost 가 self_mamba block
                # (causal_conv1d 미설치 → slow fallback path) 에서 발생함. torch.compile
                # 로 Triton/Inductor 가 작은 op 들을 fuse 하면 launch overhead 가 줄어든다.
                if hasattr(model.module, 'encoder'):
                    print(f"[torch.compile] compiling model.module.encoder with mode={compile_mode!r} ...")
                    model.module.encoder = torch.compile(model.module.encoder, mode=compile_mode)
                if hasattr(model.module, 'self_mamba_attn_layers_fuser'):
                    _n = len(model.module.self_mamba_attn_layers_fuser)
                    print(f"[torch.compile] compiling {_n} self_mamba layers with mode={compile_mode!r} ...")
                    for _i in range(_n):
                        model.module.self_mamba_attn_layers_fuser[_i] = torch.compile(
                            model.module.self_mamba_attn_layers_fuser[_i], mode=compile_mode)
            else:
                print(f"[torch.compile] compiling model.module (target={target!r}) with mode={compile_mode!r} ...")
                model.module = torch.compile(model.module, mode=compile_mode)
            print("[torch.compile] done. First forward will be slow (graph build); "
                  "use VIS_EVAL_WARMUP_FRAMES to skip those from timing.")
        except Exception as _ce:
            print(f"[torch.compile] WARN: compile failed, falling back to eager. ({_ce})")

    # tenssorRT 적용

    time_pool_ev = utils.misc.SimplePool(10000, version='np')
    eval_status = 'unsorted'

    # Initialize metric dictionaries
    # object dicts
    obj_metrics = {
        'obj_intersections': 0, 'obj_unions': 0, 'obj_0_20_intersections': 0, 'obj_0_20_unions': 0,
        'obj_20_35_intersections': 0, 'obj_20_35_unions': 0, 'obj_35_50_intersections': 0, 'obj_35_50_unions': 0
    }

    
    day_metrics = obj_metrics.copy()
    rain_metrics = obj_metrics.copy()
    night_metrics = obj_metrics.copy()
    
    global iou_labels
    global map_metrics
    
    # map dicts
    iou_labels = ['drivable_iou', 'carpark_iou', 'ped_cross_iou', 'walkway_iou', 'stop_line_iou',
                  'road_divider_iou', 'lane_divider_iou']
    


    #iou_labels = ['drivable_iou', 'carpark_iou', 'ped_cross_iou', 'walkway_iou']
    
    map_metrics = {
        'map_masks_intersections': torch.zeros(7, device=device),
        'map_masks_unions': torch.zeros(7, device=device),
        'map_masks_multi_ious_intersections': torch.zeros((7, 12), device=device),
        'map_masks_multi_ious_unions': torch.zeros((7, 12), device=device),
        'map_seg_thresholds': torch.zeros(12, device=device)
    }

    

    day_map_metrics = {k: v.clone() for k, v in map_metrics.items()}
    rain_map_metrics = {k: v.clone() for k, v in map_metrics.items()}
    night_map_metrics = {k: v.clone() for k, v in map_metrics.items()}

    map_ious = {}
    day_map_ious = {}
    rain_map_ious = {}
    night_map_ious = {}
    mean_map_iou = 0.0
    day_mean_map_iou = 0.0
    rain_mean_map_iou = 0.0
    night_mean_map_iou = 0.0

    

    while global_step < max_iters:
        global_step += 1
        
        if do_drn_val_split:
            if global_step <= val_day_len:
                eval_status = "DAY"
            if val_day_len < global_step <= (val_day_len + val_rain_len):
                eval_status = "RAIN"
            if global_step > val_day_len + val_rain_len:
                eval_status = "NIGHT"

        iter_start_time = time.time()
        read_start_time = time.time()

        sw = utils.improc.Summ_writer(
            writer=writer,
            global_step=global_step,
            log_freq=log_freq,
            fps=2,
            scalar_freq=int(log_freq / 2),
            just_gif=True)

        try:
            # print('grab next sample...')
            sample = next(iterloader)
            #map_seg_loss_fn.scene_num = global_step
            print('got SCENE: ' + str(global_step))
        except StopIteration:
            break

        read_time = time.time() - read_start_time
        
        # global egopose_c 
        # egopose_c = EgoPose_Class()

        with torch.no_grad():
            (total_loss, metrics, scene_inference_total_t,
             scene_inference_count) = run_model(
                dataloader, global_step, model, seg_loss_fn, map_seg_loss_fn, sample,
                img_dir, device, use_pseudolidar_encoder, pseudolidar_encoder_type, train_task,
                use_shallow_metadata=use_shallow_metadata,
                use_obj_layer_only_on_map=use_obj_layer_only_on_map, model_name=model_name, eval_status="All",ncams=ncams,
                iou_csv_dir=iou_csv_dir)

        # 씬별 추론 시간 CSV에 한 줄 append (paper-style forward pass)
        if scene_inference_count > 0:
            scene_mean_inference_ms = (scene_inference_total_t / scene_inference_count) * 1000.0
            scene_row = pd.DataFrame([{
                "scene_index": int(global_step),
                "eval_status": eval_status,
                "num_frames": int(scene_inference_count),
                "total_inference_time_s": round(scene_inference_total_t, 6),
                "mean_inference_time_ms": round(scene_mean_inference_ms, 3),
            }])
            write_header = (not os.path.exists(per_scene_time_csv)) or (os.path.getsize(per_scene_time_csv) == 0)
            scene_row.to_csv(per_scene_time_csv, mode="a", header=write_header, index=False)

            total_inference_time_s += scene_inference_total_t
            total_inference_frames += scene_inference_count
            total_inference_scenes += 1

        iter_time = time.time() - iter_start_time

        # range based iou clac
        # obj
        if train_task in ['both', 'object']:
            # Update overall metrics
            update_metrics(metric_prefix='obj', condition_metrics_dict=obj_metrics, metrics_model=metrics)
            update_range_metrics(metric_prefix='obj', range_metric_dict=obj_metrics, metrics_model=metrics)

            # Update day, rain, and night metrics
            if eval_status == "DAY":
                update_metrics('obj', day_metrics, metrics)
                update_range_metrics('obj', day_metrics, metrics)
            elif eval_status == "RAIN":
                update_metrics('obj', rain_metrics, metrics)
                update_range_metrics('obj', rain_metrics, metrics)
            elif eval_status == "NIGHT":
                update_metrics('obj', night_metrics, metrics)
                update_range_metrics('obj', night_metrics, metrics)

        # map
        if train_task in ['both', 'map']:
            # Calculate IOUs
            map_ious, mean_map_iou = update_and_calculate_map_metrics(eval_status='ALL', metrics=metrics,
                                                                      map_metrics=map_metrics,
                                                                      iou_labels=iou_labels)
            
            # print(f"Test map iou {mean_map_iou} ")
            # print(f"True Negative{tn}")
            # print(f"True Positive{tn}")
            # print(f"False Postive{tn}")
            # print(f"False Negative{tn}")
            

            # Update day, rain, and night map metrics
            # short version
            if eval_status == "DAY":
                day_map_ious, day_mean_map_iou = update_and_calculate_map_metrics(eval_status='DAY',
                                                                                  metrics=metrics,
                                                                                  map_metrics=day_map_metrics,
                                                                                  iou_labels=iou_labels)
            elif eval_status == "RAIN":
                rain_map_ious, rain_mean_map_iou = update_and_calculate_map_metrics(eval_status='RAIN',
                                                                                    metrics=metrics,
                                                                                    map_metrics=rain_map_metrics,
                                                                                    iou_labels=iou_labels)
            elif eval_status == "NIGHT":
                night_map_ious, night_mean_map_iou = update_and_calculate_map_metrics(eval_status='NIGHT',
                                                                                      metrics=metrics,
                                                                                      map_metrics=night_map_metrics,
                                                                                      iou_labels=iou_labels)

        time_pool_ev.update([iter_time])
        sw.summ_scalar('pooled/time_per_batch', time_pool_ev.mean())
        sw.summ_scalar('pooled/time_per_el', time_pool_ev.mean() / float(B))

        if train_task == 'object':
            print('%s; scene %04d/%d; rtime %.3f; itime %.2f; loss %.5f; iou_ev %.1f' % (
                model_name, global_step, max_iters, read_time, iter_time,
                total_loss.item(), obj_metrics['obj_iou']))

        if train_task == 'map':
            
                
            print('%s; scene %04d/%d; rtime %.3f; itime %.2f; loss %.5f; m_map_iou %.1f; driv %.1f; ' % (
                      model_name, global_step, max_iters, read_time, iter_time,
                      total_loss.item(), mean_map_iou, map_ious['drivable_iou'].item()))
            
            print('%s; scene %04d/%d; rtime %.3f; itime %.2f; loss %.5f; m_map_iou %.1f; driv %.1f; '
                  'carp %.1f; ped_cr %.1f; walkw %.1f;' % (
                      model_name, global_step, max_iters, read_time, iter_time,
                      total_loss.item(), mean_map_iou, map_ious['drivable_iou'].item(),
                      map_ious['carpark_iou'].item(), map_ious['ped_cross_iou'].item(), map_ious['walkway_iou'].item()))

        if train_task == 'both':
            print('%s; scene %04d/%d; eval_status: %s; rtime %.3f; itime %.2f; loss %.5f; iou_ev %.1f; '
                  'm_map_iou %.1f; driv %.1f; carp %.1f; ped_cr %.1f; walkw %.1f; stop %.1f; road %.1f; lane %.1f' % (
                    model_name, global_step, max_iters, eval_status, read_time, iter_time,
                    total_loss.item(), obj_metrics['obj_iou'], mean_map_iou, map_ious['drivable_iou'].item(),
                    map_ious['carpark_iou'].item(), map_ious['ped_cross_iou'].item(), map_ious['walkway_iou'].item(),
                    map_ious['stop_line_iou'].item(), map_ious['road_divider_iou'].item(),
                    map_ious['lane_divider_iou'].item()))
            
            

    # print final metrics in terminal
    display_final_results(train_task=train_task, dset=dset, obj_metrics=obj_metrics, day_metrics=day_metrics,
                          rain_metrics=rain_metrics, night_metrics=night_metrics, map_metrics=map_metrics,
                          day_map_metrics=day_map_metrics, rain_map_metrics=rain_map_metrics,
                          night_map_metrics=night_map_metrics, mean_map_iou=mean_map_iou, map_ious=map_ious,
                          day_mean_map_iou=day_mean_map_iou, day_map_ious=day_map_ious,
                          rain_mean_map_iou=rain_mean_map_iou, rain_map_ious=rain_map_ious,
                          night_mean_map_iou=night_mean_map_iou, night_map_ious=night_map_ious,
                          do_drn_val_split=do_drn_val_split)

    # 평균은 per_scene_inference_time.csv 에서 직접 계산 가능하므로 별도 CSV는 저장하지 않음
    if total_inference_frames > 0:
        mean_inference_ms_per_frame = (total_inference_time_s / total_inference_frames) * 1000.0
        print(f"[InferenceTime] mean inference time across {total_inference_scenes} scenes "
              f"({total_inference_frames} frames): {mean_inference_ms_per_frame:.3f} ms/frame")
    else:
        print("[InferenceTime] no inference frames recorded.")

    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run evaluation and visualization with model-specific config.')
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')

    args = parser.parse_args()

    # Load the config file
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    main(**config)
